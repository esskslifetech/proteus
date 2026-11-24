"""
Main simulation pipeline orchestration.

Layer ownership:
- L0 sequence: Rust core (event-sourced genome state, exact revert, diff)
- L1 variant semantics: Python (+ optional ML surrogates)
- L2 protein: Python (+ optional C++ FoldX-style ΔΔG kernel)
- L3 pathway: Python (+ optional C++ ODE integration)
- L4 tissue/neuro proxies: Python
- L5 viability: Python weighted gate
- V1-V7: runtime validation over the run artifacts

Core loop: load reference -> inject variants -> simulate the mutated and
reference scenarios across L1-L4 -> viability gate -> revert -> per-layer diff
-> report + evidence bundle.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import time
import blake3

from proteus.schema import (
    PipelineConfig,
    PipelineResult,
    SimulationReport,
    GenomeState,
    VariantEvent,
    ViabilityScore,
    LayerDiff,
    DiffEntry,
    EvidenceChain,
    ResourceRef,
    ContentHash,
)

from proteus.layers import (
    VariantSemanticsLayer,
    ProteinLayer,
    PathwayLayer,
    TissueLayer,
    ViabilityLayer,
    TimeDynamicODESolver,
    evaluate_swap_gates,
)
from proteus.layers.l5_viability import Verdict as LayerVerdict
from proteus.references.genome_loader import load_reference_genome, load_ortholog_mapper
from proteus.validation import RuntimeValidator
from proteus.cpp_adapter import cpp_status

# Try to import Rust core adapter
try:
    from proteus.core_adapter import (
        RustGenomeStateAdapter,
        create_rust_genome_state,
        variant_py_to_rs,
        variant_rs_to_py,
        layer_diff_rs_to_py,
        RUST_CORE_AVAILABLE,
    )
    HAS_RUST_CORE = True
except ImportError:
    HAS_RUST_CORE = False


@dataclass
class LayerResult:
    """Result from a single simulation layer."""
    layer: str  # L0, L1, L2, L3, L4, L5
    success: bool
    data: dict[str, Any] = field(default_factory=dict)
    evidence: list[EvidenceChain] = field(default_factory=list)
    error: str | None = None


def _decode_sequence(data: Any) -> str:
    """Decode a genome sequence (bytes/list/str) into an uppercase ASCII string."""
    if isinstance(data, str):
        return data.upper()
    if isinstance(data, list):
        data = bytes(data)
    try:
        return bytes(data).decode("ascii", errors="replace").upper()
    except Exception:
        return ""


# Aggregate fields compared between the reference and mutated scenarios.
DIFF_FIELDS: dict[str, tuple[str, ...]] = {
    "L1": ("max_pathogenicity", "mean_pathogenicity", "max_severity"),
    "L2": ("max_delta_delta_g", "mean_delta_delta_g", "max_confidence"),
    "L3": ("biomass_flux", "atp_flux", "atp_pool"),
    "L4": ("systemic_score", "mean_tissue_atp"),
}


class SimulationPipeline:
    """Orchestrates the multi-layer simulation pipeline."""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.run_id = str(uuid.uuid4())
        self.start_time = time.time_ns()
        self.layer_results: list[LayerResult] = []
        self.baseline_layer_results: list[LayerResult] = []
        self._rust_baseline: RustGenomeStateAdapter | None = None
        self._rust_mutated: RustGenomeStateAdapter | None = None
        self._rust_reverted: RustGenomeStateAdapter | None = None

    # ------------------------------------------------------------------ run
    async def run(self) -> PipelineResult:
        """Execute the full simulation pipeline."""
        try:
            baseline = await self._load_baseline()
            mutated = await self._apply_mutations(baseline)

            # Reference scenario (no variants) for cross-layer diffs.
            await self._run_layer_L1([])
            await self._run_layer_L2([])
            await self._run_layer_L3([])
            await self._run_layer_L4([])
            self.baseline_layer_results = self.layer_results
            self.layer_results = []

            # Mutated scenario.
            await self._run_layer_L0(baseline, mutated)
            await self._run_layer_L1(mutated.overlays)
            await self._run_layer_L2(mutated.overlays)
            await self._run_layer_L3(mutated.overlays)
            await self._run_layer_L4(mutated.overlays)

            # L3b: Time-dynamic ODE simulation
            time_dynamic = await self._run_layer_L3b(mutated.overlays)

            # Swap-specific gates
            swap_gate_results = await self._run_swap_gates(mutated.overlays)

            viability = await self._run_viability_gate(mutated, time_dynamic, swap_gate_results)

            # Event-sourced revert is always attempted: it is the evidence that
            # the mutation overlay can be removed exactly, independent of the
            # viability verdict.
            reverted = await self._revert_mutations(mutated)
            diffs = await self._compute_diffs(baseline, mutated, reverted)

            # Hysteresis: re-simulate physiology after the genome is reverted.
            # The genome hash returns to baseline, but irreversible damage does
            # not, so the post-revert phenotype can differ from baseline.
            revert_hysteresis = await self._run_revert_hysteresis(time_dynamic)

            report = await self._build_report(
                baseline=baseline,
                mutated=mutated,
                reverted=reverted,
                viability=viability,
                diffs=diffs,
                time_dynamic=time_dynamic,
                swap_gate_results=swap_gate_results,
                revert_hysteresis=revert_hysteresis,
            )
            artifacts = self._emit_reports(report)

            return PipelineResult(
                run_id=self.run_id,
                success=True,
                error="",
                report=report,
                artifacts=artifacts,
            )

        except Exception as e:
            empty = GenomeState(
                reference=ResourceRef(hash=ContentHash(algorithm="", digest=b""), uri=""),
                overlays=[],
                head_index=0,
            )
            return PipelineResult(
                run_id=self.run_id,
                success=False,
                error=str(e),
                report=SimulationReport(
                    simulation_id=self.run_id,
                    timestamp_ns=self.start_time,
                    baseline=empty,
                    mutated=empty,
                    reverted=empty,
                    applied_variants=[],
                    viability=ViabilityScore(
                        composite=0.0, layer_scores={}, weights={}, failed_gates=[],
                        verdict=ViabilityScore.Verdict.NON_VIABLE, confidence=0.0,
                    ),
                    diffs=[],
                    evidence=[],
                    provenance_manifest=ContentHash(algorithm="", digest=b""),
                    validation_results={},
                ),
                artifacts={},
            )

    # ------------------------------------------------------------- baseline
    async def _load_baseline(self) -> GenomeState:
        """Load reference genome and create baseline state using Rust core."""
        if HAS_RUST_CORE:
            self._rust_baseline = create_rust_genome_state(self.config.reference_genome_uri)
            return self._rust_to_py_genome_state(self._rust_baseline)
        return GenomeState(
            reference=ResourceRef(
                hash=ContentHash(algorithm="blake3", digest=b"placeholder"),
                uri=self.config.reference_genome_uri,
            ),
            overlays=[],
            head_index=0,
        )

    @staticmethod
    def _sort_variants_for_apply(variants: list[VariantEvent]) -> list[VariantEvent]:
        """Sort variants by descending start coordinate so later positions are applied first.

        This prevents coordinate shifts from insertions/deletions affecting
        the application of variants at earlier genomic positions.
        """
        return sorted(
            variants,
            key=lambda v: (v.contig, -int(v.start)),
        )

    async def _apply_mutations(self, baseline: GenomeState) -> GenomeState:
        """Apply variant events to create mutated state using Rust core."""
        sorted_variants = self._sort_variants_for_apply(self.config.variants)
        if HAS_RUST_CORE and self._rust_baseline:
            self._rust_mutated = self._rust_baseline.apply_batch(sorted_variants)
            return self._rust_to_py_genome_state(self._rust_mutated)
        return GenomeState(
            reference=baseline.reference,
            overlays=sorted_variants,
            head_index=len(sorted_variants),
        )

    def _rust_to_py_genome_state(self, rust_adapter: RustGenomeStateAdapter) -> GenomeState:
        return GenomeState(
            reference=rust_adapter.reference,
            overlays=rust_adapter.overlays,
            head_index=rust_adapter.head_index,
        )

    def _layer_config(self) -> dict[str, Any]:
        cfg: dict[str, Any] = dict(self.config.layer_configs or {})
        cfg.setdefault("seed", 42)
        return cfg

    # ------------------------------------------------------------------ L0
    async def _run_layer_L0(self, baseline: GenomeState, mutated: GenomeState) -> None:
        """L0: Sequence layer - exact sequence diff using Rust core."""
        if HAS_RUST_CORE and self._rust_baseline and self._rust_mutated:
            rust_diff = self._rust_baseline.diff_raw(self._rust_mutated)
            py_diff = layer_diff_rs_to_py(rust_diff)

            seq_changes: dict[str, dict[str, Any]] = {}
            sequence_previews: dict[str, dict[str, Any]] = {}
            max_preview_bp = 5000
            for contig in self._rust_baseline._rs_state.reference.contig_names():
                baseline_seq = self._rust_baseline.get_sequence(contig)
                mutated_seq = self._rust_mutated.get_sequence(contig)
                baseline_bytes = bytes(baseline_seq) if isinstance(baseline_seq, list) else baseline_seq
                mutated_bytes = bytes(mutated_seq) if isinstance(mutated_seq, list) else mutated_seq
                if baseline_bytes != mutated_bytes:
                    # Record changed positions only (full contigs can be large).
                    positions = [
                        i
                        for i, (a, b) in enumerate(zip(baseline_bytes, mutated_bytes))
                        if a != b
                    ]
                    seq_changes[contig] = {
                        "baseline_length": len(baseline_bytes),
                        "mutated_length": len(mutated_bytes),
                        "changed_positions_0based": positions[:1000],
                        "changed_position_count": len(positions)
                        + abs(len(baseline_bytes) - len(mutated_bytes)),
                    }

                # Human-readable sequence windows for the DNA visualizations.
                # Small contigs are shown in full; larger references get a
                # window around each injected variant so reports stay compact.
                baseline_text = _decode_sequence(baseline_bytes)
                mutated_text = _decode_sequence(mutated_bytes)
                windows: list[dict[str, Any]] = []
                if len(baseline_text) <= max_preview_bp and len(mutated_text) <= max_preview_bp:
                    windows.append({
                        "start_1based": 1,
                        "end_1based": len(baseline_text),
                        "baseline": baseline_text,
                        "mutated": mutated_text,
                        "label": "full contig",
                    })
                else:
                    variant_positions = sorted({
                        max(1, int(v.start))
                        for v in self.config.variants
                        if v.contig == contig
                    })
                    for pos in variant_positions[:6]:
                        start0 = max(0, pos - 1 - 30)
                        end0 = min(len(baseline_text), pos + 29)
                        mutated_end0 = min(len(mutated_text), end0 + max(0, len(mutated_text) - len(baseline_text)))
                        windows.append({
                            "start_1based": start0 + 1,
                            "end_1based": end0,
                            "baseline": baseline_text[start0:end0],
                            "mutated": mutated_text[start0:mutated_end0],
                            "label": f"around {contig}:{pos}",
                        })
                sequence_previews[contig] = {
                    "contig": contig,
                    "baseline_length": len(baseline_text),
                    "mutated_length": len(mutated_text),
                    "windows": windows,
                }

            self.layer_results.append(LayerResult(
                layer="L0",
                success=True,
                data={
                    "variants_applied": len(mutated.overlays),
                    "sequence_changes": seq_changes,
                    "sequence_previews": sequence_previews,
                    "variant_diff": {
                        "added": py_diff.added,
                        "removed": py_diff.removed,
                        "changed": {k: {"before": v.before, "after": v.after}
                                    for k, v in py_diff.changed.items()},
                    },
                },
            ))
        else:
            self.layer_results.append(LayerResult(
                layer="L0",
                success=True,
                data={"variants_applied": len(mutated.overlays)},
            ))

    # ------------------------------------------------------------------ L1
    async def _run_layer_L1(self, variants: list[Any]) -> None:
        """L1: Variant semantics - consequence, pathogenicity."""
        try:
            layer = VariantSemanticsLayer(config=self._layer_config())
            annotations_by_id = {
                str(getattr(v, "id", "")): (getattr(v, "annotations", {}) or {})
                for v in self.config.variants
            }
            effects = []
            for var in variants:
                annotations = (
                    annotations_by_id.get(str(getattr(var, "id", "")))
                    or getattr(var, "annotations", {})
                    or {}
                )
                effects.append(layer.predict(var, {
                    "gene_id": annotations.get("gene_id"),
                    "transcript_id": annotations.get("transcript_id"),
                }))
            agg = layer.aggregate_scores(effects)

            variant_details = []
            for effect in effects:
                variant_details.append({
                    "id": effect.variant_id,
                    "gene_id": effect.gene_id,
                    "transcript_id": effect.transcript_id,
                    "consequence": effect.consequence.value if effect.consequence else None,
                    "cds_position": effect.cds_position,
                    "protein_position": effect.protein_position,
                    "amino_acid_change": effect.amino_acid_change,
                    "pathogenicity_score": effect.pathogenicity_score,
                    "splice_score_delta": effect.splice_score_delta,
                })

            self.layer_results.append(LayerResult(
                layer="L1",
                success=True,
                data={"variant_details": variant_details, "aggregate": agg},
            ))
        except Exception as e:
            self.layer_results.append(LayerResult(layer="L1", success=False, data={}, error=str(e)))

    # ------------------------------------------------------------------ L2
    async def _run_layer_L2(self, variants: list[Any]) -> None:
        """L2: Protein structure - stability, domains."""
        try:
            layer = ProteinLayer(config=self._layer_config())
            l1_data = next((r.data for r in self.layer_results if r.layer == "L1"), {})
            variant_details = l1_data.get("variant_details", [])

            effects = []
            for var in variants:
                detail = next((d for d in variant_details if d.get("id") == var.id), {})
                effects.append(layer.predict(var, {
                    "amino_acid_change": detail.get("amino_acid_change"),
                    "uniprot_id": detail.get("gene_id"),
                }))

            agg = layer.aggregate_scores(effects)
            effect_details = [{
                "id": e.variant_id,
                "uniprot_id": e.uniprot_id,
                "domain_id": e.domain_id,
                "delta_delta_g": e.delta_delta_g,
                "confidence": e.confidence,
                "method": e.method.value,
                "structure_hash": e.structure_hash,
                "domain_impacts": [
                    {
                        "domain_id": di.domain_id,
                        "delta_delta_g": di.delta_delta_g,
                        "active_site_disrupted": di.active_site_disrupted,
                        "binding_interface_disrupted": di.binding_interface_disrupted,
                    }
                    for di in e.domain_impacts
                ],
            } for e in effects]

            self.layer_results.append(LayerResult(
                layer="L2",
                success=True,
                data={"effects": effect_details, "aggregate": agg},
            ))
        except Exception as e:
            self.layer_results.append(LayerResult(layer="L2", success=False, data={}, error=str(e)))

    # ------------------------------------------------------------------ L3
    async def _run_layer_L3(self, variants: list[Any]) -> None:
        """L3: Pathway/Cell - metabolism, GRN."""
        try:
            layer = PathwayLayer(config=self._layer_config())
            l1_data = next((r.data for r in self.layer_results if r.layer == "L1"), {})
            variant_details = l1_data.get("variant_details", [])

            results = []
            for var in variants:
                detail = next((d for d in variant_details if d.get("id") == var.id), {})
                gene_id = detail.get("gene_id")
                # LIMITATION: Without a transcript model (GTF/GFF), we cannot map
                # variants to genes. The test contig has no gene annotation.
                # For demonstration, we exercise representative metabolic genes
                # when gene_id is provided; otherwise use core glycolysis genes.
                affected = [gene_id] if gene_id else ["GAPDH", "PGK1", "PKM"]
                results.append(layer.predict(var, {"affected_genes": affected}))

            agg = layer.aggregate_scores(results)
            pathway_data = results[0][0] if results else None
            grn_data = results[0][1] if results else None

            self.layer_results.append(LayerResult(
                layer="L3",
                success=True,
                data={
                    "aggregate": agg,
                    "pathway": {
                        "model_id": pathway_data.model_id,
                        "status": pathway_data.status,
                        "flux": pathway_data.flux,
                        "objective_value": pathway_data.objective_value,
                        "metabolite_concentration": pathway_data.metabolite_concentration,
                    } if pathway_data else {},
                    "grn": {
                        "network_id": grn_data.network_id,
                        "expression": grn_data.expression,
                        "activity": grn_data.activity,
                    } if grn_data else {},
                },
            ))
        except Exception as e:
            self.layer_results.append(LayerResult(layer="L3", success=False, data={}, error=str(e)))

    # ------------------------------------------------------------------ L4
    async def _run_layer_L4(self, variants: list[Any]) -> None:
        """L4: Tissue/Organ - physiology, neuro proxies."""
        try:
            layer = TissueLayer(config=self._layer_config())
            l2_agg = next((r.data.get("aggregate") for r in self.layer_results if r.layer == "L2"), {})
            l3_agg = next((r.data.get("aggregate") for r in self.layer_results if r.layer == "L3"), {})

            if not variants:
                # Reference physiology: all tissues at baseline, systemic score 1.
                self.layer_results.append(LayerResult(
                    layer="L4",
                    success=True,
                    data={
                        "aggregate": {
                            "systemic_score": 1.0,
                            "mean_tissue_atp": 3.1,
                            "systemic": {},
                            "neuro_proxies": {},
                            "tissue_count": 10,
                        },
                        "tissues": {},
                        "limitations": layer.neuro_proxy_limitations(),
                    },
                ))
                return

            organism_state = layer.predict(variants[0], {"l2_result": l2_agg, "l3_result": l3_agg})
            agg = layer.aggregate_scores(organism_state)

            tissue_data = {
                tissue.value: {
                    "markers": state.markers,
                    "neuro_proxies": {k.value: v for k, v in state.neuro_proxies.items()},
                }
                for tissue, state in organism_state.tissues.items()
            }

            self.layer_results.append(LayerResult(
                layer="L4",
                success=True,
                data={
                    "aggregate": agg,
                    "systemic": organism_state.systemic,
                    "tissues": tissue_data,
                    "limitations": layer.neuro_proxy_limitations(),
                },
            ))
        except Exception as e:
            self.layer_results.append(LayerResult(layer="L4", success=False, data={}, error=str(e)))

    # ------------------------------------------------------------------ L3b: Time-dynamic ODE
    async def _run_layer_L3b(self, variants: list[Any]) -> dict[str, Any]:
        """L3b: Time-dynamic simulation with mRNA/protein turnover and irreversible damage."""
        try:
            layer = PathwayLayer(config=self._layer_config())
            l1_data = next((r.data for r in self.layer_results if r.layer == "L1"), {})
            variant_details = l1_data.get("variant_details", [])

            # Get pathway and GRN states for the first variant (or use reference)
            if variants:
                detail = next((d for d in variant_details if d.get("id") == variants[0].id), {})
                gene_id = detail.get("gene_id")
                affected = [gene_id] if gene_id else ["GAPDH", "PGK1", "PKM"]
            else:
                affected = []

            # Run steady-state pathway first
            pathway_state, grn_state = layer.predict(variants[0] if variants else None, {"affected_genes": affected})

            # Run time-dynamic simulation
            time_dynamic_state = layer.run_time_dynamic(pathway_state, grn_state, affected)

            # Extract key results
            damage_snapshot = {
                dtype.value: state.snapshot()
                for dtype, state in time_dynamic_state.damage_states.items()
            }
            return {
                "time_points": time_dynamic_state.time_points,
                "energy_charge": time_dynamic_state.energy_charge,
                "atp_concentration": time_dynamic_state.atp_concentration,
                "protein_concentrations": time_dynamic_state.protein_concentrations,
                "irreversible_events": time_dynamic_state.irreversible_events,
                "point_of_no_return_h": time_dynamic_state.damage_states[
                    min(time_dynamic_state.damage_states, 
                        key=lambda k: time_dynamic_state.damage_states[k].point_of_no_return or float('inf'))
                ].point_of_no_return if any(d.point_of_no_return for d in time_dynamic_state.damage_states.values()) else None,
                "survival_window_h": min(
                    [d.point_of_no_return for d in time_dynamic_state.damage_states.values() if d.point_of_no_return],
                    default=None
                ),
                "tissue_viability": time_dynamic_state.tissue_viability,
                "damage_snapshot": damage_snapshot,
                "residual_damage": {
                    dtype.value: state.effective_severity
                    for dtype, state in time_dynamic_state.damage_states.items()
                    if state.is_irreversible
                },
                "hysteresis_index": time_dynamic_state.hysteresis_index,
            }
        except Exception as e:
            import warnings
            warnings.warn(f"Time-dynamic simulation failed: {e}")
            return {
                "time_points": [],
                "energy_charge": [],
                "atp_concentration": [],
                "protein_concentrations": {},
                "irreversible_events": [],
                "point_of_no_return_h": None,
                "survival_window_h": None,
                "tissue_viability": {},
                "damage_snapshot": {},
                "residual_damage": {},
                "hysteresis_index": 0.0,
            }

    async def _run_revert_hysteresis(self, time_dynamic: dict[str, Any] | None) -> dict[str, Any]:
        """Re-simulate after the genome revert to expose residual damage.

        The genome is restored exactly, but the time-dynamic layer carries
        irreversible damage forward.  A non-zero hysteresis index therefore
        means: same DNA, different physiology.
        """
        if not time_dynamic or not time_dynamic.get("damage_snapshot"):
            return {
                "available": False,
                "genome_revert_exact": None,
                "phenotype_matches_baseline": True,
                "hysteresis_index": 0.0,
                "residual_damage": {},
                "persistent_events": [],
                "post_revert_tissue_viability": {},
                "post_revert_energy_charge": [],
            }

        try:
            layer = PathwayLayer(config=self._layer_config())
            # Healthy post-revert pathway state: no variants, no knockouts.
            pathway_state, grn_state = layer.predict(None, {"affected_genes": []})
            post = layer.run_time_dynamic(
                pathway_state,
                grn_state,
                [],
                time_span=(0, 72),
                initial_damage=time_dynamic.get("damage_snapshot", {}),
                initial_events=time_dynamic.get("irreversible_events", []),
                is_post_revert=True,
            )
            residual = {
                dtype.value: state.effective_severity
                for dtype, state in post.damage_states.items()
                if state.is_irreversible
            }
            tissue_means = {
                tissue: (sum(values) / len(values) if values else 1.0)
                for tissue, values in post.tissue_viability.items()
            }
            return {
                "available": True,
                "genome_revert_exact": None,  # filled in by _build_report
                "phenotype_matches_baseline": post.hysteresis_index <= 1e-9,
                "hysteresis_index": post.hysteresis_index,
                "residual_damage": residual,
                "persistent_events": list(time_dynamic.get("irreversible_events", [])),
                "post_revert_tissue_viability": tissue_means,
                "post_revert_energy_charge": post.energy_charge,
                "time_points": post.time_points,
                "solver_status": post.solver_status,
            }
        except Exception as e:
            import warnings
            warnings.warn(f"Post-revert hysteresis simulation failed: {e}")
            return {
                "available": False,
                "genome_revert_exact": None,
                "phenotype_matches_baseline": True,
                "hysteresis_index": 0.0,
                "residual_damage": {},
                "persistent_events": [],
                "post_revert_tissue_viability": {},
                "post_revert_energy_charge": [],
            }

    # ------------------------------------------------------------------ Swap Gates
    async def _run_swap_gates(self, variants: list[Any]) -> dict[str, Any]:
        """Evaluate swap-specific gates for cross-species simulation."""
        try:
            # Build context for swap gates from variant and layer data
            l1_data = next((r.data for r in self.layer_results if r.layer == "L1"), {})
            variant_details = l1_data.get("variant_details", [])
            
            # Get gene IDs from variants
            gene_ids = []
            for var in variants:
                detail = next((d for d in variant_details if d.get("id") == var.id), {})
                if detail.get("gene_id"):
                    gene_ids.append(detail["gene_id"])
            
            # Default to human if no specific context
            context = {
                "species": "human",
                "mtDNA_haplotype": "human",
                "nuclear_background": "human",
                "ortholog_identity": {},  # Would be filled from ortholog mapper
                "gene_presence": {gid: True for gid in gene_ids},
                "effector_type": "23BPG",
                "hb_genes": {},
                "reference_temp_c": 37.0,
                "target_temp_c": 37.0,
                "protein_stabilities": {},
                "source_karyotype": {"chr_count": 46, "sex_system": "XY"},
                "target_karyotype": {"chr_count": 46, "sex_system": "XY"},
                "variant_chromosome": None,
                "rbc_type": "enucleated",
                "simulation_aspects": ["oxygen_delivery", "gene_expression", "lifespan"],
                "source_mhc": {"class_I": ["HLA-A", "HLA-B", "HLA-C"], "class_II": ["HLA-DP", "HLA-DQ", "HLA-DR"]},
                "target_mhc": {"class_I": ["HLA-A", "HLA-B", "HLA-C"], "class_II": ["HLA-DP", "HLA-DQ", "HLA-DR"]},
                "peptides_of_interest": [],
            }
            
            # Try to load reference genome and orthologs for better context
            try:
                ref = load_reference_genome("GRCh38")
                ortho = load_ortholog_mapper("Corvus_cornix")
                # Could populate ortholog_identity from ortho
            except Exception:
                pass
            
            return evaluate_swap_gates(context)
        except Exception as e:
            import warnings
            warnings.warn(f"Swap gate evaluation failed: {e}")
            return {}

    # ------------------------------------------------------------------ L5
    async def _run_viability_gate(self, mutated: GenomeState, time_dynamic: dict[str, Any] | None = None, swap_gate_results: dict[str, Any] | None = None) -> ViabilityScore:
        """L5: Viability decision layer (weighted gate over L1-L4)."""
        def agg(layer: str) -> dict[str, Any]:
            return next((r.data.get("aggregate", {}) for r in self.layer_results if r.layer == layer), {})

        # A failed layer contributes a deliberately poor score
        def agg_or_fail(layer: str) -> dict[str, Any]:
            data = agg(layer)
            if data:
                return data
            result = next((r for r in self.layer_results if r.layer == layer), None)
            if result is not None and not result.success:
                return {"max_pathogenicity": 1.0} if layer == "L1" else (
                    {"max_delta_delta_g": 30.0} if layer == "L2" else (
                        {"biomass_flux": 0.0} if layer == "L3" else {"systemic_score": 0.0}))
            return data

        layer = ViabilityLayer(config={
            "weights": self.config.viability_weights or ViabilityLayer.DEFAULT_WEIGHTS,
        })
        result = layer.evaluate(
            agg_or_fail("L1"), agg_or_fail("L2"), agg_or_fail("L3"), agg_or_fail("L4"),
            time_dynamic=time_dynamic,
            swap_gate_results=swap_gate_results,
        )

        verdict_map = {
            LayerVerdict.VIABLE: ViabilityScore.Verdict.VIABLE,
            LayerVerdict.NON_VIABLE: ViabilityScore.Verdict.NON_VIABLE,
            LayerVerdict.UNCERTAIN: ViabilityScore.Verdict.UNCERTAIN,
        }

        return ViabilityScore(
            composite=result.composite,
            layer_scores=result.layer_scores,
            weights=result.weights,
            failed_gates=result.failed_gates,
            verdict=verdict_map.get(result.verdict, ViabilityScore.Verdict.UNCERTAIN),
            confidence=result.confidence,
            survival_window_h=result.survival_window_h,
            point_of_no_return_h=result.point_of_no_return_h,
            irreversible_events=result.irreversible_events,
            swap_gate_results=result.swap_gate_results,
            hard_gate_failures=result.hard_gate_failures,
        )

    # -------------------------------------------------------------- revert
    async def _revert_mutations(self, mutated: GenomeState) -> GenomeState:
        """Revert to baseline (exact revert) using Rust core."""
        if HAS_RUST_CORE and self._rust_mutated:
            self._rust_reverted = self._rust_mutated.revert_all()
            return self._rust_to_py_genome_state(self._rust_reverted)
        return GenomeState(reference=mutated.reference, overlays=[], head_index=0)

    # ---------------------------------------------------------------- diff
    async def _compute_diffs(
        self,
        baseline: GenomeState,
        mutated: GenomeState,
        reverted: GenomeState,
    ) -> list[LayerDiff]:
        """Compute per-layer diffs between the reference and mutated scenarios."""
        diffs: list[LayerDiff] = []

        # L0: exact sequence/variant diff from the Rust core.
        if HAS_RUST_CORE and self._rust_baseline and self._rust_mutated:
            rust_diff = self._rust_baseline.diff_raw(self._rust_mutated)
            l0_diff = layer_diff_rs_to_py(rust_diff)
            l0_diff.layer = "L0"
            diffs.append(l0_diff)
        else:
            diffs.append(LayerDiff(layer="L0", added={}, removed={}, changed={}))

        # L1-L4: numeric aggregate deltas between reference and mutated runs.
        baseline_by_layer = {r.layer: r.data.get("aggregate", {}) or {}
                             for r in self.baseline_layer_results}
        mutated_by_layer = {r.layer: r.data.get("aggregate", {}) or {}
                            for r in self.layer_results}

        for layer_name, fields in DIFF_FIELDS.items():
            before = baseline_by_layer.get(layer_name, {})
            after = mutated_by_layer.get(layer_name, {})
            changed: dict[str, DiffEntry] = {}
            for field_name in fields:
                b = before.get(field_name)
                a = after.get(field_name)
                if b is None and a is None:
                    continue
                try:
                    bf = float(b if b is not None else 0.0)
                    af = float(a if a is not None else 0.0)
                except (TypeError, ValueError):
                    continue
                magnitude = abs(af - bf)
                significance = "none"
                if magnitude > 0.0:
                    significance = "high" if magnitude > 0.25 else "moderate"
                changed[field_name] = DiffEntry(
                    before=f"{bf:.6g}",
                    after=f"{af:.6g}",
                    magnitude=magnitude,
                    significance=significance,
                )
            diffs.append(LayerDiff(layer=layer_name, added={}, removed={}, changed=changed))

        return diffs

    # ------------------------------------------------------------- reports
    def _emit_reports(self, report: SimulationReport) -> dict[str, ContentHash]:
        """Write the HTML/PDF/LaTeX/JSON-LD/RO-Crate bundle and hash each artifact."""
        if not self.config.produce_report:
            return {}

        import blake3

        from proteus.report import ReportGenerator

        output_dir = Path(self.config.output_dir or "output") / self.run_id
        viz_paths: dict[str, Path] = {}

        # Generate visualizations before the report so static figures can be
        # embedded in the HTML/PDF/RO-Crate bundle. This is best-effort; can be
        # disabled for fast runs such as CI pipelines.
        if self.config.generate_visualizations:
            try:
                from proteus.visualization import (
                    VIEW_TITLES,
                    create_comparison_audio,
                    create_dashboard,
                    create_genome_views,
                    create_gif,
                )

                viz_dir = output_dir / "visualizations"
                viz_dir.mkdir(parents=True, exist_ok=True)

                # Genome/molecule figures: show the actual sequence, codon
                # changes, layer engines, reference inputs, and pathway state.
                viz_paths.update(create_genome_views(report, viz_dir))
                # Existing multi-panel dashboard.
                viz_paths.update(create_dashboard(report, viz_dir))
                # Audio and animations (artifact-only, not embedded in reports).
                viz_paths.update(create_comparison_audio(report, viz_dir))
                viz_paths.update(create_gif(report, viz_dir))
            except Exception as e:
                # Visualization is best-effort; don't fail the pipeline.
                import warnings

                warnings.warn(f"Visualization generation failed: {e}")

        if viz_paths:
            dashboard_titles = {
                "radar": "Viability radar by layer",
                "heatmap": "Layer diff heatmap (L0–L4)",
                "variants": "Variant effect summary",
                "energy": "Energy landscape / time-dynamic trace",
                "conservation": "Conservation and constraint profile",
                "validation": "Validation layer summary (V1–V7)",
                "composite": "Composite simulation dashboard",
                "genome": "Combined genome sonification",
                "mutation": "Mutation sonification",
                "pathway": "Pathway sonification",
                "combined": "Combined sonification",
                "layer_propagation": "Layer propagation animation",
                "mutation_effect": "Mutation effect animation",
            }
            report.visualizations = [  # type: ignore[attr-defined]
                {
                    "key": key,
                    "path": path.relative_to(output_dir).as_posix(),
                    "title": VIEW_TITLES.get(key) or dashboard_titles.get(key)
                    or key.replace("_", " ").title(),
                }
                for key, path in viz_paths.items()
                if path.suffix.lower() == ".png" and path.is_file()
            ]

        outputs = ReportGenerator(output_dir).generate_all(report, base_name="report")

        artifacts: dict[str, ContentHash] = {}
        for name, path in outputs.items():
            target = path if path.is_file() else path / "ro-crate-metadata.json"
            if target.is_file():
                digest = blake3.blake3(target.read_bytes()).hexdigest()
                artifacts[name] = ContentHash(
                    algorithm="blake3", digest=bytes.fromhex(digest)
                )

        for name, path in viz_paths.items():
            if path.is_file():
                digest = blake3.blake3(path.read_bytes()).hexdigest()
                artifacts[f"viz_{name}"] = ContentHash(
                    algorithm="blake3", digest=bytes.fromhex(digest)
                )

        return artifacts

    # ------------------------------------------------------------- report
    async def _build_report(
        self,
        baseline: GenomeState,
        mutated: GenomeState,
        reverted: GenomeState | None,
        viability: ViabilityScore,
        diffs: list[LayerDiff],
        time_dynamic: dict[str, Any] | None = None,
        swap_gate_results: dict[str, Any] | None = None,
        revert_hysteresis: dict[str, Any] | None = None,
    ) -> SimulationReport:
        """Build the final simulation report with runtime validation V1-V7."""
        validator = RuntimeValidator(enabled=self.config.enabled_validations)
        checks = validator.validate(
            variants=list(self.config.variants),
            layer_results=self.layer_results,
            viability=viability,
            baseline=baseline,
            mutated=mutated,
            reverted=reverted,
            rust_baseline=self._rust_baseline,
            rust_mutated=self._rust_mutated,
            rust_reverted=self._rust_reverted,
        )
        compact = validator.compact(checks)
        validation_details = validator.details(checks)

        # Evidence: one chain per layer and per validation check, with proper hash chaining
        evidence: list[EvidenceChain] = []
        prev_output_hash: str = ""

        # Baseline state hash (input to the whole pipeline)
        if HAS_RUST_CORE and self._rust_baseline:
            baseline_state_hash = self._rust_baseline.hash_bytes()
        else:
            baseline_state_hash = baseline.hash_bytes()
        prev_output_hash = baseline_state_hash.hex()

        for layer_result in self.layer_results:
            evidence.extend(layer_result.evidence)
            # Compute a hash of this layer's output data
            layer_output = str(sorted(layer_result.data.items())).encode()
            layer_output_hash = blake3.blake3(layer_output).hexdigest()
            
            evidence.append(EvidenceChain(
                claim=f"{layer_result.layer} {'succeeded' if layer_result.success else 'failed'}",
                tool="proteus.layers",
                tool_hash=ContentHash(algorithm="blake3", digest=blake3.blake3(b"proteus.layers").digest()),
                params={
                    "layer": layer_result.layer,
                    "error": layer_result.error or "",
                    "keys": ",".join(sorted(layer_result.data.keys())),
                },
                input_hash=ContentHash(algorithm="blake3", digest=bytes.fromhex(prev_output_hash)) if prev_output_hash else ContentHash(),
                output_hash=ContentHash(algorithm="blake3", digest=bytes.fromhex(layer_output_hash)),
                upstream_ref=prev_output_hash,
                confidence=(EvidenceChain.Confidence.HIGH if layer_result.success
                            else EvidenceChain.Confidence.LOW),
                validation_passes=[],
            ))
            prev_output_hash = layer_output_hash

        for name, check in checks.items():
            check_output = f"{name}:{check.status}:{check.message}".encode()
            check_output_hash = blake3.blake3(check_output).hexdigest()
            
            evidence.append(EvidenceChain(
                claim=f"{name}: {check.message}",
                tool="proteus.runtime_validation",
                tool_hash=ContentHash(algorithm="blake3", digest=blake3.blake3(b"proteus.runtime_validation").digest()),
                params={
                    "layer": name,
                    "status": check.status,
                    "details": str(check.details)[:2000],
                },
                input_hash=ContentHash(algorithm="blake3", digest=bytes.fromhex(prev_output_hash)) if prev_output_hash else ContentHash(),
                output_hash=ContentHash(algorithm="blake3", digest=bytes.fromhex(check_output_hash)),
                upstream_ref=prev_output_hash,
                confidence=(EvidenceChain.Confidence.HIGH if check.status == "pass"
                            else EvidenceChain.Confidence.LOW),
                validation_passes=[name] if check.status == "pass" else [],
            ))
            prev_output_hash = check_output_hash

        cpp = cpp_status()
        cpp_output = f"cpp:{cpp['available']}".encode()
        cpp_output_hash = blake3.blake3(cpp_output).hexdigest()
        evidence.append(EvidenceChain(
            claim=f"C++ extension {'available' if cpp['available'] else 'unavailable'}",
            tool="proteus.cpp_adapter",
            tool_hash=ContentHash(algorithm="blake3", digest=blake3.blake3(b"proteus.cpp_adapter").digest()),
            params={
                "module_file": str(cpp.get("module_file") or ""),
                "import_error": str(cpp.get("import_error") or ""),
            },
            input_hash=ContentHash(algorithm="blake3", digest=bytes.fromhex(prev_output_hash)) if prev_output_hash else ContentHash(),
            output_hash=ContentHash(algorithm="blake3", digest=bytes.fromhex(cpp_output_hash)),
            upstream_ref=prev_output_hash,
            confidence=EvidenceChain.Confidence.HIGH,
        ))

        # Content-addressed hashes (hex strings)
        if HAS_RUST_CORE and self._rust_baseline:
            baseline_hash = self._rust_baseline.hash()
            mutated_hash = self._rust_mutated.hash() if self._rust_mutated else baseline_hash
            reverted_hash = self._rust_reverted.hash() if self._rust_reverted else baseline_hash
        else:
            baseline_hash = baseline.hash()
            mutated_hash = mutated.hash()
            reverted_hash = reverted.hash() if reverted else baseline_hash

        # Hysteresis evidence: genome revert exactness is compared against the
        # residual phenotype from the post-revert simulation.
        if revert_hysteresis is not None:
            revert_hysteresis = dict(revert_hysteresis)
            revert_hysteresis["genome_revert_exact"] = baseline_hash == reverted_hash
            revert_hysteresis["genome_baseline_hash"] = baseline_hash
            revert_hysteresis["genome_reverted_hash"] = reverted_hash
            hyst_output = (
                f"hysteresis:{revert_hysteresis.get('hysteresis_index')}:"
                f"{revert_hysteresis.get('genome_revert_exact')}:"
                f"{revert_hysteresis.get('phenotype_matches_baseline')}"
            ).encode()
            hyst_output_hash = blake3.blake3(hyst_output).hexdigest()
            evidence.append(EvidenceChain(
                claim=(
                    "genome revert exact but phenotype not restored (hysteresis)"
                    if (revert_hysteresis.get("genome_revert_exact")
                        and not revert_hysteresis.get("phenotype_matches_baseline"))
                    else "genome and phenotype revert outcome recorded"
                ),
                tool="proteus.pipeline.hysteresis",
                tool_hash=ContentHash(algorithm="blake3", digest=blake3.blake3(b"proteus.pipeline.hysteresis").digest()),
                params={
                    "genome_revert_exact": bool(revert_hysteresis.get("genome_revert_exact")),
                    "phenotype_matches_baseline": bool(revert_hysteresis.get("phenotype_matches_baseline")),
                    "hysteresis_index": float(revert_hysteresis.get("hysteresis_index", 0.0) or 0.0),
                    "residual_damage": revert_hysteresis.get("residual_damage", {}),
                },
                input_hash=ContentHash(algorithm="blake3", digest=bytes.fromhex(prev_output_hash)) if prev_output_hash else ContentHash(),
                output_hash=ContentHash(algorithm="blake3", digest=bytes.fromhex(hyst_output_hash)),
                upstream_ref=prev_output_hash,
                confidence=EvidenceChain.Confidence.HIGH,
            ))
            prev_output_hash = hyst_output_hash

        report = SimulationReport(
            simulation_id=self.run_id,
            timestamp_ns=self.start_time,
            baseline=baseline,
            mutated=mutated,
            reverted=reverted or GenomeState(reference=baseline.reference, overlays=[], head_index=0),
            applied_variants=list(self.config.variants),
            viability=viability,
            diffs=diffs,
            evidence=evidence,
            provenance_manifest=ContentHash(algorithm="blake3", digest=bytes.fromhex(mutated_hash)),
            validation_results=compact,
            # New headline outputs
            survival_window_h=viability.survival_window_h,
            point_of_no_return_event=time_dynamic.get("irreversible_events", [None])[0] if time_dynamic else None,
            irreversible_events=time_dynamic.get("irreversible_events", []) if time_dynamic else [],
            reference_genome=self.config.reference_genome_uri,
            target_genome=str(self.config.layer_configs.get("target_genome", "")),
            revert_hysteresis=revert_hysteresis or {},
        )
        # Attach rich validation details and layer payloads for report generation
        # (non-schema extensions consumed by proteus.report).
        report.validation_details = validation_details  # type: ignore[attr-defined]
        report.reverted_hash = reverted_hash  # type: ignore[attr-defined]
        report.layer_results = {r.layer: r.data for r in self.layer_results}  # type: ignore[attr-defined]
        report.baseline_layer_results = {  # type: ignore[attr-defined]
            r.layer: r.data for r in self.baseline_layer_results
        }
        report.layer_errors = {  # type: ignore[attr-defined]
            r.layer: r.error for r in self.layer_results if r.error
        }
        report.limitations = self._limitations()  # type: ignore[attr-defined]
        return report

    def _limitations(self) -> list[str]:
        """Explicit scope/limitation statements included in every report."""
        from proteus.layers import TissueLayer

        notes = [
            "This simulation is a mechanistic in silico model; outputs are not clinical "
            "or experimental evidence and must not be used for diagnosis or treatment.",
            "Variant effect, protein stability, pathway and tissue predictors are "
            "deterministic surrogates in this build. They are structured for drop-in "
            "replacement by trained models (AlphaMissense, ESMFold, FoldX, Human-GEM, "
            "PBPK) but are not those models.",
            "Reference genome is a 12-bp synthetic contig (NOT GRCh38). Real GRCh38 loading "
            "is implemented but requires FASTA/GTF download to ~/.proteus/references/.",
            "Ortholog mapping to Corvus cornix uses a hardcoded core-gene subset; "
            "full Ensembl Compara integration requires API access.",
            "Time-dynamic ODE layer (L3b) uses deterministic surrogates for turnover constants "
            "(mRNA t½~9h, protein t½ median~46h); real kinetics require experimental data. "
            "Turnover is calibrated to the supplied initial concentrations, so absolute "
            "concentrations are in arbitrary units and only relative dynamics are interpretable.",
            "Post-revert hysteresis is modeled by freezing irreversible damage at its "
            "point-of-no-return severity and applying a post-revert viability ceiling "
            "(1 - 0.5 * hysteresis_index). The persistence and magnitude of residual damage "
            "are model assumptions, not patient-specific measurements.",
            "Swap gates (mito-nuclear, urea cycle, hemoglobin effector, temperature, "
            "karyotype, enucleated-cell, MHC) use simplified models with literature citations; "
            "species-specific parameters need experimental validation.",
            "Survival window and point-of-no-return are estimates from irreversible-damage "
            "thresholds, not clinical predictions.",
            TissueLayer.neuro_proxy_limitations(),
        ]
        cpp = cpp_status()
        notes.append(
            "C++ acceleration is "
            + ("active (CVODE or RK4 fallback / SIMD / ΔΔG kernels)." if cpp["available"]
               else f"inactive ({cpp.get('import_error') or 'extension not built'}); "
                    "pure-Python fallbacks were used.")
        )
        if not HAS_RUST_CORE:
            notes.append(
                "Rust core unavailable: genome state and revert used the Python fallback, "
                "so event-sourcing guarantees are weaker in this run."
            )
        return notes


async def run_simulation(config: PipelineConfig) -> PipelineResult:
    """Entry point for running a simulation."""
    pipeline = SimulationPipeline(config)
    return await pipeline.run()
