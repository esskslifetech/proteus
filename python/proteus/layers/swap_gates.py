"""
Swap-Specific Gates Module

Implements 7 mechanistic swap gates for cross-species/organism simulation:
1. Mito-nuclear compatibility gate
2. Urea cycle presence/absence gate
3. Hemoglobin effector gate (IPP vs 2,3-BPG)
4. Temperature scaling gate (37°C vs 41°C)
5. Karyotype/segregation gate
6. Enucleated-cell fraction gate (avian vs mammalian erythrocytes)
7. MHC compatibility gate

Each gate returns a hard/soft constraint with citation.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
from enum import Enum
import logging
import math

logger = logging.getLogger(__name__)


class GateType(str, Enum):
    HARD = "hard"      # Viability = 0 if failed
    SOFT = "soft"      # Reduces viability score


class GateResult(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    WARNING = "warning"


@dataclass
class SwapGate:
    """Base class for swap gates."""
    name: str
    gate_type: GateType
    description: str
    citation: str
    
    def evaluate(self, context: dict[str, Any]) -> tuple[GateResult, float, str]:
        """
        Evaluate gate.
        Returns: (result, score_modifier, message)
        score_modifier: 1.0 = no effect, 0.0 = hard fail, 0.5 = 50% penalty
        """
        raise NotImplementedError


@dataclass
class MitoNuclearGate(SwapGate):
    """
    Mito-nuclear compatibility gate.
    
    Mitochondrial DNA (mtDNA) encodes 13 protein subunits of OXPHOS complexes.
    Nuclear DNA encodes ~1000+ mitochondrial proteins including assembly factors.
    Co-evolution creates species-specific compatibility.
    
    Citation: Bar-Yaacov et al. 2012 (Cell), Hill 2019 (Annu Rev Genom Hum Genet)
    """
    
    def __init__(self):
        super().__init__(
            name="mito_nuclear_compatibility",
            gate_type=GateType.HARD,
            description="Checks compatibility between mtDNA haplotype and nuclear-encoded mitochondrial proteins",
            citation="Bar-Yaacov et al. 2012 Cell 151:1232; Hill 2019 Annu Rev Genom Hum Genet 20:253"
        )
    
    def evaluate(self, context: dict[str, Any]) -> tuple[GateResult, float, str]:
        # Context should contain:
        # - mtDNA_haplotype: str (e.g., "human", "corvus_cornix")
        # - nuclear_background: str (e.g., "human", "corvus_cornix")
        # - ortholog_identity: dict[gene_id, float] for mitochondrial proteins
        
        mt_hap = context.get("mtDNA_haplotype", "human")
        nuc_bg = context.get("nuclear_background", "human")
        ortholog_identity = context.get("ortholog_identity", {})
        
        if mt_hap == nuc_bg:
            return GateResult.PASS, 1.0, "Matched mito-nuclear background"
        
        # Check key OXPHOS nuclear genes
        key_genes = [
            "NDUFS1", "NDUFS2", "NDUFS3", "NDUFS7", "NDUFS8",  # Complex I
            "SDHA", "SDHB", "SDHC", "SDHD",                      # Complex II
            "UQCRC1", "UQCRC2", "UQCRFS1",                       # Complex III
            "COX4I1", "COX5A", "COX6C",                          # Complex IV
            "ATP5F1A", "ATP5F1B", "ATP5F1C", "ATP5F1D", "ATP5F1E",  # Complex V
            "TFAM", "TFB2M", "POLG", "POLG2",                    # mtDNA maintenance
        ]
        
        # Compute mean identity for mitochondrial proteins
        identities = []
        for gene in key_genes:
            if gene in ortholog_identity:
                identities.append(ortholog_identity[gene])
        
        if not identities:
            return GateResult.WARNING, 0.8, f"Mito-nuclear mismatch ({mt_hap} vs {nuc_bg}); no ortholog data"
        
        mean_identity = sum(identities) / len(identities)
        
        if mean_identity < 0.85:
            return GateResult.FAIL, 0.0, f"Mito-nuclear incompatibility: mean identity {mean_identity:.2f} < 0.85"
        elif mean_identity < 0.92:
            return GateResult.WARNING, 0.6, f"Mito-nuclear divergence: mean identity {mean_identity:.2f}"
        else:
            return GateResult.PASS, 0.9, f"Mito-nuclear compatible: mean identity {mean_identity:.2f}"


@dataclass
class UreaCycleGate(SwapGate):
    """
    Urea cycle presence/absence gate.
    
    Mammals have complete urea cycle (liver).
    Birds have partial cycle (uricotelic) - lack CPS1, NAGS.
    Fish/amphibians often lack urea cycle (ammonotelic).
    
    Citation: Wright 1995 (J Exp Zool), Campbell & Anderson 2005 (Comp Biochem Physiol)
    """
    
    def __init__(self):
        super().__init__(
            name="urea_cycle",
            gate_type=GateType.SOFT,
            description="Checks for complete urea cycle enzymes; penalizes if absent",
            citation="Wright 1995 J Exp Zool 272:1; Campbell & Anderson 2005 Comp Biochem Physiol B 140:1"
        )
    
    def evaluate(self, context: dict[str, Any]) -> tuple[GateResult, float, str]:
        # Context:
        # - species: str
        # - gene_presence: dict[gene_id, bool]
        
        species = context.get("species", "human")
        gene_presence = context.get("gene_presence", {})
        
        # Urea cycle enzymes
        urea_genes = {
            "CPS1": "Carbamoyl phosphate synthetase I",
            "NAGS": "N-acetylglutamate synthase",
            "OTC": "Ornithine transcarbamylase",
            "ASS1": "Argininosuccinate synthase",
            "ASL": "Argininosuccinate lyase",
            "ARG1": "Arginase 1",
            "SLC25A15": "Ornithine transporter (ORNT1)",
            "SLC25A13": "Aspartate/glutamate carrier (Citrulin)",
        }
        
        present = sum(1 for g in urea_genes if gene_presence.get(g, False))
        total = len(urea_genes)
        completeness = present / total
        
        # Species expectations
        if species in ["human", "mouse", "rat", "mammal"]:
            expected = 1.0
        elif species in ["corvus_cornix", "chicken", "bird", "avian"]:
            expected = 0.375  # Birds have OTC, ASS1, ASL, ARG1 but lack CPS1, NAGS
        elif species in ["zebrafish", "fish"]:
            expected = 0.125  # Only ARG1
        else:
            expected = 0.5
        
        if completeness >= expected:
            return GateResult.PASS, 1.0, f"Urea cycle completeness {completeness:.2f} meets {expected:.2f} for {species}"
        else:
            penalty = completeness / expected
            return GateResult.WARNING, max(0.3, penalty), f"Urea cycle incomplete: {present}/{total} enzymes (expected {expected:.0%} for {species})"


@dataclass
class HemoglobinEffectorGate(SwapGate):
    """
    Hemoglobin effector gate (IPP vs 2,3-BPG).
    
    Mammals use 2,3-BPG (2,3-bisphosphoglycerate) as allosteric effector.
    Birds use inositol pentaphosphate (IPP).
    Both stabilize T-state (low affinity) but have different binding sites.
    
    Citation: Riggs 1998 (J Biol Chem), Weber 2004 (Respir Physiol Neurobiol)
    """
    
    def __init__(self):
        super().__init__(
            name="hemoglobin_effector",
            gate_type=GateType.SOFT,
            description="Checks for correct hemoglobin allosteric effector system",
            citation="Riggs 1998 J Biol Chem 273:16351; Weber 2004 Respir Physiol Neurobiol 144:151"
        )
    
    def evaluate(self, context: dict[str, Any]) -> tuple[GateResult, float, str]:
        # Context:
        # - species: str
        # - hb_genes: dict[gene_id, str] (sequence or variant info)
        # - effector_type: str ("23BPG" or "IPP")
        
        species = context.get("species", "human")
        effector = context.get("effector_type", "23BPG" if species != "bird" else "IPP")
        hb_genes = context.get("hb_genes", {})
        
        # Key residues for 2,3-BPG binding (human β-chain)
        # His143, Lys82, His2, Val1, His146 (His143 critical)
        bpg_site_residues = {
            "HBB": {143: "H", 82: "K", 2: "H", 1: "V", 146: "H"},
        }
        
        # Key residues for IPP binding (avian)
        # Different site, involves Lys/Arg cluster
        ipp_site_residues = {
            "HBB": {82: "R", 143: "R", 144: "K", 146: "H"},  # Avian-like
        }
        
        if effector == "23BPG":
            # Check mammalian BPG site
            if "HBB" in hb_genes:
                seq = hb_genes["HBB"]
                # Simplified: check critical His143
                if len(seq) > 143 and seq[142] == "H":
                    return GateResult.PASS, 1.0, "Mammalian 2,3-BPG binding site intact"
                else:
                    return GateResult.WARNING, 0.5, "HBB His143 mutated: impaired 2,3-BPG binding"
            return GateResult.PASS, 1.0, "2,3-BPG effector system assumed intact"
        
        elif effector == "IPP":
            # Check avian IPP site
            if "HBB" in hb_genes:
                seq = hb_genes["HBB"]
                # Avian have Arg/Lys at 82, 143
                if len(seq) > 143 and seq[81] in "RK" and seq[142] in "RK":
                    return GateResult.PASS, 1.0, "Avian IPP binding site intact"
                else:
                    return GateResult.WARNING, 0.5, "HBB lacks avian IPP-binding residues"
            return GateResult.PASS, 0.8, "IPP effector system assumed"
        
        return GateResult.WARNING, 0.7, f"Unknown effector type: {effector}"


@dataclass
class TemperatureScalingGate(SwapGate):
    """
    Temperature scaling gate (37°C vs 41°C).
    
    Mammals: 37°C (310 K)
    Birds: 41°C (314 K)
    Enzyme kinetics scale with Q10 ~ 2-3.
    Protein stability decreases at higher T.
    
    Citation: Hochachka & Somero 2002 (Biochemical Adaptation), Clarke 2017 (J Therm Biol)
    """
    
    def __init__(self):
        super().__init__(
            name="temperature_scaling",
            gate_type=GateType.SOFT,
            description="Scales kinetic rates and stability for temperature difference",
            citation="Hochachka & Somero 2002 Biochemical Adaptation; Clarke 2017 J Therm Biol 68:1"
        )
    
    def evaluate(self, context: dict[str, Any]) -> tuple[GateResult, float, str]:
        # Context:
        # - reference_temp_c: float (default 37)
        # - target_temp_c: float (default 41 for birds)
        # - protein_stabilities: dict[gene_id, float] (ΔΔG or Tm)
        
        t_ref = context.get("reference_temp_c", 37.0)
        t_target = context.get("target_temp_c", 41.0)
        stabilities = context.get("protein_stabilities", {})
        
        delta_t = t_target - t_ref
        
        if abs(delta_t) < 0.5:
            return GateResult.PASS, 1.0, f"No significant temperature difference ({t_ref}°C -> {t_target}°C)"
        
        # Q10 scaling for reaction rates
        q10 = context.get("q10", 2.5)
        rate_factor = q10 ** (delta_t / 10.0)
        
        # Protein stability decrease: ~0.5-1 kcal/mol per 10°C
        stability_loss_per_10c = context.get("stability_loss_per_10c", 0.8)  # kcal/mol
        stability_loss = stability_loss_per_10c * (delta_t / 10.0)
        
        # Check if any protein falls below stability threshold
        min_stability = 3.0  # kcal/mol minimum for functional protein
        unstable = []
        for gene, dg in stabilities.items():
            if dg - stability_loss < min_stability:
                unstable.append(gene)
        
        if unstable:
            return GateResult.WARNING, 0.5, f"Temperature {t_ref}°C->{t_target}°C: {len(unstable)} proteins near instability threshold (ΔΔG loss {stability_loss:.1f} kcal/mol)"
        
        return GateResult.PASS, 0.9, f"Temperature scaling: rate factor {rate_factor:.2f}x, stability loss {stability_loss:.1f} kcal/mol"


@dataclass
class KaryotypeGate(SwapGate):
    """
    Karyotype/segregation gate.
    
    Human: 2n=46 (23 pairs), XY sex determination
    Chicken: 2n=78 (39 pairs), ZW sex determination (female heterogametic)
    Corvus cornix: 2n=80 (40 pairs), ZW
    
    Aneuploidy tolerance differs. Meiotic drive, recombination landscapes differ.
    
    Citation: Griffin et al. 2007 (Chromosome Res), Ellegren 2010 (Nat Rev Genet)
    """
    
    def __init__(self):
        super().__init__(
            name="karyotype_segregation",
            gate_type=GateType.SOFT,
            description="Checks karyotype compatibility and segregation fidelity",
            citation="Griffin et al. 2007 Chromosome Res 15:707; Ellegren 2010 Nat Rev Genet 11:192"
        )
    
    def evaluate(self, context: dict[str, Any]) -> tuple[GateResult, float, str]:
        # Context:
        # - source_karyotype: dict (chr_count, sex_system, chromosome_sizes)
        # - target_karyotype: dict
        # - variant_chromosome: str (if variant affects segregation)
        
        src = context.get("source_karyotype", {"chr_count": 46, "sex_system": "XY"})
        tgt = context.get("target_karyotype", {"chr_count": 46, "sex_system": "XY"})
        var_chr = context.get("variant_chromosome")
        
        src_chr = src.get("chr_count", 46)
        tgt_chr = tgt.get("chr_count", 46)
        src_sex = src.get("sex_system", "XY")
        tgt_sex = tgt.get("sex_system", "XY")
        
        score = 1.0
        messages = []
        
        # Chromosome number mismatch
        if src_chr != tgt_chr:
            diff = abs(src_chr - tgt_chr)
            penalty = min(0.3, diff * 0.02)
            score -= penalty
            messages.append(f"Chromosome count mismatch: {src_chr} vs {tgt_chr} (penalty {penalty:.2f})")
        
        # Sex determination system mismatch
        if src_sex != tgt_sex:
            score *= 0.7
            messages.append(f"Sex system mismatch: {src_sex} vs {tgt_sex}")
        
        # Variant on sex chromosome
        if var_chr and var_chr in ["X", "Y", "Z", "W"]:
            score *= 0.8
            messages.append(f"Variant on sex chromosome {var_chr}")
        
        if score < 0.5:
            return GateResult.FAIL, score, "; ".join(messages)
        elif score < 0.8:
            return GateResult.WARNING, score, "; ".join(messages)
        
        return GateResult.PASS, score, "Karyotype compatible" if not messages else "; ".join(messages)


@dataclass
class EnucleatedCellGate(SwapGate):
    """
    Enucleated-cell fraction gate.
    
    Mammals: enucleated erythrocytes (no nucleus, no transcription, limited lifespan ~120d)
    Birds: nucleated erythrocytes (active transcription, longer lifespan ~30-60d in birds)
    
    Affects: oxygen delivery modeling, gene expression in blood, metabolic scaling.
    
    Citation: Maina 2002 (J Exp Biol), Storz 2007 (Physiol Genomics)
    """
    
    def __init__(self):
        super().__init__(
            name="enucleated_cell_fraction",
            gate_type=GateType.SOFT,
            description="Adjusts erythrocyte biology parameters for enucleated vs nucleated RBCs",
            citation="Maina 2002 J Exp Biol 205:1937; Storz 2007 Physiol Genomics 31:1"
        )
    
    def evaluate(self, context: dict[str, Any]) -> tuple[GateResult, float, str]:
        # Context:
        # - species: str
        # - rbc_type: str ("enucleated" or "nucleated")
        # - simulation_aspects: list[str] (what aspects affected)
        
        species = context.get("species", "human")
        rbc_type = context.get("rbc_type", "enucleated" if species != "bird" else "nucleated")
        aspects = context.get("simulation_aspects", ["oxygen_delivery", "gene_expression", "lifespan"])
        
        if rbc_type == "enucleated":
            # Mammalian: no transcription in RBCs, limited repair, 120d lifespan
            adjustments = {
                "oxygen_delivery": 1.0,  # Higher Hb concentration possible
                "gene_expression": 0.0,  # No transcription
                "repair_capacity": 0.1,  # Minimal
                "lifespan_days": 120,
                "metabolic_flexibility": 0.2,  # Glycolysis only
            }
            return GateResult.PASS, 1.0, f"Mammalian enucleated RBC model: Hb ~34g/dL, lifespan {adjustments['lifespan_days']}d"
        
        else:
            # Avian: nucleated, active transcription, 30-60d lifespan
            adjustments = {
                "oxygen_delivery": 0.85,  # Lower Hb concentration
                "gene_expression": 1.0,   # Active transcription
                "repair_capacity": 0.8,   # Can repair
                "lifespan_days": 45,
                "metabolic_flexibility": 0.8,  # Mitochondria present
            }
            return GateResult.PASS, 0.9, f"Avian nucleated RBC model: Hb ~28g/dL, lifespan {adjustments['lifespan_days']}d"


@dataclass
class MHCGate(SwapGate):
    """
    MHC compatibility gate.
    
    MHC (HLA in humans, B locus in chickens) presents peptides to T cells.
    Highly polymorphic. Cross-species compatibility very low.
    Affects: immune recognition, transplant rejection, pathogen response.
    
    Citation: Klein 1986 (Natural History of MHC), Kaufman 1999 (Rev Immunogenet)
    """
    
    def __init__(self):
        super().__init__(
            name="mhc_compatibility",
            gate_type=GateType.SOFT,
            description="Checks MHC/peptide presentation compatibility",
            citation="Klein 1986 Natural History of the MHC; Kaufman 1999 Rev Immunogenet 1:149"
        )
    
    def evaluate(self, context: dict[str, Any]) -> tuple[GateResult, float, str]:
        # Context:
        # - source_mhc: dict (class_I_genes, class_II_genes, haplotypes)
        # - target_mhc: dict
        # - peptides_of_interest: list[str] (e.g., viral epitopes)
        
        src = context.get("source_mhc", {"class_I": ["HLA-A", "HLA-B", "HLA-C"], "class_II": ["HLA-DP", "HLA-DQ", "HLA-DR"]})
        tgt = context.get("target_mhc", {"class_I": ["BF", "B-G"], "class_II": ["BLB", "BLA"]})
        peptides = context.get("peptides_of_interest", [])
        
        # Check if key MHC genes have orthologs
        src_genes = set(src.get("class_I", []) + src.get("class_II", []))
        tgt_genes = set(tgt.get("class_I", []) + tgt.get("class_II", []))
        
        # Very low cross-species compatibility expected
        if src_genes == tgt_genes:
            return GateResult.PASS, 1.0, "Identical MHC gene sets"
        
        # Different species - low compatibility
        overlap = src_genes & tgt_genes
        if not overlap:
            return GateResult.WARNING, 0.3, f"No MHC gene overlap between species; immune modeling will be approximate"
        
        return GateResult.WARNING, 0.5, f"Partial MHC compatibility ({len(overlap)} shared genes); T-cell epitope prediction unreliable"


class SwapGateRegistry:
    """Registry of all swap gates."""
    
    def __init__(self):
        self.gates: dict[str, SwapGate] = {
            "mito_nuclear": MitoNuclearGate(),
            "urea_cycle": UreaCycleGate(),
            "hemoglobin_effector": HemoglobinEffectorGate(),
            "temperature_scaling": TemperatureScalingGate(),
            "karyotype": KaryotypeGate(),
            "enucleated_cell": EnucleatedCellGate(),
            "mhc_compatibility": MHCGate(),
        }
    
    def get_gate(self, name: str) -> SwapGate | None:
        return self.gates.get(name)
    
    def evaluate_all(self, context: dict[str, Any]) -> dict[str, tuple[GateResult, float, str]]:
        """Evaluate all gates with shared context."""
        results = {}
        for name, gate in self.gates.items():
            try:
                results[name] = gate.evaluate(context)
            except Exception as e:
                logger.error(f"Gate {name} failed: {e}")
                results[name] = (GateResult.WARNING, 0.5, f"Gate evaluation error: {e}")
        return results
    
    def compute_composite_score(self, results: dict[str, tuple[GateResult, float, str]]) -> float:
        """Compute composite viability modifier from gate results."""
        # Hard gates: any FAIL -> 0
        for name, (result, score, msg) in results.items():
            gate = self.gates[name]
            if gate.gate_type == GateType.HARD and result == GateResult.FAIL:
                return 0.0
        
        # Multiply soft gate scores
        composite = 1.0
        for name, (result, score, msg) in results.items():
            gate = self.gates[name]
            if gate.gate_type == GateType.SOFT:
                composite *= score
        
        return composite
    
    def get_first_hard_failure(self, results: dict[str, tuple[GateResult, float, str]]) -> str | None:
        """Get message of first hard gate failure."""
        for name, (result, score, msg) in results.items():
            gate = self.gates[name]
            if gate.gate_type == GateType.HARD and result == GateResult.FAIL:
                return f"{gate.name}: {msg}"
        return None


# Global registry
_gate_registry: SwapGateRegistry | None = None


def get_swap_gate_registry() -> SwapGateRegistry:
    global _gate_registry
    if _gate_registry is None:
        _gate_registry = SwapGateRegistry()
    return _gate_registry


def evaluate_swap_gates(context: dict[str, Any]) -> dict[str, Any]:
    """Convenience function to evaluate all swap gates."""
    registry = get_swap_gate_registry()
    results = registry.evaluate_all(context)
    composite = registry.compute_composite_score(results)
    hard_fail = registry.get_first_hard_failure(results)
    
    return {
        "gate_results": {name: {"result": r.value, "score": s, "message": m} for name, (r, s, m) in results.items()},
        "composite_score": composite,
        "hard_failure": hard_fail,
        "viable": composite > 0 and hard_fail is None,
    }


# Export
__all__ = [
    "GateType",
    "GateResult",
    "SwapGate",
    "MitoNuclearGate",
    "UreaCycleGate",
    "HemoglobinEffectorGate",
    "TemperatureScalingGate",
    "KaryotypeGate",
    "EnucleatedCellGate",
    "MHCGate",
    "SwapGateRegistry",
    "get_swap_gate_registry",
    "evaluate_swap_gates",
]