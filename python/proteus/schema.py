"""
Protobuf-generated schema classes (placeholder until protobuf compilation).
In practice, these would be generated from proteus.proto using protoc.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
from enum import Enum


class ContentHash:
    def __init__(self, algorithm: str = "", digest: bytes = b""):
        self.algorithm = algorithm
        self.digest = digest

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ContentHash):
            return NotImplemented
        return self.algorithm == other.algorithm and self.digest == other.digest

    def __hash__(self) -> int:
        return hash((self.algorithm, self.digest))


class ResourceRef:
    def __init__(self, hash: ContentHash | None = None, uri: str = ""):
        self.hash = hash or ContentHash()
        self.uri = uri

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ResourceRef):
            return NotImplemented
        return self.hash == other.hash and self.uri == other.uri

    def __hash__(self) -> int:
        return hash((self.hash, self.uri))


class VariantEvent:
    class Type(Enum):
        TYPE_UNSPECIFIED = 0
        SNV = 1
        MNV = 2
        INSERTION = 3
        DELETION = 4
        DELINS = 5
        INVERSION = 6
        DUPLICATION = 7
        TRANSLOCATION = 8
        CNV = 9
        STR_EXPANSION = 10
        RNA_EDIT = 11
        SPLICE_SITE = 12

    def __init__(
        self,
        id: str = "",
        type: Type = Type.TYPE_UNSPECIFIED,
        contig: str = "",
        start: int = 0,
        end: int = 0,
        ref_allele: str = "",
        alt_allele: str = "",
        annotations: dict[str, str] | None = None,
        timestamp_ns: int = 0,
        source: str = "",
    ):
        self.id = id
        self.type = type
        self.contig = contig
        self.start = start
        self.end = end
        self.ref_allele = ref_allele
        self.alt_allele = alt_allele
        self.annotations = annotations or {}
        self.timestamp_ns = timestamp_ns
        self.source = source


class GenomeState:
    def __init__(
        self,
        reference: ResourceRef | None = None,
        overlays: list[VariantEvent] | None = None,
        head_index: int = 0,
    ):
        self.reference = reference or ResourceRef()
        self.overlays = overlays or []
        self.head_index = head_index

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, GenomeState):
            return NotImplemented
        return (
            self.reference == other.reference
            and self.overlays == other.overlays
            and self.head_index == other.head_index
        )

    def __hash__(self) -> int:
        return hash((self.reference, tuple(self.overlays), self.head_index))

    def hash(self) -> str:
        """Content-addressable hash of this genome state (hex string)."""
        import blake3
        data = self.reference.uri.encode() + str(self.head_index).encode()
        for v in self.overlays:
            data += v.id.encode() + v.contig.encode() + str(v.start).encode() + str(v.end).encode()
        return blake3.blake3(data).hexdigest()

    def hash_bytes(self) -> bytes:
        """Raw 32-byte blake3 digest of this genome state."""
        import blake3
        data = self.reference.uri.encode() + str(self.head_index).encode()
        for v in self.overlays:
            data += v.id.encode() + v.contig.encode() + str(v.start).encode() + str(v.end).encode()
        return blake3.blake3(data).digest()


class ViabilityScore:
    class Verdict(Enum):
        VERDICT_UNSPECIFIED = 0
        VIABLE = 1
        NON_VIABLE = 2
        UNCERTAIN = 3

    def __init__(
        self,
        composite: float = 0.0,
        layer_scores: dict[str, float] | None = None,
        weights: dict[str, float] | None = None,
        failed_gates: list[str] | None = None,
        verdict: Verdict = Verdict.VERDICT_UNSPECIFIED,
        confidence: float = 0.0,
        # New fields for time-dynamic and swap gates
        survival_window_h: float | None = None,
        point_of_no_return_h: float | None = None,
        irreversible_events: list[dict] | None = None,
        swap_gate_results: dict[str, Any] | None = None,
        hard_gate_failures: list[str] | None = None,
    ):
        self.composite = composite
        self.layer_scores = layer_scores or {}
        self.weights = weights or {}
        self.failed_gates = failed_gates or []
        self.verdict = verdict
        self.confidence = confidence
        self.survival_window_h = survival_window_h
        self.point_of_no_return_h = point_of_no_return_h
        self.irreversible_events = irreversible_events or []
        self.swap_gate_results = swap_gate_results or {}
        self.hard_gate_failures = hard_gate_failures or []


class LayerDiff:
    def __init__(
        self,
        layer: str = "",
        added: dict[str, str] | None = None,
        removed: dict[str, str] | None = None,
        changed: dict[str, DiffEntry] | None = None,
    ):
        self.layer = layer
        self.added = added or {}
        self.removed = removed or {}
        self.changed = changed or {}


class DiffEntry:
    def __init__(
        self,
        before: str = "",
        after: str = "",
        magnitude: float = 0.0,
        significance: str = "",
    ):
        self.before = before
        self.after = after
        self.magnitude = magnitude
        self.significance = significance


class EvidenceChain:
    class Confidence(Enum):
        CONFIDENCE_UNSPECIFIED = 0
        HIGH = 1
        MEDIUM = 2
        LOW = 3

    def __init__(
        self,
        claim: str = "",
        input_hash: ContentHash | None = None,
        tool: str = "",
        tool_hash: ContentHash | None = None,
        params: dict[str, str] | None = None,
        output_hash: ContentHash | None = None,
        upstream_ref: str = "",
        confidence: Confidence = Confidence.CONFIDENCE_UNSPECIFIED,
        validation_passes: list[str] | None = None,
    ):
        self.claim = claim
        self.input_hash = input_hash or ContentHash()
        self.tool = tool
        self.tool_hash = tool_hash or ContentHash()
        self.params = params or {}
        self.output_hash = output_hash or ContentHash()
        self.upstream_ref = upstream_ref
        self.confidence = confidence
        self.validation_passes = validation_passes or []


class SimulationReport:
    def __init__(
        self,
        simulation_id: str = "",
        timestamp_ns: int = 0,
        baseline: GenomeState | None = None,
        mutated: GenomeState | None = None,
        reverted: GenomeState | None = None,
        applied_variants: list[VariantEvent] | None = None,
        viability: ViabilityScore | None = None,
        diffs: list[LayerDiff] | None = None,
        evidence: list[EvidenceChain] | None = None,
        provenance_manifest: ContentHash | None = None,
        validation_results: dict[str, str] | None = None,
        # New headline outputs
        survival_window_h: float | None = None,
        point_of_no_return_event: dict | None = None,
        irreversible_events: list[dict] | None = None,
        # Reference genome info
        reference_genome: str = "",
        target_genome: str = "",
        ortholog_mapping: dict[str, Any] | None = None,
        # Post-revert residual phenotype (hysteresis)
        revert_hysteresis: dict[str, Any] | None = None,
    ):
        self.simulation_id = simulation_id
        self.timestamp_ns = timestamp_ns
        self.baseline = baseline or GenomeState()
        self.mutated = mutated or GenomeState()
        self.reverted = reverted or GenomeState()
        self.applied_variants = applied_variants or []
        self.viability = viability or ViabilityScore()
        self.diffs = diffs or []
        self.evidence = evidence or []
        self.provenance_manifest = provenance_manifest or ContentHash()
        self.validation_results = validation_results or {}
        # New headline outputs
        self.survival_window_h = survival_window_h
        self.point_of_no_return_event = point_of_no_return_event
        self.irreversible_events = irreversible_events or []
        # Reference info
        self.reference_genome = reference_genome
        self.target_genome = target_genome
        self.ortholog_mapping = ortholog_mapping or {}
        # Hysteresis: genome reverts exactly, phenotype may not
        self.revert_hysteresis = revert_hysteresis or {}


class PipelineConfig:
    def __init__(
        self,
        reference_genome_uri: str = "",
        variants: list[VariantEvent] | None = None,
        layer_configs: dict[str, str] | None = None,
        viability_weights: dict[str, float] | None = None,
        enabled_validations: list[str] | None = None,
        produce_report: bool = True,
        output_dir: str = "",
        generate_visualizations: bool = True,
    ):
        self.reference_genome_uri = reference_genome_uri
        self.variants = variants or []
        self.layer_configs = layer_configs or {}
        self.viability_weights = viability_weights or {}
        self.enabled_validations = enabled_validations or ["V1", "V2", "V3", "V4", "V5", "V6", "V7"]
        self.produce_report = produce_report
        self.output_dir = output_dir
        self.generate_visualizations = generate_visualizations


class PipelineResult:
    def __init__(
        self,
        run_id: str = "",
        success: bool = False,
        error: str = "",
        report: SimulationReport | None = None,
        artifacts: dict[str, ContentHash] | None = None,
    ):
        self.run_id = run_id
        self.success = success
        self.error = error
        self.report = report or SimulationReport()
        self.artifacts = artifacts or {}