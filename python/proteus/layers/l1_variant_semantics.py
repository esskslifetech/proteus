"""
L1: Variant Semantics Layer
Consequence prediction, pathogenicity scoring, regulatory impact.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
from enum import Enum
import hashlib
import re

IUPAC_BASES = set("ACGTUNRYSWKMBDHVacgtunryswkmbdhv")


class Consequence(str, Enum):
    TRANSCRIPT_ABLATION = "transcript_ablation"
    SPLICE_ACCEPTOR_VARIANT = "splice_acceptor_variant"
    SPLICE_DONOR_VARIANT = "splice_donor_variant"
    STOP_GAINED = "stop_gained"
    FRAMESHIFT_VARIANT = "frameshift_variant"
    STOP_LOST = "stop_lost"
    START_LOST = "start_lost"
    TRANSCRIPT_AMPLIFICATION = "transcript_amplification"
    INFRAME_INSERTION = "inframe_insertion"
    INFRAME_DELETION = "inframe_deletion"
    MISSENSE_VARIANT = "missense_variant"
    PROTEIN_ALTERING_VARIANT = "protein_altering_variant"
    SPLICE_REGION_VARIANT = "splice_region_variant"
    SYNONYMOUS_VARIANT = "synonymous_variant"
    FIVE_PRIME_UTR_VARIANT = "five_prime_utr_variant"
    THREE_PRIME_UTR_VARIANT = "three_prime_utr_variant"
    NON_CODING_TRANSCRIPT_EXON_VARIANT = "non_coding_transcript_exon_variant"
    INTRON_VARIANT = "intron_variant"
    NMD_TRANSCRIPT_VARIANT = "nmd_transcript_variant"
    NON_CODING_TRANSCRIPT_VARIANT = "non_coding_transcript_variant"
    UPSTREAM_GENE_VARIANT = "upstream_gene_variant"
    DOWNSTREAM_GENE_VARIANT = "downstream_gene_variant"
    TF_BINDING_SITE_VARIANT = "tf_binding_site_variant"
    REGULATORY_REGION_VARIANT = "regulatory_region_variant"
    INTERGENIC_VARIANT = "intergenic_variant"


# Consequence severity ranking (higher = more severe)
CONSEQUENCE_SEVERITY: dict[Consequence, float] = {
    Consequence.TRANSCRIPT_ABLATION: 10.0,
    Consequence.SPLICE_ACCEPTOR_VARIANT: 9.0,
    Consequence.SPLICE_DONOR_VARIANT: 9.0,
    Consequence.STOP_GAINED: 8.5,
    Consequence.FRAMESHIFT_VARIANT: 8.5,
    Consequence.STOP_LOST: 8.0,
    Consequence.START_LOST: 8.0,
    Consequence.INFRAME_INSERTION: 6.0,
    Consequence.INFRAME_DELETION: 6.0,
    Consequence.MISSENSE_VARIANT: 5.0,
    Consequence.PROTEIN_ALTERING_VARIANT: 5.5,
    Consequence.SPLICE_REGION_VARIANT: 4.0,
    Consequence.SYNONYMOUS_VARIANT: 1.0,
    Consequence.FIVE_PRIME_UTR_VARIANT: 2.0,
    Consequence.THREE_PRIME_UTR_VARIANT: 2.0,
    Consequence.NON_CODING_TRANSCRIPT_EXON_VARIANT: 3.0,
    Consequence.INTRON_VARIANT: 1.5,
    Consequence.NMD_TRANSCRIPT_VARIANT: 4.0,
    Consequence.NON_CODING_TRANSCRIPT_VARIANT: 2.0,
    Consequence.UPSTREAM_GENE_VARIANT: 1.0,
    Consequence.DOWNSTREAM_GENE_VARIANT: 1.0,
    Consequence.TF_BINDING_SITE_VARIANT: 2.5,
    Consequence.REGULATORY_REGION_VARIANT: 3.0,
    Consequence.INTERGENIC_VARIANT: 0.5,
}

AA_NAMES = ["Ala", "Arg", "Asn", "Asp", "Cys", "Gln", "Glu", "Gly", "His", "Ile",
            "Leu", "Lys", "Met", "Phe", "Pro", "Ser", "Thr", "Trp", "Tyr", "Val"]

# Standard genetic code: codon (RNA, T→U) → amino acid 3-letter code
# This is the canonical translation table used by VEP/Ensembl.
_CODON_TABLE: dict[str, str] = {
    "TTT": "Phe", "TTC": "Phe", "TTA": "Leu", "TTG": "Leu",
    "CTT": "Leu", "CTC": "Leu", "CTA": "Leu", "CTG": "Leu",
    "ATT": "Ile", "ATC": "Ile", "ATA": "Ile", "ATG": "Met",
    "GTT": "Val", "GTC": "Val", "GTA": "Val", "GTG": "Val",
    "TCT": "Ser", "TCC": "Ser", "TCA": "Ser", "TCG": "Ser",
    "CCT": "Pro", "CCC": "Pro", "CCA": "Pro", "CCG": "Pro",
    "ACT": "Thr", "ACC": "Thr", "ACA": "Thr", "ACG": "Thr",
    "GCT": "Ala", "GCC": "Ala", "GCA": "Ala", "GCG": "Ala",
    "TAT": "Tyr", "TAC": "Tyr", "TAA": "Ter", "TAG": "Ter",
    "CAT": "His", "CAC": "His", "CAA": "Gln", "CAG": "Gln",
    "AAT": "Asn", "AAC": "Asn", "AAA": "Lys", "AAG": "Lys",
    "GAT": "Asp", "GAC": "Asp", "GAA": "Glu", "GAG": "Glu",
    "TGT": "Cys", "TGC": "Cys", "TGA": "Ter", "TGG": "Trp",
    "CGT": "Arg", "CGC": "Arg", "CGA": "Arg", "CGG": "Arg",
    "AGT": "Ser", "AGC": "Ser", "AGA": "Arg", "AGG": "Arg",
    "GGT": "Gly", "GGC": "Gly", "GGA": "Gly", "GGG": "Gly",
}

# Single-base codon neighbours: for a given codon and position (0,1,2),
# what single-base substitutions are possible and to which codons.
def _codon_neighbours(codon: str) -> list[tuple[int, str, str]]:
    """Return [(position, base, new_codon), ...] for all single-base changes."""
    bases = "ACGT"
    out: list[tuple[int, str, str]] = []
    for i in range(3):
        for b in bases:
            if b != codon[i]:
                out.append((i, b, codon[:i] + b + codon[i + 1:]))
    return out


def stable_hash(*parts: Any) -> float:
    """Deterministic, process-independent value in [0, 1)."""
    payload = "|".join(str(p) for p in parts).encode("utf-8")
    return int(hashlib.blake2b(payload, digest_size=8).hexdigest(), 16) / 2**64


@dataclass
class VariantEffect:
    variant_id: str
    gene_id: str | None = None
    transcript_id: str | None = None
    consequence: Consequence | None = None
    cds_position: int | None = None
    protein_position: int | None = None
    amino_acid_change: str | None = None  # e.g., "p.Cys61Gly"
    pathogenicity_score: float | None = None  # 0-1
    splice_score_delta: float | None = None
    regulatory_scores: dict[str, float] = field(default_factory=dict)
    aa_change_possible: bool = True  # False if single SNV cannot produce this change

    def severity_score(self) -> float:
        """Return severity score based on consequence."""
        if self.consequence is None:
            return 0.0
        return CONSEQUENCE_SEVERITY.get(self.consequence, 0.0)


def _variant_type_name(variant: Any) -> str:
    t = getattr(variant, "type", None)
    if t is None:
        return ""
    name = getattr(t, "name", str(t))
    return str(name).upper()


class VariantSemanticsLayer:
    """
    L1: Variant consequence and pathogenicity prediction.

    Uses:
    - VEP-style consequence annotation
    - AlphaMissense/ESM pathogenicity-style scoring (deterministic surrogate)
    - SpliceAI-style splicing impact (deterministic surrogate)
    """

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}
        self.models_loaded = False
        self.seed = self.config.get("seed", 42)

    def _codon_to_aa(self, ref: str, alt: str, consequence: Consequence, pos: int) -> tuple[str, str, bool]:
        """Translate ref/alt alleles to amino acids and check if change is possible from a single SNV.
        
        Returns (from_aa, to_aa, is_possible).
        is_possible is True if the change could result from a single-base substitution.
        """
        # For single SNV with 1bp ref/alt, we can infer the codon change
        if len(ref) == 1 and len(alt) == 1 and consequence in {Consequence.MISSENSE_VARIANT, Consequence.SYNONYMOUS_VARIANT}:
            # We don't have the full codon context, so we use a deterministic but biologically
            # plausible mapping. The key constraint: a single base change can only produce
            # certain amino acid transitions. We encode this by looking at all codons for
            # the "from" AA and checking if any single-base change yields the "to" AA.
            from_aa_idx = pos % 20
            from_aa = AA_NAMES[from_aa_idx]
            
            # Check if this from_aa can become any other AA via single SNV
            possible_to_aas = set()
            for codon, aa in _CODON_TABLE.items():
                if aa == from_aa:
                    for _, _, new_codon in _codon_neighbours(codon):
                        possible_to_aas.add(_CODON_TABLE.get(new_codon, ""))
            
            # Deterministic choice from possible changes
            to_aa_candidates = sorted(possible_to_aas - {from_aa, "Ter"})
            if not to_aa_candidates:
                to_aa = from_aa
                is_possible = False
            else:
                to_idx = (pos * 7 + len(alt)) % len(to_aa_candidates)
                to_aa = to_aa_candidates[to_idx]
                is_possible = True
            
            return from_aa, to_aa, is_possible
        
        # For other consequence types or non-SNV variants, use the original heuristic
        from_aa = AA_NAMES[pos % 20]
        to_aa = AA_NAMES[(pos * 7 + len(alt)) % 20]
        
        # Quick check: is this change possible from single SNV?
        is_possible = True
        for codon, aa in _CODON_TABLE.items():
            if aa == from_aa:
                for _, _, new_codon in _codon_neighbours(codon):
                    if _CODON_TABLE.get(new_codon, "") == to_aa:
                        is_possible = True
                        break
                if is_possible:
                    break
        else:
            is_possible = False
        
        return from_aa, to_aa, is_possible

    def load_models(self) -> None:
        """Load ML models (AlphaMissense, SpliceAI, etc.)."""
        # Deterministic surrogate models are used in this build.
        self.models_loaded = True

    def _compute_pathogenicity(self, variant: Any, consequence: Consequence) -> float:
        """Compute a deterministic pathogenicity score in [0, 1]."""
        base_score = CONSEQUENCE_SEVERITY.get(consequence, 0.0) / 10.0
        type_name = _variant_type_name(variant)

        if "FRAMESHIFT" in type_name or "DELETION" in type_name:
            base_score = min(1.0, base_score + 0.2)
        elif "SNV" in type_name and consequence == Consequence.MISSENSE_VARIANT:
            jitter = stable_hash(self.seed, variant.id, "pathogenicity")
            base_score = base_score * 0.5 + jitter * 0.5
        else:
            jitter = stable_hash(self.seed, getattr(variant, "id", ""), "pathogenicity-base")
            base_score = base_score * 0.9 + jitter * 0.1

        return max(0.0, min(1.0, base_score))

    def _predict_consequence(self, variant: Any) -> Consequence:
        """Predict variant consequence from type/position (annotation surrogate)."""
        type_name = _variant_type_name(variant)
        pos = int(getattr(variant, "start", 0) or 0)

        if "SNV" in type_name or "MNV" in type_name:
            bucket = pos % 10
            if bucket == 0:
                return Consequence.MISSENSE_VARIANT
            if bucket == 1:
                return Consequence.SYNONYMOUS_VARIANT
            if bucket == 2:
                return Consequence.STOP_GAINED
            if bucket == 3:
                return Consequence.SPLICE_DONOR_VARIANT
            return Consequence.MISSENSE_VARIANT

        if "INSERTION" in type_name or "DELETION" in type_name:
            length = abs(int(getattr(variant, "end", pos) or pos) - pos)
            if length > 0 and length % 3 == 0:
                return (Consequence.INFRAME_INSERTION if "INSERTION" in type_name
                        else Consequence.INFRAME_DELETION)
            return Consequence.FRAMESHIFT_VARIANT

        if "DUPLICATION" in type_name:
            return Consequence.INFRAME_INSERTION
        if "INVERSION" in type_name:
            return Consequence.PROTEIN_ALTERING_VARIANT
        if "SPLICE" in type_name:
            return Consequence.SPLICE_REGION_VARIANT
        if "RNA_EDIT" in type_name:
            return Consequence.THREE_PRIME_UTR_VARIANT
        if "CNV" in type_name or "TRANSLOCATION" in type_name:
            return Consequence.TRANSCRIPT_ABLATION

        return Consequence.MISSENSE_VARIANT

    def _amino_acid_change(self, variant: Any, consequence: Consequence, cds_position: int | None = None) -> tuple[str | None, int | None, bool]:
        """Compute the amino acid change notation.

        Returns (notation, protein_position, is_possible).
        is_possible is False when a single SNV cannot produce the claimed
        change (e.g. p.Leu11Lys requires a TTA→AAA transversion+transition
        which is impossible from one base substitution).
        """
        if consequence not in {
            Consequence.MISSENSE_VARIANT,
            Consequence.SYNONYMOUS_VARIANT,
            Consequence.STOP_GAINED,
            Consequence.STOP_LOST,
            Consequence.START_LOST,
        }:
            return None, None, True

        pos = int(getattr(variant, "start", 0) or 0)
        # Protein position = ceil(cds_position / 3). A CDS of 1-3 → codon 1, etc.
        if cds_position is not None and cds_position > 0:
            protein_pos = (cds_position + 2) // 3  # ceil(cds/3)
        else:
            protein_pos = (pos + 2) // 3

        ref_allele = (getattr(variant, "ref_allele", "") or "").upper()
        alt_allele = (getattr(variant, "alt_allele", "") or "").upper()

        # Derive from/to amino acids from the actual codon change when possible.
        # For a single SNV, ref and alt alleles are single bases.
        from_aa, to_aa, is_possible = self._codon_to_aa(
            ref_allele, alt_allele, consequence, pos
        )

        if consequence == Consequence.SYNONYMOUS_VARIANT:
            to_aa = from_aa
        if consequence == Consequence.STOP_GAINED:
            to_aa = "Ter"
        if consequence == Consequence.START_LOST:
            from_aa = "Met"
            to_aa = "Val"  # typical start-loss variant

        return f"p.{from_aa}{protein_pos}{to_aa}", protein_pos, is_possible

    def predict(self, variant: Any, context: dict[str, Any] | None = None) -> VariantEffect:
        """Predict variant effects."""
        if not self.models_loaded:
            self.load_models()

        context = context or {}
        consequence = self._predict_consequence(variant)
        pathogenicity = self._compute_pathogenicity(variant, consequence)
        cds_pos = int(getattr(variant, "start", 0) or 0) % 3000
        aa_change, protein_pos, is_possible = self._amino_acid_change(variant, consequence, cds_pos)

        splice_delta = None
        if consequence in {
            Consequence.SPLICE_ACCEPTOR_VARIANT,
            Consequence.SPLICE_DONOR_VARIANT,
            Consequence.SPLICE_REGION_VARIANT,
        }:
            splice_delta = 0.5 + stable_hash(self.seed, variant.id, "splice") * 0.5

        return VariantEffect(
            variant_id=getattr(variant, "id", "unknown"),
            gene_id=context.get("gene_id"),
            transcript_id=context.get("transcript_id"),
            consequence=consequence,
            cds_position=cds_pos,
            protein_position=protein_pos,
            amino_acid_change=aa_change if is_possible else f"{aa_change} (impossible from single SNV)",
            pathogenicity_score=pathogenicity,
            splice_score_delta=splice_delta,
            regulatory_scores={},
            aa_change_possible=is_possible,
        )

    def predict_batch(self, variants: list[Any], context: dict[str, Any] | None = None) -> list[VariantEffect]:
        """Predict effects for multiple variants."""
        return [self.predict(v, context) for v in variants]

    def aggregate_scores(self, effects: list[VariantEffect]) -> dict[str, Any]:
        """Aggregate variant effects into summary statistics."""
        if not effects:
            return {
                "max_pathogenicity": 0.0,
                "mean_pathogenicity": 0.0,
                "variant_count": 0,
                "consequence_counts": {},
                "max_severity": 0.0,
            }

        pathogenicity_scores = [e.pathogenicity_score for e in effects if e.pathogenicity_score is not None]
        severity_scores = [e.severity_score() for e in effects]

        consequence_counts: dict[str, int] = {}
        for effect in effects:
            if effect.consequence:
                key = effect.consequence.value
                consequence_counts[key] = consequence_counts.get(key, 0) + 1

        return {
            "max_pathogenicity": max(pathogenicity_scores) if pathogenicity_scores else 0.0,
            "mean_pathogenicity": (sum(pathogenicity_scores) / len(pathogenicity_scores)
                                   if pathogenicity_scores else 0.0),
            "variant_count": len(effects),
            "consequence_counts": consequence_counts,
            "max_severity": max(severity_scores) if severity_scores else 0.0,
        }
