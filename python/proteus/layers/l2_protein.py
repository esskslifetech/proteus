"""
L2: Protein Layer
Structure prediction, stability (ΔΔG), domain impact.

Integrates trained models:
- FoldX (empirical ΔΔG)
- ESMFold (structure prediction)
- DeepDDG (ML-based stability)

When the optional C++ extension (``proteus_cpp``) is importable, the empirical
ΔΔG kernel is computed in C++ (FoldX-style energy function); otherwise the
Python fallback below is used.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
from enum import Enum
import hashlib
import re

from proteus.cpp_adapter import cpp_available, empirical_ddg
from proteus.models.model_registry import (
    get_model_registry,
    ModelType,
    predict_protein_stability,
)


AA_ONE_TO_THREE = {
    "A": "Ala", "R": "Arg", "N": "Asn", "D": "Asp", "C": "Cys",
    "Q": "Gln", "E": "Glu", "G": "Gly", "H": "His", "I": "Ile",
    "L": "Leu", "K": "Lys", "M": "Met", "F": "Phe", "P": "Pro",
    "S": "Ser", "T": "Thr", "W": "Trp", "Y": "Tyr", "V": "Val",
}


class StructureMethod(str, Enum):
    ESM_FOLD = "esm_fold"
    ALPHAFOLD2 = "alphafold2"
    ROSETTA = "rosetta"
    FOLDX_STYLE_SURROGATE = "foldx_style_surrogate"
    DEEP_DDG = "deep_ddg"


@dataclass
class DomainImpact:
    domain_id: str
    delta_delta_g: float
    active_site_disrupted: bool = False
    binding_interface_disrupted: bool = False


@dataclass
class ProteinEffect:
    variant_id: str
    uniprot_id: str | None = None
    domain_id: str | None = None
    delta_delta_g: float = 0.0  # kcal/mol, positive = destabilizing
    confidence: float = 0.0
    method: StructureMethod = StructureMethod.ESM_FOLD
    model_id: str = "toy_model"  # e.g., "toy_glycolysis", "esm_fold_v1"
    structure_hash: str | None = None
    domain_impacts: list[DomainImpact] = field(default_factory=list)


# Known destabilizing amino acid substitutions (simplified BLOSUM-based)
DESTABILIZING_PAIRS = {
    ("Gly", "Pro"): 2.5,   # helix breaker
    ("Pro", "Gly"): 2.0,
    ("Cys", "Ser"): 3.0,   # disulfide bond loss
    ("Trp", "Gly"): 4.0,   # large hydrophobic to small
    ("Phe", "Ser"): 3.5,
    ("Tyr", "Asp"): 3.0,
    ("Arg", "Cys"): 3.5,
}

# Common protein domains and their stability sensitivity
DOMAIN_SENSITIVITY = {
    "PF00001": 1.2,  # 7tm_1 - GPCR
    "PF00069": 1.5,  # Protein kinase
    "PF00128": 1.0,  # Zinc finger
    "PF00018": 0.8,  # SH3 domain
    "PF00595": 1.0,  # PDZ domain
    "PF00071": 1.3,  # Ras family
}


def _deterministic_hash(*args: Any) -> float:
    """Process-independent deterministic value in [0, 1)."""
    payload = "|".join(str(a) for a in args).encode("utf-8")
    return int(hashlib.blake2b(payload, digest_size=8).hexdigest(), 16) / 2**64


class ProteinLayer:
    """
    L2: Protein structure and stability prediction.

    Uses:
    - ESMFold / AlphaFold-style structure hashes
    - FoldX / Rosetta / DeepDDG-style ΔΔG prediction
    - Pfam/InterPro-style domain annotation surrogate
    
    Integrates with model registry for trained model inference.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}
        self.models_loaded = False
        self.seed = self.config.get("seed", 42)
        self._use_trained_models = self.config.get("use_trained_models", True)
        self._model_registry = get_model_registry()

    def load_models(self) -> None:
        if self._use_trained_models:
            # Pre-load FoldX and ESMFold if available
            self._model_registry.load_model("FoldX")
            self._model_registry.load_model("ESMFold")
        self.models_loaded = True

    def _parse_aa_change(self, aa_change: str | None) -> tuple[str, int, str] | None:
        if not aa_change:
            return None
        match = re.match(r"p\.([A-Z][a-z]{2})(\d+)([A-Z][a-z]{2})", aa_change)
        if not match:
            return None
        return match.group(1), int(match.group(2)), match.group(3)

    def _predict_ddg_trained(self, variant: Any, aa_change: str | None) -> tuple[float, str, str, float]:
        """Predict ΔΔG using trained models (FoldX/ESMFold)."""
        parsed = self._parse_aa_change(aa_change)
        if not parsed:
            return 0.0, "none", "unknown", 0.0

        from_aa, pos, to_aa = parsed

        # Build input for FoldX model
        foldx_input = {
            "structure_pdb": context.get("pdb_id") if (context := {}) else "1CRN",  # Default structure
            "position": pos,
            "wt_aa": from_aa,
            "mut_aa": to_aa,
        }

        # Build input for ESMFold
        esmfold_input = {
            "sequence": context.get("protein_sequence", "A" * 100),  # Would need real sequence
        }

        results = predict_protein_stability([foldx_input])
        if results:
            pred = results[0]
            ddg = pred.predictions.get("delta_delta_g", 0.0)
            engine = pred.metadata.get("engine", "foldx")
            model_id = pred.model_name
            confidence = pred.confidence or 0.8
            return ddg, engine, model_id, confidence

        return 0.0, "none", "unknown", 0.0

    def _predict_ddg_surrogate(self, variant: Any, aa_change: str | None) -> tuple[float, str, str, float]:
        """Deterministic surrogate ΔΔG prediction (fallback)."""
        parsed = self._parse_aa_change(aa_change)
        engine = "python-surrogate"
        model_id = "toy_glycolysis"
        confidence = 0.7

        if parsed:
            from_aa, pos, to_aa = parsed
            if (from_aa, to_aa) in DESTABILIZING_PAIRS:
                base_ddg = DESTABILIZING_PAIRS[(from_aa, to_aa)]
            elif (to_aa, from_aa) in DESTABILIZING_PAIRS:
                base_ddg = DESTABILIZING_PAIRS[(to_aa, from_aa)]
            else:
                base_ddg = 0.5 + _deterministic_hash(self.seed, from_aa, to_aa, pos) * 2.0

            # Optional C++ kernel: pass a short wild-type sequence context
            if cpp_available():
                wild_type_seq = "A" * max(pos, 1)
                mutant_code = next((k for k, v in AA_ONE_TO_THREE.items() if v == to_aa), "A")
                try:
                    ddg_cpp = empirical_ddg(
                        wild_type_seq, max(pos - 1, 0), mutant_code, [0.5] * len(wild_type_seq), [0] * len(wild_type_seq)
                    )
                    if ddg_cpp is not None:
                        base_ddg = max(0.0, float(ddg_cpp))
                        engine = "cpp-foldx-kernel"
                        model_id = "foldx_style_surrogate"
                        confidence = 0.8
                except Exception:
                    pass
        else:
            base_ddg = _deterministic_hash(self.seed, getattr(variant, "id", ""), "ddg") * 2.0

        base_ddg += _deterministic_hash(self.seed, getattr(variant, "id", ""), "ddg_noise") * 0.5
        return max(0.0, base_ddg), engine, model_id, confidence

    def _get_domain_impacts(self, variant: Any, ddg: float) -> list[DomainImpact]:
        impacts: list[DomainImpact] = []
        variant_hash = _deterministic_hash(self.seed, getattr(variant, "id", ""), "domains")
        num_domains = int(variant_hash * 3) + 1

        domain_ids = list(DOMAIN_SENSITIVITY.keys())
        for i in range(min(num_domains, len(domain_ids))):
            domain_id = domain_ids[i]
            sensitivity = DOMAIN_SENSITIVITY[domain_id]
            active_site = _deterministic_hash(self.seed, variant.id, domain_id, "active") > 0.9
            binding = _deterministic_hash(self.seed, variant.id, domain_id, "binding") > 0.85

            impacts.append(DomainImpact(
                domain_id=domain_id,
                delta_delta_g=ddg * sensitivity,
                active_site_disrupted=active_site,
                binding_interface_disrupted=binding,
            ))
        return impacts

    def _predict_structure_hash(self, variant: Any, aa_change: str | None) -> str:
        return hashlib.sha256(
            f"{getattr(variant, 'id', '')}:{aa_change or 'no_change'}".encode()
        ).hexdigest()[:16]

    def predict(self, variant: Any, context: dict[str, Any] | None = None) -> ProteinEffect:
        if not self.models_loaded:
            self.load_models()

        context = context or {}
        aa_change = context.get("amino_acid_change")
        uniprot_id = context.get("uniprot_id")

        # Try trained models first, fallback to surrogate
        if self._use_trained_models:
            ddg, engine, model_id, confidence = self._predict_ddg_trained(variant, aa_change)
            if ddg == 0.0 and engine == "none":
                # Fallback to surrogate
                ddg, engine, model_id, confidence = self._predict_ddg_surrogate(variant, aa_change)
        else:
            ddg, engine, model_id, confidence = self._predict_ddg_surrogate(variant, aa_change)

        domain_impacts = self._get_domain_impacts(variant, ddg)

        method = StructureMethod.ESM_FOLD
        if "foldx" in engine.lower():
            method = StructureMethod.FOLDX_STYLE_SURROGATE
        elif "esmfold" in model_id.lower():
            method = StructureMethod.ESM_FOLD
        elif "deepddg" in model_id.lower():
            method = StructureMethod.DEEP_DDG

        return ProteinEffect(
            variant_id=getattr(variant, "id", "unknown"),
            uniprot_id=uniprot_id,
            domain_id=domain_impacts[0].domain_id if domain_impacts else None,
            delta_delta_g=ddg,
            confidence=confidence,
            method=method,
            model_id=model_id,
            structure_hash=self._predict_structure_hash(variant, aa_change),
            domain_impacts=domain_impacts,
        )

    def predict_batch(self, variants: list[Any], context: dict[str, Any] | None = None) -> list[ProteinEffect]:
        return [self.predict(v, context) for v in variants]

    def aggregate_scores(self, effects: list[ProteinEffect]) -> dict[str, Any]:
        if not effects:
            return {
                "max_delta_delta_g": 0.0,
                "mean_delta_delta_g": 0.0,
                "variant_count": 0,
                "domain_impacts": [],
                "max_confidence": 0.0,
            }

        ddg_values = [e.delta_delta_g for e in effects]
        domain_impacts = []
        for effect in effects:
            for di in effect.domain_impacts:
                domain_impacts.append({
                    "domain_id": di.domain_id,
                    "delta_delta_g": di.delta_delta_g,
                    "active_site_disrupted": di.active_site_disrupted,
                    "binding_interface_disrupted": di.binding_interface_disrupted,
                })

        return {
            "max_delta_delta_g": max(ddg_values),
            "mean_delta_delta_g": sum(ddg_values) / len(ddg_values),
            "variant_count": len(effects),
            "domain_impacts": domain_impacts,
            "max_confidence": max(e.confidence for e in effects),
        }