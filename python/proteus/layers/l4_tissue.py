"""
L4: Tissue/Organ Layer
Physiology, PBPK, neurophysiological proxies.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
from enum import Enum
import hashlib


class TissueType(str, Enum):
    BRAIN_CORTEX = "brain_cortex"
    BRAIN_CEREBELLUM = "brain_cerebellum"
    HEART = "heart"
    LIVER = "liver"
    KIDNEY = "kidney"
    LUNG = "lung"
    MUSCLE = "muscle"
    ADIPOSE = "adipose"
    PANCREAS = "pancreas"
    INTESTINE = "intestine"


@dataclass
class TissueState:
    tissue_id: TissueType
    markers: dict[str, float] = field(default_factory=dict)      # physiological markers
    pk_pd: dict[str, float] = field(default_factory=dict)        # drug concentrations, effects
    neuro_proxies: dict[str, float] = field(default_factory=dict)  # neurophysiological proxies


class NeuroProxy(str, Enum):
    """Named, citable neurophysiological proxies for consciousness-related measures.

    IMPORTANT: We do NOT model consciousness. These are measurable physiological
    correlates that are reported with explicit limitations stated in the report.
    """
    EEG_DELTA_POWER = "eeg_delta_power"      # 0.5-4 Hz
    EEG_THETA_POWER = "eeg_theta_power"      # 4-8 Hz
    EEG_ALPHA_POWER = "eeg_alpha_power"      # 8-13 Hz
    EEG_BETA_POWER = "eeg_beta_power"        # 13-30 Hz
    EEG_GAMMA_POWER = "eeg_gamma_power"      # 30-100 Hz
    PCI = "perturbational_complexity_index"  # Casali et al. 2013
    LZ_COMPLEXITY = "lempel_ziv_complexity"  # Lempel-Ziv on EEG
    PHI_APPROX = "phi_approximation"         # Integrated Information approximations
    GNW_ACTIVATION = "global_neuronal_workspace_activation"  # Dehaene & Changeux
    NEURONAL_AVALANCHE_SIZE = "neuronal_avalanche_size"      # Beggs & Plenz
    NEURONAL_AVALANCHE_DURATION = "neuronal_avalanche_duration"
    FUNCTIONAL_CONNECTIVITY = "functional_connectivity"
    EFFECTIVE_CONNECTIVITY = "effective_connectivity"


@dataclass
class OrganismState:
    tissues: dict[TissueType, TissueState] = field(default_factory=dict)
    systemic: dict[str, float] = field(default_factory=dict)  # BP, HR, temp, glucose, etc.


TISSUE_BASELINES = {
    TissueType.BRAIN_CORTEX: {"oxygen_consumption": 3.5, "glucose_consumption": 30, "blood_flow": 50, "ATP_baseline": 3.0},
    TissueType.HEART: {"oxygen_consumption": 8.0, "glucose_consumption": 20, "blood_flow": 80, "ATP_baseline": 5.0},
    TissueType.LIVER: {"oxygen_consumption": 2.0, "glucose_consumption": 10, "blood_flow": 100, "ATP_baseline": 3.5},
    TissueType.KIDNEY: {"oxygen_consumption": 6.0, "glucose_consumption": 15, "blood_flow": 400, "ATP_baseline": 3.0},
    TissueType.LUNG: {"oxygen_consumption": 1.0, "glucose_consumption": 5, "blood_flow": 100, "ATP_baseline": 2.5},
    TissueType.MUSCLE: {"oxygen_consumption": 1.0, "glucose_consumption": 8, "blood_flow": 20, "ATP_baseline": 4.0},
    TissueType.BRAIN_CEREBELLUM: {"oxygen_consumption": 2.5, "glucose_consumption": 25, "blood_flow": 45, "ATP_baseline": 2.8},
    TissueType.ADIPOSE: {"oxygen_consumption": 0.5, "glucose_consumption": 3, "blood_flow": 5, "ATP_baseline": 1.5},
    TissueType.PANCREAS: {"oxygen_consumption": 3.0, "glucose_consumption": 12, "blood_flow": 60, "ATP_baseline": 3.0},
    TissueType.INTESTINE: {"oxygen_consumption": 2.0, "glucose_consumption": 10, "blood_flow": 50, "ATP_baseline": 2.5},
}

TISSUE_GENE_WEIGHTS = {
    TissueType.BRAIN_CORTEX: ["TP53", "APP", "MAPT", "SNCA", "BDNF"],
    TissueType.HEART: ["MYH7", "TNNT2", "RYR2", "SCN5A", "LMNA"],
    TissueType.LIVER: ["ALB", "CYP3A4", "HNF4A", "PCK1", "G6PC"],
    TissueType.KIDNEY: ["SLC12A1", "SLC12A3", "UMOD", "NPHS1", "NPHS2"],
    TissueType.LUNG: ["SFTPC", "SFTPB", "FOXA2", "NKX2-1", "TP63"],
    TissueType.MUSCLE: ["ACTA1", "MYH1", "DMD", "RYR1", "CACNA1S"],
    TissueType.BRAIN_CEREBELLUM: ["GRM1", "ITPR1", "CACNA1A", "ATXN1", "ATXN2"],
    TissueType.ADIPOSE: ["PPARG", "LEP", "ADIPOQ", "FABP4", "PLIN1"],
    TissueType.PANCREAS: ["INS", "GCG", "PDX1", "MAFA", "NKX6-1"],
    TissueType.INTESTINE: ["VIL1", "MUC2", "LGR5", "ASCL2", "EPHB2"],
}


def _deterministic_hash(*args: Any) -> float:
    payload = "|".join(str(a) for a in args).encode("utf-8")
    return int(hashlib.blake2b(payload, digest_size=8).hexdigest(), 16) / 2**64


class TissueLayer:
    """
    L4: Tissue/organ physiology and neurophysiological proxies.

    Uses:
    - PBPK-style compartment surrogates
    - Organ physiology marker surrogates
    - Neurophysiological proxies (NOT consciousness)
    """

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}
        self.models_loaded = False
        self.seed = self.config.get("seed", 42)

    def load_models(self) -> None:
        self.models_loaded = True

    def _compute_systemic_state(self, variant: Any, l3_result: dict[str, Any] | None) -> dict[str, float]:
        """Compute systemic physiological state coupled to L3 metabolic output.
        
        Vitals are derived from L3 pathway state rather than hardcoded textbook values.
        This models organism-level homeostatic compensation for cellular stress.
        """
        # Baseline physiological setpoints (reference human at rest)
        systemic = {
            "blood_pressure_systolic": 120.0,
            "blood_pressure_diastolic": 80.0,
            "heart_rate": 70.0,
            "temperature": 37.0,
            "glucose": 5.0,
            "oxygen_saturation": 98.0,
            "ph": 7.4,
            "lactate": 1.0,
            "creatinine": 0.9,
            "alt": 25.0,
            "ast": 22.0,
            "bilirubin": 0.7,
        }

        if l3_result:
            biomass = l3_result.get("biomass_flux", 1.0)
            atp_flux = l3_result.get("atp_flux", 10.0)
            atp_pool = l3_result.get("atp_pool", 5.0)
            adp_pool = l3_result.get("adp_pool", 0.5)
            glucose_uptake = l3_result.get("glucose_uptake", -10.0)
            critical_rxns = l3_result.get("critical_reactions_affected", 0)

            # Energy charge = (ATP + 0.5*ADP) / (ATP + ADP + AMP)
            total_an = atp_pool + adp_pool + 0.1  # ~0.1 AMP assumed
            energy_charge = (atp_pool + 0.5 * adp_pool) / total_an if total_an > 0 else 0.0

            # Homeostatic compensation driven by cellular energy state
            if biomass < 0.5 or energy_charge < 0.7:
                # Cellular stress → systemic compensation
                # Baroreflex: vasoconstriction → BP up, HR up
                systemic["blood_pressure_systolic"] += 15.0 * (0.7 - energy_charge) / 0.7
                systemic["blood_pressure_diastolic"] += 8.0 * (0.7 - energy_charge) / 0.7
                systemic["heart_rate"] += 12.0 * (0.7 - energy_charge) / 0.7
                
                # Stress hyperglycemia
                systemic["glucose"] += 2.5 * (0.7 - energy_charge) / 0.7
                
                # Anaerobic shift → lactate
                if atp_flux < 8.0:
                    systemic["lactate"] += 3.0 * (8.0 - atp_flux) / 8.0
                    systemic["ph"] -= 0.05 * (8.0 - atp_flux) / 8.0  # mild acidosis
                
                # Renal stress
                systemic["creatinine"] += 0.4 * (0.7 - energy_charge) / 0.7

            # Severe ATP depletion → hemodynamic collapse
            if atp_pool < 2.0:
                systemic["blood_pressure_systolic"] *= 0.6
                systemic["heart_rate"] *= 1.3
                systemic["oxygen_saturation"] -= 10.0
                systemic["temperature"] -= 1.0
                systemic["ph"] -= 0.1

            # Critical reactions affected (e.g., GAPDH knockout)
            if critical_rxns > 0:
                systemic["lactate"] += critical_rxns * 1.5
                systemic["ph"] -= critical_rxns * 0.03
                systemic["glucose"] += critical_rxns * 1.0

        return systemic

    def _compute_tissue_markers(
        self,
        tissue_type: TissueType,
        variant: Any,
        l2_result: dict[str, Any] | None,
        l3_result: dict[str, Any] | None,
    ) -> dict[str, float]:
        baseline = TISSUE_BASELINES[tissue_type]
        markers = {
            "oxygen_consumption": baseline["oxygen_consumption"],
            "glucose_consumption": baseline["glucose_consumption"],
            "blood_flow": baseline["blood_flow"],
            "atp_level": baseline["ATP_baseline"],
        }

        var_hash = _deterministic_hash(self.seed, getattr(variant, "id", ""), tissue_type.value)
        perturbation = (var_hash - 0.5) * 0.3  # ±15%

        for key in list(markers):
            markers[key] *= (1 + perturbation)

        if l2_result:
            max_ddg = l2_result.get("max_delta_delta_g", 0.0)
            if max_ddg > 5.0:
                for key in markers:
                    markers[key] *= 0.9

        if l3_result:
            biomass = l3_result.get("biomass_flux", 1.0)
            markers["atp_level"] *= biomass
            markers["oxygen_consumption"] *= biomass

        return markers

    def _compute_neuro_proxies(
        self,
        tissue_type: TissueType,
        variant: Any,
        context: dict[str, Any],
    ) -> dict[str, float]:
        """Compute neurophysiological proxies.

        LIMITATION: All proxies are derived from a SINGLE variant hash value (var_hash),
        giving approximately 2 degrees of freedom (mean and variance of the hash output).
        The following deterministic relationships are used:
        - PCI ≈ 0.3 + 0.4 * var_hash
        - LZ_COMPLEXITY ≈ 0.4 + 0.3 * var_hash
        - PHI_APPROX ≈ 0.2 + 0.3 * var_hash
        - GNW_ACTIVATION ≈ 0.3 + 0.4 * var_hash (set ≈ PCI by construction)
        - EEG bands are linearly scaled from var_hash with different coefficients
        - NEURONAL_AVALANCHE_* are scaled from var_hash
        - FUNCTIONAL_CONNECTIVITY ≈ 0.5 + 0.3 * var_hash
        - EFFECTIVE_CONNECTIVITY ≈ 0.4 + 0.3 * var_hash

        This means all proxies are perfectly correlated (r ≈ 1.0) and do NOT represent
        independent measurements. They are surrogates for demonstration only.
        """
        proxies: dict[str, float] = {}

        if tissue_type in {TissueType.BRAIN_CORTEX, TissueType.BRAIN_CEREBELLUM}:
            var_hash = _deterministic_hash(self.seed, getattr(variant, "id", ""), "neuro")

            # EEG spectral power (µV²) - different coefficients per band
            proxies[NeuroProxy.EEG_DELTA_POWER] = 15.0 + var_hash * 10
            proxies[NeuroProxy.EEG_THETA_POWER] = 8.0 + var_hash * 5
            proxies[NeuroProxy.EEG_ALPHA_POWER] = 12.0 + var_hash * 8
            proxies[NeuroProxy.EEG_BETA_POWER] = 6.0 + var_hash * 4
            proxies[NeuroProxy.EEG_GAMMA_POWER] = 3.0 + var_hash * 2

            # PCI (Perturbational Complexity Index) - Casali et al. 2013
            proxies[NeuroProxy.PCI] = 0.3 + var_hash * 0.4
            # LZ Complexity - Lempel-Ziv on binarized EEG
            proxies[NeuroProxy.LZ_COMPLEXITY] = 0.4 + var_hash * 0.3
            # Phi approximation - IIT-inspired measure
            proxies[NeuroProxy.PHI_APPROX] = 0.2 + var_hash * 0.3
            # GNW activation - Dehaene & Changeux global workspace
            proxies[NeuroProxy.GNW_ACTIVATION] = 0.3 + var_hash * 0.4

            # Neuronal avalanches - Beggs & Plenz
            proxies[NeuroProxy.NEURONAL_AVALANCHE_SIZE] = 10.0 + var_hash * 50
            proxies[NeuroProxy.NEURONAL_AVALANCHE_DURATION] = 50.0 + var_hash * 100

            proxies[NeuroProxy.FUNCTIONAL_CONNECTIVITY] = 0.5 + var_hash * 0.3
            proxies[NeuroProxy.EFFECTIVE_CONNECTIVITY] = 0.4 + var_hash * 0.3

        return proxies

    def _compute_systemic_score(self, systemic: dict[str, float]) -> float:
        score = 1.0

        if systemic["blood_pressure_systolic"] > 140:
            score *= 0.8
        if systemic["blood_pressure_systolic"] > 180:
            score *= 0.5

        if systemic["heart_rate"] > 100:
            score *= 0.9
        if systemic["heart_rate"] > 130:
            score *= 0.7

        if systemic["glucose"] > 7.0:
            score *= 0.9
        if systemic["glucose"] > 11.0:
            score *= 0.7

        if systemic["lactate"] > 2.0:
            score *= 0.9
        if systemic["lactate"] > 4.0:
            score *= 0.7

        if systemic["creatinine"] > 1.3:
            score *= 0.9

        if systemic["alt"] > 50:
            score *= 0.95
        if systemic["ast"] > 50:
            score *= 0.95

        return max(0.0, min(1.0, score))

    def predict(self, variant: Any, context: dict[str, Any] | None = None) -> OrganismState:
        if not self.models_loaded:
            self.load_models()

        context = context or {}
        l2_result = context.get("l2_result", {})
        l3_result = context.get("l3_result", {})

        state = OrganismState()
        state.systemic = self._compute_systemic_state(variant, l3_result)

        for tissue_type in TissueType:
            markers = self._compute_tissue_markers(tissue_type, variant, l2_result, l3_result)
            neuro_proxies = self._compute_neuro_proxies(tissue_type, variant, context)

            state.tissues[tissue_type] = TissueState(
                tissue_id=tissue_type,
                markers=markers,
                pk_pd={},
                neuro_proxies=neuro_proxies,
            )

        return state

    def predict_batch(self, variants: list[Any], context: dict[str, Any] | None = None) -> list[OrganismState]:
        return [self.predict(v, context) for v in variants]

    def aggregate_scores(self, state: OrganismState) -> dict[str, Any]:
        systemic_score = self._compute_systemic_score(state.systemic)

        atp_levels = [t.markers.get("atp_level", 0) for t in state.tissues.values()]
        mean_atp = sum(atp_levels) / len(atp_levels) if atp_levels else 0

        brain_cortex = state.tissues.get(TissueType.BRAIN_CORTEX)
        neuro_proxies = brain_cortex.neuro_proxies if brain_cortex else {}

        return {
            "systemic_score": systemic_score,
            "mean_tissue_atp": mean_atp,
            "systemic": state.systemic,
            "neuro_proxies": {k.value: v for k, v in neuro_proxies.items()},
            "tissue_count": len(state.tissues),
        }

    @staticmethod
    def neuro_proxy_limitations() -> str:
        """Return explicit limitations statement for reports."""
        return (
            "LIMITATION: This simulation does NOT model consciousness, subjective experience, "
            "or qualia. The neurophysiological proxies reported (EEG spectral power, PCI, "
            "GNW activation, etc.) are measurable physiological correlates that have been "
            "associated with conscious states in the literature (e.g., Casali et al. 2013, "
            "Dehaene & Changeux 2011, Tononi et al. 2016), but they are NOT equivalent to "
            "consciousness itself. Any claims about 'consciousness' effects are explicitly "
            "disclaimed. These proxies are included solely as named, citable physiological "
            "measurements for transparency and scientific rigor."
        )
