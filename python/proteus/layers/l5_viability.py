"""
L5: Viability Layer
Composite survival score, decision gate, survival-window estimation.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
from enum import Enum


class Verdict(str, Enum):
    VIABLE = "viable"
    NON_VIABLE = "non_viable"
    UNCERTAIN = "uncertain"


@dataclass
class ViabilityScore:
    composite: float  # 0-1
    layer_scores: dict[str, float] = field(default_factory=dict)  # L1-L4 contributions
    weights: dict[str, float] = field(default_factory=dict)
    failed_gates: list[str] = field(default_factory=list)  # hard thresholds violated
    verdict: Verdict = Verdict.UNCERTAIN
    confidence: float = 0.0  # 0-1
    # New headline outputs
    survival_window_h: float | None = None  # estimated hours to failure/death
    point_of_no_return_h: float | None = None  # time of first irreversible event
    irreversible_events: list[dict] = field(default_factory=list)
    swap_gate_results: dict[str, Any] = field(default_factory=dict)
    hard_gate_failures: list[str] = field(default_factory=list)


class ViabilityLayer:
    """
    L5: Viability decision layer.
    
    Fuses evidence from L1-L4 using weighted scoring with hard gates.
    
    Hard gates are evaluated BEFORE weighted scoring and can independently
    force NON_VIABLE. This ensures that single catastrophic failures
    (essential gene KO, stop-gain, ATP collapse) are not masked by
    high scores in other layers.
    
    Survival window is estimated from time-dynamic irreversible events
    (L3b) when available; otherwise inferred from damage severity.
    """

    # Default weights (configurable) - re-weighted so sensitive layers matter
    # L1 (variant semantics) increased because stop-gain/essential KO is detected here
    # L3 (pathway) increased because biomass/ATP collapse is detected here
    DEFAULT_WEIGHTS = {
        "L1": 0.25,  # Variant semantics - detects stop-gain, essential gene KO
        "L2": 0.20,  # Protein stability
        "L3": 0.35,  # Pathway/cellular - detects biomass/ATP collapse
        "L4": 0.20,  # Tissue/organ - systemic compensation
    }

    # Hard gates (if any fails -> NON_VIABLE)
    HARD_GATES = {
        "L2_delta_delta_g_max": 10.0,    # kcal/mol, >10 = likely non-viable
        "L3_biomass_min": 0.01,          # minimum biomass flux
        "L4_systemic_collapse": 0.5,     # systemic score threshold
    }

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}
        self.weights = self.config.get("weights", self.DEFAULT_WEIGHTS)
        self.hard_gates = self.config.get("hard_gates", self.HARD_GATES)

    def evaluate(
        self,
        l1_results: dict[str, Any],
        l2_results: dict[str, Any],
        l3_results: dict[str, Any],
        l4_results: dict[str, Any],
        time_dynamic: dict[str, Any] | None = None,
        swap_gate_results: dict[str, Any] | None = None,
    ) -> ViabilityScore:
        """Compute viability score from layer results.

        Args:
            l1_results: L1 variant semantics aggregate.
            l2_results: L2 protein stability aggregate.
            l3_results: L3 pathway aggregate.
            l4_results: L4 tissue aggregate.
            time_dynamic: Optional L3b time-dynamic results with irreversible
                events, survival window estimates.
            swap_gate_results: Optional swap-specific gate results.
        """
        time_dynamic = time_dynamic or {}
        swap_gate_results = swap_gate_results or {}

        layer_scores = {}
        failed_gates = []
        hard_gate_failures = []

        # === HARD GATES (evaluated first - can independently force NON_VIABLE) ===
        
        # Gate 1: Essential gene knockout (GAPDH, etc.) detected in L1/L3
        # L1 consequence shows stop_gained on essential gene, or L3 biomass = 0
        l1_consequence = l1_results.get("max_severity", 0.0)
        if l1_consequence >= 8.5:  # stop_gained, frameshift, transcript_ablation severity
            failed_gates.append("L1_essential_gene_stop_gain: high-severity consequence detected")
            hard_gate_failures.append("L1_essential_gene_stop_gain")
        
        # Gate 2: GAPDH (or other essential metabolic gene) knockout in L3
        biomass = l3_results.get("biomass_flux", 1.0)
        critical_rxns = l3_results.get("critical_reactions_affected", 0)
        if critical_rxns > 0 or biomass <= 0.01:
            failed_gates.append(f"L3_essential_gene_knockout: biomass={biomass:.4f}, critical_rxns={critical_rxns}")
            hard_gate_failures.append("L3_essential_gene_knockout")

        # Gate 3: ADP/ATP collapse (energy charge < 0.3)
        atp_pool = l3_results.get("atp_pool", 5.0)
        adp_pool = l3_results.get("adp_pool", 0.5)
        total_an = atp_pool + adp_pool + 0.1
        energy_charge = (atp_pool + 0.5 * adp_pool) / total_an if total_an > 0 else 0.0
        if energy_charge < 0.3:
            failed_gates.append(f"L3_energy_charge_collapse: EC={energy_charge:.2f} < 0.3 (ATP={atp_pool:.1f}, ADP={adp_pool:.1f})")
            hard_gate_failures.append("L3_energy_charge_collapse")

        # Gate 4: Systemic collapse (from L4)
        systemic_score = l4_results.get("systemic_score", 1.0)
        if systemic_score < self.hard_gates["L4_systemic_collapse"]:
            failed_gates.append(f"L4_systemic_collapse: {systemic_score:.2f} < {self.hard_gates['L4_systemic_collapse']}")
            hard_gate_failures.append("L4_systemic_collapse")

        # Gate 5: Protein stability catastrophic (L2)
        max_ddg = l2_results.get("max_delta_delta_g", 0.0)
        if max_ddg > self.hard_gates["L2_delta_delta_g_max"]:
            failed_gates.append(f"L2_delta_delta_g_max: {max_ddg:.1f} > {self.hard_gates['L2_delta_delta_g_max']}")
            hard_gate_failures.append("L2_delta_delta_g_max")

        # Gate 6: Swap-specific hard gates (mito-nuclear, etc.)
        for gate_name, gate_result in swap_gate_results.items():
            if isinstance(gate_result, dict):
                result = gate_result.get("result", "")
                score = gate_result.get("score", 1.0)
                msg = gate_result.get("message", "")
                if result == "fail":
                    failed_gates.append(f"{gate_name}: {msg}")
                    hard_gate_failures.append(gate_name)

        # === WEIGHTED SCORING (always computed so composite reflects model) ===
        
        # L1 score: based on pathogenicity burden
        l1_score = self._score_l1(l1_results)
        layer_scores["L1"] = l1_score

        # L2 score: protein stability
        l2_score, l2_failed = self._score_l2(l2_results)
        layer_scores["L2"] = l2_score
        failed_gates.extend(l2_failed)

        # L3 score: pathway viability
        l3_score, l3_failed = self._score_l3(l3_results)
        layer_scores["L3"] = l3_score
        failed_gates.extend(l3_failed)

        # L4 score: organism physiology
        l4_score, l4_failed = self._score_l4(l4_results)
        layer_scores["L4"] = l4_score
        failed_gates.extend(l4_failed)

        # Composite score (weighted average)
        composite = sum(
            layer_scores.get(layer, 0.0) * self.weights.get(layer, 0.0)
            for layer in ["L1", "L2", "L3", "L4"]
        )

        # Apply swap gate modifiers (soft gates multiply score)
        swap_composite = 1.0
        for gate_name, gate_result in swap_gate_results.items():
            if isinstance(gate_result, dict):
                result = gate_result.get("result", "")
                score = gate_result.get("score", 1.0)
                if result != "fail":  # hard fails already handled above
                    swap_composite *= score
        composite *= swap_composite

        # If any hard gate fails, short-circuit to NON_VIABLE but keep layer_scores
        if hard_gate_failures:
            return ViabilityScore(
                composite=0.0,
                layer_scores=layer_scores,
                weights=self.weights,
                failed_gates=failed_gates,
                verdict=Verdict.NON_VIABLE,
                confidence=0.95,
                hard_gate_failures=hard_gate_failures,
                survival_window_h=self._estimate_survival_window(time_dynamic),
                point_of_no_return_h=time_dynamic.get("point_of_no_return_h"),
                irreversible_events=time_dynamic.get("irreversible_events", []),
                swap_gate_results=swap_gate_results,
            )

        # Soft gate check
        if composite > 0.7:
            verdict = Verdict.VIABLE
            confidence = 0.8
        elif composite < 0.3:
            verdict = Verdict.NON_VIABLE
            confidence = 0.8
        else:
            verdict = Verdict.UNCERTAIN
            confidence = 0.5

        return ViabilityScore(
            composite=composite,
            layer_scores=layer_scores,
            weights=self.weights,
            failed_gates=failed_gates,
            verdict=verdict,
            confidence=confidence,
            survival_window_h=self._estimate_survival_window(time_dynamic),
            point_of_no_return_h=time_dynamic.get("point_of_no_return_h"),
            irreversible_events=time_dynamic.get("irreversible_events", []),
            swap_gate_results=swap_gate_results,
            hard_gate_failures=hard_gate_failures,
        )

    def _estimate_survival_window(self, time_dynamic: dict[str, Any]) -> float | None:
        """Estimate survival window in hours from time-dynamic results."""
        point = time_dynamic.get("point_of_no_return_h")
        if point is not None:
            return point
        # Fallback: infer from irreversible events
        events = time_dynamic.get("irreversible_events", [])
        if events:
            times = [e.get("time") for e in events if e.get("time") is not None]
            if times:
                return min(times)
        return None

    def _score_l1(self, results: dict[str, Any]) -> float:
        """Score variant semantics layer."""
        # Based on max pathogenicity score
        max_path = results.get("max_pathogenicity", 0.0)
        return 1.0 - max_path  # lower pathogenicity = higher viability

    def _score_l2(self, results: dict[str, Any]) -> tuple[float, list[str]]:
        """Score protein layer."""
        failed = []
        max_ddg = results.get("max_delta_delta_g", 0.0)
        
        if max_ddg > self.hard_gates["L2_delta_delta_g_max"]:
            failed.append(f"L2_delta_delta_g_max: {max_ddg:.1f} > {self.hard_gates['L2_delta_delta_g_max']}")

        # Score: exponential decay with ΔΔG
        score = max(0.0, 1.0 - max_ddg / 20.0)  # 20 kcal/mol = 0 score
        return score, failed

    def _score_l3(self, results: dict[str, Any]) -> tuple[float, list[str]]:
        """Score pathway layer."""
        failed = []
        biomass = results.get("biomass_flux", 1.0)

        if biomass < self.hard_gates["L3_biomass_min"]:
            failed.append(f"L3_biomass_min: {biomass:.4f} < {self.hard_gates['L3_biomass_min']}")

        score = min(1.0, biomass * 10)  # normalize
        return score, failed

    def _score_l4(self, results: dict[str, Any]) -> tuple[float, list[str]]:
        """Score tissue/organ layer."""
        failed = []
        systemic_score = results.get("systemic_score", 1.0)

        if systemic_score < self.hard_gates["L4_systemic_collapse"]:
            failed.append(f"L4_systemic_collapse: {systemic_score:.2f} < {self.hard_gates['L4_systemic_collapse']}")

        return systemic_score, failed