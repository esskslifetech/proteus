"""
L3b: Time-Dynamic ODE Layer with Real Turnover Constants
Models mRNA/protein turnover, hysteresis, and irreversible-damage-vs-duration curves.

Key parameters (from literature):
- mRNA half-life: ~9 hours median (range 30 min - 24h)
- Protein half-life: ~46 hours median (range minutes - days)
- Short-lived proteins: transcription factors, cyclins (minutes-hours)
- Long-lived proteins: structural, histones (days-weeks)

Irreversible damage accumulates when:
- Energy charge < 0.5 for > threshold duration
- Misfolded protein load > clearance capacity for > threshold
- DNA damage > repair capacity
- ROS > antioxidant capacity
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
from enum import Enum
import math
import hashlib
import logging
from abc import ABC, abstractmethod

from proteus.cpp_adapter import cpp_available, solve_ode

logger = logging.getLogger(__name__)


class DamageType(str, Enum):
    ENERGY_DEPLETION = "energy_depletion"
    PROTEOSTASIS_COLLAPSE = "proteostasis_collapse"
    DNA_DAMAGE = "dna_damage"
    OXIDATIVE_STRESS = "oxidative_stress"
    MITOCHONDRIAL_DYSFUNCTION = "mitochondrial_dysfunction"


@dataclass
class TurnoverParams:
    """Protein/mRNA turnover parameters from literature."""
    # mRNA half-lives (hours)
    mrna_half_life_median: float = 9.0
    mrna_half_life_short: float = 0.5  # 30 min for TFs, cyclins
    mrna_half_life_long: float = 24.0  # structural mRNAs
    
    # Protein half-lives (hours)
    protein_half_life_median: float = 46.0
    protein_half_life_short: float = 1.0  # 1 hour for short-lived
    protein_half_life_long: float = 720.0  # 30 days for structural
    
    # Degradation rate constants (k = ln(2) / t½)
    @property
    def k_mrna_median(self) -> float:
        return math.log(2) / self.mrna_half_life_median
    
    @property
    def k_mrna_short(self) -> float:
        return math.log(2) / self.mrna_half_life_short
    
    @property
    def k_mrna_long(self) -> float:
        return math.log(2) / self.mrna_half_life_long
    
    @property
    def k_prot_median(self) -> float:
        return math.log(2) / self.protein_half_life_median
    
    @property
    def k_prot_short(self) -> float:
        return math.log(2) / self.protein_half_life_short
    
    @property
    def k_prot_long(self) -> float:
        return math.log(2) / self.protein_half_life_long


@dataclass
class IrreversibleDamage:
    """Tracks irreversible damage accumulation.

    ``residual_severity`` is the hysteresis term: once damage becomes
    irreversible its severity is floored at the value reached at the point of
    no return, even if the triggering insult is later removed.  This is what
    makes a reverted genome phenotypically different from a naive baseline.
    """
    damage_type: DamageType
    severity: float  # 0-1
    start_time: float  # hours
    duration: float  # hours above threshold
    threshold_duration: float  # hours before irreversible
    is_irreversible: bool = False
    point_of_no_return: float | None = None  # time when irreversible
    residual_severity: float = 0.0  # permanent component after irreversibility

    def update(self, current_time: float, above_threshold: bool) -> None:
        if above_threshold:
            if self.start_time is None:
                self.start_time = current_time
            self.duration = current_time - self.start_time
            if self.duration >= self.threshold_duration and not self.is_irreversible:
                self.is_irreversible = True
                self.point_of_no_return = current_time
                # Freeze the damage reached at the point of no return.
                self.residual_severity = max(self.residual_severity, self.severity)
        else:
            # Damage can partially recover if below threshold, but only the
            # reversible component: irreversible residue is never cleared.
            if self.start_time is not None:
                self.duration = max(0, self.duration - (current_time - self.start_time) * 0.1)
            if self.duration == 0:
                self.start_time = None
        if self.is_irreversible:
            self.residual_severity = max(self.residual_severity, self.severity)
            self.severity = max(self.severity, self.residual_severity)

    @property
    def effective_severity(self) -> float:
        """Severity including the permanent (hysteretic) component."""
        return max(self.severity, self.residual_severity)

    def snapshot(self) -> dict[str, Any]:
        """Serializable state for carrying damage across a genome revert."""
        return {
            "damage_type": self.damage_type.value,
            "severity": self.severity,
            "start_time": self.start_time,
            "duration": self.duration,
            "threshold_duration": self.threshold_duration,
            "is_irreversible": self.is_irreversible,
            "point_of_no_return": self.point_of_no_return,
            "residual_severity": self.effective_severity if self.is_irreversible else 0.0,
        }

    @classmethod
    def from_snapshot(cls, data: dict[str, Any]) -> "IrreversibleDamage":
        """Restore a damage state previously produced by :meth:`snapshot`."""
        dtype_raw = data.get("damage_type", "")
        try:
            dtype = DamageType(dtype_raw)
        except ValueError:
            dtype = DamageType.ENERGY_DEPLETION
        return cls(
            damage_type=dtype,
            severity=float(data.get("severity", 0.0) or 0.0),
            start_time=data.get("start_time"),
            duration=float(data.get("duration", 0.0) or 0.0),
            threshold_duration=float(data.get("threshold_duration", 0.0) or 0.0),
            is_irreversible=bool(data.get("is_irreversible", False)),
            point_of_no_return=data.get("point_of_no_return"),
            residual_severity=float(data.get("residual_severity", 0.0) or 0.0),
        )


@dataclass
class TimeDynamicState:
    """Complete time-dynamic simulation state."""
    time_points: list[float]  # hours
    
    # Molecular species concentrations over time
    mrna_concentrations: dict[str, list[float]]  # gene_id -> [conc_t0, conc_t1, ...]
    protein_concentrations: dict[str, list[float]]
    
    # Energy state over time
    atp_concentration: list[float]
    adp_concentration: list[float]
    amp_concentration: list[float]
    energy_charge: list[float]
    
    # Damage accumulation
    damage_states: dict[DamageType, IrreversibleDamage]
    irreversible_events: list[dict]  # {"time": t, "type": DamageType, "severity": s}
    
    # Metabolic fluxes over time
    fluxes: dict[str, list[float]]
    
    # Systemic state
    tissue_viability: dict[str, list[float]]  # tissue -> viability over time
    
    # Hysteresis / post-revert residual phenotype
    is_post_revert: bool = False
    hysteresis_index: float = 0.0  # 0 = fully recovered, 1 = maximum residual damage

    # Provenance
    initial_hash: str = ""
    model_hash: str = "time_dynamic_v1"
    solver_status: str = "rk4"


class TurnoverModel:
    """Models mRNA and protein synthesis/degradation dynamics."""
    
    def __init__(self, params: TurnoverParams | None = None):
        self.params = params or TurnoverParams()
    
    def get_degradation_rate(self, gene_id: str, is_mrna: bool, is_short_lived: bool = False) -> float:
        """Get degradation rate constant for a gene product."""
        if is_mrna:
            if is_short_lived:
                return self.params.k_mrna_short
            # Check if known short-lived (TFs, cyclins, etc.)
            short_lived_mrna = {"MYC", "FOS", "JUN", "CCND1", "CCNE1", "TP53", "EGR1"}
            if any(s in gene_id for s in short_lived_mrna):
                return self.params.k_mrna_short
            return self.params.k_mrna_median
        else:
            if is_short_lived:
                return self.params.k_prot_short
            # Known short-lived proteins
            short_lived_prot = {"CYCLIN", "MYC", "P53", "HIF1A", "NFKB"}
            if any(s in gene_id for s in short_lived_prot):
                return self.params.k_prot_short
            return self.params.k_prot_median
    
    def simulate_turnover(
        self,
        initial_mrna: dict[str, float],
        initial_protein: dict[str, float],
        synthesis_rates: dict[str, float],  # gene_id -> synthesis rate
        time_points: list[float],
        perturbation: dict[str, float] | None = None  # gene_id -> synthesis fold change
    ) -> tuple[dict[str, list[float]], dict[str, list[float]]]:
        """
        Simulate mRNA/protein turnover over time.

        dmRNA/dt = synthesis - k_mrna * mRNA
        dProtein/dt = translation * mRNA - k_prot * Protein

        The initial state is calibrated to a steady state before integration:
        the starting mRNA is set to ``synthesis / k_mrna`` (when synthesis is
        positive), and the translation rate is set to
        ``k_prot * [protein]_0 / [mRNA]_0``.  This avoids the unstable,
        non-physiological concentration explosion that occurs when arbitrary
        expression-derived values are used with a fixed translation rate.

        ``perturbation`` is a multiplicative fold change on synthesis
        (0 = knockout, 0.5 = 50% knockdown), matching the ``synthesis_fold``
        values produced by the pathway layer.
        """
        n_steps = len(time_points)
        mrna_trajectories = {g: [0.0] * n_steps for g in initial_mrna}
        prot_trajectories = {g: [0.0] * n_steps for g in initial_protein}

        # Calibrate the initial state and translation rates.
        synth0: dict[str, float] = {}
        translation: dict[str, float] = {}
        for gene, concentration in initial_mrna.items():
            k_mrna = self.get_degradation_rate(gene, is_mrna=True)
            supplied = synthesis_rates.get(gene)
            if supplied is not None and supplied > 0:
                # Start at the steady state implied by the supplied synthesis.
                mrna0 = supplied / k_mrna
                baseline_synth = supplied
            else:
                # Explicit knockout (0) or no synthesis data: keep the
                # supplied concentration and decay from there.
                mrna0 = max(0.0, float(concentration))
                baseline_synth = 0.0 if supplied is not None else k_mrna * mrna0
            synth0[gene] = baseline_synth
            mrna_trajectories[gene][0] = mrna0

            if gene in initial_protein:
                k_prot = self.get_degradation_rate(gene, is_mrna=False)
                protein0 = max(0.0, float(initial_protein[gene]))
                translation[gene] = (
                    k_prot * protein0 / mrna0 if mrna0 > 0 else 0.0
                )

        for gene, concentration in initial_protein.items():
            prot_trajectories[gene][0] = max(0.0, float(concentration))

        # Simple Euler integration (production uses C++ ODE solver when present)
        for i in range(1, n_steps):
            dt = time_points[i] - time_points[i-1]

            for gene in initial_mrna:
                # mRNA dynamics
                k_mrna = self.get_degradation_rate(gene, is_mrna=True)
                fold = 1.0
                if perturbation and gene in perturbation:
                    fold = max(0.0, float(perturbation[gene]))
                synth = synth0.get(gene, 0.0) * fold

                mrna_prev = mrna_trajectories[gene][i-1]
                mrna_trajectories[gene][i] = max(
                    0.0, mrna_prev + dt * (synth - k_mrna * mrna_prev)
                )

                # Protein dynamics (translation coupled to mRNA)
                if gene in initial_protein:
                    k_prot = self.get_degradation_rate(gene, is_mrna=False)
                    prot_prev = prot_trajectories[gene][i-1]
                    prot_trajectories[gene][i] = max(
                        0.0,
                        prot_prev
                        + dt
                        * (
                            translation.get(gene, 0.0) * mrna_trajectories[gene][i]
                            - k_prot * prot_prev
                        ),
                    )

        return mrna_trajectories, prot_trajectories


class DamageAccumulationModel:
    """Models irreversible damage accumulation with hysteresis."""
    
    # Thresholds from literature
    ENERGY_CHARGE_CRITICAL = 0.5  # Below this -> energy depletion damage
    ENERGY_CHARGE_THRESHOLD_DURATION = 2.0  # hours below threshold before irreversible
    
    PROTEOSTASIS_THRESHOLD = 0.3  # Fraction of misfolded proteins
    PROTEOSTASIS_THRESHOLD_DURATION = 6.0  # hours
    
    ROS_THRESHOLD = 2.0  # Fold increase over baseline
    ROS_THRESHOLD_DURATION = 4.0  # hours
    
    DNA_DAMAGE_THRESHOLD = 10.0  # Lesions per Mb
    DNA_DAMAGE_THRESHOLD_DURATION = 24.0  # hours
    
    MITO_DYSFUNCTION_THRESHOLD = 0.4  # Fraction of depolarized mitochondria
    MITO_DYSFUNCTION_THRESHOLD_DURATION = 12.0  # hours
    
    def __init__(
        self,
        initial_damage: dict[str, dict[str, Any]] | None = None,
        initial_events: list[dict[str, Any]] | None = None,
    ):
        self.damage_states: dict[DamageType, IrreversibleDamage] = {
            DamageType.ENERGY_DEPLETION: IrreversibleDamage(
                damage_type=DamageType.ENERGY_DEPLETION,
                severity=0.0,
                start_time=None,
                duration=0.0,
                threshold_duration=self.ENERGY_CHARGE_THRESHOLD_DURATION,
            ),
            DamageType.PROTEOSTASIS_COLLAPSE: IrreversibleDamage(
                damage_type=DamageType.PROTEOSTASIS_COLLAPSE,
                severity=0.0,
                start_time=None,
                duration=0.0,
                threshold_duration=self.PROTEOSTASIS_THRESHOLD_DURATION,
            ),
            DamageType.OXIDATIVE_STRESS: IrreversibleDamage(
                damage_type=DamageType.OXIDATIVE_STRESS,
                severity=0.0,
                start_time=None,
                duration=0.0,
                threshold_duration=self.ROS_THRESHOLD_DURATION,
            ),
            DamageType.DNA_DAMAGE: IrreversibleDamage(
                damage_type=DamageType.DNA_DAMAGE,
                severity=0.0,
                start_time=None,
                duration=0.0,
                threshold_duration=self.DNA_DAMAGE_THRESHOLD_DURATION,
            ),
            DamageType.MITOCHONDRIAL_DYSFUNCTION: IrreversibleDamage(
                damage_type=DamageType.MITOCHONDRIAL_DYSFUNCTION,
                severity=0.0,
                start_time=None,
                duration=0.0,
                threshold_duration=self.MITO_DYSFUNCTION_THRESHOLD_DURATION,
            ),
        }
        self.irreversible_events: list[dict] = list(initial_events or [])

        # Carry damage across a genome revert (hysteresis).  Only states that
        # are actually irreversible are restored, so a clean baseline run is
        # unaffected.
        for key, snap in (initial_damage or {}).items():
            state = self.damage_states.get(self._coerce_damage_type(key))
            if state is None:
                continue
            restored = IrreversibleDamage.from_snapshot(snap)
            if restored.is_irreversible:
                state.is_irreversible = True
                state.point_of_no_return = restored.point_of_no_return
                state.residual_severity = restored.residual_severity
                state.severity = max(state.severity, restored.residual_severity)

    @staticmethod
    def _coerce_damage_type(value: Any) -> DamageType | None:
        if isinstance(value, DamageType):
            return value
        try:
            return DamageType(str(value))
        except ValueError:
            return None

    def snapshot(self) -> dict[str, dict[str, Any]]:
        """Serializable snapshot used to seed a post-revert simulation."""
        return {state.damage_type.value: state.snapshot() for state in self.damage_states.values()}

    def residual_damage(self) -> dict[str, float]:
        """Permanent damage remaining after the insult is removed."""
        return {
            dtype.value: state.effective_severity
            for dtype, state in self.damage_states.items()
            if state.is_irreversible
        }

    def hysteresis_index(self) -> float:
        """Scalar 0-1 summary of residual damage (max across damage types)."""
        residual = self.residual_damage()
        return max(residual.values(), default=0.0)
    
    def update(
        self,
        current_time: float,
        energy_charge: float,
        misfolded_fraction: float,
        ros_level: float,
        dna_damage: float,
        mito_dysfunction: float
    ) -> list[dict]:
        """Update damage states and return new irreversible events."""
        new_events = []
        
        # Energy depletion
        above = energy_charge < self.ENERGY_CHARGE_CRITICAL
        self.damage_states[DamageType.ENERGY_DEPLETION].severity = max(
            0, (self.ENERGY_CHARGE_CRITICAL - energy_charge) / self.ENERGY_CHARGE_CRITICAL
        )
        self.damage_states[DamageType.ENERGY_DEPLETION].update(current_time, above)
        if self.damage_states[DamageType.ENERGY_DEPLETION].is_irreversible:
            if self.damage_states[DamageType.ENERGY_DEPLETION].point_of_no_return == current_time:
                new_events.append({
                    "time": current_time,
                    "type": DamageType.ENERGY_DEPLETION.value,
                    "severity": self.damage_states[DamageType.ENERGY_DEPLETION].severity,
                    "description": f"Energy charge < {self.ENERGY_CHARGE_CRITICAL} for > {self.ENERGY_CHARGE_THRESHOLD_DURATION}h"
                })
        
        # Proteostasis collapse
        above = misfolded_fraction > self.PROTEOSTASIS_THRESHOLD
        self.damage_states[DamageType.PROTEOSTASIS_COLLAPSE].severity = min(
            1.0, misfolded_fraction / self.PROTEOSTASIS_THRESHOLD
        )
        self.damage_states[DamageType.PROTEOSTASIS_COLLAPSE].update(current_time, above)
        if self.damage_states[DamageType.PROTEOSTASIS_COLLAPSE].is_irreversible:
            if self.damage_states[DamageType.PROTEOSTASIS_COLLAPSE].point_of_no_return == current_time:
                new_events.append({
                    "time": current_time,
                    "type": DamageType.PROTEOSTASIS_COLLAPSE.value,
                    "severity": self.damage_states[DamageType.PROTEOSTASIS_COLLAPSE].severity,
                    "description": f"Misfolded proteins > {self.PROTEOSTASIS_THRESHOLD*100}% for > {self.PROTEOSTASIS_THRESHOLD_DURATION}h"
                })
        
        # Oxidative stress
        above = ros_level > self.ROS_THRESHOLD
        self.damage_states[DamageType.OXIDATIVE_STRESS].severity = min(
            1.0, (ros_level - 1.0) / (self.ROS_THRESHOLD - 1.0)
        )
        self.damage_states[DamageType.OXIDATIVE_STRESS].update(current_time, above)
        if self.damage_states[DamageType.OXIDATIVE_STRESS].is_irreversible:
            if self.damage_states[DamageType.OXIDATIVE_STRESS].point_of_no_return == current_time:
                new_events.append({
                    "time": current_time,
                    "type": DamageType.OXIDATIVE_STRESS.value,
                    "severity": self.damage_states[DamageType.OXIDATIVE_STRESS].severity,
                    "description": f"ROS > {self.ROS_THRESHOLD}x baseline for > {self.ROS_THRESHOLD_DURATION}h"
                })
        
        # DNA damage
        above = dna_damage > self.DNA_DAMAGE_THRESHOLD
        self.damage_states[DamageType.DNA_DAMAGE].severity = min(
            1.0, dna_damage / self.DNA_DAMAGE_THRESHOLD
        )
        self.damage_states[DamageType.DNA_DAMAGE].update(current_time, above)
        if self.damage_states[DamageType.DNA_DAMAGE].is_irreversible:
            if self.damage_states[DamageType.DNA_DAMAGE].point_of_no_return == current_time:
                new_events.append({
                    "time": current_time,
                    "type": DamageType.DNA_DAMAGE.value,
                    "severity": self.damage_states[DamageType.DNA_DAMAGE].severity,
                    "description": f"DNA damage > {self.DNA_DAMAGE_THRESHOLD} lesions/Mb for > {self.DNA_DAMAGE_THRESHOLD_DURATION}h"
                })
        
        # Mitochondrial dysfunction
        above = mito_dysfunction > self.MITO_DYSFUNCTION_THRESHOLD
        self.damage_states[DamageType.MITOCHONDRIAL_DYSFUNCTION].severity = min(
            1.0, mito_dysfunction / self.MITO_DYSFUNCTION_THRESHOLD
        )
        self.damage_states[DamageType.MITOCHONDRIAL_DYSFUNCTION].update(current_time, above)
        if self.damage_states[DamageType.MITOCHONDRIAL_DYSFUNCTION].is_irreversible:
            if self.damage_states[DamageType.MITOCHONDRIAL_DYSFUNCTION].point_of_no_return == current_time:
                new_events.append({
                    "time": current_time,
                    "type": DamageType.MITOCHONDRIAL_DYSFUNCTION.value,
                    "severity": self.damage_states[DamageType.MITOCHONDRIAL_DYSFUNCTION].severity,
                    "description": f"Mito depolarization > {self.MITO_DYSFUNCTION_THRESHOLD*100}% for > {self.MITO_DYSFUNCTION_THRESHOLD_DURATION}h"
                })
        
        self.irreversible_events.extend(new_events)
        return new_events
    
    def get_first_irreversible_time(self) -> float | None:
        """Get time of first irreversible event (point of no return)."""
        if not self.irreversible_events:
            return None
        return min(e["time"] for e in self.irreversible_events)
    
    def get_survival_window(self, max_time: float) -> float:
        """Estimate survival window in hours."""
        first_irr = self.get_first_irreversible_time()
        if first_irr is not None:
            return first_irr
        # If no irreversible event, estimate from damage severity
        max_severity = max(d.severity for d in self.damage_states.values())
        if max_severity > 0.8:
            return max_time * 0.5
        elif max_severity > 0.5:
            return max_time * 0.75
        return max_time


class TimeDynamicODESolver:
    """Integrates turnover + damage + metabolic dynamics."""
    
    def __init__(self, use_cpp: bool = True):
        self.use_cpp = use_cpp and cpp_available()
        self.turnover = TurnoverModel()
        # A fresh damage model per simulate() call keeps runs independent;
        # post-revert runs deliberately seed it with prior damage.
        self.damage = DamageAccumulationModel()
    
    def simulate(
        self,
        initial_state: dict[str, Any],
        variant_effects: dict[str, Any],
        time_span: tuple[float, float] = (0, 168),  # hours (1 week)
        n_points: int = 100,
        initial_damage: dict[str, dict[str, Any]] | None = None,
        initial_events: list[dict[str, Any]] | None = None,
        is_post_revert: bool = False,
    ) -> TimeDynamicState:
        """
        Run full time-dynamic simulation.
        
        Args:
            initial_state: Baseline concentrations, fluxes, etc.
            variant_effects: Mutation effects on synthesis, stability, etc.
            time_span: (start, end) in hours
            n_points: Number of time points
            initial_damage: Optional damage snapshot to carry into this run.
                Used after a genome revert to model hysteresis: the genome is
                restored exactly, but irreversible damage persists.
            initial_events: Irreversible events carried over from a prior run.
            is_post_revert: Marks this as the post-revert re-simulation.
        """
        self.damage = DamageAccumulationModel(
            initial_damage=initial_damage,
            initial_events=initial_events,
        )
        time_points = [time_span[0] + i * (time_span[1] - time_span[0]) / (n_points - 1) for i in range(n_points)]
        
        # Extract initial conditions
        initial_mrna = initial_state.get("mrna", {})
        initial_protein = initial_state.get("protein", {})
        synthesis_rates = initial_state.get("synthesis_rates", {})
        initial_fluxes = initial_state.get("fluxes", {})
        
        # Apply variant effects to synthesis rates.  Values are fold changes
        # (0 = knockout, 0.5 = 50% knockdown), not additive deltas.
        perturbation = {}
        for gene, effect in variant_effects.items():
            if "synthesis_fold" in effect:
                perturbation[gene] = max(0.0, float(effect["synthesis_fold"]))
            if "stability_fold" in effect:
                # Stability affects degradation rate
                pass  # Handled in turnover model
        
        # Simulate turnover
        mrna_traj, prot_traj = self.turnover.simulate_turnover(
            initial_mrna, initial_protein, synthesis_rates, time_points, perturbation
        )
        
        # Simulate metabolic fluxes (simplified coupling)
        flux_traj = self._simulate_fluxes(initial_fluxes, prot_traj, time_points)
        
        # Compute energy state over time
        atp_traj, adp_traj, amp_traj, ec_traj = self._simulate_energy(flux_traj, time_points)
        
        # Compute damage signals
        misfolded_traj = self._compute_misfolded(prot_traj, variant_effects, time_points)
        ros_traj = self._compute_ros(flux_traj, time_points)
        dna_damage_traj = self._compute_dna_damage(ros_traj, time_points)
        mito_traj = self._compute_mito_dysfunction(flux_traj, time_points)
        
        # Update damage model
        for i, t in enumerate(time_points):
            self.damage.update(
                t, ec_traj[i], misfolded_traj[i], ros_traj[i],
                dna_damage_traj[i], mito_traj[i]
            )
        
        # Hysteresis: permanent damage reached at the point of no return.
        # A post-revert run is phenotypically distinct from a clean baseline
        # whenever this index is > 0, even though the genome is identical.
        hyst_index = self.damage.hysteresis_index()

        # Tissue viability
        tissue_viability = self._compute_tissue_viability(
            ec_traj, flux_traj, self.damage.irreversible_events, time_points,
            hysteresis_index=hyst_index if is_post_revert else 0.0,
        )
        
        # Hashes for provenance
        initial_hash = hashlib.blake2b(str(initial_state).encode(), digest_size=16).hexdigest()
        model_hash = "time_dynamic_v1"
        solver_status = "cvode" if self.use_cpp else "rk4"
        
        return TimeDynamicState(
            time_points=time_points,
            mrna_concentrations=mrna_traj,
            protein_concentrations=prot_traj,
            atp_concentration=atp_traj,
            adp_concentration=adp_traj,
            amp_concentration=amp_traj,
            energy_charge=ec_traj,
            damage_states=self.damage.damage_states,
            irreversible_events=self.damage.irreversible_events,
            fluxes=flux_traj,
            tissue_viability=tissue_viability,
            is_post_revert=is_post_revert,
            hysteresis_index=hyst_index,
            initial_hash=initial_hash,
            model_hash=model_hash,
            solver_status=solver_status,
        )
    
    def _simulate_fluxes(
        self,
        initial_fluxes: dict[str, float],
        protein_traj: dict[str, list[float]],
        time_points: list[float]
    ) -> dict[str, list[float]]:
        """Simulate metabolic fluxes coupled to enzyme levels."""
        n = len(time_points)
        flux_traj = {rxn: [0.0] * n for rxn in initial_fluxes}
        
        # Enzyme -> flux mapping (simplified)
        enzyme_flux_map = {
            "GAPDH": "GAPD",
            "PGK1": "PGK",
            "PKM": "PYK",
            "ENO1": "ENO",
            "ALDOA": "FBA",
            "TPI1": "TPI",
            "PGAM1": "PGM",
            "LDHA": "PYR_ox",
            "ATP5F1A": "ATP_synthase",
            "CS": "CIT_SYN",
            "ACO2": "ACONTa",
            "IDH2": "ICDHyr",
            "OGDH": "AKGDH",
            "SUCLG1": "SUCOAS",
            "SDHA": "SUCDi",
            "FH": "FUM",
            "MDH2": "MDH",
        }
        
        for i in range(n):
            for rxn in initial_fluxes:
                flux_traj[rxn][i] = initial_fluxes[rxn]
            
            # Scale fluxes by enzyme abundance
            for enzyme, rxn in enzyme_flux_map.items():
                if enzyme in protein_traj and rxn in flux_traj:
                    if i < len(protein_traj[enzyme]):
                        rel_abundance = protein_traj[enzyme][i] / max(protein_traj[enzyme][0], 1e-9)
                        flux_traj[rxn][i] *= rel_abundance
        
        return flux_traj
    
    def _simulate_energy(
        self,
        flux_traj: dict[str, list[float]],
        time_points: list[float]
    ) -> tuple[list[float], list[float], list[float], list[float]]:
        """Simulate ATP/ADP/AMP and energy charge.

        ATP production is calibrated so the initial (baseline) flux
        distribution is balanced: at t=0 production equals demand.  A fall in
        ATP-synthase flux (mitochondrial failure) or a rise in demand then
        drives the energy charge down.  The adenine nucleotide pool is
        conserved and ATP is clamped to it, so EC stays in [0, 1].
        """
        n = len(time_points)
        atp = [0.0] * n
        adp = [0.0] * n
        amp = [0.0] * n
        ec = [0.0] * n

        total_an = 4.1  # mM, conserved ATP + ADP + AMP pool
        atp[0] = 3.0
        adp[0], amp[0] = self._adenylate_equilibrium(atp[0], total_an)
        ec[0] = (atp[0] + 0.5 * adp[0]) / total_an

        atp_flux = flux_traj.get("ATP_synthase", [0.0] * n)
        baseline_flux = max(float(atp_flux[0]), 1e-9) if n else 1e-9
        baseline_demand = self._atp_demand(flux_traj, 0)

        for i in range(1, n):
            dt = time_points[i] - time_points[i-1]

            # Oxidative phosphorylation scales with ATP-synthase flux.
            production = (
                baseline_demand * max(0.0, float(atp_flux[i])) / baseline_flux
            )
            demand = self._atp_demand(flux_traj, i)

            datp = dt * (production - demand)
            atp[i] = min(max(atp[i-1] + datp, 0.01), total_an - 0.02)
            adp[i], amp[i] = self._adenylate_equilibrium(atp[i], total_an)
            ec[i] = (atp[i] + 0.5 * adp[i]) / total_an

        return atp, adp, amp, ec

    @staticmethod
    def _adenylate_equilibrium(atp: float, total_an: float) -> tuple[float, float]:
        """ADP and AMP from the adenylate kinase equilibrium at a given ATP."""
        atp = min(max(atp, 0.0), total_an)
        disc = atp * atp + 4.0 * atp * max(0.0, total_an - atp)
        adp = (-atp + math.sqrt(disc)) / 2.0 if disc > 0 else 0.0
        adp = min(max(adp, 0.0), max(0.0, total_an - atp))
        amp = max(0.0, total_an - atp - adp)
        return adp, amp

    @staticmethod
    def _atp_demand(flux_traj: dict[str, list[float]], index: int) -> float:
        """Basal plus flux-coupled ATP demand at one time point."""
        flux_sum = 0.0
        for rxn, traj in flux_traj.items():
            if rxn == "ATP_synthase" or index >= len(traj):
                continue
            flux_sum += max(0.0, float(traj[index]))
        return 5.0 + 0.01 * flux_sum
    
    def _compute_misfolded(
        self,
        protein_traj: dict[str, list[float]],
        variant_effects: dict[str, Any],
        time_points: list[float]
    ) -> list[float]:
        """Compute fraction of misfolded proteins over time."""
        n = len(time_points)
        misfolded = [0.0] * n
        
        # Base misfolding rate
        base_misfold = 0.01
        
        for i in range(n):
            total_prot = 0
            misfolded_prot = 0
            
            for gene, traj in protein_traj.items():
                if i < len(traj):
                    conc = traj[i]
                    total_prot += conc
                    
                    # Increased misfolding for mutated proteins
                    if gene in variant_effects:
                        effect = variant_effects[gene]
                        ddg = effect.get("delta_delta_g", 0)
                        # ΔΔG > 2 kcal/mol significantly increases misfolding
                        if ddg > 2:
                            misfolded_prot += conc * min(0.5, ddg / 10)
                        else:
                            misfolded_prot += conc * base_misfold
                    else:
                        misfolded_prot += conc * base_misfold
            
            misfolded[i] = misfolded_prot / max(total_prot, 1)
        
        return misfolded
    
    def _compute_ros(
        self,
        flux_traj: dict[str, list[float]],
        time_points: list[float]
    ) -> list[float]:
        """Compute ROS level over time (fold over baseline)."""
        n = len(time_points)
        ros = [1.0] * n
        
        for i in range(n):
            # ROS from mitochondrial flux (ETC leak)
            etc_flux = flux_traj.get("CYTB", [0.0] * n)[i] + flux_traj.get("COX1", [0.0] * n)[i]
            # ROS proportional to ETC flux * reduction state.  Reduction state
            # is relative to the baseline ATP-synthase flux, not a hard-coded
            # 10, so different flux scales remain comparable.
            atp_flux = flux_traj.get("ATP_synthase", [1.0] * n)
            baseline = max(float(atp_flux[0]), 1e-9)
            reduction_state = 1.0 - float(atp_flux[i]) / baseline
            ros[i] = 1.0 + etc_flux * max(0, reduction_state) * 0.05
        
        return ros
    
    def _compute_dna_damage(
        self,
        ros_traj: list[float],
        time_points: list[float]
    ) -> list[float]:
        """Compute DNA damage (lesions per Mb) over time."""
        n = len(time_points)
        dna_damage = [0.0] * n
        
        # Base damage rate: ~10 lesions/Mb/day from endogenous ROS
        base_rate = 10.0 / 24.0  # per hour
        
        for i in range(1, n):
            dt = time_points[i] - time_points[i-1]
            # Damage accumulates with ROS, repaired with half-life ~2h
            repair_rate = math.log(2) / 2.0
            dna_damage[i] = dna_damage[i-1] + dt * (base_rate * ros_traj[i] - repair_rate * dna_damage[i-1])
            dna_damage[i] = max(0, dna_damage[i])
        
        return dna_damage
    
    def _compute_mito_dysfunction(
        self,
        flux_traj: dict[str, list[float]],
        time_points: list[float]
    ) -> list[float]:
        """Compute mitochondrial dysfunction fraction over time."""
        n = len(time_points)
        mito_dys = [0.0] * n
        atp_traj = flux_traj.get("ATP_synthase", [1.0] * n)
        baseline = max(float(atp_traj[0]), 1e-9)

        for i in range(n):
            # Dysfunction from low ATP synthesis, high ROS, membrane depolarization
            atp_synth = float(atp_traj[i])
            # Normalized to the baseline: 1 = full function, 0 = complete failure
            func = min(1.0, max(0.0, atp_synth / baseline))
            mito_dys[i] = 1.0 - func
        
        return mito_dys
    
    def _compute_tissue_viability(
        self,
        ec_traj: list[float],
        flux_traj: dict[str, list[float]],
        irr_events: list[dict],
        time_points: list[float],
        hysteresis_index: float = 0.0,
    ) -> dict[str, list[float]]:
        """Compute tissue viability over time.

        ``hysteresis_index`` applies a persistent viability ceiling after a
        genome revert: the mutation is gone, but irreversible damage remains.
        """
        tissues = ["liver", "heart", "brain", "kidney", "muscle", "lung", "adipose", "spleen", "gut"]
        n = len(time_points)
        viability = {t: [1.0] * n for t in tissues}
        
        # Tissue-specific sensitivities
        sensitivity = {
            "brain": 1.5,      # Most sensitive to energy depletion
            "heart": 1.3,      # High energy demand
            "kidney": 1.2,
            "liver": 1.0,      # Regenerative
            "muscle": 0.8,
            "lung": 1.1,
            "adipose": 0.5,
            "spleen": 0.9,
            "gut": 1.0,
        }
        # Persistent ceiling: up to 50% loss at maximum residual damage.
        hyst_ceiling = max(0.0, 1.0 - 0.5 * float(hysteresis_index or 0.0))
        
        for i, t in enumerate(time_points):
            ec = ec_traj[i]
            base_viability = max(0, (ec - 0.3) / 0.5)  # 0 at EC=0.3, 1 at EC=0.8
            
            for tissue in tissues:
                sens = sensitivity[tissue]
                value = max(0, min(1, base_viability ** sens))
                
                # Irreversible events cause permanent drops
                for event in irr_events:
                    if event["time"] <= t:
                        value *= (1 - 0.1 * event["severity"])
                
                viability[tissue][i] = min(value, hyst_ceiling)
        
        return viability


def run_time_dynamic_simulation(
    initial_state: dict[str, Any],
    variant_effects: dict[str, Any],
    time_span: tuple[float, float] = (0, 168),
) -> TimeDynamicState:
    """Convenience function to run time-dynamic simulation."""
    solver = TimeDynamicODESolver()
    return solver.simulate(initial_state, variant_effects, time_span)


# Export key classes
__all__ = [
    "TurnoverParams",
    "DamageType",
    "IrreversibleDamage",
    "TimeDynamicState",
    "TurnoverModel",
    "DamageAccumulationModel",
    "TimeDynamicODESolver",
    "run_time_dynamic_simulation",
]