"""Unit tests for the L3b time-dynamic layer and revert hysteresis."""

from __future__ import annotations

from proteus.layers.l3b_time_dynamics import (
    DamageAccumulationModel,
    DamageType,
    TimeDynamicODESolver,
    TurnoverParams,
)


def test_turnover_half_lives_from_literature():
    params = TurnoverParams()
    assert 8.0 <= params.mrna_half_life_median <= 10.0
    assert 40.0 <= params.protein_half_life_median <= 50.0


def test_damage_becomes_irreversible_after_threshold_duration():
    model = DamageAccumulationModel()
    # Energy charge below 0.5 for 3 hours (> 2 h threshold).
    for hour in range(4):
        model.update(
            current_time=float(hour),
            energy_charge=0.2,
            misfolded_fraction=0.0,
            ros_level=1.0,
            dna_damage=0.0,
            mito_dysfunction=0.0,
        )
    state = model.damage_states[DamageType.ENERGY_DEPLETION]
    assert state.is_irreversible
    assert state.point_of_no_return is not None
    assert model.hysteresis_index() > 0.0


def test_snapshot_roundtrip_preserves_residual_damage():
    model = DamageAccumulationModel()
    for hour in range(25):
        model.update(float(hour), 0.2, 0.0, 1.0, 0.0, 0.0)
    snapshot = model.snapshot()
    restored = DamageAccumulationModel(initial_damage=snapshot)
    original = model.residual_damage()
    roundtrip = restored.residual_damage()
    assert original
    for key, value in original.items():
        assert abs(roundtrip[key] - value) < 1e-9


def test_post_revert_run_retains_damage_but_genome_is_not_involved():
    initial_state = {
        "mrna": {"ATP5F1A": 1.0},
        "protein": {"ATP5F1A": 1.0},
        "synthesis_rates": {"ATP5F1A": 1.0},
        "fluxes": {"ATP_synthase": 10.0},
    }
    solver = TimeDynamicODESolver()
    mutated = solver.simulate(
        initial_state, {"ATP5F1A": {"synthesis_fold": 0.0}}
    )
    assert mutated.irreversible_events

    snapshot = {
        dtype.value: state.snapshot()
        for dtype, state in mutated.damage_states.items()
    }
    post = TimeDynamicODESolver().simulate(
        initial_state,
        {},
        time_span=(0, 72),
        initial_damage=snapshot,
        initial_events=mutated.irreversible_events,
        is_post_revert=True,
    )
    assert post.is_post_revert
    assert post.hysteresis_index > 0.0
    # The genome is reverted and energy physiology is healthy again; the
    # residual phenotype is the persistent viability ceiling, not an ongoing
    # energy collapse.
    assert min(post.energy_charge) > 0.8
    means = [sum(v) / len(v) for v in post.tissue_viability.values()]
    assert means and all(abs(m - 0.5) < 1e-9 for m in means)


def test_clean_run_has_no_hysteresis():
    solver = TimeDynamicODESolver()
    state = solver.simulate(
        {
            "mrna": {"GAPDH": 1.0},
            "protein": {"GAPDH": 1.0},
            "synthesis_rates": {"GAPDH": 1.0},
            "fluxes": {"ATP_synthase": 10.0},
        },
        {},
        time_span=(0, 24),
    )
    assert state.hysteresis_index == 0.0


def test_baseline_energy_charge_is_stable():
    """An unperturbed baseline must not drift into energy failure."""
    solver = TimeDynamicODESolver()
    state = solver.simulate(
        {
            "mrna": {"GAPDH": 1.0, "ATP5F1A": 1.0},
            "protein": {"GAPDH": 1.0, "ATP5F1A": 1.0},
            "synthesis_rates": {"GAPDH": 1.0, "ATP5F1A": 1.0},
            "fluxes": {"ATP_synthase": 10.0, "GAPD": 10.0},
        },
        {},
        time_span=(0, 72),
    )
    assert min(state.energy_charge) > 0.8
    assert max(state.energy_charge) - min(state.energy_charge) < 1e-6
    assert state.irreversible_events == []
