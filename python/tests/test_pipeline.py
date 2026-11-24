"""
End-to-end pipeline tests.
"""

from __future__ import annotations
import pytest
from proteus.pipeline import run_simulation
from proteus.schema import PipelineConfig, VariantEvent


@pytest.fixture
def minimal_config():
    """Minimal valid pipeline configuration."""
    return PipelineConfig(
        reference_genome_uri="test_genome.fasta",
        variants=[
            VariantEvent(
                id="test_v1",
                type=VariantEvent.Type.SNV,
                contig="chr1",
                start=10,
                end=10,
                ref_allele="T",  # 1-based position 10 is 'T' in the test genome ATCGATCGATCG
                alt_allele="A",
                annotations={},
                timestamp_ns=0,
                source="test",
            )
        ],
        output_dir="./test_output",
        generate_visualizations=False,  # report formats are covered separately
    )


@pytest.mark.asyncio
async def test_pipeline_runs(minimal_config):
    """Test that pipeline executes without errors and emits headline outputs."""
    result = await run_simulation(minimal_config)
    assert result.success
    assert result.run_id is not None
    assert result.report is not None
    # Headline outputs: survival window and point of no return
    assert result.report.survival_window_h is not None
    assert result.report.point_of_no_return_event is not None
    assert result.report.irreversible_events
    # Hysteresis: genome revert is exact but the phenotype retains damage
    hysteresis = result.report.revert_hysteresis
    assert hysteresis.get("available") is True
    assert hysteresis.get("genome_revert_exact") is True
    assert hysteresis.get("phenotype_matches_baseline") is False
    assert hysteresis.get("hysteresis_index", 0.0) > 0.0


@pytest.mark.asyncio
async def test_viability_verdict(minimal_config):
    """Test that viability verdict is produced."""
    result = await run_simulation(minimal_config)
    assert result.report.viability.verdict is not None
    assert result.report.viability.composite >= 0.0
    assert result.report.viability.composite <= 1.0


@pytest.mark.asyncio
async def test_revert_produces_identical_state(minimal_config):
    """Test that revert produces bit-identical baseline."""
    result = await run_simulation(minimal_config)
    if result.report.viability.verdict.name == "VIABLE":
        # Reverted state should match baseline
        assert result.report.reverted.hash() == result.report.baseline.hash()