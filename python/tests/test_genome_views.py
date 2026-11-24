"""
Tests for genome/molecule-centric visualizations.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from proteus.pipeline import run_simulation
from proteus.schema import PipelineConfig, VariantEvent
from proteus.visualization import (
    VIEW_TITLES,
    create_genome_views,
    plot_genome_browser,
    plot_dna_helix,
    plot_central_dogma,
    plot_molecular_cascade,
    plot_inputs_provenance,
    plot_glycolysis_map,
)


@pytest.fixture(scope="module")
def demo_report():
    """Run a minimal simulation to get a report with all layers populated."""
    config = PipelineConfig(
        reference_genome_uri="test_genome.fasta",
        variants=[
            VariantEvent(
                id="v1_snv",
                type=VariantEvent.Type.SNV,
                contig="chr1",
                start=10,
                end=10,
                ref_allele="T",
                alt_allele="A",
                annotations={"gene_id": "GAPDH"},
                timestamp_ns=0,
                source="demo",
            ),
            VariantEvent(
                id="v2_snv",
                type=VariantEvent.Type.SNV,
                contig="chr1",
                start=2,
                end=2,
                ref_allele="T",
                alt_allele="C",
                annotations={"gene_id": "PGK1"},
                timestamp_ns=0,
                source="demo",
            ),
            VariantEvent(
                id="v3_ins",
                type=VariantEvent.Type.INSERTION,
                contig="chr1",
                start=8,
                end=8,
                ref_allele="",
                alt_allele="A",
                annotations={"gene_id": "PKM"},
                timestamp_ns=0,
                source="demo",
            ),
        ],
        output_dir="./test_views_output",
        generate_visualizations=False,
    )
    return asyncio.run(run_simulation(config)).report


def test_view_titles_exist():
    """All expected visualization titles are defined."""
    expected = [
        "genome_browser",
        "dna_helix",
        "central_dogma",
        "molecular_cascade",
        "inputs_provenance",
        "glycolysis_map",
    ]
    for key in expected:
        assert key in VIEW_TITLES
        assert isinstance(VIEW_TITLES[key], str)
        assert len(VIEW_TITLES[key]) > 0


def test_genome_browser_creates_figure(demo_report, tmp_path):
    """plot_genome_browser produces a figure with expected elements."""
    out = tmp_path / "browser.png"
    fig = plot_genome_browser(demo_report, out)
    assert fig is not None
    assert out.exists()
    assert out.stat().st_size > 1000
    import matplotlib.pyplot as plt
    plt.close(fig)


def test_dna_helix_creates_figure(demo_report, tmp_path):
    """plot_dna_helix produces a figure with expected elements."""
    out = tmp_path / "helix.png"
    fig = plot_dna_helix(demo_report, out)
    assert fig is not None
    assert out.exists()
    assert out.stat().st_size > 1000
    import matplotlib.pyplot as plt
    plt.close(fig)


def test_central_dogma_creates_figure(demo_report, tmp_path):
    """plot_central_dogma produces a figure with expected elements."""
    out = tmp_path / "central.png"
    fig = plot_central_dogma(demo_report, out)
    assert fig is not None
    assert out.exists()
    assert out.stat().st_size > 1000
    import matplotlib.pyplot as plt
    plt.close(fig)


def test_molecular_cascade_creates_figure(demo_report, tmp_path):
    """plot_molecular_cascade produces a figure with expected elements."""
    out = tmp_path / "cascade.png"
    fig = plot_molecular_cascade(demo_report, out)
    assert fig is not None
    assert out.exists()
    assert out.stat().st_size > 1000
    import matplotlib.pyplot as plt
    plt.close(fig)


def test_inputs_provenance_creates_figure(demo_report, tmp_path):
    """plot_inputs_provenance produces a figure with expected elements."""
    out = tmp_path / "provenance.png"
    fig = plot_inputs_provenance(demo_report, out)
    assert fig is not None
    assert out.exists()
    assert out.stat().st_size > 1000
    import matplotlib.pyplot as plt
    plt.close(fig)


def test_glycolysis_map_creates_figure(demo_report, tmp_path):
    """plot_glycolysis_map produces a figure with expected elements."""
    out = tmp_path / "glycolysis.png"
    fig = plot_glycolysis_map(demo_report, out)
    assert fig is not None
    assert out.exists()
    assert out.stat().st_size > 1000
    import matplotlib.pyplot as plt
    plt.close(fig)


def test_create_genome_views_aggregator(demo_report, tmp_path):
    """create_genome_views generates all six figures."""
    paths = create_genome_views(demo_report, tmp_path)
    assert len(paths) == 6
    for key in [
        "genome_browser",
        "dna_helix",
        "central_dogma",
        "molecular_cascade",
        "inputs_provenance",
        "glycolysis_map",
    ]:
        assert key in paths
        assert paths[key].exists()
        assert paths[key].stat().st_size > 1000


def test_l0_sequence_previews_attached(demo_report):
    """L0 layer_results contains sequence_previews with correct structure."""
    l0 = demo_report.layer_results.get("L0", {})
    previews = l0.get("sequence_previews", {})
    assert isinstance(previews, dict)
    assert "chr1" in previews
    preview = previews["chr1"]
    assert "baseline_length" in preview
    assert "mutated_length" in preview
    assert "windows" in preview
    window = preview["windows"][0]
    assert "start_1based" in window
    assert "end_1based" in window
    assert "baseline" in window
    assert "mutated" in window
    assert "label" in window
    # The mutated sequence should reflect v1(T10→A), v2(T2→C), v3(ins at 8)
    assert window["mutated"] == "ACCGATCAGATCG"


def test_l1_variant_details_have_gene_ids(demo_report):
    """L1 layer_results.variant_details includes gene_id from annotations."""
    l1 = demo_report.layer_results.get("L1", {})
    details = l1.get("variant_details", [])
    assert len(details) == 3
    gene_ids = {d["gene_id"] for d in details if d.get("gene_id")}
    assert "GAPDH" in gene_ids
    assert "PGK1" in gene_ids
    assert "PKM" in gene_ids


def test_placeholder_when_missing_data(tmp_path):
    """Each plot function gracefully returns a placeholder figure when data is absent."""
    empty_report = MagicMock()
    empty_report.layer_results = {}
    empty_report.applied_variants = []
    empty_report.reference_genome = "unknown"
    empty_report.baseline = None
    empty_report.viability = None
    empty_report.revert_hysteresis = {}
    empty_report.evidence = []

    out = tmp_path / "placeholder.png"
    fig = plot_genome_browser(empty_report, out)
    assert fig is not None
    assert out.exists()
    import matplotlib.pyplot as plt
    plt.close(fig)

    out2 = tmp_path / "placeholder2.png"
    fig2 = plot_dna_helix(empty_report, out2)
    assert fig2 is not None
    assert out2.exists()
    plt.close(fig2)

    out3 = tmp_path / "placeholder3.png"
    fig3 = plot_central_dogma(empty_report, out3)
    assert fig3 is not None
    assert out3.exists()
    plt.close(fig3)

    out4 = tmp_path / "placeholder4.png"
    fig4 = plot_molecular_cascade(empty_report, out4)
    assert fig4 is not None
    assert out4.exists()
    plt.close(fig4)

    out5 = tmp_path / "placeholder5.png"
    fig5 = plot_inputs_provenance(empty_report, out5)
    assert fig5 is not None
    assert out5.exists()
    plt.close(fig5)

    out6 = tmp_path / "placeholder6.png"
    fig6 = plot_glycolysis_map(empty_report, out6)
    assert fig6 is not None
    assert out6.exists()
    plt.close(fig6)