"""
Report generation tests: every format and the evidence bundle.
"""

from __future__ import annotations

import json

import pytest

from proteus.report import ReportGenerator, build_context
from proteus.schema import (
    ContentHash,
    EvidenceChain,
    GenomeState,
    LayerDiff,
    PipelineConfig,
    ResourceRef,
    SimulationReport,
    VariantEvent,
    ViabilityScore,
)


@pytest.fixture
def report() -> SimulationReport:
    reference = ResourceRef(
        hash=ContentHash(algorithm="blake3", digest=b"reference-digest"),
        uri="test_genome.fasta",
    )
    variant = VariantEvent(
        id="v1",
        type=VariantEvent.Type.SNV,
        contig="chr1",
        start=10,
        end=10,
        ref_allele="T",
        alt_allele="A",
        annotations={},
        timestamp_ns=0,
        source="test",
    )
    baseline = GenomeState(reference=reference, overlays=[], head_index=0)
    mutated = GenomeState(reference=reference, overlays=[variant], head_index=1)
    reverted = GenomeState(reference=reference, overlays=[], head_index=0)

    rep = SimulationReport(
        simulation_id="sim-test-001",
        timestamp_ns=123456789,
        baseline=baseline,
        mutated=mutated,
        reverted=reverted,
        applied_variants=[variant],
        viability=ViabilityScore(
            composite=0.87,
            layer_scores={"L1": 0.9, "L2": 0.8, "L3": 0.85, "L4": 0.95},
            weights={"L1": 0.15, "L2": 0.25, "L3": 0.30, "L4": 0.30},
            failed_gates=[],
            verdict=ViabilityScore.Verdict.VIABLE,
            confidence=0.8,
        ),
        diffs=[LayerDiff(layer="L0", added={"v1": "v1"}, removed={}, changed={})],
        evidence=[
            EvidenceChain(
                claim="L2 succeeded",
                tool="proteus.layers",
                params={"layer": "L2"},
                confidence=EvidenceChain.Confidence.HIGH,
                validation_passes=["V3"],
            )
        ],
        provenance_manifest=ContentHash(algorithm="blake3", digest=b"manifest"),
        validation_results={"V1": "pass", "V2": "pass", "V3": "pass"},
    )
    rep.validation_details = {  # type: ignore[attr-defined]
        "V1": {"layer": "V1", "status": "pass", "message": "ok", "details": {"n": 1}}
    }
    rep.limitations = [  # type: ignore[attr-defined]
        "Not a clinical result.",
        "Neurophysiological proxies are not consciousness.",
    ]
    return rep


def test_build_context(report):
    ctx = build_context(report)
    assert ctx.simulation_id == "sim-test-001"
    assert len(ctx.variants) == 1
    assert ctx.viability["verdict"] == "VIABLE"
    assert ctx.validation[0]["name"] == "V1"
    assert ctx.state_baseline_hash == ctx.state_reverted_hash


def test_generate_all_formats(report, tmp_path):
    outputs = ReportGenerator(tmp_path).generate_all(report, base_name="report")

    assert outputs["html"].is_file()
    assert outputs["latex"].is_file()
    assert outputs["jsonld"].is_file()
    assert outputs["pdf"].is_file()
    assert outputs["ro_crate"].is_dir()

    html = outputs["html"].read_text(encoding="utf-8")
    assert "Proteus Simulation Report" in html
    assert "sim-test-001" in html
    assert "VIABLE" in html
    assert "v1" in html
    assert "EXACT" in html

    tex = outputs["latex"].read_text(encoding="utf-8")
    assert r"\documentclass" in tex
    assert "sim-test-001" in tex

    jsonld = json.loads(outputs["jsonld"].read_text(encoding="utf-8"))
    assert jsonld["identifier"] == "sim-test-001"
    assert jsonld["proteus:viability"]["proteus:verdict"] == "VIABLE"
    assert jsonld["@type"] == ["prov:Activity", "Dataset"]

    crate_dir = outputs["ro_crate"]
    crate = json.loads((crate_dir / "ro-crate-metadata.json").read_text(encoding="utf-8"))
    assert crate["@context"].startswith("https://w3id.org/ro/crate/")
    graph_ids = {node["@id"] for node in crate["@graph"]}
    assert "run_manifest.json" in graph_ids
    assert "evidence.json" in graph_ids

    manifest = json.loads((crate_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["states"]["revert_exact"] is True
    assert manifest["validation"] == {"V1": "pass", "V2": "pass", "V3": "pass"}

    evidence = json.loads((crate_dir / "evidence.json").read_text(encoding="utf-8"))
    assert evidence["revert_exact"] is True
    assert evidence["evidence"][0]["tool"] == "proteus.layers"
    assert evidence["limitations"]
