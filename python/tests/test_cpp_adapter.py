"""
Optional C++ acceleration adapter tests.

The extension targets the interpreter it was compiled for (MSYS2 CPython);
these tests skip cleanly when ``proteus_cpp`` cannot be imported, and exercise
the Python fallbacks either way.
"""

from __future__ import annotations

import math

import pytest

from proteus import cpp_adapter
from proteus.layers import ProteinLayer, PathwayLayer
from proteus.schema import VariantEvent


HAS_CPP = cpp_adapter.cpp_available()


def _variant(pos: int = 10, ref: str = "T", alt: str = "A") -> VariantEvent:
    return VariantEvent(
        id="v1",
        type=VariantEvent.Type.SNV,
        contig="chr1",
        start=pos,
        end=pos,
        ref_allele=ref,
        alt_allele=alt,
        annotations={},
        timestamp_ns=0,
        source="test",
    )


class TestAdapterContract:
    """The adapter must always expose the same API, with or without C++."""

    def test_status_shape(self):
        status = cpp_adapter.cpp_status()
        # Updated shape: module_hash instead of module_file, no search_paths
        assert set(status) == {"available", "module_hash", "import_error"}
        assert isinstance(status["available"], bool)
        assert isinstance(status["module_hash"], str)

    def test_helpers_return_none_when_unavailable(self):
        if HAS_CPP:
            pytest.skip("C++ extension is available; fallback behaviour not applicable")
        assert cpp_adapter.mean([1, 2, 3]) is None
        assert cpp_adapter.stddev([1, 2, 3]) is None
        assert cpp_adapter.empirical_ddg("AAAA", 1, "A") is None
        with pytest.raises(RuntimeError):
            cpp_adapter.solve_ode(lambda t, y: [-y[0]], [1.0], 0.0, 1.0)


@pytest.mark.skipif(not HAS_CPP, reason="proteus_cpp extension not importable in this interpreter")
class TestCppKernels:
    def test_mean_variance_stddev(self):
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        assert cpp_adapter.mean(data) == pytest.approx(3.0)
        assert cpp_adapter.stddev(data) == pytest.approx(math.sqrt(2.5))
        # 1/sqrt(n) population convention
        assert cpp_adapter.stddev(data) == pytest.approx(1.5811, rel=1e-3)

    def test_empirical_ddg_is_finite(self):
        ddg = cpp_adapter.empirical_ddg("AAAAA", 2, "G", [0.5] * 5, [0] * 5)
        assert ddg is not None
        assert math.isfinite(ddg)

    def test_ode_exponential_decay(self):
        times, values = cpp_adapter.solve_ode(
            lambda t, y: [-y[0]], [1.0], 0.0, 1.0, [1.0]
        )
        assert times and values
        assert values[-1][0] == pytest.approx(math.exp(-1.0), abs=1e-4)


class TestLayersUseFallback:
    def test_protein_layer_works_without_cpp(self):
        effect = ProteinLayer(config={"seed": 42}).predict(_variant())
        assert effect.delta_delta_g >= 0.0
        assert effect.method.value in {"esm_fold", "foldx"}

    def test_pathway_layer_works_without_cpp(self):
        pathway, grn = PathwayLayer(config={"seed": 42}).predict(
            _variant(), {"affected_genes": ["TP53"]}
        )
        assert pathway.status in {"optimal", "suboptimal", "infeasible"}
        assert pathway.metabolite_concentration["atp"] >= 0.0
        assert grn.activity["TP53"] > 0.0
