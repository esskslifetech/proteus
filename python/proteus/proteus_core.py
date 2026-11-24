"""
py-proteus: optional Python bindings for proteus-core via PyO3.

This module re-exports the Rust core types under stable Python names. The
extension is optional: when it is not installed, ``__all__`` stays empty and
the pure-Python implementation is used instead.
"""

from __future__ import annotations

__all__: list[str] = []

try:
    from proteus_core import (
        PyGenomeState as GenomeState,
        PyLayerDiff as LayerDiff,
        PyReferenceGenome as ReferenceGenome,
        PyVariantEvent as VariantEvent,
        PyVariantType as VariantType,
    )
except ImportError:
    pass
else:
    __all__ = [
        "GenomeState",
        "ReferenceGenome",
        "VariantEvent",
        "VariantType",
        "LayerDiff",
    ]
