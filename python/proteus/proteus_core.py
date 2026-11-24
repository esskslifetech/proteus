"""
py-proteus: Python bindings for proteus-core via PyO3.

This module re-exports the Rust core types for use from Python.
"""

from __future__ import annotations
from proteus_core import (
    GenomeState,
    ReferenceGenome,
    ReferenceGenomeBuilder,
    VariantEvent,
    VariantType,
    LayerDiff,
    StateTransition,
)

__all__ = [
    "GenomeState",
    "ReferenceGenome",
    "ReferenceGenomeBuilder",
    "VariantEvent",
    "VariantType",
    "LayerDiff",
    "StateTransition",
]
