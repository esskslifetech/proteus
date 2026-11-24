"""Proteus: human genome mutation simulation and viability analysis."""

from __future__ import annotations

__version__ = "0.1.0"

from .proteus_core import *  # noqa: F403
from . import proteus_core

__all__ = list(proteus_core.__all__)
