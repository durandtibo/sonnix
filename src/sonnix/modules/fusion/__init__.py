r"""Contain fusion modules."""

from __future__ import annotations

__all__ = [
    "AverageFusion",
    "BaseFusion",
    "ConcatFusion",
    "MultiplicationFusion",
    "SumFusion",
]

from sonnix.modules.fusion.aggregation import (
    AverageFusion,
    MultiplicationFusion,
    SumFusion,
)
from sonnix.modules.fusion.base import BaseFusion
from sonnix.modules.fusion.concat import ConcatFusion
