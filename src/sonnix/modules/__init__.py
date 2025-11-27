r"""Contain modules."""

from __future__ import annotations

__all__ = [
    "Asinh",
    "AverageFusion",
    "BaseAlphaActivation",
    "Clamp",
    "ConcatFusion",
    "Exp",
    "ExpSin",
    "Expm1",
    "Gaussian",
    "Laplacian",
    "Log",
    "Log1p",
    "MultiQuadratic",
    "MultiplicationFusion",
    "NLinear",
    "Quadratic",
    "ReLUn",
    "SafeExp",
    "SafeLog",
    "Sin",
    "Sinh",
    "Snake",
    "SquaredReLU",
    "SumFusion",
]

from sonnix.modules.activations import (
    Asinh,
    BaseAlphaActivation,
    Exp,
    Expm1,
    ExpSin,
    Gaussian,
    Laplacian,
    Log,
    Log1p,
    MultiQuadratic,
    Quadratic,
    ReLUn,
    SafeExp,
    SafeLog,
    Sin,
    Sinh,
    Snake,
    SquaredReLU,
)
from sonnix.modules.fusion import (
    AverageFusion,
    ConcatFusion,
    MultiplicationFusion,
    SumFusion,
)
from sonnix.modules.fusion.clamp import Clamp
from sonnix.modules.nlinear import NLinear
