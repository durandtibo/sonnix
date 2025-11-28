r"""Contain modules."""

from __future__ import annotations

__all__ = [
    "Asinh",
    "AverageFusion",
    "BaseAlphaActivation",
    "Clamp",
    "ConcatFusion",
    "ExU",
    "Exp",
    "ExpSin",
    "Expm1",
    "Gaussian",
    "Laplacian",
    "Log",
    "Log1p",
    "MultiQuadratic",
    "MulticlassFlatten",
    "MultiplicationFusion",
    "NLinear",
    "Quadratic",
    "ReLUn",
    "ResidualBlock",
    "SafeExp",
    "SafeLog",
    "Sin",
    "Sinh",
    "Snake",
    "SquaredReLU",
    "Squeeze",
    "SumFusion",
    "ToFloat",
    "ToLong",
    "View",
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
from sonnix.modules.dtype import ToFloat, ToLong
from sonnix.modules.exu import ExU
from sonnix.modules.fusion import (
    AverageFusion,
    ConcatFusion,
    MultiplicationFusion,
    SumFusion,
)
from sonnix.modules.fusion.clamp import Clamp
from sonnix.modules.nlinear import NLinear
from sonnix.modules.residual import ResidualBlock
from sonnix.modules.shape import MulticlassFlatten, Squeeze, View
