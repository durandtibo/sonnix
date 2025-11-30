r"""Contain activation modules."""

from __future__ import annotations

__all__ = [
    "Asinh",
    "BaseAlphaActivation",
    "DynamicAsinh",
    "DynamicTanh",
    "Exp",
    "ExpSin",
    "Expm1",
    "Gaussian",
    "Laplacian",
    "Log",
    "Log1p",
    "MultiQuadratic",
    "Pow",
    "Quadratic",
    "ReLUn",
    "RectifierAsinhUnit",
    "SafeExp",
    "SafeLog",
    "Sin",
    "Sinh",
    "Snake",
    "Square",
    "SquaredReLU",
]

from sonnix.modules.activation.alpha import (
    BaseAlphaActivation,
    ExpSin,
    Gaussian,
    Laplacian,
    MultiQuadratic,
    Quadratic,
)
from sonnix.modules.activation.dynamic import DynamicAsinh, DynamicTanh
from sonnix.modules.activation.hyperbolic import RectifierAsinhUnit
from sonnix.modules.activation.math import (
    Asinh,
    Exp,
    Expm1,
    Log,
    Log1p,
    Pow,
    SafeExp,
    SafeLog,
    Sin,
    Sinh,
    Square,
)
from sonnix.modules.activation.relu import ReLUn, SquaredReLU
from sonnix.modules.activation.snake import Snake
