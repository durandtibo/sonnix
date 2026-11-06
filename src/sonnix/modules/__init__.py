r"""Provide custom PyTorch modules and layers.

This subpackage contains custom ``torch.nn.Module`` implementations
including activation functions, loss functions, fusion layers, numerical
encoders, and other building blocks for neural networks.
"""

from __future__ import annotations

__all__ = [
    "Asinh",
    "AsinhCosSinNumericalEncoder",
    "AsinhNumericalEncoder",
    "AverageFusion",
    "BaseAlphaActivation",
    "BaseBinaryFocalLoss",
    "BaseBinaryPoly1Loss",
    "BaseDynamicNorm",
    "BaseFusion",
    "BaseLoss",
    "BinaryFocalLoss",
    "BinaryFocalLossWithLogits",
    "BinaryPoly1Loss",
    "BinaryPoly1LossWithLogits",
    "Clamp",
    "ConcatFusion",
    "CosSinNumericalEncoder",
    "DynamicAsinh",
    "DynamicTanh",
    "ExU",
    "Exp",
    "ExpSin",
    "Expm1",
    "Gaussian",
    "GeneralRobustRegressionLoss",
    "Laplacian",
    "Log",
    "Log1p",
    "LogCoshLoss",
    "MultiQuadratic",
    "MulticlassFlatten",
    "MultiplicationFusion",
    "NLinear",
    "PiecewiseLinearNumericalEncoder",
    "PoissonRegressionLoss",
    "Pow",
    "Quadratic",
    "QuantileRegressionLoss",
    "ReLUn",
    "RectifierAsinhUnit",
    "RelativeLoss",
    "ResidualBlock",
    "SafeExp",
    "SafeLog",
    "ScaleAndShift",
    "Sin",
    "Sinh",
    "Snake",
    "Square",
    "SquaredReLU",
    "Squeeze",
    "SumFusion",
    "ToBinaryLabel",
    "ToCategoricalLabel",
    "ToDtype",
    "ToFloat",
    "ToLong",
    "TransformedLoss",
    "View",
]

from sonnix.modules.activation.alpha import (
    BaseAlphaActivation,
    ExpSin,
    Gaussian,
    Laplacian,
    MultiQuadratic,
    Quadratic,
)
from sonnix.modules.activation.clamp import Clamp
from sonnix.modules.activation.dynamic import BaseDynamicNorm, DynamicAsinh, DynamicTanh
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
from sonnix.modules.classification import ToBinaryLabel, ToCategoricalLabel
from sonnix.modules.dtype import ToDtype, ToFloat, ToLong
from sonnix.modules.exu import ExU
from sonnix.modules.fusion.aggregation import (
    AverageFusion,
    MultiplicationFusion,
    SumFusion,
)
from sonnix.modules.fusion.base import BaseFusion
from sonnix.modules.fusion.concat import ConcatFusion
from sonnix.modules.loss.base import BaseLoss
from sonnix.modules.loss.focal import (
    BaseBinaryFocalLoss,
    BinaryFocalLoss,
    BinaryFocalLossWithLogits,
)
from sonnix.modules.loss.general_robust import GeneralRobustRegressionLoss
from sonnix.modules.loss.log import LogCoshLoss
from sonnix.modules.loss.multiclass import MulticlassFlatten
from sonnix.modules.loss.poisson import PoissonRegressionLoss
from sonnix.modules.loss.poly import (
    BaseBinaryPoly1Loss,
    BinaryPoly1Loss,
    BinaryPoly1LossWithLogits,
)
from sonnix.modules.loss.quantile import QuantileRegressionLoss
from sonnix.modules.loss.relative import (
    RelativeLoss,
)
from sonnix.modules.loss.transform import TransformedLoss
from sonnix.modules.nlinear import NLinear
from sonnix.modules.numerical.asinh import AsinhNumericalEncoder
from sonnix.modules.numerical.piecewise import PiecewiseLinearNumericalEncoder
from sonnix.modules.numerical.sine import (
    AsinhCosSinNumericalEncoder,
    CosSinNumericalEncoder,
)
from sonnix.modules.residual import ResidualBlock
from sonnix.modules.scale_and_shift import ScaleAndShift
from sonnix.modules.shape import Squeeze, View
