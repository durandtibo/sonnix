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

from sonnix.modules.activation import (
    Asinh,
    BaseAlphaActivation,
    BaseDynamicNorm,
    Clamp,
    DynamicAsinh,
    DynamicTanh,
    Exp,
    Expm1,
    ExpSin,
    Gaussian,
    Laplacian,
    Log,
    Log1p,
    MultiQuadratic,
    Pow,
    Quadratic,
    RectifierAsinhUnit,
    ReLUn,
    SafeExp,
    SafeLog,
    Sin,
    Sinh,
    Snake,
    Square,
    SquaredReLU,
)
from sonnix.modules.classification import ToBinaryLabel, ToCategoricalLabel
from sonnix.modules.dtype import ToDtype, ToFloat, ToLong
from sonnix.modules.exu import ExU
from sonnix.modules.fusion import (
    AverageFusion,
    BaseFusion,
    ConcatFusion,
    MultiplicationFusion,
    SumFusion,
)
from sonnix.modules.loss import (
    BaseBinaryFocalLoss,
    BaseBinaryPoly1Loss,
    BaseLoss,
    BinaryFocalLoss,
    BinaryFocalLossWithLogits,
    BinaryPoly1Loss,
    BinaryPoly1LossWithLogits,
    GeneralRobustRegressionLoss,
    LogCoshLoss,
    MulticlassFlatten,
    PoissonRegressionLoss,
    QuantileRegressionLoss,
    RelativeLoss,
    TransformedLoss,
)
from sonnix.modules.nlinear import NLinear
from sonnix.modules.numerical import (
    AsinhCosSinNumericalEncoder,
    AsinhNumericalEncoder,
    CosSinNumericalEncoder,
    PiecewiseLinearNumericalEncoder,
)
from sonnix.modules.residual import ResidualBlock
from sonnix.modules.scale_and_shift import ScaleAndShift
from sonnix.modules.shape import Squeeze, View
