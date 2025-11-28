r"""Contain loss functions."""

from __future__ import annotations

__all__ = [
    "ArithmeticalMeanIndicator",
    "AsinhMSELoss",
    "AsinhSmoothL1Loss",
    "BaseRelativeIndicator",
    "BinaryFocalLoss",
    "BinaryFocalLossWithLogits",
    "BinaryPoly1Loss",
    "BinaryPoly1LossWithLogits",
    "ClassicalRelativeIndicator",
    "GeneralRobustRegressionLoss",
    "GeometricMeanIndicator",
    "MaximumMeanIndicator",
    "MinimumMeanIndicator",
    "MomentMeanIndicator",
    "PoissonRegressionLoss",
    "QuantileRegressionLoss",
    "RelativeLoss",
    "RelativeMSELoss",
    "RelativeSmoothL1Loss",
    "ReversedRelativeIndicator",
    "TransformedLoss",
]

from sonnix.modules.loss.asinh import AsinhMSELoss, AsinhSmoothL1Loss
from sonnix.modules.loss.focal import BinaryFocalLoss, BinaryFocalLossWithLogits
from sonnix.modules.loss.general_robust import GeneralRobustRegressionLoss
from sonnix.modules.loss.indicators import (
    ArithmeticalMeanIndicator,
    BaseRelativeIndicator,
    ClassicalRelativeIndicator,
    GeometricMeanIndicator,
    MaximumMeanIndicator,
    MinimumMeanIndicator,
    MomentMeanIndicator,
    ReversedRelativeIndicator,
)
from sonnix.modules.loss.poisson import PoissonRegressionLoss
from sonnix.modules.loss.poly import BinaryPoly1Loss, BinaryPoly1LossWithLogits
from sonnix.modules.loss.quantile import QuantileRegressionLoss
from sonnix.modules.loss.relative import (
    RelativeLoss,
    RelativeMSELoss,
    RelativeSmoothL1Loss,
)
from sonnix.modules.loss.transform import TransformedLoss
