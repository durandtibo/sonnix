r"""Contain loss functions."""

from __future__ import annotations

__all__ = [
    "BaseBinaryFocalLoss",
    "BaseBinaryPoly1Loss",
    "BaseLoss",
    "BinaryFocalLoss",
    "BinaryFocalLossWithLogits",
    "BinaryPoly1Loss",
    "BinaryPoly1LossWithLogits",
    "GeneralRobustRegressionLoss",
    "LogCoshLoss",
    "MulticlassFlatten",
    "PoissonRegressionLoss",
    "QuantileRegressionLoss",
    "RelativeLoss",
    "TransformedLoss",
]

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
