from __future__ import annotations

import inspect
from typing import TYPE_CHECKING

import pytest

from sonnix.modules import (
    AsinhMSELoss,
    AsinhSmoothL1Loss,
    BinaryFocalLoss,
    BinaryFocalLossWithLogits,
    BinaryPoly1Loss,
    BinaryPoly1LossWithLogits,
    GeneralRobustRegressionLoss,
    PoissonRegressionLoss,
    QuantileRegressionLoss,
    RelativeLoss,
    RelativeMSELoss,
    RelativeSmoothL1Loss,
)

if TYPE_CHECKING:
    from torch import nn

#################################################
#     Tests for the signature of the losses     #
#################################################

LOSSES = [
    AsinhMSELoss,
    AsinhSmoothL1Loss,
    BinaryFocalLoss,
    BinaryFocalLossWithLogits,
    BinaryPoly1Loss,
    BinaryPoly1LossWithLogits,
    GeneralRobustRegressionLoss,
    PoissonRegressionLoss,
    QuantileRegressionLoss,
    RelativeLoss,
    RelativeMSELoss,
    RelativeSmoothL1Loss,
]


@pytest.mark.parametrize("loss", LOSSES)
def test_loss_reduction_is_keyword_only(loss: type[nn.Module]) -> None:
    parameters = inspect.signature(loss).parameters
    assert parameters["reduction"].kind is inspect.Parameter.KEYWORD_ONLY
