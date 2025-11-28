from __future__ import annotations

import pytest
import torch
from torch import nn

from sonnix.modules import Asinh, Clamp, Sinh, TransformedLoss
from sonnix.utils.loss import is_loss_decreasing_with_sgd

CRITERIA = [nn.MSELoss(), nn.SmoothL1Loss(), nn.L1Loss()]
TRANSFORMS = [Asinh(), Sinh(), Clamp()]


#####################################
#     Tests for TransformedLoss     #
#####################################


@pytest.mark.parametrize("criterion", CRITERIA)
@pytest.mark.parametrize("prediction", TRANSFORMS)
@pytest.mark.parametrize("target", TRANSFORMS)
def test_transformed_loss_loss_decreasing(
    criterion: nn.Module, prediction: nn.Module, target: nn.Module
) -> None:
    assert is_loss_decreasing_with_sgd(
        module=nn.Linear(8, 8),
        criterion=TransformedLoss(criterion=criterion, prediction=prediction, target=target),
        feature=torch.randn(16, 8),
        target=torch.randn(16, 8),
    )
