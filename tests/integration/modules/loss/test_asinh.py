from __future__ import annotations

import torch
from torch import nn

from sonnix.modules import AsinhMSELoss, AsinhSmoothL1Loss
from sonnix.utils.loss import is_loss_decreasing_with_sgd

##################################
#     Tests for AsinhMSELoss     #
##################################


def test_asinh_mse_loss_is_loss_decreasing() -> None:
    assert is_loss_decreasing_with_sgd(
        module=nn.Linear(8, 8),
        criterion=AsinhMSELoss(),
        feature=torch.rand(16, 8),
        target=torch.rand(16, 8),
    )


#######################################
#     Tests for AsinhSmoothL1Loss     #
#######################################


def test_asinh_smooth_l1_loss_is_loss_decreasing() -> None:
    assert is_loss_decreasing_with_sgd(
        module=nn.Linear(8, 8),
        criterion=AsinhSmoothL1Loss(),
        feature=torch.rand(16, 8),
        target=torch.rand(16, 8),
    )
