from __future__ import annotations

import torch
from torch import nn

from sonnix.functional import asinh_mse_loss, asinh_smooth_l1_loss
from sonnix.utils.loss import is_loss_decreasing_with_sgd

####################################
#     Tests for asinh_mse_loss     #
####################################


def test_asinh_mse_loss_is_loss_decreasing() -> None:
    assert is_loss_decreasing_with_sgd(
        module=nn.Linear(8, 8),
        criterion=asinh_mse_loss,
        feature=torch.rand(16, 8),
        target=torch.rand(16, 8),
    )


##########################################
#     Tests for asinh_smooth_l1_loss     #
##########################################


def test_asinh_smooth_l1_loss_is_loss_decreasing() -> None:
    assert is_loss_decreasing_with_sgd(
        module=nn.Linear(8, 8),
        criterion=asinh_smooth_l1_loss,
        feature=torch.rand(16, 8),
        target=torch.rand(16, 8),
    )
