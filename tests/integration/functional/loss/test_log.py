from __future__ import annotations

import torch
from torch import nn

from sonnix.functional import log_cosh_loss, msle_loss
from sonnix.utils.loss import is_loss_decreasing_with_sgd

###################################
#     Tests for log_cosh_loss     #
###################################


def test_log_cosh_loss_is_loss_decreasing() -> None:
    assert is_loss_decreasing_with_sgd(
        module=nn.Linear(8, 8),
        criterion=log_cosh_loss,
        feature=torch.rand(16, 8),
        target=torch.rand(16, 8),
    )


###############################
#     Tests for msle_loss     #
###############################


def test_msle_loss_is_loss_decreasing() -> None:
    assert is_loss_decreasing_with_sgd(
        module=nn.Sequential(nn.Linear(8, 8), nn.Sigmoid()),
        criterion=msle_loss,
        feature=torch.rand(16, 8),
        target=torch.rand(16, 8),
    )
