from __future__ import annotations

import torch
from torch import nn

from sonnix.functional import general_robust_regression_loss
from sonnix.utils.loss import is_loss_decreasing_with_sgd

####################################################
#     Tests for general_robust_regression_loss     #
####################################################


def test_general_robust_regression_loss_decreasing() -> None:
    assert is_loss_decreasing_with_sgd(
        module=nn.Linear(8, 8),
        criterion=general_robust_regression_loss,
        feature=torch.rand(16, 8),
        target=torch.rand(16, 8),
    )
