from __future__ import annotations

import torch
from torch import nn

from sonnix.functional import poisson_regression_loss
from sonnix.utils.loss import is_loss_decreasing_with_adam

#############################################
#     Tests for poisson_regression_loss     #
#############################################


def test_poisson_regression_loss() -> None:
    assert is_loss_decreasing_with_adam(
        module=nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 4), nn.Softplus()),
        criterion=poisson_regression_loss,
        feature=torch.randn(8, 4),
        target=torch.randn(8, 4),
        num_iterations=5,
    )
