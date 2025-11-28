from __future__ import annotations

import torch
from torch import nn

from sonnix.modules import PoissonRegressionLoss
from sonnix.utils.loss import is_loss_decreasing_with_adam


def test_poisson_regression_loss() -> None:
    assert is_loss_decreasing_with_adam(
        module=nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 4), nn.Softplus()),
        criterion=PoissonRegressionLoss(),
        feature=torch.randn(8, 4),
        target=torch.randn(8, 4),
        num_iterations=5,
    )
