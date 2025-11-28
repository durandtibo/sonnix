from __future__ import annotations

import pytest
import torch
from torch import nn

from sonnix.modules import ExpSin, Gaussian, Laplacian, MultiQuadratic, Quadratic
from sonnix.utils.loss import is_loss_decreasing_with_adam


@pytest.mark.parametrize(
    "activation",
    [
        ExpSin(),
        ExpSin(num_parameters=6),
        Gaussian(),
        Gaussian(num_parameters=6),
        Laplacian(),
        Laplacian(num_parameters=6),
        MultiQuadratic(),
        MultiQuadratic(num_parameters=6),
        Quadratic(),
        Quadratic(num_parameters=6),
    ],
)
def test_activation_is_loss_decreasing(activation: nn.Module) -> None:
    assert is_loss_decreasing_with_adam(
        module=nn.Sequential(nn.Linear(6, 6), activation),
        criterion=nn.MSELoss(),
        feature=torch.randn(16, 6),
        target=torch.randn(16, 6),
    )
