from __future__ import annotations

import pytest
import torch
from torch import nn

from sonnix.modules import (
    RectifierAsinhUnit,
)
from sonnix.utils.loss import is_loss_decreasing_with_adam


@pytest.mark.parametrize(
    "activation",
    [RectifierAsinhUnit()],
)
def test_activation_is_loss_decreasing(activation: nn.Module) -> None:
    assert is_loss_decreasing_with_adam(
        module=nn.Sequential(nn.Linear(6, 6), activation),
        criterion=nn.MSELoss(),
        feature=torch.randn(16, 6),
        target=torch.randn(16, 6),
    )
