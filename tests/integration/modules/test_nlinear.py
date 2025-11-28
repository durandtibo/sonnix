from __future__ import annotations

import torch
from torch import nn
from torch.nn import MSELoss

from sonnix.modules import NLinear
from sonnix.utils.loss import is_loss_decreasing_with_adam


def test_nlinear_loss() -> None:
    assert is_loss_decreasing_with_adam(
        module=nn.Sequential(
            NLinear(n=3, in_features=4, out_features=8),
            nn.ReLU(),
            NLinear(n=3, in_features=8, out_features=6),
        ),
        criterion=MSELoss(),
        feature=torch.randn(8, 3, 4),
        target=torch.randn(8, 3, 6),
    )
