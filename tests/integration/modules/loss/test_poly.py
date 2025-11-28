from __future__ import annotations

import torch
from torch import nn

from sonnix.modules import BinaryPoly1Loss, BinaryPoly1LossWithLogits
from sonnix.utils.loss import is_loss_decreasing_with_adam


def test_binary_poly1_loss() -> None:
    assert is_loss_decreasing_with_adam(
        module=nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 4), nn.Sigmoid()),
        criterion=BinaryPoly1Loss(),
        feature=torch.randn(8, 4),
        target=torch.randint(0, 2, size=(8, 4), dtype=torch.float),
    )


def test_binary_poly1_loss_with_logits() -> None:
    assert is_loss_decreasing_with_adam(
        module=nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 4)),
        criterion=BinaryPoly1LossWithLogits(),
        feature=torch.randn(8, 4),
        target=torch.randint(0, 2, size=(8, 4), dtype=torch.float),
    )
