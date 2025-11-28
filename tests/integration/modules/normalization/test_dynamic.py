from __future__ import annotations

import torch
from torch.nn import MSELoss

from sonnix.modules import DynamicSinh, DynamicTanh
from sonnix.utils.loss import is_loss_decreasing_with_adam


def test_dynamic_sinh_loss() -> None:
    assert is_loss_decreasing_with_adam(
        module=DynamicSinh(normalized_shape=6),
        criterion=MSELoss(),
        feature=torch.randn(8, 6),
        target=torch.randn(8, 6),
    )


def test_dynamic_tanh_loss() -> None:
    assert is_loss_decreasing_with_adam(
        module=DynamicTanh(normalized_shape=6),
        criterion=MSELoss(),
        feature=torch.randn(8, 6),
        target=torch.randn(8, 6),
    )
