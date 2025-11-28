from __future__ import annotations

import torch
from torch.nn import MSELoss

from sonnix.modules import ScaleAndShift
from sonnix.utils.loss import is_loss_decreasing_with_adam


def test_scale_and_shift_loss() -> None:
    assert is_loss_decreasing_with_adam(
        module=ScaleAndShift(6),
        criterion=MSELoss(),
        feature=torch.randn(8, 6),
        target=torch.randn(8, 6),
    )
