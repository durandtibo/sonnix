from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

import pytest
import torch
from torch import nn
from torch.nn.functional import l1_loss, mse_loss, smooth_l1_loss

from sonnix.functional import relative_loss
from sonnix.functional.loss import (
    arithmetical_mean_indicator,
    classical_relative_indicator,
    geometric_mean_indicator,
    maximum_mean_indicator,
    minimum_mean_indicator,
    moment_mean_indicator,
    reversed_relative_indicator,
)
from sonnix.utils.loss import is_loss_decreasing_with_sgd

if TYPE_CHECKING:
    from collections.abc import Callable


LOSSES = [
    partial(mse_loss, reduction="none"),
    partial(smooth_l1_loss, reduction="none"),
    partial(l1_loss, reduction="none"),
]
INDICATORS = [
    arithmetical_mean_indicator,
    classical_relative_indicator,
    geometric_mean_indicator,
    maximum_mean_indicator,
    minimum_mean_indicator,
    moment_mean_indicator,
    partial(moment_mean_indicator, k=2),
    partial(moment_mean_indicator, k=3),
    reversed_relative_indicator,
]
REDUCTIONS = ["mean", "sum"]


@pytest.fixture
def module() -> nn.Module:
    return nn.Sequential(nn.Linear(10, 32), nn.ReLU(), nn.Linear(32, 8), nn.Softplus())


###################################
#     Tests for relative_loss     #
###################################


@pytest.mark.parametrize("base_loss", LOSSES)
@pytest.mark.parametrize("indicator", INDICATORS)
@pytest.mark.parametrize("reduction", REDUCTIONS)
def test_relative_loss_loss_decreasing(
    module: nn.Module,
    base_loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    indicator: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    reduction: str,
) -> None:
    def my_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return relative_loss(
            loss=base_loss(prediction, target),
            indicator=indicator(prediction, target),
            reduction=reduction,
        )

    assert is_loss_decreasing_with_sgd(
        module=module,
        criterion=my_loss,
        feature=torch.randn(16, 10).clamp(-1.0, 1.0),
        target=torch.rand(16, 8),
        num_iterations=50,
    )
