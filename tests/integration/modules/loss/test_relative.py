from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

import pytest
import torch
from torch import nn

from sonnix.functional import (
    arithmetical_mean_indicator,
    classical_relative_indicator,
    geometric_mean_indicator,
    maximum_mean_indicator,
    minimum_mean_indicator,
    moment_mean_indicator,
    reversed_relative_indicator,
)
from sonnix.modules import RelativeLoss
from sonnix.utils.loss import is_loss_decreasing_with_sgd

if TYPE_CHECKING:
    from collections.abc import Callable

CRITERIA = [
    nn.MSELoss(reduction="none"),
    nn.SmoothL1Loss(reduction="none"),
    nn.L1Loss(reduction="none"),
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


##################################
#     Tests for RelativeLoss     #
##################################


@pytest.mark.parametrize("criterion", CRITERIA)
@pytest.mark.parametrize("indicator", INDICATORS)
@pytest.mark.parametrize("reduction", REDUCTIONS)
def test_relative_loss_loss_decreasing(
    module: nn.Module,
    criterion: nn.Module,
    reduction: str,
    indicator: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
) -> None:
    assert is_loss_decreasing_with_sgd(
        module=module,
        criterion=RelativeLoss(
            criterion=criterion, reduction=reduction, indicator=indicator, eps=1e-5
        ),
        feature=torch.randn(16, 10).clamp(-1.0, 1.0),
        target=torch.rand(16, 8),
        num_iterations=50,
    )
