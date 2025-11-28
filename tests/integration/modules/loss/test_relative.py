from __future__ import annotations

import pytest
import torch
from torch import nn

from sonnix.modules import RelativeLoss, RelativeMSELoss, RelativeSmoothL1Loss
from sonnix.modules.loss import (
    ArithmeticalMeanIndicator,
    BaseRelativeIndicator,
    ClassicalRelativeIndicator,
    GeometricMeanIndicator,
    MaximumMeanIndicator,
    MinimumMeanIndicator,
    MomentMeanIndicator,
    ReversedRelativeIndicator,
)
from sonnix.utils.loss import is_loss_decreasing_with_sgd

CRITERIA = [
    nn.MSELoss(reduction="none"),
    nn.SmoothL1Loss(reduction="none"),
    nn.L1Loss(reduction="none"),
]
INDICATORS = [
    ArithmeticalMeanIndicator(),
    ClassicalRelativeIndicator(),
    GeometricMeanIndicator(),
    MaximumMeanIndicator(),
    MinimumMeanIndicator(),
    MomentMeanIndicator(),
    MomentMeanIndicator(k=2),
    MomentMeanIndicator(k=3),
    ReversedRelativeIndicator(),
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
    indicator: BaseRelativeIndicator,
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


#####################################
#     Tests for RelativeMSELoss     #
#####################################


@pytest.mark.parametrize("indicator", INDICATORS)
@pytest.mark.parametrize("reduction", REDUCTIONS)
def test_relative_mse_loss_loss_decreasing(
    module: nn.Module, reduction: str, indicator: BaseRelativeIndicator
) -> None:
    assert is_loss_decreasing_with_sgd(
        module=module,
        criterion=RelativeMSELoss(reduction=reduction, indicator=indicator),
        feature=torch.randn(16, 10).clamp(-1.0, 1.0),
        target=torch.rand(16, 8),
        num_iterations=10,
    )


##########################################
#     Tests for RelativeSmoothL1Loss     #
##########################################


@pytest.mark.parametrize("indicator", INDICATORS)
@pytest.mark.parametrize("reduction", REDUCTIONS)
def test_relative_smooth_l1_loss_loss_decreasing(
    module: nn.Module, reduction: str, indicator: BaseRelativeIndicator
) -> None:
    assert is_loss_decreasing_with_sgd(
        module=module,
        criterion=RelativeSmoothL1Loss(reduction=reduction, indicator=indicator),
        feature=torch.randn(16, 10).clamp(-1.0, 1.0),
        target=torch.rand(16, 8),
        num_iterations=10,
    )
