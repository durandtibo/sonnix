from __future__ import annotations

import pytest
import torch
from coola.utils.tensor import get_available_devices
from torch import nn
from torch.optim import SGD

from sonnix.utils.loss import (
    is_loss_decreasing,
    is_loss_decreasing_with_adam,
    is_loss_decreasing_with_sgd,
)

########################################
#     Tests for is_loss_decreasing     #
########################################


@pytest.mark.parametrize("device", get_available_devices())
def test_is_loss_decreasing_true(device: str) -> None:
    device = torch.device(device)
    module = nn.Linear(4, 6)
    assert is_loss_decreasing(
        module=module.to(device=device),
        criterion=nn.MSELoss().to(device=device),
        optimizer=SGD(module.parameters(), lr=0.01),
        feature=torch.rand(8, 4, device=device),
        target=torch.rand(8, 6, device=device),
    )


def test_is_loss_decreasing_false() -> None:
    module = nn.Linear(4, 6)
    assert not is_loss_decreasing(
        module=module,
        criterion=nn.MSELoss(),
        optimizer=SGD(module.parameters(), lr=0.01),
        feature=torch.rand(8, 4),
        target=torch.rand(8, 6),
        num_iterations=0,
    )


def test_is_loss_decreasing_train_mode() -> None:
    module = nn.Linear(4, 6)
    module.train()
    assert is_loss_decreasing(
        module=module,
        criterion=nn.MSELoss(),
        optimizer=SGD(module.parameters(), lr=0.01),
        feature=torch.rand(8, 4),
        target=torch.rand(8, 6),
    )
    assert module.training


def test_is_loss_decreasing_eval_mode() -> None:
    module = nn.Linear(4, 6)
    module.eval()
    assert is_loss_decreasing(
        module=module,
        criterion=nn.MSELoss(),
        optimizer=SGD(module.parameters(), lr=0.01),
        feature=torch.rand(8, 4),
        target=torch.rand(8, 6),
    )
    assert not module.training


def test_is_loss_decreasing_criterion_functional() -> None:
    module = nn.Linear(4, 6)
    optimizer = SGD(module.parameters(), lr=0.01)
    assert is_loss_decreasing(
        module=module,
        criterion=nn.functional.mse_loss,
        optimizer=optimizer,
        feature=torch.rand(8, 4),
        target=torch.rand(8, 6),
    )


##################################################
#     Tests for is_loss_decreasing_with_adam     #
##################################################


def test_is_loss_decreasing_with_adam_true() -> None:
    assert is_loss_decreasing_with_adam(
        module=nn.Linear(4, 6),
        criterion=nn.MSELoss(),
        feature=torch.rand(8, 4),
        target=torch.rand(8, 6),
    )


#################################################
#     Tests for is_loss_decreasing_with_sgd     #
#################################################


def test_is_loss_decreasing_with_sgd_true() -> None:
    assert is_loss_decreasing_with_sgd(
        module=nn.Linear(4, 6),
        criterion=nn.MSELoss(),
        feature=torch.rand(8, 4),
        target=torch.rand(8, 6),
    )
