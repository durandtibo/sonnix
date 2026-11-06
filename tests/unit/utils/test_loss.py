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


def test_is_loss_decreasing_train_mode_check1() -> None:
    module = nn.Linear(4, 6)
    module.train()
    assert is_loss_decreasing(
        module=module,
        criterion=nn.MSELoss(),
        optimizer=SGD(module.parameters(), lr=0.01),
        feature=torch.rand(8, 4),
        target=torch.rand(8, 6),
    )


def test_is_loss_decreasing_train_mode_training() -> None:
    module = nn.Linear(4, 6)
    module.train()
    assert module.training


def test_is_loss_decreasing_eval_mode_check1() -> None:
    module = nn.Linear(4, 6)
    module.eval()
    assert is_loss_decreasing(
        module=module,
        criterion=nn.MSELoss(),
        optimizer=SGD(module.parameters(), lr=0.01),
        feature=torch.rand(8, 4),
        target=torch.rand(8, 6),
    )


def test_is_loss_decreasing_eval_mode_training() -> None:
    module = nn.Linear(4, 6)
    module.eval()
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


#########################################
#     Tests for the max_norm option     #
#########################################


def _scalar_linear() -> nn.Linear:
    module = nn.Linear(1, 1, bias=False)
    module.weight.data.fill_(1.0)
    return module


def test_is_loss_decreasing_max_norm_default_clips_gradient() -> None:
    module = _scalar_linear()
    assert is_loss_decreasing(
        module=module,
        criterion=nn.MSELoss(),
        optimizer=SGD(module.parameters(), lr=0.1),
        feature=torch.full((1, 1), 10.0),
        target=torch.zeros(1, 1),
    )


def test_is_loss_decreasing_max_norm_none_does_not_clip_gradient() -> None:
    module = _scalar_linear()
    assert not is_loss_decreasing(
        module=module,
        criterion=nn.MSELoss(),
        optimizer=SGD(module.parameters(), lr=0.1),
        feature=torch.full((1, 1), 10.0),
        target=torch.zeros(1, 1),
        max_norm=None,
    )


def test_is_loss_decreasing_with_sgd_max_norm_none_does_not_clip_gradient() -> None:
    assert not is_loss_decreasing_with_sgd(
        module=_scalar_linear(),
        criterion=nn.MSELoss(),
        feature=torch.full((1, 1), 10.0),
        target=torch.zeros(1, 1),
        lr=0.1,
        max_norm=None,
    )


def test_is_loss_decreasing_with_sgd_max_norm_default_clips_gradient() -> None:
    assert is_loss_decreasing_with_sgd(
        module=_scalar_linear(),
        criterion=nn.MSELoss(),
        feature=torch.full((1, 1), 10.0),
        target=torch.zeros(1, 1),
        lr=0.1,
    )


def test_is_loss_decreasing_with_adam_max_norm_is_forwarded() -> None:
    assert is_loss_decreasing_with_adam(
        module=_scalar_linear(),
        criterion=nn.MSELoss(),
        feature=torch.full((1, 1), 10.0),
        target=torch.zeros(1, 1),
        max_norm=0.5,
    )


#############################################
#     Tests for the options and the seed    #
#############################################


def test_is_loss_decreasing_num_iterations_is_keyword_only() -> None:
    module = nn.Linear(4, 6)
    with pytest.raises(TypeError, match=r"positional argument"):
        is_loss_decreasing(
            module,
            nn.MSELoss(),
            SGD(module.parameters(), lr=0.01),
            torch.rand(8, 4),
            torch.rand(8, 6),
            2,
        )


@pytest.mark.parametrize("function", [is_loss_decreasing_with_adam, is_loss_decreasing_with_sgd])
def test_is_loss_decreasing_with_optimizer_lr_is_keyword_only(function: object) -> None:
    with pytest.raises(TypeError, match=r"positional argument"):
        function(nn.Linear(4, 6), nn.MSELoss(), torch.rand(8, 4), torch.rand(8, 6), 0.01)


def _train_with_dropout(random_seed: int) -> torch.Tensor:
    torch.manual_seed(0)
    module = nn.Sequential(nn.Linear(4, 6), nn.Dropout(0.5))
    is_loss_decreasing(
        module=module,
        criterion=nn.MSELoss(),
        optimizer=SGD(module.parameters(), lr=0.1),
        feature=torch.ones(8, 4),
        target=torch.ones(8, 6),
        num_iterations=3,
        random_seed=random_seed,
    )
    return module[0].weight.detach().clone()


def test_is_loss_decreasing_training_steps_are_deterministic() -> None:
    torch.manual_seed(5)
    first = _train_with_dropout(random_seed=1)
    torch.manual_seed(99)
    second = _train_with_dropout(random_seed=1)
    assert first.equal(second)
