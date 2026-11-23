from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import torch
from coola.utils.tensor import get_available_devices
from torch import nn
from torch.optim import SGD

from sonnix.testing.loss import (
    is_loss_decreasing,
    is_loss_decreasing_with_adam,
    is_loss_decreasing_with_sgd,
)

if TYPE_CHECKING:
    from collections.abc import Callable

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
        optimizer=SGD(module.parameters(), lr=0.0),
        feature=torch.rand(8, 4),
        target=torch.rand(8, 6),
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


def test_is_loss_decreasing_n_iterations_is_keyword_only() -> None:
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
        n_iterations=3,
        random_seed=random_seed,
    )
    return module[0].weight.detach().clone()


def test_is_loss_decreasing_training_steps_are_deterministic() -> None:
    torch.manual_seed(5)
    first = _train_with_dropout(random_seed=1)
    torch.manual_seed(99)
    second = _train_with_dropout(random_seed=1)
    assert first.equal(second)


#####################################################
#     Tests for the validation of the arguments     #
#####################################################

FUNCTIONS = [is_loss_decreasing_with_adam, is_loss_decreasing_with_sgd]


def _call(function: Callable[..., bool], **kwargs: object) -> bool:
    module = nn.Linear(4, 2)
    arguments = (module, nn.MSELoss(), torch.rand(8, 4), torch.rand(8, 2))
    if function is is_loss_decreasing:
        arguments = (
            module,
            nn.MSELoss(),
            SGD(module.parameters(), lr=0.01),
            *arguments[2:],
        )
    return function(*arguments, **kwargs)


ALL_FUNCTIONS = [is_loss_decreasing, *FUNCTIONS]


@pytest.mark.parametrize("function", ALL_FUNCTIONS)
@pytest.mark.parametrize("value", [0, -1])
def test_is_loss_decreasing_rejects_a_number_of_iterations_lower_than_1(
    function: Callable[..., bool], value: int
) -> None:
    with pytest.raises(ValueError, match=r"n_iterations must be greater than or equal to 1"):
        _call(function, n_iterations=value)


@pytest.mark.parametrize("function", ALL_FUNCTIONS)
@pytest.mark.parametrize("value", [1.5, True, "1"])
def test_is_loss_decreasing_rejects_a_non_integer_number_of_iterations(
    function: Callable[..., bool], value: object
) -> None:
    with pytest.raises(TypeError, match=r"n_iterations must be an int"):
        _call(function, n_iterations=value)


@pytest.mark.parametrize("function", ALL_FUNCTIONS)
@pytest.mark.parametrize("value", [1.5, True])
def test_is_loss_decreasing_rejects_a_non_integer_random_seed(
    function: Callable[..., bool], value: object
) -> None:
    with pytest.raises(TypeError, match=r"random_seed must be an int"):
        _call(function, random_seed=value)


@pytest.mark.parametrize("function", ALL_FUNCTIONS)
@pytest.mark.parametrize("value", [0.0, -1.0])
def test_is_loss_decreasing_rejects_a_max_norm_that_is_not_positive(
    function: Callable[..., bool], value: float
) -> None:
    with pytest.raises(ValueError, match=r"max_norm must be greater than 0"):
        _call(function, max_norm=value)


@pytest.mark.parametrize("function", ALL_FUNCTIONS)
@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_is_loss_decreasing_rejects_a_max_norm_that_is_not_finite(
    function: Callable[..., bool], value: float
) -> None:
    with pytest.raises(ValueError, match=r"max_norm must be finite"):
        _call(function, max_norm=value)


@pytest.mark.parametrize("function", ALL_FUNCTIONS)
def test_is_loss_decreasing_rejects_a_max_norm_that_is_not_a_number(
    function: Callable[..., bool],
) -> None:
    with pytest.raises(TypeError, match=r"max_norm must be a real number"):
        _call(function, max_norm="a")


@pytest.mark.parametrize("function", ALL_FUNCTIONS)
def test_is_loss_decreasing_accepts_no_clipping(function: Callable[..., bool]) -> None:
    assert isinstance(_call(function, max_norm=None), bool)


@pytest.mark.parametrize("function", FUNCTIONS)
@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_is_loss_decreasing_with_a_learning_rate_that_is_not_finite(
    function: Callable[..., bool], value: float
) -> None:
    with pytest.raises(ValueError, match=r"lr must be finite"):
        _call(function, lr=value)


@pytest.mark.parametrize("function", FUNCTIONS)
@pytest.mark.parametrize("value", [0.0, -0.1])
def test_is_loss_decreasing_with_a_learning_rate_that_is_not_positive(
    function: Callable[..., bool], value: float
) -> None:
    with pytest.raises(ValueError, match=r"lr must be greater than 0"):
        _call(function, lr=value)


@pytest.mark.parametrize("function", FUNCTIONS)
def test_is_loss_decreasing_with_a_learning_rate_that_is_not_a_number(
    function: Callable[..., bool],
) -> None:
    with pytest.raises(TypeError, match=r"lr must be a real number"):
        _call(function, lr="a")


@pytest.mark.parametrize("function", FUNCTIONS)
def test_is_loss_decreasing_does_not_train_when_an_argument_is_invalid(
    function: Callable[..., bool],
) -> None:
    module = nn.Linear(4, 2)
    weight = module.weight.detach().clone()
    with pytest.raises(ValueError, match=r"n_iterations must be greater than or equal to 1"):
        function(module, nn.MSELoss(), torch.rand(8, 4), torch.rand(8, 2), n_iterations=0)
    assert module.weight.equal(weight)
