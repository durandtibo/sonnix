from __future__ import annotations

import pytest
import torch
from coola.utils.tensor import get_available_devices
from torch.nn.functional import poisson_nll_loss

from sonnix.functional import poisson_regression_loss

SHAPES = [(2,), (2, 3), (2, 3, 4)]

#############################################
#     Tests for poisson_regression_loss     #
#############################################


@pytest.mark.parametrize("device", get_available_devices())
def test_poisson_regression_loss_correct(device: str) -> None:
    device = torch.device(device)
    assert poisson_regression_loss(
        torch.ones(2, 3, device=device), torch.ones(2, 3, device=device)
    ).equal(torch.tensor(1.0, device=device))


@pytest.mark.parametrize("device", get_available_devices())
def test_poisson_regression_loss_incorrect(device: str) -> None:
    device = torch.device(device)
    assert poisson_regression_loss(
        2 * torch.ones(2, 3, device=device), torch.ones(2, 3, device=device)
    ).allclose(torch.tensor(1.3068528194400546, device=device))


@pytest.mark.parametrize("device", get_available_devices())
def test_poisson_regression_loss_partially_correct(device: str) -> None:
    device = torch.device(device)
    assert poisson_regression_loss(
        torch.eye(2, device=device) + 1, torch.ones(2, 2, device=device)
    ).allclose(torch.tensor(1.1534264097200273, device=device))


@pytest.mark.parametrize("device", get_available_devices())
def test_poisson_regression_loss_reduction_sum(device: str) -> None:
    device = torch.device(device)
    assert poisson_regression_loss(
        torch.eye(2, device=device) + 1, torch.ones(2, 2, device=device), reduction="sum"
    ).allclose(torch.tensor(4.613705638880109, device=device))


@pytest.mark.parametrize("device", get_available_devices())
def test_poisson_regression_loss_reduction_none(device: str) -> None:
    device = torch.device(device)
    assert poisson_regression_loss(
        torch.eye(2, device=device) + 1, torch.ones(2, 2, device=device), reduction="none"
    ).allclose(
        torch.tensor([[1.3068528194400546, 1.0], [1.0, 1.3068528194400546]], device=device),
    )


def test_poisson_regression_loss_reduction_incorrect() -> None:
    with pytest.raises(ValueError, match=r"reduction must be one of"):
        poisson_regression_loss(torch.ones(2, 2), torch.eye(2), reduction="incorrect")


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("shape", SHAPES)
def test_poisson_regression_loss_shape(device: str, shape: tuple[int, ...]) -> None:
    device = torch.device(device)
    assert poisson_regression_loss(
        torch.ones(*shape, device=device), torch.ones(*shape, device=device)
    ).equal(torch.tensor(1.0, device=device))


@pytest.mark.parametrize("device", get_available_devices())
def test_poisson_regression_loss_zero(device: str) -> None:
    device = torch.device(device)
    assert poisson_regression_loss(
        torch.zeros(2, 2, device=device), torch.zeros(2, 2, device=device)
    ).allclose(torch.tensor(1e-8, device=device))


@pytest.mark.parametrize("device", get_available_devices())
def test_poisson_regression_loss_pred_zero(device: str) -> None:
    device = torch.device(device)
    assert poisson_regression_loss(
        torch.zeros(2, 2, device=device), torch.ones(2, 2, device=device)
    ).allclose(torch.tensor(18.420680743952367, device=device))


@pytest.mark.parametrize("device", get_available_devices())
def test_poisson_regression_loss_target_zero(device: str) -> None:
    device = torch.device(device)
    assert poisson_regression_loss(
        torch.ones(2, 2, device=device), torch.zeros(2, 2, device=device)
    ).equal(torch.tensor(1.0, device=device))


@pytest.mark.parametrize("eps", [0.0, -1.0])
def test_poisson_regression_loss_invalid_eps(eps: float) -> None:
    with pytest.raises(ValueError, match=r"eps must be greater than 0"):
        poisson_regression_loss(torch.ones(2, 3), torch.ones(2, 3), eps=eps)


@pytest.mark.parametrize("prediction", [-5.0, -50.0, -500.0])
def test_poisson_regression_loss_negative_prediction_is_bounded_below(prediction: float) -> None:
    # The prediction is clamped to eps: the loss is eps - log(eps).
    assert poisson_regression_loss(torch.tensor([prediction]), torch.tensor([1.0])).allclose(
        torch.tensor(18.420680753952367)
    )


def test_poisson_regression_loss_negative_prediction_gradient_is_zero() -> None:
    prediction = torch.tensor([-5.0], requires_grad=True)
    poisson_regression_loss(prediction, torch.tensor([1.0])).backward()
    assert prediction.grad.equal(torch.zeros(1))


def test_poisson_regression_loss_log_input() -> None:
    # exp(0.5) - 2 * 0.5
    assert poisson_regression_loss(
        torch.tensor([0.5]), torch.tensor([2.0]), log_input=True
    ).allclose(torch.tensor(0.6487212707001282))


def test_poisson_regression_loss_log_input_matches_torch() -> None:
    prediction, target = torch.randn(4, 3, generator=torch.manual_seed(1)), torch.rand(4, 3)
    assert poisson_regression_loss(prediction, target, log_input=True).allclose(
        poisson_nll_loss(prediction, target, log_input=True)
    )


def test_poisson_regression_loss_log_input_negative_prediction_gradient() -> None:
    prediction = torch.tensor([-5.0], requires_grad=True)
    poisson_regression_loss(prediction, torch.tensor([1.0]), log_input=True).backward()
    # exp(-5) - 1
    assert prediction.grad.allclose(torch.tensor([-0.9932620530009145]))
