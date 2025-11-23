from __future__ import annotations

import pytest
import torch
from coola.utils.tensor import get_available_devices

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
    with pytest.raises(ValueError, match=r"Incorrect reduction: incorrect"):
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
    ).equal(torch.tensor(0.0, device=device))


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
