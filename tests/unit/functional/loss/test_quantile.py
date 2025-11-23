from __future__ import annotations

import pytest
import torch
from coola.utils.tensor import get_available_devices

from sonnix.functional import quantile_regression_loss

SHAPES = [(2,), (2, 3), (2, 3, 4)]

##############################################
#     Tests for quantile_regression_loss     #
##############################################


@pytest.mark.parametrize("device", get_available_devices())
def test_quantile_regression_loss_correct(device: str) -> None:
    device = torch.device(device)
    assert quantile_regression_loss(
        torch.ones(2, 3, device=device), torch.ones(2, 3, device=device)
    ).equal(torch.tensor(0.0, device=device))


@pytest.mark.parametrize("device", get_available_devices())
def test_quantile_regression_loss_incorrect(device: str) -> None:
    device = torch.device(device)
    assert quantile_regression_loss(
        torch.ones(2, 3, device=device), -torch.ones(2, 3, device=device)
    ).equal(torch.tensor(1.0, device=device))


@pytest.mark.parametrize("device", get_available_devices())
def test_quantile_regression_loss_partially_correct(device: str) -> None:
    device = torch.device(device)
    assert quantile_regression_loss(
        torch.ones(2, 2, device=device), 2 * torch.eye(2, device=device) - 1
    ).equal(torch.tensor(0.5, device=device))


@pytest.mark.parametrize("device", get_available_devices())
def test_quantile_regression_loss_reduction_sum(device: str) -> None:
    device = torch.device(device)
    assert quantile_regression_loss(
        torch.ones(2, 2, device=device),
        2 * torch.eye(2, device=device) - 1,
        reduction="sum",
    ).allclose(torch.tensor(2.0, device=device))


@pytest.mark.parametrize("device", get_available_devices())
def test_quantile_regression_loss_reduction_none(device: str) -> None:
    device = torch.device(device)
    assert quantile_regression_loss(
        torch.ones(2, 2, device=device),
        2 * torch.eye(2, device=device) - 1,
        reduction="none",
    ).allclose(
        torch.tensor([[0.0, 1.0], [1.0, 0.0]], device=device),
    )


def test_quantile_regression_loss_reduction_incorrect() -> None:
    with pytest.raises(ValueError, match=r"Incorrect reduction: incorrect"):
        quantile_regression_loss(torch.ones(2, 2), torch.eye(2), reduction="incorrect")


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("shape", SHAPES)
def test_quantile_regression_loss_shape(device: str, shape: tuple[int, ...]) -> None:
    device = torch.device(device)
    assert quantile_regression_loss(
        torch.ones(*shape, device=device), torch.ones(*shape, device=device)
    ).equal(torch.tensor(0.0, device=device))


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize(("q", "loss"), [(0.5, 0.5), (0.25, 0.25), (0.75, 0.75)])
def test_quantile_regression_loss_q_underestimate(device: str, q: float, loss: float) -> None:
    device = torch.device(device)
    assert quantile_regression_loss(
        2 * torch.eye(2, device=device) - 1, torch.ones(2, 2, device=device), q=q
    ).equal(torch.tensor(loss, device=device))


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize(("q", "loss"), [(0.5, 0.5), (0.25, 0.75), (0.75, 0.25)])
def test_quantile_regression_loss_q_overestimate(device: str, q: float, loss: float) -> None:
    device = torch.device(device)
    assert quantile_regression_loss(
        torch.ones(2, 2, device=device), 2 * torch.eye(2, device=device) - 1, q=q
    ).equal(torch.tensor(loss, device=device))
