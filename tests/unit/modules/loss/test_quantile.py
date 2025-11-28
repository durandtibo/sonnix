from __future__ import annotations

import pytest
import torch
from coola import objects_are_allclose, objects_are_equal
from coola.utils.tensor import get_available_devices

from sonnix.modules import QuantileRegressionLoss

############################################
#     Tests for QuantileRegressionLoss     #
############################################


def test_quantile_regression_loss_str() -> None:
    assert str(QuantileRegressionLoss()).startswith("QuantileRegressionLoss(")


@pytest.mark.parametrize("device", get_available_devices())
def test_quantile_regression_loss_correct(device: str) -> None:
    device = torch.device(device)
    criterion = QuantileRegressionLoss().to(device=device)
    loss = criterion(torch.ones(2, 3, device=device), torch.ones(2, 3, device=device))
    assert objects_are_equal(loss, torch.tensor(0.0, device=device))


@pytest.mark.parametrize("device", get_available_devices())
def test_quantile_regression_loss_incorrect(device: str) -> None:
    device = torch.device(device)
    criterion = QuantileRegressionLoss().to(device=device)
    loss = criterion(torch.ones(2, 3, device=device), -torch.ones(2, 3, device=device))
    assert objects_are_allclose(loss, torch.tensor(1.0, device=device))


@pytest.mark.parametrize("device", get_available_devices())
def test_quantile_regression_loss_partially_correct(device: str) -> None:
    device = torch.device(device)
    criterion = QuantileRegressionLoss().to(device=device)
    loss = criterion(torch.ones(2, 2, device=device), 2 * torch.eye(2, device=device) - 1)
    assert objects_are_allclose(loss, torch.tensor(0.5, device=device))


@pytest.mark.parametrize("device", get_available_devices())
def test_quantile_regression_loss_reduction_sum(device: str) -> None:
    device = torch.device(device)
    criterion = QuantileRegressionLoss(reduction="sum").to(device=device)
    loss = criterion(torch.ones(2, 2, device=device), 2 * torch.eye(2, device=device) - 1)
    assert objects_are_allclose(loss, torch.tensor(2.0, device=device))


@pytest.mark.parametrize("device", get_available_devices())
def test_quantile_regression_loss_reduction_none(device: str) -> None:
    device = torch.device(device)
    criterion = QuantileRegressionLoss(reduction="none").to(device=device)
    loss = criterion(torch.ones(2, 2, device=device), 2 * torch.eye(2, device=device) - 1)
    assert objects_are_allclose(loss, torch.tensor([[0.0, 1.0], [1.0, 0.0]], device=device))


def test_quantile_regression_loss_reduction_incorrect() -> None:
    with pytest.raises(ValueError, match=r"Incorrect reduction: incorrect"):
        QuantileRegressionLoss(reduction="incorrect")


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("shape", [(2,), (2, 3), (2, 3, 4)])
def test_quantile_regression_loss_shape(device: str, shape: tuple[int, ...]) -> None:
    device = torch.device(device)
    criterion = QuantileRegressionLoss().to(device=device)
    loss = criterion(torch.ones(*shape, device=device), torch.ones(*shape, device=device))
    assert objects_are_equal(loss, torch.tensor(0.0, device=device))


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize(("q", "value"), [(0.5, 0.5), (0.25, 0.25), (0.75, 0.75)])
def test_quantile_regression_loss_q_underestimate(device: str, q: float, value: float) -> None:
    device = torch.device(device)
    criterion = QuantileRegressionLoss(q=q).to(device=device)
    loss = criterion(2 * torch.eye(2, device=device) - 1, torch.ones(2, 2, device=device))
    assert objects_are_equal(loss, torch.tensor(value, device=device))


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize(("q", "value"), [(0.5, 0.5), (0.25, 0.75), (0.75, 0.25)])
def test_quantile_regression_loss_q_overestimate(device: str, q: float, value: float) -> None:
    device = torch.device(device)
    criterion = QuantileRegressionLoss(q=q).to(device=device)
    loss = criterion(torch.ones(2, 2, device=device), 2 * torch.eye(2, device=device) - 1)
    assert objects_are_equal(loss, torch.tensor(value, device=device))
