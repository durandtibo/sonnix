from __future__ import annotations

import pytest
import torch
from coola.equality import objects_are_allclose, objects_are_equal
from coola.utils.tensor import get_available_devices

from sonnix.modules import QuantileRegressionLoss
from tests.helpers import make_generator

############################################
#     Tests for QuantileRegressionLoss     #
############################################


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
def test_quantile_regression_loss_reduction_none(device: str) -> None:
    device = torch.device(device)
    criterion = QuantileRegressionLoss(reduction="none").to(device=device)
    loss = criterion(torch.ones(2, 2, device=device), 2 * torch.eye(2, device=device) - 1)
    assert objects_are_allclose(loss, torch.tensor([[0.0, 1.0], [1.0, 0.0]], device=device))


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


@pytest.mark.parametrize("q", [-0.1, 1.1])
def test_quantile_regression_loss_module_incorrect_q(q: float) -> None:
    with pytest.raises(ValueError, match="q must be in the range"):
        QuantileRegressionLoss(q=q)


def test_quantile_regression_loss_median_is_half_of_l1_loss() -> None:
    generator = make_generator(1)
    prediction = torch.randn(6, 5, generator=generator)
    target = torch.randn(6, 5, generator=generator)
    assert QuantileRegressionLoss(q=0.5)(prediction, target).allclose(
        0.5 * torch.nn.functional.l1_loss(prediction, target)
    )


def test_quantile_regression_loss_extra_repr() -> None:
    assert QuantileRegressionLoss(q=0.3).extra_repr() == "reduction=mean, q=0.3"
