from __future__ import annotations

import pytest
import torch
from coola.equality import objects_are_allclose, objects_are_equal
from coola.utils.tensor import get_available_devices

from sonnix.modules import AsinhMSELoss, AsinhSmoothL1Loss

##################################
#     Tests for AsinhMSELoss     #
##################################


def test_asinh_mse_loss_str() -> None:
    assert str(AsinhMSELoss()).startswith("AsinhMSELoss(")


@pytest.mark.parametrize("device", get_available_devices())
def test_asinh_mse_loss_correct(device: str) -> None:
    device = torch.device(device)
    criterion = AsinhMSELoss().to(device=device)
    loss = criterion(torch.ones(2, 3, device=device), torch.ones(2, 3, device=device))
    assert objects_are_equal(loss, torch.tensor(0.0, device=device))


@pytest.mark.parametrize("device", get_available_devices())
def test_asinh_mse_loss_incorrect(device: str) -> None:
    device = torch.device(device)
    criterion = AsinhMSELoss().to(device=device)
    loss = criterion(torch.ones(2, 3, device=device), -torch.ones(2, 3, device=device))
    assert objects_are_allclose(loss, torch.tensor(3.107277599582784, device=device))


@pytest.mark.parametrize("device", get_available_devices())
def test_asinh_mse_loss_partially_correct(device: str) -> None:
    device = torch.device(device)
    criterion = AsinhMSELoss().to(device=device)
    loss = criterion(torch.ones(2, 2, device=device), 2 * torch.eye(2, device=device) - 1)
    assert objects_are_allclose(loss, torch.tensor(1.553638799791392, device=device))


@pytest.mark.parametrize("device", get_available_devices())
def test_asinh_mse_loss_reduction_sum(device: str) -> None:
    device = torch.device(device)
    criterion = AsinhMSELoss(reduction="sum").to(device=device)
    loss = criterion(torch.ones(2, 2, device=device), 2 * torch.eye(2, device=device) - 1)
    assert objects_are_allclose(loss, torch.tensor(6.214555199165568, device=device))


@pytest.mark.parametrize("device", get_available_devices())
def test_asinh_mse_loss_reduction_none(device: str) -> None:
    device = torch.device(device)
    criterion = AsinhMSELoss(reduction="none").to(device=device)
    loss = criterion(torch.ones(2, 2, device=device), 2 * torch.eye(2, device=device) - 1)
    assert objects_are_allclose(
        loss, torch.tensor([[0.0, 3.107277599582784], [3.107277599582784, 0.0]], device=device)
    )


def test_asinh_mse_loss_reduction_incorrect() -> None:
    with pytest.raises(ValueError, match=r"Incorrect reduction: incorrect"):
        AsinhMSELoss(reduction="incorrect")


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("shape", [(2,), (2, 3), (2, 3, 4)])
def test_asinh_mse_loss_shape(device: str, shape: tuple[int, ...]) -> None:
    device = torch.device(device)
    criterion = AsinhMSELoss().to(device=device)
    loss = criterion(torch.ones(*shape, device=device), torch.ones(*shape, device=device))
    assert objects_are_equal(loss, torch.tensor(0.0, device=device))


#######################################
#     Tests for AsinhSmoothL1Loss     #
#######################################


def test_asinh_smooth_l1_loss_str() -> None:
    assert str(AsinhSmoothL1Loss()).startswith("AsinhSmoothL1Loss(")


@pytest.mark.parametrize("device", get_available_devices())
def test_asinh_smooth_l1_loss_correct(device: str) -> None:
    device = torch.device(device)
    criterion = AsinhSmoothL1Loss().to(device=device)
    loss = criterion(torch.ones(2, 3, device=device), torch.ones(2, 3, device=device))
    assert objects_are_equal(loss, torch.tensor(0.0, device=device))


@pytest.mark.parametrize("device", get_available_devices())
def test_asinh_smooth_l1_loss_incorrect(device: str) -> None:
    device = torch.device(device)
    criterion = AsinhSmoothL1Loss().to(device=device)
    loss = criterion(torch.ones(2, 3, device=device), -torch.ones(2, 3, device=device))
    assert objects_are_allclose(loss, torch.tensor(1.262747174039086, device=device))


@pytest.mark.parametrize("device", get_available_devices())
def test_asinh_smooth_l1_loss_partially_correct(device: str) -> None:
    device = torch.device(device)
    criterion = AsinhSmoothL1Loss().to(device=device)
    loss = criterion(torch.ones(2, 2, device=device), 2 * torch.eye(2, device=device) - 1)
    assert objects_are_allclose(loss, torch.tensor(0.631373587019543, device=device))


@pytest.mark.parametrize("device", get_available_devices())
def test_asinh_smooth_l1_loss_reduction_sum(device: str) -> None:
    device = torch.device(device)
    criterion = AsinhSmoothL1Loss(reduction="sum").to(device=device)
    loss = criterion(torch.ones(2, 2, device=device), 2 * torch.eye(2, device=device) - 1)
    assert objects_are_allclose(loss, torch.tensor(2.525494348078172, device=device))


@pytest.mark.parametrize("device", get_available_devices())
def test_asinh_smooth_l1_loss_reduction_none(device: str) -> None:
    device = torch.device(device)
    criterion = AsinhSmoothL1Loss(reduction="none").to(device=device)
    loss = criterion(torch.ones(2, 2, device=device), 2 * torch.eye(2, device=device) - 1)
    assert objects_are_allclose(
        loss, torch.tensor([[0.0, 1.262747174039086], [1.262747174039086, 0.0]], device=device)
    )


def test_asinh_smooth_l1_loss_reduction_incorrect() -> None:
    with pytest.raises(ValueError, match=r"Incorrect reduction: incorrect"):
        AsinhSmoothL1Loss(reduction="incorrect")


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("shape", [(2,), (2, 3), (2, 3, 4)])
def test_asinh_smooth_l1_loss_shape(device: str, shape: tuple[int, ...]) -> None:
    device = torch.device(device)
    criterion = AsinhSmoothL1Loss().to(device=device)
    loss = criterion(torch.ones(*shape, device=device), torch.ones(*shape, device=device))
    assert objects_are_equal(loss, torch.tensor(0.0, device=device))
