from __future__ import annotations

import pytest
import torch
from coola.utils.tensor import get_available_devices

from sonnix.modules import BinaryPoly1Loss, BinaryPoly1LossWithLogits

SIZES = (1, 2)
TOLERANCE = 1e-6

SHAPES = [(2,), (2, 3), (2, 3, 4)]


#####################################
#     Tests for BinaryPoly1Loss     #
#####################################


def test_binary_focal_loss_str() -> None:
    assert str(BinaryPoly1Loss()).startswith("BinaryPoly1Loss(")


@pytest.mark.parametrize("alpha", [-1.0, 0.0, 0.5, 1])
def test_binary_focal_loss_alpha(alpha: float) -> None:
    assert BinaryPoly1Loss(alpha=alpha)._alpha == alpha


@pytest.mark.parametrize("device", get_available_devices())
def test_binary_focal_loss_forward_correct(device: str) -> None:
    device = torch.device(device)
    criterion = BinaryPoly1Loss().to(device=device)
    loss = criterion(
        torch.tensor([[0.9, 0.1, 0.9], [0.1, 0.9, 0.1]], device=device),
        torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]], device=device),
    )
    assert loss.allclose(torch.tensor(0.2053605156578263, device=device))


@pytest.mark.parametrize("device", get_available_devices())
def test_binary_focal_loss_forward_incorrect(device: str) -> None:
    device = torch.device(device)
    criterion = BinaryPoly1Loss().to(device=device)
    loss = criterion(
        torch.tensor([[0.9, 0.1, 0.9], [0.1, 0.9, 0.1]], device=device),
        torch.tensor([[0.0, 1.0, 0.0], [1.0, 0.0, 1.0]], device=device),
    )
    assert loss.allclose(torch.tensor(3.2025850929940454, device=device))


@pytest.mark.parametrize("device", get_available_devices())
def test_binary_focal_loss_forward_partially_correct(device: str) -> None:
    device = torch.device(device)
    criterion = BinaryPoly1Loss().to(device=device)
    loss = criterion(
        torch.tensor([[0.9, 0.1, 0.9], [0.1, 0.9, 0.9]], device=device),
        torch.tensor([[1.0, 1.0, 0.0], [0.0, 1.0, 0.0]], device=device),
    )
    assert loss.allclose(torch.tensor(1.703972804325936, device=device))


@pytest.mark.parametrize("device", get_available_devices())
def test_binary_focal_loss_forward_reduction_sum(device: str) -> None:
    device = torch.device(device)
    criterion = BinaryPoly1Loss(reduction="sum").to(device=device)
    loss = criterion(
        torch.tensor([[0.9, 0.1, 0.9], [0.1, 0.9, 0.9]], device=device),
        torch.tensor([[1.0, 1.0, 0.0], [0.0, 1.0, 0.0]], device=device),
    )
    assert loss.allclose(torch.tensor(10.223836825955615, device=device))


@pytest.mark.parametrize("device", get_available_devices())
def test_binary_focal_loss_forward_reduction_none(device: str) -> None:
    device = torch.device(device)
    criterion = BinaryPoly1Loss(reduction="none").to(device=device)
    loss = criterion(
        torch.tensor([[0.9, 0.1, 0.9], [0.1, 0.9, 0.9]], device=device),
        torch.tensor([[1.0, 1.0, 0.0], [0.0, 1.0, 0.0]], device=device),
    )
    assert loss.allclose(
        torch.tensor(
            [
                [0.2053605156578263, 3.2025850929940454, 3.2025850929940454],
                [0.2053605156578263, 0.2053605156578263, 3.2025850929940454],
            ],
            device=device,
        ),
    )


def test_binary_focal_loss_forward_incorrect_reduction() -> None:
    with pytest.raises(ValueError, match=r"Incorrect reduction: incorrect reduction."):
        BinaryPoly1Loss(reduction="incorrect reduction")


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("shape", SHAPES)
def test_binary_focal_loss_forward_shape(device: str, shape: tuple[int, ...]) -> None:
    device = torch.device(device)
    criterion = BinaryPoly1Loss().to(device=device)
    loss = criterion(torch.ones(*shape, device=device), torch.ones(*shape, device=device))
    assert loss.allclose(torch.tensor(0.0, device=device))


@pytest.mark.parametrize("device", get_available_devices())
def test_binary_focal_loss_forward_alpha_0_5(device: str) -> None:
    device = torch.device(device)
    criterion = BinaryPoly1Loss(alpha=0.5).to(device=device)
    loss = criterion(
        torch.tensor([[0.9, 0.1, 0.9], [0.1, 0.9, 0.9]], device=device),
        torch.tensor([[1.0, 1.0, 0.0], [0.0, 1.0, 0.0]], device=device),
    )
    assert loss.allclose(torch.tensor(1.453972804325936, device=device))


@pytest.mark.parametrize("device", get_available_devices())
def test_binary_focal_loss_backward(device: str) -> None:
    device = torch.device(device)
    criterion = BinaryPoly1Loss().to(device=device)
    loss = criterion(
        torch.tensor([[0.9, 0.1, 0.9], [0.1, 0.9, 0.1]], device=device, requires_grad=True),
        torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]], device=device),
    )
    loss.backward()
    assert loss.allclose(torch.tensor(0.2053605156578263, device=device))


###############################################
#     Tests for BinaryPoly1LossWithLogits     #
###############################################


def test_binary_focal_loss_with_logits_str() -> None:
    assert str(BinaryPoly1LossWithLogits()).startswith("BinaryPoly1LossWithLogits(")


@pytest.mark.parametrize("alpha", [-1.0, 0.0, 0.5, 1])
def test_binary_focal_loss_with_logits_alpha(alpha: float) -> None:
    assert BinaryPoly1LossWithLogits(alpha=alpha)._alpha == alpha


@pytest.mark.parametrize("device", get_available_devices())
def test_binary_focal_loss_with_logits_forward_correct(device: str) -> None:
    device = torch.device(device)
    criterion = BinaryPoly1LossWithLogits().to(device=device)
    loss = criterion(
        torch.tensor([[1.0, -1.0, 1.0], [-1.0, 1.0, -1.0]], device=device),
        torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]], device=device),
    )
    assert loss.allclose(torch.tensor(0.5822031346214558, device=device))


@pytest.mark.parametrize("device", get_available_devices())
def test_binary_focal_loss_with_logits_forward_incorrect(device: str) -> None:
    device = torch.device(device)
    criterion = BinaryPoly1LossWithLogits().to(device=device)
    loss = criterion(
        torch.tensor([[1.0, -1.0, 1.0], [-1.0, 1.0, -1.0]], device=device),
        torch.tensor([[0.0, 1.0, 0.0], [1.0, 0.0, 1.0]], device=device),
    )
    assert loss.allclose(torch.tensor(2.044320355487445, device=device))


@pytest.mark.parametrize("device", get_available_devices())
def test_binary_focal_loss_with_logits_forward_partially_correct(device: str) -> None:
    device = torch.device(device)
    criterion = BinaryPoly1LossWithLogits().to(device=device)
    loss = criterion(
        torch.tensor([[1.0, -1.0, 1.0], [-1.0, 1.0, 1.0]], device=device),
        torch.tensor([[1.0, 1.0, 0.0], [0.0, 1.0, 0.0]], device=device),
    )
    assert loss.allclose(torch.tensor(1.3132617450544504, device=device))


@pytest.mark.parametrize("device", get_available_devices())
def test_binary_focal_loss_with_logits_forward_reduction_sum(device: str) -> None:
    device = torch.device(device)
    criterion = BinaryPoly1LossWithLogits(reduction="sum").to(device=device)
    loss = criterion(
        torch.tensor([[1.0, -1.0, 1.0], [-1.0, 1.0, 1.0]], device=device),
        torch.tensor([[1.0, 1.0, 0.0], [0.0, 1.0, 0.0]], device=device),
    )
    assert loss.allclose(torch.tensor(7.879570470326702, device=device))


@pytest.mark.parametrize("device", get_available_devices())
def test_binary_focal_loss_with_logits_forward_reduction_none(device: str) -> None:
    device = torch.device(device)
    criterion = BinaryPoly1LossWithLogits(reduction="none").to(device=device)
    loss = criterion(
        torch.tensor([[1.0, -1.0, 1.0], [-1.0, 1.0, 1.0]], device=device),
        torch.tensor([[1.0, 1.0, 0.0], [0.0, 1.0, 0.0]], device=device),
    )
    assert loss.allclose(
        torch.tensor(
            [
                [0.5822031346214558, 2.044320355487445, 2.044320355487445],
                [0.5822031346214558, 0.5822031346214558, 2.044320355487445],
            ],
            device=device,
        ),
    )


def test_binary_focal_loss_with_logits_forward_incorrect_reduction() -> None:
    with pytest.raises(ValueError, match=r"Incorrect reduction: incorrect reduction."):
        BinaryPoly1LossWithLogits(reduction="incorrect reduction")


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("shape", SHAPES)
def test_binary_focal_loss_with_logits_forward_shape(device: str, shape: tuple[int, ...]) -> None:
    device = torch.device(device)
    criterion = BinaryPoly1LossWithLogits().to(device=device)
    loss = criterion(torch.ones(*shape, device=device), torch.ones(*shape, device=device))
    assert loss.allclose(torch.tensor(0.5822031346214558, device=device))


@pytest.mark.parametrize("device", get_available_devices())
def test_binary_focal_loss_with_logits_forward_alpha_0_5(device: str) -> None:
    device = torch.device(device)
    criterion = BinaryPoly1LossWithLogits(alpha=0.5).to(device=device)
    loss = criterion(
        torch.tensor([[1.0, -1.0, 1.0], [-1.0, 1.0, 1.0]], device=device),
        torch.tensor([[1.0, 1.0, 0.0], [0.0, 1.0, 0.0]], device=device),
    )
    assert loss.allclose(torch.tensor(1.0632617376038698, device=device))


@pytest.mark.parametrize("device", get_available_devices())
def test_binary_focal_loss_with_logits_backward(device: str) -> None:
    device = torch.device(device)
    criterion = BinaryPoly1LossWithLogits().to(device=device)
    loss = criterion(
        torch.tensor([[1.0, -1.0, 1.0], [-1.0, 1.0, -1.0]], device=device, requires_grad=True),
        torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]], device=device),
    )
    loss.backward()
    assert loss.allclose(torch.tensor(0.5822031346214558, device=device))
