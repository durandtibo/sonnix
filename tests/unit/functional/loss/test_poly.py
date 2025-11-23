from __future__ import annotations

import pytest
import torch
from coola.utils.tensor import get_available_devices

from sonnix.functional import binary_poly1_loss, binary_poly1_loss_with_logits

SHAPES = [(2,), (2, 3), (2, 3, 4)]


#######################################
#     Tests for binary_poly1_loss     #
#######################################


@pytest.mark.parametrize("device", get_available_devices())
def test_binary_poly1_loss_correct(device: str) -> None:
    device = torch.device(device)
    assert binary_poly1_loss(
        torch.tensor([[0.9, 0.1, 0.9], [0.1, 0.9, 0.1]], device=device),
        torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]], device=device),
    ).allclose(torch.tensor(0.2053605156578263, device=device))


@pytest.mark.parametrize("device", get_available_devices())
def test_binary_poly1_loss_incorrect(device: str) -> None:
    device = torch.device(device)
    assert binary_poly1_loss(
        torch.tensor([[0.9, 0.1, 0.9], [0.1, 0.9, 0.1]], device=device),
        torch.tensor([[0.0, 1.0, 0.0], [1.0, 0.0, 1.0]], device=device),
    ).allclose(torch.tensor(3.2025850929940454, device=device))


@pytest.mark.parametrize("device", get_available_devices())
def test_binary_poly1_loss_partially_correct(device: str) -> None:
    device = torch.device(device)
    assert binary_poly1_loss(
        torch.tensor([[0.9, 0.1, 0.9], [0.1, 0.9, 0.9]], device=device),
        torch.tensor([[1.0, 1.0, 0.0], [0.0, 1.0, 0.0]], device=device),
    ).allclose(torch.tensor(1.703972804325936, device=device))


@pytest.mark.parametrize("device", get_available_devices())
def test_binary_poly1_loss_reduction_sum(device: str) -> None:
    device = torch.device(device)
    assert binary_poly1_loss(
        torch.tensor([[0.9, 0.1, 0.9], [0.1, 0.9, 0.9]], device=device),
        torch.tensor([[1.0, 1.0, 0.0], [0.0, 1.0, 0.0]], device=device),
        reduction="sum",
    ).allclose(torch.tensor(10.223836825955615, device=device))


@pytest.mark.parametrize("device", get_available_devices())
def test_binary_poly1_loss_reduction_none(device: str) -> None:
    device = torch.device(device)
    assert binary_poly1_loss(
        torch.tensor([[0.9, 0.1, 0.9], [0.1, 0.9, 0.9]], device=device),
        torch.tensor([[1.0, 1.0, 0.0], [0.0, 1.0, 0.0]], device=device),
        reduction="none",
    ).allclose(
        torch.tensor(
            [
                [0.2053605156578263, 3.2025850929940454, 3.2025850929940454],
                [0.2053605156578263, 0.2053605156578263, 3.2025850929940454],
            ],
            device=device,
        ),
    )


def test_binary_poly1_loss_reduction_incorrect() -> None:
    with pytest.raises(ValueError, match=r"Incorrect reduction: incorrect"):
        binary_poly1_loss(torch.ones(2, 3), torch.ones(2, 3), reduction="incorrect")


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("shape", SHAPES)
def test_binary_poly1_loss_shape(device: str, shape: tuple[int, ...]) -> None:
    device = torch.device(device)
    assert binary_poly1_loss(
        torch.ones(*shape, device=device), torch.ones(*shape, device=device)
    ).allclose(torch.tensor(0.0, device=device))


@pytest.mark.parametrize("device", get_available_devices())
def test_binary_poly1_loss_alpha_0_5(device: str) -> None:
    device = torch.device(device)
    assert binary_poly1_loss(
        torch.tensor([[0.9, 0.1, 0.9], [0.1, 0.9, 0.9]], device=device),
        torch.tensor([[1.0, 1.0, 0.0], [0.0, 1.0, 0.0]], device=device),
        alpha=0.5,
    ).allclose(torch.tensor(1.453972804325936, device=device))


@pytest.mark.parametrize("device", get_available_devices())
def test_binary_poly1_loss_backward(device: str) -> None:
    device = torch.device(device)
    loss = binary_poly1_loss(
        torch.tensor([[0.9, 0.1, 0.9], [0.1, 0.9, 0.1]], device=device, requires_grad=True),
        torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]], device=device),
    )
    loss.backward()
    assert loss.allclose(torch.tensor(0.2053605156578263, device=device))


########################################################
#     Tests for binary_poly1_loss_with_logits_loss     #
########################################################


@pytest.mark.parametrize("device", get_available_devices())
def test_binary_poly1_loss_with_logits_loss_correct(device: str) -> None:
    device = torch.device(device)
    assert binary_poly1_loss_with_logits(
        torch.tensor([[1.0, -1.0, 1.0], [-1.0, 1.0, -1.0]], device=device),
        torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]], device=device),
    ).allclose(torch.tensor(0.5822031346214558, device=device))


@pytest.mark.parametrize("device", get_available_devices())
def test_binary_poly1_loss_with_logits_loss_incorrect(device: str) -> None:
    device = torch.device(device)
    assert binary_poly1_loss_with_logits(
        torch.tensor([[1.0, -1.0, 1.0], [-1.0, 1.0, -1.0]], device=device),
        torch.tensor([[0.0, 1.0, 0.0], [1.0, 0.0, 1.0]], device=device),
    ).allclose(torch.tensor(2.044320355487445, device=device))


@pytest.mark.parametrize("device", get_available_devices())
def test_binary_poly1_loss_with_logits_loss_partially_correct(device: str) -> None:
    device = torch.device(device)
    assert binary_poly1_loss_with_logits(
        torch.tensor([[1.0, -1.0, 1.0], [-1.0, 1.0, 1.0]], device=device),
        torch.tensor([[1.0, 1.0, 0.0], [0.0, 1.0, 0.0]], device=device),
    ).allclose(torch.tensor(1.3132617450544504, device=device))


@pytest.mark.parametrize("device", get_available_devices())
def test_binary_poly1_loss_with_logits_loss_reduction_sum(device: str) -> None:
    device = torch.device(device)
    assert binary_poly1_loss_with_logits(
        torch.tensor([[1.0, -1.0, 1.0], [-1.0, 1.0, 1.0]], device=device),
        torch.tensor([[1.0, 1.0, 0.0], [0.0, 1.0, 0.0]], device=device),
        reduction="sum",
    ).allclose(torch.tensor(7.879570470326702, device=device))


@pytest.mark.parametrize("device", get_available_devices())
def test_binary_poly1_loss_with_logits_loss_reduction_none(device: str) -> None:
    device = torch.device(device)
    assert binary_poly1_loss_with_logits(
        torch.tensor([[1.0, -1.0, 1.0], [-1.0, 1.0, 1.0]], device=device),
        torch.tensor([[1.0, 1.0, 0.0], [0.0, 1.0, 0.0]], device=device),
        reduction="none",
    ).allclose(
        torch.tensor(
            [
                [0.5822031346214558, 2.044320355487445, 2.044320355487445],
                [0.5822031346214558, 0.5822031346214558, 2.044320355487445],
            ],
            device=device,
        ),
    )


def test_binary_poly1_loss_with_logits_loss_reduction_incorrect() -> None:
    with pytest.raises(ValueError, match=r"Incorrect reduction: incorrect"):
        binary_poly1_loss_with_logits(torch.ones(2, 3), torch.ones(2, 3), reduction="incorrect")


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("shape", SHAPES)
def test_binary_poly1_loss_with_logits_loss_shape(device: str, shape: tuple[int, ...]) -> None:
    device = torch.device(device)
    assert binary_poly1_loss_with_logits(
        torch.ones(*shape, device=device), torch.ones(*shape, device=device)
    ).allclose(torch.tensor(0.5822031346214558, device=device))


@pytest.mark.parametrize("device", get_available_devices())
def test_binary_poly1_loss_with_logits_loss_alpha_0_5(device: str) -> None:
    device = torch.device(device)
    assert binary_poly1_loss_with_logits(
        torch.tensor([[1.0, -1.0, 1.0], [-1.0, 1.0, 1.0]], device=device),
        torch.tensor([[1.0, 1.0, 0.0], [0.0, 1.0, 0.0]], device=device),
        alpha=0.5,
    ).allclose(torch.tensor(1.0632617376038698, device=device))


@pytest.mark.parametrize("device", get_available_devices())
def test_binary_poly1_loss_with_logits_loss_backward(device: str) -> None:
    device = torch.device(device)
    loss = binary_poly1_loss_with_logits(
        torch.tensor([[1.0, -1.0, 1.0], [-1.0, 1.0, -1.0]], device=device, requires_grad=True),
        torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]], device=device),
    )
    loss.backward()
    assert loss.allclose(torch.tensor(0.5822031346214558, device=device))
