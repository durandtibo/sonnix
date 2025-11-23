from __future__ import annotations

import pytest
import torch
from coola.utils.tensor import get_available_devices

from sonnix.functional import binary_focal_loss, binary_focal_loss_with_logits

SHAPES = [(2,), (2, 3), (2, 3, 4)]


#######################################
#     Tests for binary_focal_loss     #
#######################################


@pytest.mark.parametrize("device", get_available_devices())
def test_binary_focal_loss_correct(device: str) -> None:
    device = torch.device(device)
    assert binary_focal_loss(
        torch.tensor([[0.9, 0.1, 0.9], [0.1, 0.9, 0.1]], device=device),
        torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]], device=device),
    ).allclose(torch.tensor(0.0005268025782891315, device=device))


@pytest.mark.parametrize("device", get_available_devices())
def test_binary_focal_loss_incorrect(device: str) -> None:
    device = torch.device(device)
    assert binary_focal_loss(
        torch.tensor([[0.9, 0.1, 0.9], [0.1, 0.9, 0.1]], device=device),
        torch.tensor([[0.0, 1.0, 0.0], [1.0, 0.0, 1.0]], device=device),
    ).allclose(torch.tensor(0.9325469626625885, device=device))


@pytest.mark.parametrize("device", get_available_devices())
def test_binary_focal_loss_partially_correct(device: str) -> None:
    device = torch.device(device)
    assert binary_focal_loss(
        torch.tensor([[0.9, 0.1, 0.9], [0.1, 0.9, 0.9]], device=device),
        torch.tensor([[1.0, 1.0, 0.0], [0.0, 1.0, 0.0]], device=device),
    ).allclose(torch.tensor(0.5442052292941304, device=device))


@pytest.mark.parametrize("device", get_available_devices())
def test_binary_focal_loss_reduction_sum(device: str) -> None:
    device = torch.device(device)
    assert binary_focal_loss(
        torch.tensor([[0.9, 0.1, 0.9], [0.1, 0.9, 0.9]], device=device),
        torch.tensor([[1.0, 1.0, 0.0], [0.0, 1.0, 0.0]], device=device),
        reduction="sum",
    ).allclose(torch.tensor(3.2652313757647824, device=device))


@pytest.mark.parametrize("device", get_available_devices())
def test_binary_focal_loss_reduction_none(device: str) -> None:
    device = torch.device(device)
    assert binary_focal_loss(
        torch.tensor([[0.9, 0.1, 0.9], [0.1, 0.9, 0.9]], device=device),
        torch.tensor([[1.0, 1.0, 0.0], [0.0, 1.0, 0.0]], device=device),
        reduction="none",
    ).allclose(
        torch.tensor(
            [
                [0.00026340128914456557, 0.46627348133129426, 1.3988204439938827],
                [0.0007902038674336973, 0.00026340128914456557, 1.3988204439938827],
            ],
            device=device,
        ),
    )


def test_binary_focal_loss_reduction_incorrect() -> None:
    with pytest.raises(ValueError, match=r"Incorrect reduction: incorrect"):
        binary_focal_loss(torch.ones(2, 3), torch.ones(2, 3), reduction="incorrect")


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("shape", SHAPES)
def test_binary_focal_loss_shape(device: str, shape: tuple[int, ...]) -> None:
    device = torch.device(device)
    assert binary_focal_loss(
        torch.ones(*shape, device=device), torch.ones(*shape, device=device)
    ).allclose(torch.tensor(0.0, device=device))


@pytest.mark.parametrize("device", get_available_devices())
def test_binary_focal_loss_alpha_0_5(device: str) -> None:
    device = torch.device(device)
    assert binary_focal_loss(
        torch.tensor([[0.9, 0.1, 0.9], [0.1, 0.9, 0.9]], device=device),
        torch.tensor([[1.0, 1.0, 0.0], [0.0, 1.0, 0.0]], device=device),
        alpha=0.5,
    ).allclose(torch.tensor(0.4665368826204388, device=device))


@pytest.mark.parametrize("device", get_available_devices())
def test_binary_focal_loss_no_alpha(device: str) -> None:
    device = torch.device(device)
    assert binary_focal_loss(
        torch.tensor([[0.9, 0.1, 0.9], [0.1, 0.9, 0.9]], device=device),
        torch.tensor([[1.0, 1.0, 0.0], [0.0, 1.0, 0.0]], device=device),
        alpha=-1.0,
    ).allclose(torch.tensor(0.9330737652408776, device=device))


@pytest.mark.parametrize("device", get_available_devices())
def test_binary_focal_loss_gamma_1(device: str) -> None:
    device = torch.device(device)
    assert binary_focal_loss(
        torch.tensor([[0.9, 0.1, 0.9], [0.1, 0.9, 0.9]], device=device),
        torch.tensor([[1.0, 1.0, 0.0], [0.0, 1.0, 0.0]], device=device),
        gamma=1.0,
    ).allclose(torch.tensor(0.6066235976538085, device=device))


@pytest.mark.parametrize("device", get_available_devices())
def test_binary_focal_loss_loss_backward(device: str) -> None:
    device = torch.device(device)
    loss = binary_focal_loss(
        torch.tensor([[0.9, 0.1, 0.9], [0.1, 0.9, 0.9]], device=device, requires_grad=True),
        torch.tensor([[1.0, 1.0, 0.0], [0.0, 1.0, 0.0]], device=device),
    )
    loss.backward()
    assert loss.allclose(torch.tensor(0.5442052292941304, device=device))


###################################################
#     Tests for binary_focal_loss_with_logits     #
###################################################


@pytest.mark.parametrize("device", get_available_devices())
def test_binary_focal_loss_with_logits_correct(device: str) -> None:
    device = torch.device(device)
    assert binary_focal_loss_with_logits(
        torch.tensor([[1.0, -1.0, 1.0], [-1.0, 1.0, -1.0]], device=device),
        torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]], device=device),
    ).allclose(torch.tensor(0.01132902921115496, device=device))


@pytest.mark.parametrize("device", get_available_devices())
def test_binary_focal_loss_with_logits_incorrect(device: str) -> None:
    device = torch.device(device)
    assert binary_focal_loss_with_logits(
        torch.tensor([[1.0, -1.0, 1.0], [-1.0, 1.0, -1.0]], device=device),
        torch.tensor([[0.0, 1.0, 0.0], [1.0, 0.0, 1.0]], device=device),
    ).allclose(torch.tensor(0.3509341741420062, device=device))


@pytest.mark.parametrize("device", get_available_devices())
def test_binary_focal_loss_with_logits_partially_correct(device: str) -> None:
    device = torch.device(device)
    assert binary_focal_loss_with_logits(
        torch.tensor([[1.0, -1.0, 1.0], [-1.0, 1.0, 1.0]], device=device),
        torch.tensor([[1.0, 1.0, 0.0], [0.0, 1.0, 0.0]], device=device),
    ).allclose(torch.tensor(0.20943205058574677, device=device))


@pytest.mark.parametrize("device", get_available_devices())
def test_binary_focal_loss_with_logits_reduction_sum(device: str) -> None:
    device = torch.device(device)
    assert binary_focal_loss_with_logits(
        torch.tensor([[1.0, -1.0, 1.0], [-1.0, 1.0, 1.0]], device=device),
        torch.tensor([[1.0, 1.0, 0.0], [0.0, 1.0, 0.0]], device=device),
        reduction="sum",
    ).allclose(torch.tensor(1.2565922737121582, device=device))


@pytest.mark.parametrize("device", get_available_devices())
def test_binary_focal_loss_with_logits_reduction_none(device: str) -> None:
    device = torch.device(device)
    assert binary_focal_loss_with_logits(
        torch.tensor([[1.0, -1.0, 1.0], [-1.0, 1.0, 1.0]], device=device),
        torch.tensor([[1.0, 1.0, 0.0], [0.0, 1.0, 0.0]], device=device),
        reduction="none",
    ).allclose(
        torch.tensor(
            [
                [0.005664513111161829, 0.1754670652368954, 0.5264012830471171],
                [0.016993545311148092, 0.005664513111161829, 0.5264012830471171],
            ],
            device=device,
        ),
    )


def test_binary_focal_loss_with_logits_reduction_incorrect() -> None:
    with pytest.raises(ValueError, match=r"Incorrect reduction: incorrect"):
        binary_focal_loss_with_logits(torch.ones(2, 3), torch.ones(2, 3), reduction="incorrect")


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("shape", SHAPES)
def test_binary_focal_loss_with_logits_shape(device: str, shape: tuple[int, ...]) -> None:
    device = torch.device(device)
    assert binary_focal_loss_with_logits(
        torch.ones(*shape, device=device), torch.ones(*shape, device=device)
    ).allclose(torch.tensor(0.005664513111161829, device=device))


@pytest.mark.parametrize("device", get_available_devices())
def test_binary_focal_loss_with_logits_alpha_0_5(device: str) -> None:
    device = torch.device(device)
    assert binary_focal_loss_with_logits(
        torch.tensor([[1.0, -1.0, 1.0], [-1.0, 1.0, 1.0]], device=device),
        torch.tensor([[1.0, 1.0, 0.0], [0.0, 1.0, 0.0]], device=device),
        alpha=0.5,
    ).allclose(torch.tensor(0.18113157834805724, device=device))


@pytest.mark.parametrize("device", get_available_devices())
def test_binary_focal_loss_with_logits_no_alpha(device: str) -> None:
    device = torch.device(device)
    assert binary_focal_loss_with_logits(
        torch.tensor([[1.0, -1.0, 1.0], [-1.0, 1.0, 1.0]], device=device),
        torch.tensor([[1.0, 1.0, 0.0], [0.0, 1.0, 0.0]], device=device),
        alpha=-1.0,
    ).allclose(torch.tensor(0.3622631566961145, device=device))


@pytest.mark.parametrize("device", get_available_devices())
def test_binary_focal_loss_with_logits_gamma_1(device: str) -> None:
    device = torch.device(device)
    assert binary_focal_loss_with_logits(
        torch.tensor([[1.0, -1.0, 1.0], [-1.0, 1.0, 1.0]], device=device),
        torch.tensor([[1.0, 1.0, 0.0], [0.0, 1.0, 0.0]], device=device),
        gamma=1.0,
    ).allclose(torch.tensor(0.2975726744437273, device=device))


@pytest.mark.parametrize("device", get_available_devices())
def test_binary_focal_loss_with_logits_loss_backward(device: str) -> None:
    device = torch.device(device)
    loss = binary_focal_loss_with_logits(
        torch.tensor([[1.0, -1.0, 1.0], [-1.0, 1.0, 1.0]], device=device, requires_grad=True),
        torch.tensor([[1.0, 1.0, 0.0], [0.0, 1.0, 0.0]], device=device),
    )
    loss.backward()
    assert loss.allclose(torch.tensor(0.20943205058574677, device=device))
