from __future__ import annotations

import pytest
import torch
from coola.utils.tensor import get_available_devices

from sonnix.modules import BinaryFocalLoss, BinaryFocalLossWithLogits
from sonnix.modules.loss import BaseBinaryFocalLoss

SIZES = (1, 2)
TOLERANCE = 1e-6

SHAPES = [(2,), (2, 3), (2, 3, 4)]


#####################################
#     Tests for BinaryFocalLoss     #
#####################################


def test_binary_focal_loss_str() -> None:
    assert str(BinaryFocalLoss()).startswith("BinaryFocalLoss(")


@pytest.mark.parametrize("alpha", [None, 0.0, 0.5, 1])
def test_binary_focal_loss_alpha(alpha: float) -> None:
    assert BinaryFocalLoss(alpha=alpha)._alpha == alpha


@pytest.mark.parametrize("gamma", [0, 0.5, 1])
def test_binary_focal_loss_valid_gamma(gamma: float) -> None:
    assert BinaryFocalLoss(gamma=gamma)._gamma == gamma


@pytest.mark.parametrize("gamma", [-1, -0.5])
def test_binary_focal_loss_invalid_gamma(gamma: float) -> None:
    with pytest.raises(ValueError, match=r"gamma must be greater than or equal to 0"):
        BinaryFocalLoss(gamma=gamma)


@pytest.mark.parametrize("device", get_available_devices())
def test_binary_focal_loss_forward_correct(device: str) -> None:
    device = torch.device(device)
    criterion = BinaryFocalLoss().to(device=device)
    loss = criterion(
        torch.tensor([[0.9, 0.1, 0.9], [0.1, 0.9, 0.1]], device=device),
        torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]], device=device),
    )
    assert loss.allclose(torch.tensor(0.0005268025782891315, device=device))


@pytest.mark.parametrize("device", get_available_devices())
def test_binary_focal_loss_forward_incorrect(device: str) -> None:
    device = torch.device(device)
    criterion = BinaryFocalLoss().to(device=device)
    loss = criterion(
        torch.tensor([[0.9, 0.1, 0.9], [0.1, 0.9, 0.1]], device=device),
        torch.tensor([[0.0, 1.0, 0.0], [1.0, 0.0, 1.0]], device=device),
    )
    assert loss.allclose(torch.tensor(0.9325469626625885, device=device))


@pytest.mark.parametrize("device", get_available_devices())
def test_binary_focal_loss_forward_partially_correct(device: str) -> None:
    device = torch.device(device)
    criterion = BinaryFocalLoss().to(device=device)
    loss = criterion(
        torch.tensor([[0.9, 0.1, 0.9], [0.1, 0.9, 0.9]], device=device),
        torch.tensor([[1.0, 1.0, 0.0], [0.0, 1.0, 0.0]], device=device),
    )
    assert loss.allclose(torch.tensor(0.5442052292941304, device=device))


@pytest.mark.parametrize("device", get_available_devices())
def test_binary_focal_loss_forward_reduction_sum(device: str) -> None:
    device = torch.device(device)
    criterion = BinaryFocalLoss(reduction="sum").to(device=device)
    loss = criterion(
        torch.tensor([[0.9, 0.1, 0.9], [0.1, 0.9, 0.9]], device=device),
        torch.tensor([[1.0, 1.0, 0.0], [0.0, 1.0, 0.0]], device=device),
    )
    assert loss.allclose(torch.tensor(3.2652313757647824, device=device))


@pytest.mark.parametrize("device", get_available_devices())
def test_binary_focal_loss_forward_reduction_none(device: str) -> None:
    device = torch.device(device)
    criterion = BinaryFocalLoss(reduction="none").to(device=device)
    loss = criterion(
        torch.tensor([[0.9, 0.1, 0.9], [0.1, 0.9, 0.9]], device=device),
        torch.tensor([[1.0, 1.0, 0.0], [0.0, 1.0, 0.0]], device=device),
    )
    assert loss.allclose(
        torch.tensor(
            [
                [0.00026340128914456557, 0.46627348133129426, 1.3988204439938827],
                [0.0007902038674336973, 0.00026340128914456557, 1.3988204439938827],
            ],
            device=device,
        ),
    )


def test_binary_focal_loss_forward_incorrect_reduction() -> None:
    with pytest.raises(ValueError, match=r"reduction must be one of"):
        BinaryFocalLoss(reduction="incorrect reduction")


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("shape", SHAPES)
def test_binary_focal_loss_forward_shape(device: str, shape: tuple[int, ...]) -> None:
    device = torch.device(device)
    criterion = BinaryFocalLoss().to(device=device)
    loss = criterion(torch.ones(*shape, device=device), torch.ones(*shape, device=device))
    assert loss.allclose(torch.tensor(0.0, device=device))


@pytest.mark.parametrize("device", get_available_devices())
def test_binary_focal_loss_forward_alpha_0_5(device: str) -> None:
    device = torch.device(device)
    criterion = BinaryFocalLoss(alpha=0.5).to(device=device)
    loss = criterion(
        torch.tensor([[0.9, 0.1, 0.9], [0.1, 0.9, 0.9]], device=device),
        torch.tensor([[1.0, 1.0, 0.0], [0.0, 1.0, 0.0]], device=device),
    )
    assert loss.allclose(torch.tensor(0.4665368826204388, device=device))


@pytest.mark.parametrize("device", get_available_devices())
def test_binary_focal_loss_forward_no_alpha(device: str) -> None:
    device = torch.device(device)
    criterion = BinaryFocalLoss(alpha=None).to(device=device)
    loss = criterion(
        torch.tensor([[0.9, 0.1, 0.9], [0.1, 0.9, 0.9]], device=device),
        torch.tensor([[1.0, 1.0, 0.0], [0.0, 1.0, 0.0]], device=device),
    )
    assert loss.allclose(torch.tensor(0.9330737652408776, device=device))


@pytest.mark.parametrize("device", get_available_devices())
def test_binary_focal_loss_forward_gamma_1(device: str) -> None:
    device = torch.device(device)
    criterion = BinaryFocalLoss(gamma=1.0).to(device=device)
    loss = criterion(
        torch.tensor([[0.9, 0.1, 0.9], [0.1, 0.9, 0.9]], device=device),
        torch.tensor([[1.0, 1.0, 0.0], [0.0, 1.0, 0.0]], device=device),
    )
    assert loss.allclose(torch.tensor(0.6066235976538085, device=device))


@pytest.mark.parametrize("device", get_available_devices())
def test_binary_focal_loss_backward(device: str) -> None:
    device = torch.device(device)
    criterion = BinaryFocalLoss().to(device=device)
    loss = criterion(
        torch.tensor([[0.9, 0.1, 0.9], [0.1, 0.9, 0.9]], device=device, requires_grad=True),
        torch.tensor([[1.0, 1.0, 0.0], [0.0, 1.0, 0.0]], device=device),
    )
    loss.backward()
    assert loss.allclose(torch.tensor(0.5442052292941304, device=device))


###############################################
#     Tests for BinaryFocalLossWithLogits     #
###############################################


def test_binary_focal_loss_with_logits_str() -> None:
    assert str(BinaryFocalLossWithLogits()).startswith("BinaryFocalLossWithLogits(")


@pytest.mark.parametrize("alpha", [None, 0.0, 0.5, 1])
def test_binary_focal_loss_with_logits_alpha(alpha: float) -> None:
    assert BinaryFocalLossWithLogits(alpha=alpha)._alpha == alpha


@pytest.mark.parametrize("gamma", [0, 0.5, 1])
def test_binary_focal_loss_with_logits_valid_gamma(gamma: float) -> None:
    assert BinaryFocalLossWithLogits(gamma=gamma)._gamma == gamma


@pytest.mark.parametrize("gamma", [-1, -0.5])
def test_binary_focal_loss_with_logits_invalid_gamma(gamma: float) -> None:
    with pytest.raises(ValueError, match=r"gamma must be greater than or equal to 0"):
        BinaryFocalLossWithLogits(gamma=gamma)


@pytest.mark.parametrize("device", get_available_devices())
def test_binary_focal_loss_with_logits_forward_correct(device: str) -> None:
    device = torch.device(device)
    criterion = BinaryFocalLossWithLogits().to(device=device)
    loss = criterion(
        torch.tensor([[1.0, -1.0, 1.0], [-1.0, 1.0, -1.0]], device=device),
        torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]], device=device),
    )
    assert loss.allclose(torch.tensor(0.01132902921115496, device=device))


@pytest.mark.parametrize("device", get_available_devices())
def test_binary_focal_loss_with_logits_forward_incorrect(device: str) -> None:
    device = torch.device(device)
    criterion = BinaryFocalLossWithLogits().to(device=device)
    loss = criterion(
        torch.tensor([[1.0, -1.0, 1.0], [-1.0, 1.0, -1.0]], device=device),
        torch.tensor([[0.0, 1.0, 0.0], [1.0, 0.0, 1.0]], device=device),
    )
    assert loss.allclose(torch.tensor(0.3509341741420062, device=device))


@pytest.mark.parametrize("device", get_available_devices())
def test_binary_focal_loss_with_logits_forward_partially_correct(device: str) -> None:
    device = torch.device(device)
    criterion = BinaryFocalLossWithLogits().to(device=device)
    loss = criterion(
        torch.tensor([[1.0, -1.0, 1.0], [-1.0, 1.0, 1.0]], device=device),
        torch.tensor([[1.0, 1.0, 0.0], [0.0, 1.0, 0.0]], device=device),
    )
    assert loss.allclose(torch.tensor(0.20943205058574677, device=device))


@pytest.mark.parametrize("device", get_available_devices())
def test_binary_focal_loss_with_logits_forward_reduction_sum(device: str) -> None:
    device = torch.device(device)
    criterion = BinaryFocalLossWithLogits(reduction="sum").to(device=device)
    loss = criterion(
        torch.tensor([[1.0, -1.0, 1.0], [-1.0, 1.0, 1.0]], device=device),
        torch.tensor([[1.0, 1.0, 0.0], [0.0, 1.0, 0.0]], device=device),
    )
    assert loss.allclose(torch.tensor(1.2565922737121582, device=device))


@pytest.mark.parametrize("device", get_available_devices())
def test_binary_focal_loss_with_logits_forward_reduction_none(device: str) -> None:
    device = torch.device(device)
    criterion = BinaryFocalLossWithLogits(reduction="none").to(device=device)
    loss = criterion(
        torch.tensor([[1.0, -1.0, 1.0], [-1.0, 1.0, 1.0]], device=device),
        torch.tensor([[1.0, 1.0, 0.0], [0.0, 1.0, 0.0]], device=device),
    )
    assert loss.allclose(
        torch.tensor(
            [
                [0.005664513111161829, 0.1754670652368954, 0.5264012830471171],
                [0.016993545311148092, 0.005664513111161829, 0.5264012830471171],
            ],
            device=device,
        ),
    )


def test_binary_focal_loss_with_logits_forward_incorrect_reduction() -> None:
    with pytest.raises(ValueError, match=r"reduction must be one of"):
        BinaryFocalLossWithLogits(reduction="incorrect reduction")


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("shape", SHAPES)
def test_binary_focal_loss_with_logits_forward_shape(device: str, shape: tuple[int, ...]) -> None:
    device = torch.device(device)
    criterion = BinaryFocalLossWithLogits().to(device=device)
    loss = criterion(torch.ones(*shape, device=device), torch.ones(*shape, device=device))
    assert loss.allclose(torch.tensor(0.005664513111161829, device=device))


@pytest.mark.parametrize("device", get_available_devices())
def test_binary_focal_loss_with_logits_forward_alpha_0_5(device: str) -> None:
    device = torch.device(device)
    criterion = BinaryFocalLossWithLogits(alpha=0.5).to(device=device)
    loss = criterion(
        torch.tensor([[1.0, -1.0, 1.0], [-1.0, 1.0, 1.0]], device=device),
        torch.tensor([[1.0, 1.0, 0.0], [0.0, 1.0, 0.0]], device=device),
    )
    assert loss.allclose(torch.tensor(0.18113157834805724, device=device))


@pytest.mark.parametrize("device", get_available_devices())
def test_binary_focal_loss_with_logits_forward_no_alpha(device: str) -> None:
    device = torch.device(device)
    criterion = BinaryFocalLossWithLogits(alpha=None).to(device=device)
    loss = criterion(
        torch.tensor([[1.0, -1.0, 1.0], [-1.0, 1.0, 1.0]], device=device),
        torch.tensor([[1.0, 1.0, 0.0], [0.0, 1.0, 0.0]], device=device),
    )
    assert loss.allclose(torch.tensor(0.3622631566961145, device=device))


@pytest.mark.parametrize("device", get_available_devices())
def test_binary_focal_loss_with_logits_forward_gamma_1(device: str) -> None:
    device = torch.device(device)
    criterion = BinaryFocalLossWithLogits(gamma=1.0).to(device=device)
    loss = criterion(
        torch.tensor([[1.0, -1.0, 1.0], [-1.0, 1.0, 1.0]], device=device),
        torch.tensor([[1.0, 1.0, 0.0], [0.0, 1.0, 0.0]], device=device),
    )
    assert loss.allclose(torch.tensor(0.2975726744437273, device=device))


@pytest.mark.parametrize("device", get_available_devices())
def test_binary_focal_loss_with_logits_backward(device: str) -> None:
    device = torch.device(device)
    criterion = BinaryFocalLossWithLogits().to(device=device)
    loss = criterion(
        torch.tensor([[1.0, -1.0, 1.0], [-1.0, 1.0, 1.0]], device=device, requires_grad=True),
        torch.tensor([[1.0, 1.0, 0.0], [0.0, 1.0, 0.0]], device=device),
    )
    loss.backward()
    assert loss.allclose(torch.tensor(0.20943205058574677, device=device))


@pytest.mark.parametrize("loss", [BinaryFocalLoss, BinaryFocalLossWithLogits])
@pytest.mark.parametrize("alpha", [5.0, -1.0])
def test_binary_focal_loss_module_incorrect_alpha(loss: type, alpha: float) -> None:
    with pytest.raises(ValueError, match=r"alpha must be in the range \[0, 1\]"):
        loss(alpha=alpha)


#########################################
#     Tests for BaseBinaryFocalLoss     #
#########################################


def test_basebinaryfocalloss_repr() -> None:
    assert (
        repr(BaseBinaryFocalLoss()) == "BaseBinaryFocalLoss(alpha=0.25, gamma=2.0, reduction=mean)"
    )


@pytest.mark.parametrize("loss", [BinaryFocalLoss, BinaryFocalLossWithLogits])
def test_basebinaryfocalloss_subclass(loss: type) -> None:
    assert issubclass(loss, BaseBinaryFocalLoss)


def test_basebinaryfocalloss_pair_are_independent() -> None:
    assert not issubclass(BinaryFocalLoss, BinaryFocalLossWithLogits)
