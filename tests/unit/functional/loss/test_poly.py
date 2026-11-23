from __future__ import annotations

import pytest
import torch
from coola.utils.tensor import get_available_devices

from sonnix.functional import binary_poly1_loss, binary_poly1_loss_with_logits
from sonnix.functional.loss.poly import _poly1
from tests.helpers import make_generator

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
    with pytest.raises(ValueError, match=r"reduction must be one of"):
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
    with pytest.raises(ValueError, match=r"reduction must be one of"):
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


def test_binary_poly1_loss_with_logits_matches_probability_version() -> None:
    logits = torch.randn(6, 5, dtype=torch.float64)
    target = torch.randint(0, 2, (6, 5)).double()
    assert binary_poly1_loss_with_logits(logits, target).allclose(
        binary_poly1_loss(logits.sigmoid(), target)
    )


def test_binary_poly1_loss_with_logits_matches_probability_version_soft_target() -> None:
    generator = make_generator(1)
    logits = torch.randn(6, 5, generator=generator, dtype=torch.float64)
    target = torch.rand(6, 5, generator=generator, dtype=torch.float64)
    assert binary_poly1_loss_with_logits(logits, target).allclose(
        binary_poly1_loss(logits.sigmoid(), target)
    )


def test_binary_poly1_loss_with_logits_soft_target_reduction_none() -> None:
    generator = make_generator(2)
    logits = torch.randn(6, 5, generator=generator, dtype=torch.float64)
    target = torch.rand(6, 5, generator=generator, dtype=torch.float64)
    assert binary_poly1_loss_with_logits(logits, target, reduction="none").allclose(
        binary_poly1_loss(logits.sigmoid(), target, reduction="none")
    )


############################
#     Tests for _poly1     #
############################

CE_LOSS = torch.tensor([0.5, 2.0])
P_T = torch.tensor([0.5, 0.25])


def test_poly1_alpha_one() -> None:
    # ce_loss + 1 * (1 - p_t)
    assert _poly1(CE_LOSS, P_T, alpha=1.0, reduction="none").equal(torch.tensor([1.0, 2.75]))


def test_poly1_alpha_zero_is_the_cross_entropy() -> None:
    assert _poly1(CE_LOSS, P_T, alpha=0.0, reduction="none").equal(CE_LOSS)


def test_poly1_alpha_two() -> None:
    assert _poly1(CE_LOSS, P_T, alpha=2.0, reduction="none").equal(torch.tensor([1.5, 3.5]))


def test_poly1_negative_alpha() -> None:
    assert _poly1(CE_LOSS, P_T, alpha=-1.0, reduction="none").equal(torch.tensor([0.0, 1.25]))


@pytest.mark.parametrize(
    ("reduction", "expected"),
    [("sum", 3.75), ("mean", 1.875), ("batchmean", 1.875)],
)
def test_poly1_reduction(reduction: str, expected: float) -> None:
    assert _poly1(CE_LOSS, P_T, alpha=1.0, reduction=reduction).equal(torch.tensor(expected))


def test_poly1_gradient_of_p_t() -> None:
    p_t = torch.tensor([0.5, 0.25], requires_grad=True)
    _poly1(CE_LOSS, p_t, alpha=2.0, reduction="sum").backward()
    assert p_t.grad.equal(torch.tensor([-2.0, -2.0]))


def test_poly1_does_not_modify_the_inputs() -> None:
    ce_loss, p_t = CE_LOSS.clone(), P_T.clone()
    _poly1(ce_loss, p_t, alpha=1.0, reduction="none")
    assert ce_loss.equal(CE_LOSS)
    assert p_t.equal(P_T)


def test_binary_poly1_loss_with_logits_matches_the_probability_variant() -> None:
    generator = make_generator()
    logits = torch.randn(4, 3, generator=generator)
    target = torch.randint(0, 2, (4, 3), generator=generator).float()
    assert binary_poly1_loss_with_logits(logits, target).allclose(
        binary_poly1_loss(logits.sigmoid(), target)
    )
