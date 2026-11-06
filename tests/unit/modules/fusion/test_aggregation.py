from __future__ import annotations

import pytest
import torch
from coola.equality import objects_are_allclose, objects_are_equal
from coola.utils.tensor import get_available_devices

from sonnix.modules import AverageFusion, MultiplicationFusion, SumFusion

SIZES = (1, 2, 3)

###################################
#     Tests for AverageFusion     #
###################################


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("batch_size", SIZES)
@pytest.mark.parametrize("input_size", SIZES)
def test_average_fusion_forward_1_input(device: str, batch_size: int, input_size: int) -> None:
    device = torch.device(device)
    module = AverageFusion().to(device=device)
    out = module(torch.ones(batch_size, input_size, device=device))
    assert objects_are_equal(out, torch.ones(batch_size, input_size, device=device))


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("batch_size", SIZES)
@pytest.mark.parametrize("input_size", SIZES)
def test_average_fusion_forward_2_inputs(device: str, batch_size: int, input_size: int) -> None:
    device = torch.device(device)
    module = AverageFusion().to(device=device)
    out = module(
        torch.ones(batch_size, input_size, device=device),
        torch.full(size=(batch_size, input_size), fill_value=3.0, device=device),
    )
    assert objects_are_allclose(
        out, torch.full(size=(batch_size, input_size), fill_value=2.0, device=device)
    )


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("batch_size", SIZES)
@pytest.mark.parametrize("input_size", SIZES)
def test_average_fusion_forward_3_inputs(device: str, batch_size: int, input_size: int) -> None:
    device = torch.device(device)
    module = AverageFusion().to(device=device)
    out = module(
        torch.ones(batch_size, input_size, device=device),
        torch.full(size=(batch_size, input_size), fill_value=2.0, device=device),
        torch.full(size=(batch_size, input_size), fill_value=3.0, device=device),
    )
    assert objects_are_allclose(
        out, torch.full(size=(batch_size, input_size), fill_value=2.0, device=device)
    )


@pytest.mark.parametrize("device", get_available_devices())
def test_average_fusion_backward(device: str) -> None:
    device = torch.device(device)
    module = AverageFusion().to(device=device)
    out = module(
        torch.ones(2, 4, device=device, requires_grad=True),
        torch.ones(2, 4, device=device, requires_grad=True),
        torch.ones(2, 4, device=device, requires_grad=True),
    )
    out.mean().backward()
    assert objects_are_equal(out, torch.ones(2, 4, device=device))


##########################################
#     Tests for MultiplicationFusion     #
##########################################


def test_multiplication_fusion_forward_0_input() -> None:
    module = MultiplicationFusion()
    with pytest.raises(
        ValueError, match=r"MultiplicationFusion needs at least one tensor as input"
    ):
        module()


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("batch_size", SIZES)
@pytest.mark.parametrize("input_size", SIZES)
def test_multiplication_fusion_forward_1_input(
    device: str, batch_size: int, input_size: int
) -> None:
    device = torch.device(device)
    module = MultiplicationFusion().to(device=device)
    out = module(torch.ones(batch_size, input_size, device=device))
    assert objects_are_equal(out, torch.ones(batch_size, input_size, device=device))


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("batch_size", SIZES)
@pytest.mark.parametrize("input_size", SIZES)
def test_multiplication_fusion_forward_2_inputs(
    device: str, batch_size: int, input_size: int
) -> None:
    device = torch.device(device)
    module = MultiplicationFusion().to(device=device)
    out = module(
        torch.ones(batch_size, input_size, device=device).mul(0.5),
        torch.ones(batch_size, input_size, device=device).mul(0.5),
    )
    assert objects_are_allclose(out, torch.ones(batch_size, input_size, device=device).mul(0.25))


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("batch_size", SIZES)
@pytest.mark.parametrize("input_size", SIZES)
def test_multiplication_fusion_forward_3_inputs(
    device: str, batch_size: int, input_size: int
) -> None:
    device = torch.device(device)
    module = MultiplicationFusion().to(device=device)
    out = module(
        torch.ones(batch_size, input_size, device=device).mul(0.5),
        torch.ones(batch_size, input_size, device=device).mul(0.5),
        torch.ones(batch_size, input_size, device=device).mul(0.5),
    )
    assert objects_are_allclose(out, torch.ones(batch_size, input_size, device=device).mul(0.125))


@pytest.mark.parametrize("device", get_available_devices())
def test_multiplication_fusion_backward(device: str) -> None:
    device = torch.device(device)
    module = MultiplicationFusion().to(device=device)
    out = module(
        torch.ones(2, 4, device=device, requires_grad=True),
        torch.ones(2, 4, device=device, requires_grad=True),
        torch.ones(2, 4, device=device, requires_grad=True),
    )
    out.mean().backward()
    assert out.equal(torch.ones(2, 4, device=device))


###############################
#     Tests for SumFusion     #
###############################


def test_sum_fusion_str() -> None:
    assert str(SumFusion()).startswith("SumFusion(")


def test_sum_fusion_forward_0_input() -> None:
    module = SumFusion()
    with pytest.raises(ValueError, match=r"SumFusion needs at least one tensor as input"):
        module()


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("batch_size", SIZES)
@pytest.mark.parametrize("input_size", SIZES)
def test_sum_fusion_forward_1_input(device: str, batch_size: int, input_size: int) -> None:
    device = torch.device(device)
    module = SumFusion().to(device=device)
    out = module(torch.ones(batch_size, input_size, device=device))
    assert objects_are_equal(out, torch.ones(batch_size, input_size, device=device))


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("batch_size", SIZES)
@pytest.mark.parametrize("input_size", SIZES)
def test_sum_fusion_forward_2_inputs(device: str, batch_size: int, input_size: int) -> None:
    device = torch.device(device)
    module = SumFusion().to(device=device)
    out = module(
        torch.ones(batch_size, input_size, device=device),
        torch.full(size=(batch_size, input_size), fill_value=3.0, device=device),
    )
    assert objects_are_equal(out, torch.ones(batch_size, input_size, device=device).mul(4))


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("batch_size", SIZES)
@pytest.mark.parametrize("input_size", SIZES)
def test_sum_fusion_forward_3_inputs(device: str, batch_size: int, input_size: int) -> None:
    device = torch.device(device)
    module = SumFusion().to(device=device)
    out = module(
        torch.ones(batch_size, input_size, device=device),
        torch.full(size=(batch_size, input_size), fill_value=2.0, device=device),
        torch.full(size=(batch_size, input_size), fill_value=3.0, device=device),
    )
    assert objects_are_allclose(
        out, torch.full(size=(batch_size, input_size), fill_value=6.0, device=device)
    )


@pytest.mark.parametrize("device", get_available_devices())
def test_sum_fusion_backward(device: str) -> None:
    device = torch.device(device)
    module = SumFusion().to(device=device)
    out = module(
        torch.ones(2, 4, device=device, requires_grad=True),
        torch.ones(2, 4, device=device, requires_grad=True),
        torch.ones(2, 4, device=device, requires_grad=True),
    )
    out.mean().backward()
    assert objects_are_equal(out, torch.ones(2, 4, device=device).mul(3))


def test_sum_fusion_repr() -> None:
    assert repr(SumFusion()) == "SumFusion()"


def test_sum_fusion_no_normalized_argument() -> None:
    with pytest.raises(TypeError, match=r"normalized"):
        SumFusion(normalized=True)


def test_average_fusion_repr() -> None:
    assert repr(AverageFusion()) == "AverageFusion()"


def test_average_fusion_is_not_sum_fusion() -> None:
    assert not isinstance(AverageFusion(), SumFusion)


def test_average_fusion_forward_values() -> None:
    assert objects_are_equal(
        AverageFusion()(
            torch.tensor([1.0, 2.0]), torch.tensor([3.0, 6.0]), torch.tensor([5.0, 1.0])
        ),
        torch.tensor([3.0, 3.0]),
    )
