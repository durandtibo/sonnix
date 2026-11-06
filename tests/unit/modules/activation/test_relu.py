from __future__ import annotations

import pytest
import torch
from coola.equality import objects_are_equal
from coola.utils.tensor import get_available_devices

from sonnix.modules import ReLUn, SquaredReLU

SHAPES = [(2,), (2, 3), (2, 3, 4), (2, 3, 4, 5)]

###########################
#     Tests for ReLUn     #
###########################


def test_relun_str() -> None:
    assert str(ReLUn()).startswith("ReLUn(")


@pytest.mark.parametrize("device", get_available_devices())
def test_relun_forward(device: str) -> None:
    device = torch.device(device)
    module = ReLUn().to(device=device)
    assert objects_are_equal(
        module(torch.arange(-1, 4, dtype=torch.float, device=device)),
        torch.tensor([0.0, 0.0, 1.0, 1.0, 1.0], dtype=torch.float, device=device),
    )


@pytest.mark.parametrize("device", get_available_devices())
def test_relun_forward_max_value_2(device: str) -> None:
    device = torch.device(device)
    module = ReLUn(max_val=2).to(device=device)
    assert objects_are_equal(
        module(torch.arange(-1, 4, dtype=torch.float, device=device)),
        torch.tensor([0.0, 0.0, 1.0, 2.0, 2.0], dtype=torch.float, device=device),
    )


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("shape", SHAPES)
def test_relun_forward_shape_value(device: str, shape: tuple[int, ...]) -> None:
    device = torch.device(device)
    module = ReLUn().to(device=device)
    out = module(torch.randn(*shape, device=device))
    assert out.shape == shape


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("shape", SHAPES)
def test_relun_forward_shape_dtype(device: str, shape: tuple[int, ...]) -> None:
    device = torch.device(device)
    module = ReLUn().to(device=device)
    out = module(torch.randn(*shape, device=device))
    assert out.dtype == torch.float


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("shape", SHAPES)
def test_relun_forward_shape_device(device: str, shape: tuple[int, ...]) -> None:
    device = torch.device(device)
    module = ReLUn().to(device=device)
    out = module(torch.randn(*shape, device=device))
    assert out.device == device


#################################
#     Tests for SquaredReLU     #
#################################


@pytest.mark.parametrize("device", get_available_devices())
def test_squared_relu_forward(device: str) -> None:
    device = torch.device(device)
    module = SquaredReLU().to(device=device)
    assert objects_are_equal(
        module(torch.arange(-1, 4, dtype=torch.float, device=device)),
        torch.tensor([0.0, 0.0, 1.0, 4.0, 9.0], dtype=torch.float, device=device),
    )


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("shape", SHAPES)
def test_squared_relu_forward_shape_value(device: str, shape: tuple[int, ...]) -> None:
    device = torch.device(device)
    module = SquaredReLU().to(device=device)
    out = module(torch.randn(*shape, device=device))
    assert out.shape == shape


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("shape", SHAPES)
def test_squared_relu_forward_shape_dtype(device: str, shape: tuple[int, ...]) -> None:
    device = torch.device(device)
    module = SquaredReLU().to(device=device)
    out = module(torch.randn(*shape, device=device))
    assert out.dtype == torch.float


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("shape", SHAPES)
def test_squared_relu_forward_shape_device(device: str, shape: tuple[int, ...]) -> None:
    device = torch.device(device)
    module = SquaredReLU().to(device=device)
    out = module(torch.randn(*shape, device=device))
    assert out.device == device


def test_relun_invalid_max_val() -> None:
    with pytest.raises(ValueError, match=r"max_val must be greater than or equal to 0"):
        ReLUn(max_val=-1.0)
