from __future__ import annotations

import pytest
import torch
from coola.equality import objects_are_equal
from coola.utils.tensor import get_available_devices

from sonnix.modules import ExU

SIZES = (1, 2, 3)


#########################
#     Tests for ExU     #
#########################


def test_exu_str() -> None:
    assert str(ExU(in_features=4, out_features=6)).startswith("ExU(")


@pytest.mark.parametrize("in_features", SIZES)
@pytest.mark.parametrize("out_features", SIZES)
def test_exu_in_out_features_in_features(in_features: int, out_features: int) -> None:
    module = ExU(in_features=in_features, out_features=out_features)
    assert module.in_features == in_features


@pytest.mark.parametrize("in_features", SIZES)
@pytest.mark.parametrize("out_features", SIZES)
def test_exu_in_out_features_value(in_features: int, out_features: int) -> None:
    module = ExU(in_features=in_features, out_features=out_features)
    assert module.out_features == out_features


@pytest.mark.parametrize("in_features", SIZES)
@pytest.mark.parametrize("out_features", SIZES)
def test_exu_in_out_features_shape(in_features: int, out_features: int) -> None:
    module = ExU(in_features=in_features, out_features=out_features)
    assert module.weight.shape == (out_features, in_features)
    assert module.bias.shape == (in_features,)


@pytest.mark.parametrize("device", get_available_devices())
def test_exu_parameters_device_with_bias_shape_dtype_device(device: str) -> None:
    device = torch.device(device)
    module = ExU(in_features=4, out_features=6, device=device).to(device=device)
    assert module.weight.shape == (6, 4)
    assert module.weight.dtype == torch.float
    assert module.weight.device == device


@pytest.mark.parametrize("device", get_available_devices())
def test_exu_parameters_device_with_bias_data(device: str) -> None:
    device = torch.device(device)
    module = ExU(in_features=4, out_features=6, device=device).to(device=device)

    assert objects_are_equal(module.bias.data, torch.zeros(4, device=device))


@pytest.mark.parametrize("device", get_available_devices())
def test_exu_parameters_device_without_bias_shape_dtype_device(device: str) -> None:
    device = torch.device(device)
    module = ExU(in_features=4, out_features=6, bias=False, device=device).to(device=device)
    assert module.weight.shape == (6, 4)
    assert module.weight.dtype == torch.float
    assert module.weight.device == device


@pytest.mark.parametrize("device", get_available_devices())
def test_exu_parameters_device_without_bias_value(device: str) -> None:
    device = torch.device(device)
    module = ExU(in_features=4, out_features=6, bias=False, device=device).to(device=device)

    assert module.bias is None


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("bias", [True, False])
def test_exu_forward_1d(device: str, bias: bool) -> None:
    device = torch.device(device)
    module = ExU(in_features=4, out_features=6, bias=bias).to(device=device)
    output = module(torch.randn(4, device=device))
    assert output.shape == (6,)
    assert output.dtype == torch.float
    assert output.device == device


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("batch_size", SIZES)
@pytest.mark.parametrize("bias", [True, False])
def test_exu_forward_2d(device: str, batch_size: int, bias: bool) -> None:
    device = torch.device(device)
    module = ExU(in_features=4, out_features=6, bias=bias).to(device=device)
    output = module(torch.randn(batch_size, 4, device=device))
    assert output.shape == (batch_size, 6)
    assert output.dtype == torch.float
    assert output.device == device


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("batch_size", SIZES)
@pytest.mark.parametrize("bias", [True, False])
def test_exu_forward_3d(device: str, batch_size: int, bias: bool) -> None:
    device = torch.device(device)
    module = ExU(in_features=4, out_features=6, bias=bias).to(device=device)
    output = module(torch.randn(batch_size, 3, 4, device=device))
    assert output.shape == (batch_size, 3, 6)
    assert output.dtype == torch.float
    assert output.device == device


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("batch_size", SIZES)
@pytest.mark.parametrize("bias", [True, False])
def test_exu_forward_4d(device: str, batch_size: int, bias: bool) -> None:
    device = torch.device(device)
    module = ExU(in_features=4, out_features=6, bias=bias).to(device=device)
    output = module(torch.randn(batch_size, 3, 4, 4, device=device))
    assert output.shape == (batch_size, 3, 4, 6)
    assert output.dtype == torch.float
    assert output.device == device


def test_exu_bias_is_subtracted() -> None:
    module = ExU(in_features=2, out_features=1)
    with torch.no_grad():
        module.weight.zero_()  # exp(0) = 1
        module.bias.copy_(torch.tensor([1.0, 2.0]))
    # y = exp(W) (x - b) = (3 - 1) + (5 - 2) = 5
    assert module(torch.tensor([[3.0, 5.0]])).allclose(torch.tensor([[5.0]]))


#####################################
#     Tests for the size checks     #
#####################################


@pytest.mark.parametrize("value", [0, -1])
@pytest.mark.parametrize("argument", ["in_features", "out_features"])
def test_exu_incorrect_size(argument: str, value: int) -> None:
    kwargs = {"in_features": 4, "out_features": 5, argument: value}
    with pytest.raises(ValueError, match=rf"{argument} must be greater than or equal to 1"):
        ExU(**kwargs)


def test_exu_incorrect_size_type() -> None:
    with pytest.raises(TypeError, match=r"in_features must be an int"):
        ExU(in_features=2.5, out_features=5)
