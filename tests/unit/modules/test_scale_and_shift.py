from __future__ import annotations

import pytest
import torch
from coola.equality import objects_are_allclose
from coola.utils.tensor import get_available_devices

from sonnix.modules import ScaleAndShift

SIZES = (1, 2, 3)


###################################
#     Tests for ScaleAndShift     #
###################################


def test_scale_and_shift_str() -> None:
    assert (
        str(ScaleAndShift(normalized_shape=5)) == "ScaleAndShift(normalized_shape=(5,), bias=True)"
    )


@pytest.mark.parametrize(("shape_input", "expected"), [(5, (5,)), ([3], (3,)), ((4, 2), (4, 2))])
def test_scale_and_shift_normalized_shape_normalized_shape(
    shape_input: int | list[int] | tuple[int, ...], expected: tuple[int, ...]
) -> None:
    module = ScaleAndShift(shape_input)
    assert module.normalized_shape == expected


@pytest.mark.parametrize(("shape_input", "expected"), [(5, (5,)), ([3], (3,)), ((4, 2), (4, 2))])
def test_scale_and_shift_normalized_shape_weight_shape(
    shape_input: int | list[int] | tuple[int, ...], expected: tuple[int, ...]
) -> None:
    module = ScaleAndShift(shape_input)
    assert module.weight.shape == expected


@pytest.mark.parametrize(("shape_input", "expected"), [(5, (5,)), ([3], (3,)), ((4, 2), (4, 2))])
def test_scale_and_shift_normalized_shape_bias_shape(
    shape_input: int | list[int] | tuple[int, ...], expected: tuple[int, ...]
) -> None:
    module = ScaleAndShift(shape_input)
    assert module.bias.shape == expected


def test_scale_and_shift_normalized_shape_int_weight() -> None:
    module = ScaleAndShift(normalized_shape=5)
    assert module.weight.equal(torch.ones(5))


def test_scale_and_shift_normalized_shape_int_bias() -> None:
    module = ScaleAndShift(normalized_shape=5)
    assert module.bias.equal(torch.zeros(5))


def test_scale_and_shift_normalized_shape_list_weight() -> None:
    module = ScaleAndShift(normalized_shape=[5])
    assert module.weight.equal(torch.ones(5))


def test_scale_and_shift_normalized_shape_list_bias() -> None:
    module = ScaleAndShift(normalized_shape=[5])
    assert module.bias.equal(torch.zeros(5))


def test_scale_and_shift_normalized_shape_tuple_1_weight() -> None:
    module = ScaleAndShift(normalized_shape=(5,))
    assert module.weight.equal(torch.ones(5))


def test_scale_and_shift_normalized_shape_tuple_1_bias() -> None:
    module = ScaleAndShift(normalized_shape=(5,))
    assert module.bias.equal(torch.zeros(5))


def test_scale_and_shift_normalized_shape_tuple_2_weight() -> None:
    module = ScaleAndShift(normalized_shape=(4, 5))
    assert module.weight.equal(torch.ones(4, 5))


def test_scale_and_shift_normalized_shape_tuple_2_bias() -> None:
    module = ScaleAndShift(normalized_shape=(4, 5))
    assert module.bias.equal(torch.zeros(4, 5))


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("batch_size", SIZES)
@pytest.mark.parametrize("dim", SIZES)
@pytest.mark.parametrize("mode", [True, False])
def test_scale_and_shift_forward_2d_shape(
    device: str, batch_size: int, dim: int, mode: bool
) -> None:
    device = torch.device(device)
    module = ScaleAndShift(normalized_shape=dim).to(device=device)
    module.train(mode)
    out = module(torch.rand(batch_size, dim, device=device))
    assert out.shape == (batch_size, dim)


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("batch_size", SIZES)
@pytest.mark.parametrize("dim", SIZES)
@pytest.mark.parametrize("mode", [True, False])
def test_scale_and_shift_forward_2d_device(
    device: str, batch_size: int, dim: int, mode: bool
) -> None:
    device = torch.device(device)
    module = ScaleAndShift(normalized_shape=dim).to(device=device)
    module.train(mode)
    out = module(torch.rand(batch_size, dim, device=device))
    assert out.device == device


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("batch_size", SIZES)
@pytest.mark.parametrize("dim", SIZES)
@pytest.mark.parametrize("mode", [True, False])
def test_scale_and_shift_forward_2d_dtype(
    device: str, batch_size: int, dim: int, mode: bool
) -> None:
    device = torch.device(device)
    module = ScaleAndShift(normalized_shape=dim).to(device=device)
    module.train(mode)
    out = module(torch.rand(batch_size, dim, device=device))
    assert out.dtype == torch.float


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("batch_size", SIZES)
@pytest.mark.parametrize("dims", [(5,), (2, 3), (3, 3), (2, 3, 4)])
@pytest.mark.parametrize("mode", [True, False])
def test_scale_and_shift_forward_2d_plus_shape(
    device: str, batch_size: int, dims: tuple[int, ...], mode: bool
) -> None:
    device = torch.device(device)
    module = ScaleAndShift(normalized_shape=dims).to(device=device)
    module.train(mode)
    out = module(torch.rand(batch_size, *dims, device=device))
    assert out.shape == (batch_size, *dims)


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("batch_size", SIZES)
@pytest.mark.parametrize("dims", [(5,), (2, 3), (3, 3), (2, 3, 4)])
@pytest.mark.parametrize("mode", [True, False])
def test_scale_and_shift_forward_2d_plus_device(
    device: str, batch_size: int, dims: tuple[int, ...], mode: bool
) -> None:
    device = torch.device(device)
    module = ScaleAndShift(normalized_shape=dims).to(device=device)
    module.train(mode)
    out = module(torch.rand(batch_size, *dims, device=device))
    assert out.device == device


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("batch_size", SIZES)
@pytest.mark.parametrize("dims", [(5,), (2, 3), (3, 3), (2, 3, 4)])
@pytest.mark.parametrize("mode", [True, False])
def test_scale_and_shift_forward_2d_plus_dtype(
    device: str, batch_size: int, dims: tuple[int, ...], mode: bool
) -> None:
    device = torch.device(device)
    module = ScaleAndShift(normalized_shape=dims).to(device=device)
    module.train(mode)
    out = module(torch.rand(batch_size, *dims, device=device))
    assert out.dtype == torch.float


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("batch_size", SIZES)
@pytest.mark.parametrize("seq_len", SIZES)
@pytest.mark.parametrize("dim", SIZES)
@pytest.mark.parametrize("mode", [True, False])
def test_scale_and_shift_forward_3d_seq_shape(
    device: str, batch_size: int, seq_len: int, dim: int, mode: bool
) -> None:
    device = torch.device(device)
    module = ScaleAndShift(normalized_shape=dim).to(device=device)
    module.train(mode)
    out = module(torch.rand(batch_size, seq_len, dim, device=device))
    assert out.shape == (batch_size, seq_len, dim)


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("batch_size", SIZES)
@pytest.mark.parametrize("seq_len", SIZES)
@pytest.mark.parametrize("dim", SIZES)
@pytest.mark.parametrize("mode", [True, False])
def test_scale_and_shift_forward_3d_seq_device(
    device: str, batch_size: int, seq_len: int, dim: int, mode: bool
) -> None:
    device = torch.device(device)
    module = ScaleAndShift(normalized_shape=dim).to(device=device)
    module.train(mode)
    out = module(torch.rand(batch_size, seq_len, dim, device=device))
    assert out.device == device


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("batch_size", SIZES)
@pytest.mark.parametrize("seq_len", SIZES)
@pytest.mark.parametrize("dim", SIZES)
@pytest.mark.parametrize("mode", [True, False])
def test_scale_and_shift_forward_3d_seq_dtype(
    device: str, batch_size: int, seq_len: int, dim: int, mode: bool
) -> None:
    device = torch.device(device)
    module = ScaleAndShift(normalized_shape=dim).to(device=device)
    module.train(mode)
    out = module(torch.rand(batch_size, seq_len, dim, device=device))
    assert out.dtype == torch.float


def test_scale_and_shift_forward_2d_init_weights() -> None:
    module = ScaleAndShift(normalized_shape=5)
    assert objects_are_allclose(
        module(torch.tensor([[-2.0, -1.0, 0.0, 1.0, 2.0], [3.0, 2.0, 1.0, 2.0, 3.0]])),
        torch.tensor([[-2.0, -1.0, 0.0, 1.0, 2.0], [3.0, 2.0, 1.0, 2.0, 3.0]]),
    )


def test_scale_and_shift_forward_2d_custom_weights() -> None:
    module = ScaleAndShift(normalized_shape=5)
    with torch.no_grad():
        module.weight.data += 1.0
        module.bias.data += 1.0
    assert objects_are_allclose(
        module(torch.tensor([[-2.0, -1.0, 0.0, 1.0, 2.0], [3.0, 2.0, 1.0, 2.0, 3.0]])),
        torch.tensor([[-3.0, -1.0, 1.0, 3.0, 5.0], [7.0, 5.0, 3.0, 5.0, 7.0]]),
    )


def test_scale_and_shift_reset_parameters_weight() -> None:
    module = ScaleAndShift(normalized_shape=5)
    with torch.no_grad():
        module.weight.data += 1.0
        module.bias.data += 1.0
    module.reset_parameters()
    assert module.weight.equal(torch.ones(5))


def test_scale_and_shift_reset_parameters_bias() -> None:
    module = ScaleAndShift(normalized_shape=5)
    with torch.no_grad():
        module.weight.data += 1.0
        module.bias.data += 1.0
    module.reset_parameters()
    assert module.bias.equal(torch.zeros(5))


@pytest.mark.parametrize("dtype", [torch.float64, torch.bfloat16])
def test_scale_and_shift_dtype_weight_dtype(dtype: torch.dtype) -> None:
    module = ScaleAndShift(4, dtype=dtype)
    assert module.weight.dtype == dtype


@pytest.mark.parametrize("dtype", [torch.float64, torch.bfloat16])
def test_scale_and_shift_dtype_bias_dtype(dtype: torch.dtype) -> None:
    module = ScaleAndShift(4, dtype=dtype)
    assert module.bias.dtype == dtype


def test_scale_and_shift_bias_false_has_no_bias() -> None:
    assert ScaleAndShift(normalized_shape=3, bias=False).bias is None


def test_scale_and_shift_bias_false_parameters() -> None:
    module = ScaleAndShift(normalized_shape=3, bias=False)
    assert [name for name, _ in module.named_parameters()] == ["weight"]


def test_scale_and_shift_bias_false_forward() -> None:
    module = ScaleAndShift(normalized_shape=3, bias=False)
    with torch.no_grad():
        module.weight.copy_(torch.tensor([1.0, 2.0, 3.0]))
    assert module(torch.ones(2, 3)).equal(torch.tensor([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]))


def test_scale_and_shift_bias_false_extra_repr() -> None:
    assert (
        ScaleAndShift(normalized_shape=3, bias=False).extra_repr()
        == "normalized_shape=(3,), bias=False"
    )


def test_scale_and_shift_bias_false_reset_parameters() -> None:
    module = ScaleAndShift(normalized_shape=3, bias=False)
    with torch.no_grad():
        module.weight.fill_(5.0)
    module.reset_parameters()
    assert module.weight.equal(torch.ones(3))


def test_scale_and_shift_reset_parameters_with_bias() -> None:
    module = ScaleAndShift(normalized_shape=3)
    with torch.no_grad():
        module.weight.fill_(5.0)
        module.bias.fill_(2.0)
    module.reset_parameters()
    assert module.bias.equal(torch.zeros(3))


######################################
#     Tests for the shape checks     #
######################################


@pytest.mark.parametrize("normalized_shape", [-1, 0, (3, 0), [2, -1]])
def test_scale_and_shift_incorrect_normalized_shape(normalized_shape: object) -> None:
    with pytest.raises(ValueError, match=r"normalized_shape must be greater than or equal to 1"):
        ScaleAndShift(normalized_shape=normalized_shape)


def test_scale_and_shift_incorrect_normalized_shape_type() -> None:
    with pytest.raises(TypeError, match=r"normalized_shape must be an int"):
        ScaleAndShift(normalized_shape=(2.5,))


#############################
#     Tests for _affine     #
#############################


def test_scale_and_shift_affine() -> None:
    module = ScaleAndShift(normalized_shape=3)
    module.weight.data.fill_(2.0)
    module.bias.data.fill_(1.0)
    assert module._affine(torch.tensor([[1.0, 2.0, 3.0]])).equal(torch.tensor([[3.0, 5.0, 7.0]]))


def test_scale_and_shift_affine_without_bias() -> None:
    module = ScaleAndShift(normalized_shape=3, bias=False)
    module.weight.data.fill_(2.0)
    assert module._affine(torch.tensor([[1.0, 2.0, 3.0]])).equal(torch.tensor([[2.0, 4.0, 6.0]]))


def test_scale_and_shift_forward_calls_affine() -> None:
    module = ScaleAndShift(normalized_shape=3)
    x = torch.rand(2, 3, generator=torch.manual_seed(1))
    assert module(x).equal(module._affine(x))
