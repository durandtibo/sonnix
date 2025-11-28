from __future__ import annotations

import pytest
import torch
from coola import objects_are_allclose
from coola.utils.tensor import get_available_devices

from sonnix.modules import ScaleAndShift

SIZES = (1, 2, 3)


###################################
#     Tests for ScaleAndShift     #
###################################


def test_scale_and_shift_str() -> None:
    assert str(ScaleAndShift(normalized_shape=5)) == "ScaleAndShift(normalized_shape=(5,))"


@pytest.mark.parametrize(("shape_input", "expected"), [(5, (5,)), ([3], (3,)), ((4, 2), (4, 2))])
def test_scale_and_shift_normalized_shape(
    shape_input: int | list[int] | tuple[int, ...], expected: tuple[int, ...]
) -> None:
    module = ScaleAndShift(shape_input)
    assert module._normalized_shape == expected
    assert module.weight.shape == expected
    assert module.bias.shape == expected


def test_scale_and_shift_normalized_shape_int() -> None:
    module = ScaleAndShift(normalized_shape=5)
    assert module.weight.equal(torch.ones(5))
    assert module.bias.equal(torch.zeros(5))


def test_scale_and_shift_normalized_shape_list() -> None:
    module = ScaleAndShift(normalized_shape=[5])
    assert module.weight.equal(torch.ones(5))
    assert module.bias.equal(torch.zeros(5))


def test_scale_and_shift_normalized_shape_tuple_1() -> None:
    module = ScaleAndShift(normalized_shape=(5,))
    assert module.weight.equal(torch.ones(5))
    assert module.bias.equal(torch.zeros(5))


def test_scale_and_shift_normalized_shape_tuple_2() -> None:
    module = ScaleAndShift(normalized_shape=(4, 5))
    assert module.weight.equal(torch.ones(4, 5))
    assert module.bias.equal(torch.zeros(4, 5))


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("batch_size", SIZES)
@pytest.mark.parametrize("dim", SIZES)
@pytest.mark.parametrize("mode", [True, False])
def test_scale_and_shift_forward_2d(device: str, batch_size: int, dim: int, mode: bool) -> None:
    device = torch.device(device)
    module = ScaleAndShift(normalized_shape=dim).to(device=device)
    module.train(mode)
    out = module(torch.rand(batch_size, dim, device=device))
    assert out.shape == (batch_size, dim)
    assert out.device == device
    assert out.dtype == torch.float


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("batch_size", SIZES)
@pytest.mark.parametrize("dims", [(5,), (2, 3), (3, 3), (2, 3, 4)])
@pytest.mark.parametrize("mode", [True, False])
def test_scale_and_shift_forward_2d_plus(
    device: str, batch_size: int, dims: tuple[int, ...], mode: bool
) -> None:
    device = torch.device(device)
    module = ScaleAndShift(normalized_shape=dims).to(device=device)
    module.train(mode)
    out = module(torch.rand(batch_size, *dims, device=device))
    assert out.shape == (batch_size, *dims)
    assert out.device == device
    assert out.dtype == torch.float


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("batch_size", SIZES)
@pytest.mark.parametrize("seq_len", SIZES)
@pytest.mark.parametrize("dim", SIZES)
@pytest.mark.parametrize("mode", [True, False])
def test_scale_and_shift_forward_3d_seq(
    device: str, batch_size: int, seq_len: int, dim: int, mode: bool
) -> None:
    device = torch.device(device)
    module = ScaleAndShift(normalized_shape=dim).to(device=device)
    module.train(mode)
    out = module(torch.rand(batch_size, seq_len, dim, device=device))
    assert out.shape == (batch_size, seq_len, dim)
    assert out.device == device
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


def test_scale_and_shift_reset_parameters() -> None:
    module = ScaleAndShift(normalized_shape=5)
    with torch.no_grad():
        module.weight.data += 1.0
        module.bias.data += 1.0
    module.reset_parameters()
    assert module.weight.equal(torch.ones(5))
    assert module.bias.equal(torch.zeros(5))
