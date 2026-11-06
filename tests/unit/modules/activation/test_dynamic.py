from __future__ import annotations

import pytest
import torch
from coola.equality import objects_are_allclose
from coola.utils.tensor import get_available_devices

from sonnix.modules import DynamicAsinh, DynamicTanh, ScaleAndShift
from sonnix.modules.activation.dynamic import BaseDynamicNorm

SIZES = (1, 2, 3)


##################################
#     Tests for DynamicAsinh     #
##################################


def test_dynamic_asinh_str() -> None:
    assert str(DynamicAsinh(normalized_shape=5)) == "DynamicAsinh(normalized_shape=(5,), bias=True)"


@pytest.mark.parametrize(("shape_input", "expected"), [(5, (5,)), ([3], (3,)), ((4, 2), (4, 2))])
def test_dynamic_asinh_normalized_shape_value(
    shape_input: int | list[int] | tuple[int, ...], expected: tuple[int, ...]
) -> None:
    module = DynamicAsinh(shape_input)
    assert module.normalized_shape == expected


@pytest.mark.parametrize(("shape_input", "expected"), [(5, (5,)), ([3], (3,)), ((4, 2), (4, 2))])
def test_dynamic_asinh_normalized_shape_weight_shape(
    shape_input: int | list[int] | tuple[int, ...], expected: tuple[int, ...]
) -> None:
    module = DynamicAsinh(shape_input)
    assert module.weight.shape == expected


@pytest.mark.parametrize(("shape_input", "expected"), [(5, (5,)), ([3], (3,)), ((4, 2), (4, 2))])
def test_dynamic_asinh_normalized_shape_bias_shape(
    shape_input: int | list[int] | tuple[int, ...], expected: tuple[int, ...]
) -> None:
    module = DynamicAsinh(shape_input)
    assert module.bias.shape == expected


def test_dynamic_asinh_normalized_shape_int_weight() -> None:
    module = DynamicAsinh(normalized_shape=5)
    assert module.weight.equal(torch.ones(5))


def test_dynamic_asinh_normalized_shape_int_bias() -> None:
    module = DynamicAsinh(normalized_shape=5)
    assert module.bias.equal(torch.zeros(5))


def test_dynamic_asinh_normalized_shape_list_weight() -> None:
    module = DynamicAsinh(normalized_shape=[5])
    assert module.weight.equal(torch.ones(5))


def test_dynamic_asinh_normalized_shape_list_bias() -> None:
    module = DynamicAsinh(normalized_shape=[5])
    assert module.bias.equal(torch.zeros(5))


def test_dynamic_asinh_normalized_shape_tuple_1_weight() -> None:
    module = DynamicAsinh(normalized_shape=(5,))
    assert module.weight.equal(torch.ones(5))


def test_dynamic_asinh_normalized_shape_tuple_1_bias() -> None:
    module = DynamicAsinh(normalized_shape=(5,))
    assert module.bias.equal(torch.zeros(5))


def test_dynamic_asinh_normalized_shape_tuple_2_weight() -> None:
    module = DynamicAsinh(normalized_shape=(4, 5))
    assert module.weight.equal(torch.ones(4, 5))


def test_dynamic_asinh_normalized_shape_tuple_2_bias() -> None:
    module = DynamicAsinh(normalized_shape=(4, 5))
    assert module.bias.equal(torch.zeros(4, 5))


def test_dynamic_asinh_alpha_default() -> None:
    module = DynamicAsinh(normalized_shape=[5])
    assert module.alpha.equal(torch.tensor([0.5]))


def test_dynamic_asinh_alpha_2() -> None:
    module = DynamicAsinh(normalized_shape=[5], alpha_init_value=2.0)
    assert module.alpha.equal(torch.tensor([2.0]))


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("batch_size", SIZES)
@pytest.mark.parametrize("dim", SIZES)
@pytest.mark.parametrize("mode", [True, False])
def test_dynamic_asinh_forward_2d_shape(device: str, batch_size: int, dim: int, mode: bool) -> None:
    device = torch.device(device)
    module = DynamicAsinh(normalized_shape=dim).to(device=device)
    module.train(mode)
    out = module(torch.rand(batch_size, dim, device=device))
    assert out.shape == (batch_size, dim)


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("batch_size", SIZES)
@pytest.mark.parametrize("dim", SIZES)
@pytest.mark.parametrize("mode", [True, False])
def test_dynamic_asinh_forward_2d_device(
    device: str, batch_size: int, dim: int, mode: bool
) -> None:
    device = torch.device(device)
    module = DynamicAsinh(normalized_shape=dim).to(device=device)
    module.train(mode)
    out = module(torch.rand(batch_size, dim, device=device))
    assert out.device == device


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("batch_size", SIZES)
@pytest.mark.parametrize("dim", SIZES)
@pytest.mark.parametrize("mode", [True, False])
def test_dynamic_asinh_forward_2d_dtype(device: str, batch_size: int, dim: int, mode: bool) -> None:
    device = torch.device(device)
    module = DynamicAsinh(normalized_shape=dim).to(device=device)
    module.train(mode)
    out = module(torch.rand(batch_size, dim, device=device))
    assert out.dtype == torch.float


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("batch_size", SIZES)
@pytest.mark.parametrize("dims", [(5,), (2, 3), (3, 3), (2, 3, 4)])
@pytest.mark.parametrize("mode", [True, False])
def test_dynamic_asinh_forward_2d_plus_shape(
    device: str, batch_size: int, dims: tuple[int, ...], mode: bool
) -> None:
    device = torch.device(device)
    module = DynamicAsinh(normalized_shape=dims).to(device=device)
    module.train(mode)
    out = module(torch.rand(batch_size, *dims, device=device))
    assert out.shape == (batch_size, *dims)


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("batch_size", SIZES)
@pytest.mark.parametrize("dims", [(5,), (2, 3), (3, 3), (2, 3, 4)])
@pytest.mark.parametrize("mode", [True, False])
def test_dynamic_asinh_forward_2d_plus_device(
    device: str, batch_size: int, dims: tuple[int, ...], mode: bool
) -> None:
    device = torch.device(device)
    module = DynamicAsinh(normalized_shape=dims).to(device=device)
    module.train(mode)
    out = module(torch.rand(batch_size, *dims, device=device))
    assert out.device == device


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("batch_size", SIZES)
@pytest.mark.parametrize("dims", [(5,), (2, 3), (3, 3), (2, 3, 4)])
@pytest.mark.parametrize("mode", [True, False])
def test_dynamic_asinh_forward_2d_plus_dtype(
    device: str, batch_size: int, dims: tuple[int, ...], mode: bool
) -> None:
    device = torch.device(device)
    module = DynamicAsinh(normalized_shape=dims).to(device=device)
    module.train(mode)
    out = module(torch.rand(batch_size, *dims, device=device))
    assert out.dtype == torch.float


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("batch_size", SIZES)
@pytest.mark.parametrize("seq_len", SIZES)
@pytest.mark.parametrize("dim", SIZES)
@pytest.mark.parametrize("mode", [True, False])
def test_dynamic_asinh_forward_3d_seq_shape(
    device: str, batch_size: int, seq_len: int, dim: int, mode: bool
) -> None:
    device = torch.device(device)
    module = DynamicAsinh(normalized_shape=dim).to(device=device)
    module.train(mode)
    out = module(torch.rand(batch_size, seq_len, dim, device=device))
    assert out.shape == (batch_size, seq_len, dim)


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("batch_size", SIZES)
@pytest.mark.parametrize("seq_len", SIZES)
@pytest.mark.parametrize("dim", SIZES)
@pytest.mark.parametrize("mode", [True, False])
def test_dynamic_asinh_forward_3d_seq_device(
    device: str, batch_size: int, seq_len: int, dim: int, mode: bool
) -> None:
    device = torch.device(device)
    module = DynamicAsinh(normalized_shape=dim).to(device=device)
    module.train(mode)
    out = module(torch.rand(batch_size, seq_len, dim, device=device))
    assert out.device == device


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("batch_size", SIZES)
@pytest.mark.parametrize("seq_len", SIZES)
@pytest.mark.parametrize("dim", SIZES)
@pytest.mark.parametrize("mode", [True, False])
def test_dynamic_asinh_forward_3d_seq_dtype(
    device: str, batch_size: int, seq_len: int, dim: int, mode: bool
) -> None:
    device = torch.device(device)
    module = DynamicAsinh(normalized_shape=dim).to(device=device)
    module.train(mode)
    out = module(torch.rand(batch_size, seq_len, dim, device=device))
    assert out.dtype == torch.float


def test_dynamic_asinh_forward_2d_init_weights() -> None:
    module = DynamicAsinh(normalized_shape=5)
    assert objects_are_allclose(
        module(torch.tensor([[-2.0, -1.0, 0.0, 1.0, 2.0], [3.0, 2.0, 1.0, 2.0, 3.0]])),
        torch.tensor(
            [
                [
                    -0.8813735842704773,
                    -0.4812118113040924,
                    0.0,
                    0.4812118113040924,
                    0.8813735842704773,
                ],
                [
                    1.19476318359375,
                    0.8813735842704773,
                    0.4812118113040924,
                    0.8813735842704773,
                    1.19476318359375,
                ],
            ]
        ),
    )


def test_dynamic_asinh_forward_2d_custom_weights() -> None:
    module = DynamicAsinh(normalized_shape=5)
    with torch.no_grad():
        module.weight.data += 1.0
        module.bias.data += 1.0
    assert objects_are_allclose(
        module(torch.tensor([[-2.0, -1.0, 0.0, 1.0, 2.0], [3.0, 2.0, 1.0, 2.0, 3.0]])),
        torch.tensor(
            [
                [
                    -0.7627471685409546,
                    0.037576377391815186,
                    1.0,
                    1.96242356300354,
                    2.762747287750244,
                ],
                [
                    3.3895263671875,
                    2.762747287750244,
                    1.96242356300354,
                    2.762747287750244,
                    3.3895263671875,
                ],
            ]
        ),
    )


def test_dynamic_asinh_reset_parameters_alpha() -> None:
    module = DynamicAsinh(normalized_shape=5)
    with torch.no_grad():
        module.alpha.data += 1.0
        module.weight.data += 1.0
        module.bias.data += 1.0
    module.reset_parameters()
    assert module.alpha.equal(torch.tensor([0.5]))


def test_dynamic_asinh_reset_parameters_weight() -> None:
    module = DynamicAsinh(normalized_shape=5)
    with torch.no_grad():
        module.alpha.data += 1.0
        module.weight.data += 1.0
        module.bias.data += 1.0
    module.reset_parameters()
    assert module.weight.equal(torch.ones(5))


def test_dynamic_asinh_reset_parameters_bias() -> None:
    module = DynamicAsinh(normalized_shape=5)
    with torch.no_grad():
        module.alpha.data += 1.0
        module.weight.data += 1.0
        module.bias.data += 1.0
    module.reset_parameters()
    assert module.bias.equal(torch.zeros(5))


#################################
#     Tests for DynamicTanh     #
#################################


def test_dynamic_tanh_str() -> None:
    assert str(DynamicTanh(normalized_shape=5)) == "DynamicTanh(normalized_shape=(5,), bias=True)"


@pytest.mark.parametrize(("shape_input", "expected"), [(5, (5,)), ([3], (3,)), ((4, 2), (4, 2))])
def test_dynamic_tanh_normalized_shape_value(
    shape_input: int | list[int] | tuple[int, ...], expected: tuple[int, ...]
) -> None:
    module = DynamicTanh(shape_input)
    assert module.normalized_shape == expected


@pytest.mark.parametrize(("shape_input", "expected"), [(5, (5,)), ([3], (3,)), ((4, 2), (4, 2))])
def test_dynamic_tanh_normalized_shape_weight_shape(
    shape_input: int | list[int] | tuple[int, ...], expected: tuple[int, ...]
) -> None:
    module = DynamicTanh(shape_input)
    assert module.weight.shape == expected


@pytest.mark.parametrize(("shape_input", "expected"), [(5, (5,)), ([3], (3,)), ((4, 2), (4, 2))])
def test_dynamic_tanh_normalized_shape_bias_shape(
    shape_input: int | list[int] | tuple[int, ...], expected: tuple[int, ...]
) -> None:
    module = DynamicTanh(shape_input)
    assert module.bias.shape == expected


def test_dynamic_tanh_normalized_shape_int_weight() -> None:
    module = DynamicTanh(normalized_shape=5)
    assert module.weight.equal(torch.ones(5))


def test_dynamic_tanh_normalized_shape_int_bias() -> None:
    module = DynamicTanh(normalized_shape=5)
    assert module.bias.equal(torch.zeros(5))


def test_dynamic_tanh_normalized_shape_list_weight() -> None:
    module = DynamicTanh(normalized_shape=[5])
    assert module.weight.equal(torch.ones(5))


def test_dynamic_tanh_normalized_shape_list_bias() -> None:
    module = DynamicTanh(normalized_shape=[5])
    assert module.bias.equal(torch.zeros(5))


def test_dynamic_tanh_normalized_shape_tuple_1_weight() -> None:
    module = DynamicTanh(normalized_shape=(5,))
    assert module.weight.equal(torch.ones(5))


def test_dynamic_tanh_normalized_shape_tuple_1_bias() -> None:
    module = DynamicTanh(normalized_shape=(5,))
    assert module.bias.equal(torch.zeros(5))


def test_dynamic_tanh_normalized_shape_tuple_2_weight() -> None:
    module = DynamicTanh(normalized_shape=(4, 5))
    assert module.weight.equal(torch.ones(4, 5))


def test_dynamic_tanh_normalized_shape_tuple_2_bias() -> None:
    module = DynamicTanh(normalized_shape=(4, 5))
    assert module.bias.equal(torch.zeros(4, 5))


def test_dynamic_tanh_alpha_default() -> None:
    module = DynamicTanh(normalized_shape=[5])
    assert module.alpha.equal(torch.tensor([0.5]))


def test_dynamic_tanh_alpha_2() -> None:
    module = DynamicTanh(normalized_shape=[5], alpha_init_value=2.0)
    assert module.alpha.equal(torch.tensor([2.0]))


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("batch_size", SIZES)
@pytest.mark.parametrize("dim", SIZES)
@pytest.mark.parametrize("mode", [True, False])
def test_dynamic_tanh_forward_2d_shape(device: str, batch_size: int, dim: int, mode: bool) -> None:
    device = torch.device(device)
    module = DynamicTanh(normalized_shape=dim).to(device=device)
    module.train(mode)
    out = module(torch.rand(batch_size, dim, device=device))
    assert out.shape == (batch_size, dim)


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("batch_size", SIZES)
@pytest.mark.parametrize("dim", SIZES)
@pytest.mark.parametrize("mode", [True, False])
def test_dynamic_tanh_forward_2d_device(device: str, batch_size: int, dim: int, mode: bool) -> None:
    device = torch.device(device)
    module = DynamicTanh(normalized_shape=dim).to(device=device)
    module.train(mode)
    out = module(torch.rand(batch_size, dim, device=device))
    assert out.device == device


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("batch_size", SIZES)
@pytest.mark.parametrize("dim", SIZES)
@pytest.mark.parametrize("mode", [True, False])
def test_dynamic_tanh_forward_2d_dtype(device: str, batch_size: int, dim: int, mode: bool) -> None:
    device = torch.device(device)
    module = DynamicTanh(normalized_shape=dim).to(device=device)
    module.train(mode)
    out = module(torch.rand(batch_size, dim, device=device))
    assert out.dtype == torch.float


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("batch_size", SIZES)
@pytest.mark.parametrize("dims", [(5,), (2, 3), (3, 3), (2, 3, 4)])
@pytest.mark.parametrize("mode", [True, False])
def test_dynamic_tanh_forward_2d_plus_shape(
    device: str, batch_size: int, dims: tuple[int, ...], mode: bool
) -> None:
    device = torch.device(device)
    module = DynamicTanh(normalized_shape=dims).to(device=device)
    module.train(mode)
    out = module(torch.rand(batch_size, *dims, device=device))
    assert out.shape == (batch_size, *dims)


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("batch_size", SIZES)
@pytest.mark.parametrize("dims", [(5,), (2, 3), (3, 3), (2, 3, 4)])
@pytest.mark.parametrize("mode", [True, False])
def test_dynamic_tanh_forward_2d_plus_device(
    device: str, batch_size: int, dims: tuple[int, ...], mode: bool
) -> None:
    device = torch.device(device)
    module = DynamicTanh(normalized_shape=dims).to(device=device)
    module.train(mode)
    out = module(torch.rand(batch_size, *dims, device=device))
    assert out.device == device


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("batch_size", SIZES)
@pytest.mark.parametrize("dims", [(5,), (2, 3), (3, 3), (2, 3, 4)])
@pytest.mark.parametrize("mode", [True, False])
def test_dynamic_tanh_forward_2d_plus_dtype(
    device: str, batch_size: int, dims: tuple[int, ...], mode: bool
) -> None:
    device = torch.device(device)
    module = DynamicTanh(normalized_shape=dims).to(device=device)
    module.train(mode)
    out = module(torch.rand(batch_size, *dims, device=device))
    assert out.dtype == torch.float


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("batch_size", SIZES)
@pytest.mark.parametrize("seq_len", SIZES)
@pytest.mark.parametrize("dim", SIZES)
@pytest.mark.parametrize("mode", [True, False])
def test_dynamic_tanh_forward_3d_seq_shape(
    device: str, batch_size: int, seq_len: int, dim: int, mode: bool
) -> None:
    device = torch.device(device)
    module = DynamicTanh(normalized_shape=dim).to(device=device)
    module.train(mode)
    out = module(torch.rand(batch_size, seq_len, dim, device=device))
    assert out.shape == (batch_size, seq_len, dim)


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("batch_size", SIZES)
@pytest.mark.parametrize("seq_len", SIZES)
@pytest.mark.parametrize("dim", SIZES)
@pytest.mark.parametrize("mode", [True, False])
def test_dynamic_tanh_forward_3d_seq_device(
    device: str, batch_size: int, seq_len: int, dim: int, mode: bool
) -> None:
    device = torch.device(device)
    module = DynamicTanh(normalized_shape=dim).to(device=device)
    module.train(mode)
    out = module(torch.rand(batch_size, seq_len, dim, device=device))
    assert out.device == device


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("batch_size", SIZES)
@pytest.mark.parametrize("seq_len", SIZES)
@pytest.mark.parametrize("dim", SIZES)
@pytest.mark.parametrize("mode", [True, False])
def test_dynamic_tanh_forward_3d_seq_dtype(
    device: str, batch_size: int, seq_len: int, dim: int, mode: bool
) -> None:
    device = torch.device(device)
    module = DynamicTanh(normalized_shape=dim).to(device=device)
    module.train(mode)
    out = module(torch.rand(batch_size, seq_len, dim, device=device))
    assert out.dtype == torch.float


def test_dynamic_tanh_forward_2d_init_weights() -> None:
    module = DynamicTanh(normalized_shape=5)
    assert objects_are_allclose(
        module(torch.tensor([[-2.0, -1.0, 0.0, 1.0, 2.0], [3.0, 2.0, 1.0, 2.0, 3.0]])),
        torch.tensor(
            [
                [
                    -0.7615941762924194,
                    -0.46211716532707214,
                    0.0,
                    0.46211716532707214,
                    0.7615941762924194,
                ],
                [
                    0.9051482677459717,
                    0.7615941762924194,
                    0.46211716532707214,
                    0.7615941762924194,
                    0.9051482677459717,
                ],
            ]
        ),
    )


def test_dynamic_tanh_forward_2d_custom_weights() -> None:
    module = DynamicTanh(normalized_shape=5)
    with torch.no_grad():
        module.weight.data += 1.0
        module.bias.data += 1.0
    assert objects_are_allclose(
        module(torch.tensor([[-2.0, -1.0, 0.0, 1.0, 2.0], [3.0, 2.0, 1.0, 2.0, 3.0]])),
        torch.tensor(
            [
                [
                    -0.5231883525848389,
                    0.07576566934585571,
                    1.0,
                    1.924234390258789,
                    2.523188352584839,
                ],
                [
                    2.8102965354919434,
                    2.523188352584839,
                    1.924234390258789,
                    2.523188352584839,
                    2.8102965354919434,
                ],
            ]
        ),
    )


def test_dynamic_tanh_reset_parameters_alpha() -> None:
    module = DynamicTanh(normalized_shape=5)
    with torch.no_grad():
        module.alpha.data += 1.0
        module.weight.data += 1.0
        module.bias.data += 1.0
    module.reset_parameters()
    assert module.alpha.equal(torch.tensor([0.5]))


def test_dynamic_tanh_reset_parameters_weight() -> None:
    module = DynamicTanh(normalized_shape=5)
    with torch.no_grad():
        module.alpha.data += 1.0
        module.weight.data += 1.0
        module.bias.data += 1.0
    module.reset_parameters()
    assert module.weight.equal(torch.ones(5))


def test_dynamic_tanh_reset_parameters_bias() -> None:
    module = DynamicTanh(normalized_shape=5)
    with torch.no_grad():
        module.alpha.data += 1.0
        module.weight.data += 1.0
        module.bias.data += 1.0
    module.reset_parameters()
    assert module.bias.equal(torch.zeros(5))


@pytest.mark.parametrize("dtype", [torch.float64, torch.bfloat16])
def test_dynamic_tanh_dtype_alpha_dtype(dtype: torch.dtype) -> None:
    module = DynamicTanh(4, dtype=dtype)
    assert module.alpha.dtype == dtype


@pytest.mark.parametrize("dtype", [torch.float64, torch.bfloat16])
def test_dynamic_tanh_dtype_weight_dtype(dtype: torch.dtype) -> None:
    module = DynamicTanh(4, dtype=dtype)
    assert module.weight.dtype == dtype


@pytest.mark.parametrize("dtype", [torch.float64, torch.bfloat16])
def test_dynamic_tanh_dtype_bias_dtype(dtype: torch.dtype) -> None:
    module = DynamicTanh(4, dtype=dtype)
    assert module.bias.dtype == dtype


#####################################
#     Tests for BaseDynamicNorm     #
#####################################


class MyDynamicNorm(BaseDynamicNorm):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return ScaleAndShift.forward(self, x.mul(self.alpha))


def test_base_dynamic_norm_is_scale_and_shift() -> None:
    assert isinstance(MyDynamicNorm(normalized_shape=3), ScaleAndShift)


def test_base_dynamic_norm_abstract() -> None:
    with pytest.raises(TypeError, match=r"abstract"):
        BaseDynamicNorm(normalized_shape=3)


def test_base_dynamic_norm_initial_parameters_alpha() -> None:
    module = MyDynamicNorm(normalized_shape=3, alpha_init_value=0.2)
    assert module.alpha.equal(torch.tensor([0.2]))


def test_base_dynamic_norm_initial_parameters_weight() -> None:
    module = MyDynamicNorm(normalized_shape=3, alpha_init_value=0.2)
    assert module.weight.equal(torch.ones(3))


def test_base_dynamic_norm_initial_parameters_bias() -> None:
    module = MyDynamicNorm(normalized_shape=3, alpha_init_value=0.2)
    assert module.bias.equal(torch.zeros(3))


def test_base_dynamic_norm_reset_parameters_alpha() -> None:
    module = MyDynamicNorm(normalized_shape=3, alpha_init_value=0.2)
    with torch.no_grad():
        module.alpha.fill_(3.0)
        module.weight.fill_(5.0)
        module.bias.fill_(2.0)
    module.reset_parameters()
    assert module.alpha.equal(torch.tensor([0.2]))


def test_base_dynamic_norm_reset_parameters_weight() -> None:
    module = MyDynamicNorm(normalized_shape=3, alpha_init_value=0.2)
    with torch.no_grad():
        module.alpha.fill_(3.0)
        module.weight.fill_(5.0)
        module.bias.fill_(2.0)
    module.reset_parameters()
    assert module.weight.equal(torch.ones(3))


def test_base_dynamic_norm_reset_parameters_bias() -> None:
    module = MyDynamicNorm(normalized_shape=3, alpha_init_value=0.2)
    with torch.no_grad():
        module.alpha.fill_(3.0)
        module.weight.fill_(5.0)
        module.bias.fill_(2.0)
    module.reset_parameters()
    assert module.bias.equal(torch.zeros(3))


def test_base_dynamic_norm_bias_false() -> None:
    assert MyDynamicNorm(normalized_shape=3, bias=False).bias is None


def test_base_dynamic_norm_extra_repr() -> None:
    assert MyDynamicNorm(normalized_shape=3).extra_repr() == "normalized_shape=(3,), bias=True"


@pytest.mark.parametrize("module", [DynamicAsinh, DynamicTanh])
def test_dynamic_norm_bias_false_parameters(module: type[BaseDynamicNorm]) -> None:
    assert sorted(
        name for name, _ in module(normalized_shape=3, bias=False).named_parameters()
    ) == [
        "alpha",
        "weight",
    ]


def test_dynamic_asinh_bias_false_forward() -> None:
    module = DynamicAsinh(normalized_shape=2, alpha_init_value=1.0, bias=False)
    assert module(torch.tensor([[1.0, -1.0]])).allclose(torch.tensor([[0.8813736, -0.8813736]]))


def test_dynamic_tanh_bias_false_forward() -> None:
    module = DynamicTanh(normalized_shape=2, alpha_init_value=1.0, bias=False)
    assert module(torch.tensor([[1.0, -1.0]])).allclose(torch.tensor([[0.7615942, -0.7615942]]))
