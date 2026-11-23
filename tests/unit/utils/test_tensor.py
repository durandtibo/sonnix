from __future__ import annotations

import pytest
import torch
from coola.equality import objects_are_equal
from torch import nn

from sonnix.utils.tensor import (
    _get_float_dtype,
    register_tensor,
    resolve_eps,
    resolve_max_exponent,
)

#####################################
#     Tests for register_tensor     #
#####################################


@pytest.mark.parametrize("learnable", [1, 0, "yes", None])
def test_register_tensor_rejects_a_learnable_that_is_not_a_bool(learnable: object) -> None:
    with pytest.raises(TypeError, match=r"learnable must be an instance of bool"):
        register_tensor(nn.Module(), "alpha", torch.ones(2), learnable=learnable)


def test_register_tensor_does_not_register_when_learnable_is_invalid() -> None:
    module = nn.Module()
    with pytest.raises(TypeError, match=r"learnable must be an instance of bool"):
        register_tensor(module, "alpha", torch.ones(2), learnable="yes")
    assert not hasattr(module, "alpha")


def test_register_tensor_learnable_is_a_parameter() -> None:
    module = nn.Module()
    register_tensor(module, "alpha", torch.ones(2), learnable=True)
    assert isinstance(module.alpha, nn.Parameter)


def test_register_tensor_learnable_requires_grad() -> None:
    module = nn.Module()
    register_tensor(module, "alpha", torch.ones(2), learnable=True)
    assert module.alpha.requires_grad


def test_register_tensor_frozen_is_a_buffer() -> None:
    module = nn.Module()
    register_tensor(module, "alpha", torch.ones(2), learnable=False)
    assert [name for name, _ in module.named_buffers()] == ["alpha"]


def test_register_tensor_frozen_is_not_a_parameter() -> None:
    module = nn.Module()
    register_tensor(module, "alpha", torch.ones(2), learnable=False)
    assert list(module.parameters()) == []


def test_register_tensor_frozen_is_not_a_tensor_with_grad() -> None:
    module = nn.Module()
    register_tensor(module, "alpha", torch.ones(2), learnable=False)
    assert not module.alpha.requires_grad


def test_register_tensor_state_dict_key_does_not_depend_on_learnable() -> None:
    learnable, frozen = nn.Module(), nn.Module()
    register_tensor(learnable, "alpha", torch.ones(2), learnable=True)
    register_tensor(frozen, "alpha", torch.ones(2), learnable=False)
    assert list(learnable.state_dict()) == list(frozen.state_dict())


def test_register_tensor_keeps_the_value() -> None:
    module = nn.Module()
    register_tensor(module, "alpha", torch.tensor([1.0, 2.0]), learnable=True)
    assert objects_are_equal(module.alpha.data, torch.tensor([1.0, 2.0]))


def test_register_tensor_keeps_the_dtype() -> None:
    module = nn.Module()
    register_tensor(module, "alpha", torch.ones(2, dtype=torch.float64), learnable=False)
    assert module.alpha.dtype == torch.float64


def test_register_tensor_copies_the_tensor() -> None:
    tensor = torch.ones(2)
    module = nn.Module()
    register_tensor(module, "alpha", tensor, learnable=False)
    tensor.add_(1.0)
    assert objects_are_equal(module.alpha, torch.ones(2))


def test_register_tensor_detaches_the_tensor() -> None:
    module = nn.Module()
    register_tensor(module, "alpha", torch.ones(2, requires_grad=True) * 2, learnable=False)
    assert module.alpha.grad_fn is None


def test_register_tensor_does_not_modify_the_tensor() -> None:
    tensor = torch.ones(2)
    register_tensor(nn.Module(), "alpha", tensor, learnable=True)
    assert not tensor.requires_grad


#################################
#     Tests for resolve_eps     #
#################################


def test_resolve_eps_float16_raises_the_guard() -> None:
    assert resolve_eps(1e-8, torch.zeros(1, dtype=torch.float16)) == 6.103515625e-05


def test_resolve_eps_float16_keeps_a_guard_that_is_large_enough() -> None:
    assert resolve_eps(1e-3, torch.zeros(1, dtype=torch.float16)) == 1e-3


def test_resolve_eps_float16_boundary_is_the_smallest_normal_number() -> None:
    assert resolve_eps(6.103515625e-05, torch.zeros(1, dtype=torch.float16)) == 6.103515625e-05


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_resolve_eps_keeps_the_default_guard(dtype: torch.dtype) -> None:
    assert resolve_eps(1e-8, torch.zeros(1, dtype=dtype)) == 1e-8


def test_resolve_eps_float64_keeps_the_default_guard() -> None:
    assert resolve_eps(1e-8, torch.zeros(1, dtype=torch.float64)) == 1e-8


def test_resolve_eps_bfloat16_raises_a_guard_below_the_smallest_normal_number() -> None:
    assert resolve_eps(1e-40, torch.zeros(1, dtype=torch.bfloat16)) == 1.1754943508222875e-38


def test_resolve_eps_float64_raises_a_guard_below_the_smallest_normal_number() -> None:
    assert resolve_eps(1e-320, torch.zeros(1, dtype=torch.float64)) == 2.2250738585072014e-308


def test_resolve_eps_integer_tensor_uses_the_default_dtype() -> None:
    assert resolve_eps(1e-40, torch.zeros(1, dtype=torch.long)) == 1.1754943508222875e-38


def test_resolve_eps_returns_a_float() -> None:
    assert isinstance(resolve_eps(1e-8, torch.zeros(1, dtype=torch.float16)), float)


##########################################
#     Tests for resolve_max_exponent     #
##########################################


@pytest.mark.parametrize(
    ("dtype", "expected"),
    [
        (torch.float16, 10.089866488461016),
        (torch.bfloat16, 87.71892521235186),
        (torch.float32, 87.72283905206835),
        (torch.float64, 708.782712893384),
    ],
)
def test_resolve_max_exponent_lowers_a_large_bound(dtype: torch.dtype, expected: float) -> None:
    assert resolve_max_exponent(1e4, torch.zeros(1, dtype=dtype)) == expected


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_resolve_max_exponent_keeps_a_small_bound(dtype: torch.dtype) -> None:
    assert resolve_max_exponent(5.0, torch.zeros(1, dtype=dtype)) == 5.0


def test_resolve_max_exponent_default_bound_is_lowered_in_float16() -> None:
    assert resolve_max_exponent(20.0, torch.zeros(1, dtype=torch.float16)) == 10.089866488461016


def test_resolve_max_exponent_default_bound_is_kept_in_float32() -> None:
    assert resolve_max_exponent(20.0, torch.zeros(1)) == 20.0


def test_resolve_max_exponent_negative_bound_is_kept() -> None:
    assert resolve_max_exponent(-3.0, torch.zeros(1, dtype=torch.float16)) == -3.0


def test_resolve_max_exponent_integer_tensor_uses_the_default_dtype() -> None:
    assert resolve_max_exponent(1e4, torch.zeros(1, dtype=torch.long)) == 87.72283905206835


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_resolve_max_exponent_exp_is_finite(dtype: torch.dtype) -> None:
    tensor = torch.zeros(1, dtype=dtype)
    bound = resolve_max_exponent(1e4, tensor)
    assert torch.isfinite(torch.full_like(tensor, bound).exp()).all()


######################################
#     Tests for _get_float_dtype     #
######################################


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_get_float_dtype_floating_tensor(dtype: torch.dtype) -> None:
    assert _get_float_dtype(torch.zeros(1, dtype=dtype)) == dtype


@pytest.mark.parametrize("dtype", [torch.long, torch.int32, torch.bool])
def test_get_float_dtype_not_floating_tensor_returns_the_default_dtype(dtype: torch.dtype) -> None:
    assert _get_float_dtype(torch.zeros(1, dtype=dtype)) == torch.get_default_dtype()
