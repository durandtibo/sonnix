from __future__ import annotations

import pytest
import torch
from torch import nn

from tests.registry import MODULE_CASES, ModuleCase

DTYPES = (torch.float32, torch.float64, torch.bfloat16)
DIFFERENTIABLE_CASES = [case for case in MODULE_CASES if case.differentiable]


def _floating_tensors(module: nn.Module) -> list[torch.Tensor]:
    return [
        tensor for tensor in (*module.parameters(), *module.buffers()) if tensor.is_floating_point()
    ]


def _gradients(module: nn.Module, inputs: tuple[torch.Tensor, ...]) -> list[torch.Tensor]:
    return [t.grad for t in (*module.parameters(), *inputs) if t.grad is not None]


def _forward(case: ModuleCase, dtype: torch.dtype) -> tuple[nn.Module, tuple, torch.Tensor]:
    module = case.build().to(dtype)
    inputs = tuple(x.to(dtype).requires_grad_() for x in case.make_inputs())
    return module, inputs, module(*inputs)


######################################
#     Tests for module.to(dtype)     #
######################################


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("case", MODULE_CASES, ids=repr)
def test_to_dtype_converts_parameters_and_buffers(case: ModuleCase, dtype: torch.dtype) -> None:
    module = case.build().to(dtype)
    assert {tensor.dtype for tensor in _floating_tensors(module)} <= {dtype}


###############################################
#     Tests for forward in several dtypes     #
###############################################


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("case", DIFFERENTIABLE_CASES, ids=repr)
def test_forward_keeps_dtype(case: ModuleCase, dtype: torch.dtype) -> None:
    assert _forward(case, dtype)[2].dtype == dtype


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("case", DIFFERENTIABLE_CASES, ids=repr)
def test_forward_is_finite(case: ModuleCase, dtype: torch.dtype) -> None:
    assert _forward(case, dtype)[2].isfinite().all()


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("case", DIFFERENTIABLE_CASES, ids=repr)
def test_backward_gradients_have_the_dtype(case: ModuleCase, dtype: torch.dtype) -> None:
    module, inputs, output = _forward(case, dtype)
    output.sum().backward()
    assert {grad.dtype for grad in _gradients(module, inputs)} <= {dtype}


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("case", DIFFERENTIABLE_CASES, ids=repr)
def test_backward_gradients_are_finite(case: ModuleCase, dtype: torch.dtype) -> None:
    module, inputs, output = _forward(case, dtype)
    output.sum().backward()
    assert {bool(grad.isfinite().all()) for grad in _gradients(module, inputs)} == {True}
