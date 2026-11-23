from __future__ import annotations

import copy
from itertools import product

import pytest
import torch

from sonnix.modules import SafeExp, SafeLog
from tests.registry import EDGE_VALUES, FUNCTION_CASES, MODULE_CASES, ModuleCase

DTYPES = (torch.float16, torch.bfloat16)
CASES_AND_EDGES = [
    pytest.param(case, edge, id=f"{case.name}-{edge}")
    for case, edge in product((*MODULE_CASES, *FUNCTION_CASES), EDGE_VALUES)
    if case.differentiable and edge in case.edges
]


def _make_edge_inputs(case: ModuleCase, edge: str, dtype: torch.dtype) -> tuple[torch.Tensor, ...]:
    # Every input that is not fixed is filled with the edge value, rounded to ``dtype``. The
    # second input has the opposite sign, so that the difference of a prediction and a target is
    # not zero.
    return tuple(
        x.to(dtype)
        if i in case.fixed
        else torch.full_like(x, EDGE_VALUES[edge] * (-1) ** i).to(dtype)
        for i, x in enumerate(case.make_inputs())
    )


def _run(
    module: torch.nn.Module, inputs: tuple[torch.Tensor, ...]
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    # Return the output and the gradients of the parameters and of the floating inputs.
    inputs = tuple(x.detach().requires_grad_() if x.is_floating_point() else x for x in inputs)
    output = module(*inputs)
    output.backward(torch.ones_like(output))
    tensors = (*module.parameters(), *inputs)
    return output.detach(), [t.grad for t in tensors if t.grad is not None]


def _is_representable(value: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    # A result that is finite in float32 and far enough from the maximum of the dtype.
    return value.isfinite() & (value.abs() < torch.finfo(dtype).max / 4)


######################################################
#     Tests for the finite values on edge inputs     #
######################################################


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize(("case", "edge"), CASES_AND_EDGES)
def test_finite_wherever_float32_fits(case: ModuleCase, edge: str, dtype: torch.dtype) -> None:
    # The rule of the property P6: where the float32 result is finite and representable in
    # ``dtype``, the result in ``dtype`` is finite. It excludes by construction what the
    # dtype cannot represent (the ``Square`` of ``1000`` in ``float16``).
    reference = case.build()
    inputs = _make_edge_inputs(case, edge, dtype)
    out_ref, grads_ref = _run(reference, tuple(x.float() for x in inputs))
    out, grads = _run(copy.deepcopy(reference).to(dtype), inputs)
    finite = [_is_representable(out_ref, dtype).logical_not() | out.isfinite()]
    finite += [
        _is_representable(g_ref, dtype).logical_not() | g.isfinite()
        for g_ref, g in zip(grads_ref, grads, strict=True)
    ]
    assert all(bool(x.all()) for x in finite)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("edge", EDGE_VALUES)
@pytest.mark.parametrize("module", [SafeExp, SafeLog])
def test_finite_wherever_float32_fits_with_preserve_gradient(
    module: type[torch.nn.Module], edge: str, dtype: torch.dtype
) -> None:
    # The option ``preserve_gradient`` changes the gradient only: the output stays finite.
    x = torch.full((2, 3), EDGE_VALUES[edge], dtype=dtype)
    x[0] = -x[0]
    assert module(preserve_gradient=True)(x).isfinite().all()


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("module", [SafeExp, SafeLog])
def test_preserve_gradient_keeps_the_value_on_an_infinite_input(
    module: type[torch.nn.Module], dtype: torch.dtype
) -> None:
    x = torch.tensor([-torch.inf, torch.inf], dtype=dtype)
    assert module(preserve_gradient=True)(x).equal(module()(x))
