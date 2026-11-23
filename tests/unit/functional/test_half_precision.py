from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import torch

import sonnix.functional as sf
from tests.helpers import make_generator

if TYPE_CHECKING:
    from collections.abc import Callable

HALF_DTYPES = [torch.float16, torch.bfloat16]


def _large() -> tuple[torch.Tensor, torch.Tensor]:
    r"""Return seeded tensors with values in ``[-300, 300]``."""
    generator = make_generator()
    return (
        torch.rand(4, 6, generator=generator) * 600 - 300,
        torch.rand(4, 6, generator=generator) * 600 - 300,
    )


def _positive() -> tuple[torch.Tensor, torch.Tensor]:
    r"""Return seeded tensors with values in ``[1, 301]``."""
    prediction, target = _large()
    return prediction.abs() + 1, target.abs() + 1


def _probability() -> tuple[torch.Tensor, torch.Tensor]:
    r"""Return seeded probabilities and binary targets."""
    generator = make_generator()
    return (
        torch.rand(4, 6, generator=generator) * 0.9 + 0.05,
        (torch.rand(4, 6, generator=generator) > 0.5).float(),
    )


def _logit() -> tuple[torch.Tensor, torch.Tensor]:
    r"""Return seeded logits and binary targets."""
    _, target = _probability()
    prediction, _ = _large()
    return prediction / 30, target


def _extreme() -> tuple[torch.Tensor, torch.Tensor]:
    r"""Return seeded tensors with values in ``[-1000, 1000]``."""
    prediction, target = _large()
    return prediction * (1000 / 300), target * (1000 / 300)


def _zeros() -> tuple[torch.Tensor, torch.Tensor]:
    r"""Return two tensors of zeros."""
    return torch.zeros(4, 6), torch.zeros(4, 6)


def _tiny() -> tuple[torch.Tensor, torch.Tensor]:
    r"""Return two tensors of ``1e-10``, which is ``0`` in half precision."""
    return torch.full((4, 6), 1e-10), torch.full((4, 6), 1e-10)


def _general_robust(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    # ``alpha=2`` is the squared error, which ``float16`` cannot hold for a difference of 300.
    return sf.general_robust_regression_loss(prediction, target, alpha=1.0)


def _relative_geometric(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return sf.relative_loss(
        sf.absolute_error(prediction, target), sf.geometric_mean_indicator(prediction, target)
    )


def _relative_moment(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return sf.relative_loss(
        sf.absolute_error(prediction, target), sf.moment_mean_indicator(prediction, target, k=2)
    )


# (function, inputs): the functions are compared in half precision and
# in float32 on large values.
CASES = {
    "absolute_error": (sf.absolute_error, _large),
    "absolute_relative_error": (sf.absolute_relative_error, _large),
    "symmetric_absolute_relative_error": (sf.symmetric_absolute_relative_error, _large),
    "log_cosh_loss": (sf.log_cosh_loss, _large),
    "general_robust_regression_loss": (_general_robust, _extreme),
    "poisson_regression_loss": (sf.poisson_regression_loss, _positive),
    "quantile_regression_loss": (sf.quantile_regression_loss, _large),
    "binary_focal_loss": (sf.binary_focal_loss, _probability),
    "binary_focal_loss_with_logits": (sf.binary_focal_loss_with_logits, _logit),
    "binary_poly1_loss": (sf.binary_poly1_loss, _probability),
    "binary_poly1_loss_with_logits": (sf.binary_poly1_loss_with_logits, _logit),
    "geometric_mean_indicator": (sf.geometric_mean_indicator, _large),
    "relative_loss_geometric": (_relative_geometric, _large),
    "relative_loss_moment": (_relative_moment, _large),
}


# The functions that accept any real input: they are also run on zeros and on tiny values.
EDGE_CASES = [
    "absolute_error",
    "absolute_relative_error",
    "symmetric_absolute_relative_error",
    "log_cosh_loss",
    "general_robust_regression_loss",
    "poisson_regression_loss",
    "quantile_regression_loss",
    "geometric_mean_indicator",
    "relative_loss_geometric",
    "relative_loss_moment",
]


#####################################################
#     Tests of the functions in half precision      #
#####################################################


@pytest.mark.parametrize("dtype", HALF_DTYPES)
@pytest.mark.parametrize("name", CASES)
def test_half_precision_is_finite(name: str, dtype: torch.dtype) -> None:
    function, make_inputs = CASES[name]
    prediction, target = make_inputs()
    assert function(prediction.to(dtype), target.to(dtype)).isfinite().all()


@pytest.mark.parametrize("dtype", HALF_DTYPES)
@pytest.mark.parametrize("name", CASES)
def test_half_precision_matches_float32(name: str, dtype: torch.dtype) -> None:
    function: Callable[..., torch.Tensor] = CASES[name][0]
    prediction, target = CASES[name][1]()
    expected = function(prediction, target)
    out = function(prediction.to(dtype), target.to(dtype))
    assert out.float().allclose(expected, rtol=5e-2, atol=expected.abs().max().item() * 1e-2)


@pytest.mark.parametrize("dtype", HALF_DTYPES)
@pytest.mark.parametrize("make_inputs", [_zeros, _tiny])
@pytest.mark.parametrize("name", EDGE_CASES)
def test_half_precision_edge_inputs_are_finite(
    name: str, make_inputs: Callable[[], tuple[torch.Tensor, torch.Tensor]], dtype: torch.dtype
) -> None:
    prediction, target = make_inputs()
    assert CASES[name][0](prediction.to(dtype), target.to(dtype)).isfinite().all()


@pytest.mark.parametrize("dtype", HALF_DTYPES)
def test_half_precision_general_robust_extreme_inputs_gradient_is_finite(
    dtype: torch.dtype,
) -> None:
    prediction, target = _extreme()
    prediction = prediction.to(dtype).requires_grad_()
    _general_robust(prediction, target.to(dtype)).backward()
    assert prediction.grad.isfinite().all()
