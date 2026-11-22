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


def _scaled() -> tuple[torch.Tensor, torch.Tensor]:
    r"""Return seeded tensors with values in ``[-30, 30]``."""
    prediction, target = _large()
    return prediction / 10, target / 10


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
    "general_robust_regression_loss": (sf.general_robust_regression_loss, _scaled),
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


#########################################################
#     Tests of the functions in half precision          #
#########################################################


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
