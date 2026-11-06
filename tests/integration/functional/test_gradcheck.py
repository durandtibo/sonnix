from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

import pytest
import torch

from sonnix import functional as F  # noqa: N812

if TYPE_CHECKING:
    from collections.abc import Callable


def _generator() -> torch.Generator:
    return torch.Generator().manual_seed(42)


def _normal() -> tuple[torch.Tensor, torch.Tensor]:
    generator = _generator()
    return (
        torch.randn(3, 4, generator=generator, dtype=torch.float64),
        torch.randn(3, 4, generator=generator, dtype=torch.float64),
    )


def _positive() -> tuple[torch.Tensor, torch.Tensor]:
    generator = _generator()
    return (
        torch.rand(3, 4, generator=generator, dtype=torch.float64) + 0.5,
        torch.rand(3, 4, generator=generator, dtype=torch.float64) + 0.5,
    )


def _normal_target() -> tuple[torch.Tensor, torch.Tensor]:
    prediction, target = _normal()
    return prediction, target.sigmoid()


def _probability() -> tuple[torch.Tensor, torch.Tensor]:
    generator = _generator()
    return (
        torch.rand(3, 4, generator=generator, dtype=torch.float64) * 0.8 + 0.1,
        torch.rand(3, 4, generator=generator, dtype=torch.float64),
    )


# (function, factory of the inputs). The gradient is checked with respect to
# the first input (the prediction). The inputs are random, so none of them is
# at a non-differentiable point.
LOSSES: dict[str, tuple[Callable[..., torch.Tensor], Callable[[], tuple[torch.Tensor, ...]]]] = {
    "asinh_mse_loss": (F.asinh_mse_loss, _normal),
    "asinh_smooth_l1_loss": (F.asinh_smooth_l1_loss, _normal),
    "binary_focal_loss": (F.binary_focal_loss, _probability),
    "binary_focal_loss_with_logits": (F.binary_focal_loss_with_logits, _normal_target),
    "binary_poly1_loss": (F.binary_poly1_loss, _probability),
    "binary_poly1_loss_with_logits": (F.binary_poly1_loss_with_logits, _normal_target),
    "general_robust_regression_loss": (F.general_robust_regression_loss, _normal),
    "general_robust_regression_loss_alpha_0": (
        partial(F.general_robust_regression_loss, alpha=0.0),
        _normal,
    ),
    "log_cosh_loss": (F.log_cosh_loss, _normal),
    "msle_loss": (F.msle_loss, _positive),
    "poisson_regression_loss": (F.poisson_regression_loss, _positive),
    "quantile_regression_loss": (F.quantile_regression_loss, _normal),
    "relative_loss": (F.relative_loss, _positive),
    "absolute_error": (F.absolute_error, _normal),
    "absolute_relative_error": (F.absolute_relative_error, _positive),
    "symmetric_absolute_relative_error": (F.symmetric_absolute_relative_error, _positive),
    "arithmetical_mean_indicator": (F.arithmetical_mean_indicator, _positive),
    "classical_relative_indicator": (F.classical_relative_indicator, _positive),
    "geometric_mean_indicator": (F.geometric_mean_indicator, _positive),
    "maximum_mean_indicator": (F.maximum_mean_indicator, _positive),
    "minimum_mean_indicator": (F.minimum_mean_indicator, _positive),
    "moment_mean_indicator": (F.moment_mean_indicator, _positive),
    "reversed_relative_indicator": (F.reversed_relative_indicator, _positive),
}

ACTIVATIONS: dict[str, Callable[..., torch.Tensor]] = {
    "rectifier_asinh_unit": F.rectifier_asinh_unit,
    "safe_exp": F.safe_exp,
    "safe_exp_preserve_gradient": partial(F.safe_exp, preserve_gradient=True),
    "safe_log": F.safe_log,
    "safe_log_preserve_gradient": partial(F.safe_log, preserve_gradient=True),
}


@pytest.mark.parametrize("name", LOSSES)
def test_gradcheck_loss(name: str) -> None:
    function, make_inputs = LOSSES[name]
    prediction, target = make_inputs()
    prediction.requires_grad_()
    assert torch.autograd.gradcheck(lambda x: function(x, target), (prediction,))


@pytest.mark.parametrize("name", ACTIVATIONS)
def test_gradcheck_activation(name: str) -> None:
    x = _positive()[0].requires_grad_()
    assert torch.autograd.gradcheck(ACTIVATIONS[name], (x,))


# ``classical_relative_indicator`` returns the target, so it does not depend on the prediction.
DEPEND_ON_PREDICTION = [name for name in LOSSES if name != "classical_relative_indicator"]


@pytest.mark.parametrize("name", DEPEND_ON_PREDICTION)
def test_loss_gradient_is_finite_when_prediction_equals_target(name: str) -> None:
    function, make_inputs = LOSSES[name]
    target = make_inputs()[1]
    prediction = target.clone().requires_grad_()
    function(prediction, target).sum().backward()
    assert prediction.grad.isfinite().all()
