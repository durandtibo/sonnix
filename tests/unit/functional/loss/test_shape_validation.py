r"""Test that the functions with two tensors of the same role require
the same shape (property P4)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import torch

import sonnix.functional as sf

if TYPE_CHECKING:
    from collections.abc import Callable

FUNCTIONS = [
    sf.asinh_mse_loss,
    sf.asinh_smooth_l1_loss,
    sf.binary_focal_loss,
    sf.binary_focal_loss_with_logits,
    sf.binary_poly1_loss,
    sf.binary_poly1_loss_with_logits,
    sf.general_robust_regression_loss,
    sf.log_cosh_loss,
    sf.msle_loss,
    sf.poisson_regression_loss,
    sf.quantile_regression_loss,
    sf.arithmetical_mean_indicator,
    sf.classical_relative_indicator,
    sf.geometric_mean_indicator,
    sf.maximum_mean_indicator,
    sf.minimum_mean_indicator,
    sf.moment_mean_indicator,
    sf.reversed_relative_indicator,
]


@pytest.mark.parametrize("function", FUNCTIONS, ids=lambda f: f.__name__)
def test_different_shapes_raise(function: Callable[..., torch.Tensor]) -> None:
    with pytest.raises(ValueError, match=r"prediction and target must have the same shape"):
        function(torch.full((4, 3), 0.5), torch.full((1, 3), 0.5))


@pytest.mark.parametrize("function", FUNCTIONS, ids=lambda f: f.__name__)
def test_column_vs_vector_raise(function: Callable[..., torch.Tensor]) -> None:
    with pytest.raises(ValueError, match=r"got \(4,\) and \(4, 1\)"):
        function(torch.full((4,), 0.5), torch.full((4, 1), 0.5))


@pytest.mark.parametrize("function", FUNCTIONS, ids=lambda f: f.__name__)
def test_same_shapes_do_not_raise(function: Callable[..., torch.Tensor]) -> None:
    assert function(torch.full((4, 3), 0.5), torch.full((4, 3), 0.5)).isfinite().all()


def test_quantile_regression_loss_does_not_broadcast() -> None:
    with pytest.raises(ValueError, match=r"got \(4, 3\) and \(1, 3\)"):
        sf.quantile_regression_loss(torch.ones(4, 3), torch.ones(1, 3))
