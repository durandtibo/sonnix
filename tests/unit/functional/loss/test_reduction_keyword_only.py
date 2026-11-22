from __future__ import annotations

import inspect
from typing import TYPE_CHECKING

import pytest

from sonnix.functional import (
    binary_focal_loss,
    binary_focal_loss_with_logits,
    binary_poly1_loss,
    binary_poly1_loss_with_logits,
    general_robust_regression_loss,
    log_cosh_loss,
    poisson_regression_loss,
    quantile_regression_loss,
    relative_loss,
)

if TYPE_CHECKING:
    from collections.abc import Callable

#################################################
#     Tests for the signature of the losses     #
#################################################

LOSSES = [
    binary_focal_loss,
    binary_focal_loss_with_logits,
    binary_poly1_loss,
    binary_poly1_loss_with_logits,
    general_robust_regression_loss,
    log_cosh_loss,
    poisson_regression_loss,
    quantile_regression_loss,
    relative_loss,
]


@pytest.mark.parametrize("loss", LOSSES)
def test_loss_reduction_is_keyword_only(loss: Callable) -> None:
    parameters = inspect.signature(loss).parameters
    assert parameters["reduction"].kind is inspect.Parameter.KEYWORD_ONLY


@pytest.mark.parametrize("loss", LOSSES)
def test_loss_reduction_is_last_parameter(loss: Callable) -> None:
    assert list(inspect.signature(loss).parameters)[-1] == "reduction"
