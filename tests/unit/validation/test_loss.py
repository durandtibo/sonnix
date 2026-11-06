from __future__ import annotations

from typing import get_args

import pytest

from sonnix.functional import Reduction
from sonnix.validation.loss import (
    VALID_REDUCTIONS,
    validate_focal_params,
    validate_general_robust_params,
    validate_loss_reduction_strategy,
    validate_moment_order,
)

######################################################
#     Tests for validate_loss_reduction_strategy     #
######################################################


@pytest.mark.parametrize("reduction", ["none", "mean", "sum", "batchmean"])
def test_validate_loss_reduction_strategy_valid(reduction: str) -> None:
    validate_loss_reduction_strategy(reduction)


def test_validate_loss_reduction_strategy_incorrect() -> None:
    with pytest.raises(ValueError, match=r"reduction must be one of"):
        validate_loss_reduction_strategy("incorrect")


def test_valid_reductions_match_reduction_type() -> None:
    assert get_args(Reduction) == VALID_REDUCTIONS


###########################################
#     Tests for validate_moment_order     #
###########################################


@pytest.mark.parametrize("k", [1, 2, 5])
def test_validate_moment_order_valid(k: int) -> None:
    assert validate_moment_order(k) is None


@pytest.mark.parametrize("k", [0, -1])
def test_validate_moment_order_invalid(k: int) -> None:
    with pytest.raises(ValueError, match=r"k must be greater than or equal to 1"):
        validate_moment_order(k)


###########################################
#     Tests for validate_focal_params     #
###########################################


@pytest.mark.parametrize("alpha", [None, 0.0, 0.25, 1.0])
def test_validate_focal_params_valid_alpha(alpha: float | None) -> None:
    validate_focal_params(alpha=alpha, gamma=2.0)


@pytest.mark.parametrize("gamma", [0.0, 2.0])
def test_validate_focal_params_valid_gamma(gamma: float) -> None:
    validate_focal_params(alpha=0.25, gamma=gamma)


@pytest.mark.parametrize("alpha", [-0.1, 1.1])
def test_validate_focal_params_alpha_out_of_range(alpha: float) -> None:
    with pytest.raises(ValueError, match=r"alpha"):
        validate_focal_params(alpha=alpha, gamma=2.0)


def test_validate_focal_params_negative_gamma() -> None:
    with pytest.raises(ValueError, match=r"gamma"):
        validate_focal_params(alpha=0.25, gamma=-1.0)


################################################
#     Tests for validate_general_robust_params     #
################################################


@pytest.mark.parametrize("alpha", [float("-inf"), -2.0, 0.0, 1.0, 2.0, 100.0])
def test_validate_general_robust_params_valid_alpha(alpha: float) -> None:
    validate_general_robust_params(alpha=alpha, scale=1.0, max_val=None)


@pytest.mark.parametrize("alpha", [float("nan"), float("inf")])
def test_validate_general_robust_params_invalid_alpha(alpha: float) -> None:
    with pytest.raises(ValueError, match=r"alpha cannot be NaN or \+inf"):
        validate_general_robust_params(alpha=alpha, scale=1.0, max_val=None)


@pytest.mark.parametrize("scale", [0.0, -1.0])
def test_validate_general_robust_params_invalid_scale(scale: float) -> None:
    with pytest.raises(ValueError, match=r"scale must be greater than 0"):
        validate_general_robust_params(alpha=1.0, scale=scale, max_val=None)


@pytest.mark.parametrize("max_val", [None, 0.5, 10.0])
def test_validate_general_robust_params_valid_max_val(max_val: float | None) -> None:
    validate_general_robust_params(alpha=1.0, scale=1.0, max_val=max_val)


@pytest.mark.parametrize("max_val", [0.0, -1.0])
def test_validate_general_robust_params_invalid_max_val(max_val: float) -> None:
    with pytest.raises(ValueError, match=r"max_val must be greater than 0"):
        validate_general_robust_params(alpha=1.0, scale=1.0, max_val=max_val)
