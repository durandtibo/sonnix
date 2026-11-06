from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest
import torch

import sonnix.functional as sf
import sonnix.modules as sm

if TYPE_CHECKING:
    from collections.abc import Callable

    from torch import nn

# (id, function, module class, invalid keyword arguments, error message)
CASES = [
    (
        "asinh_smooth_l1_beta",
        sf.asinh_smooth_l1_loss,
        sm.AsinhSmoothL1Loss,
        {"beta": -1.0},
        r"beta must be greater than or equal to 0",
    ),
    (
        "focal_alpha",
        sf.binary_focal_loss,
        sm.BinaryFocalLoss,
        {"alpha": 2.0},
        r"alpha must be in the range",
    ),
    (
        "focal_gamma",
        sf.binary_focal_loss,
        sm.BinaryFocalLoss,
        {"gamma": -1.0},
        r"gamma must be greater than or equal to 0",
    ),
    (
        "focal_logits_alpha",
        sf.binary_focal_loss_with_logits,
        sm.BinaryFocalLossWithLogits,
        {"alpha": 2.0},
        r"alpha must be in the range",
    ),
    (
        "poisson_eps",
        sf.poisson_regression_loss,
        sm.PoissonRegressionLoss,
        {"eps": 0.0},
        r"eps must be greater than 0",
    ),
    (
        "log_cosh_scale",
        sf.log_cosh_loss,
        sm.LogCoshLoss,
        {"scale": 0.0},
        r"scale must be greater than 0",
    ),
    (
        "general_robust_max_val",
        sf.general_robust_regression_loss,
        sm.GeneralRobustRegressionLoss,
        {"max_val": -1.0},
        r"max_val must be greater than 0",
    ),
    (
        "general_robust_scale",
        sf.general_robust_regression_loss,
        sm.GeneralRobustRegressionLoss,
        {"scale": 0.0},
        r"scale must be greater than 0",
    ),
    (
        "general_robust_alpha",
        sf.general_robust_regression_loss,
        sm.GeneralRobustRegressionLoss,
        {"alpha": float("nan")},
        r"alpha cannot be NaN or \+inf",
    ),
    (
        "quantile_q",
        sf.quantile_regression_loss,
        sm.QuantileRegressionLoss,
        {"q": 2.0},
        r"q must be in the range",
    ),
    (
        "log_cosh_reduction",
        sf.log_cosh_loss,
        sm.LogCoshLoss,
        {"reduction": "bad"},
        r"reduction must be one of",
    ),
    (
        "msle_reduction",
        sf.msle_loss,
        sm.MSLELoss,
        {"reduction": "bad"},
        r"reduction must be one of",
    ),
]
IDS = [case[0] for case in CASES]


#########################################################
#     Tests that the functions and modules validate     #
#########################################################


@pytest.mark.parametrize("case", CASES, ids=IDS)
def test_function_raises(case: tuple[str, Callable[..., Any], type[nn.Module], dict, str]) -> None:
    _, function, _, kwargs, message = case
    with pytest.raises(ValueError, match=message):
        function(torch.ones(2, 3), torch.ones(2, 3), **kwargs)


@pytest.mark.parametrize("case", CASES, ids=IDS)
def test_module_raises(case: tuple[str, Callable[..., Any], type[nn.Module], dict, str]) -> None:
    _, _, module_class, kwargs, message = case
    with pytest.raises(ValueError, match=message):
        module_class(**kwargs)
