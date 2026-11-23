from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import torch
from coola.equality import objects_are_equal
from torch import nn

import sonnix.functional as sf
from tests.registry import MODULE_CASES, ModuleCase

if TYPE_CHECKING:
    from collections.abc import Callable

# The module of a function of ``sonnix.functional`` stores the hyperparameters and calls the
# function (property P1). Each entry calls the function with the hyperparameters of the case of
# the registry.
PARITY: dict[str, Callable[..., torch.Tensor]] = {
    "BinaryFocalLoss": sf.binary_focal_loss,
    "BinaryFocalLossWithLogits": sf.binary_focal_loss_with_logits,
    "BinaryPoly1Loss": sf.binary_poly1_loss,
    "BinaryPoly1LossWithLogits": sf.binary_poly1_loss_with_logits,
    "GeneralRobustRegressionLoss": lambda p, t: sf.general_robust_regression_loss(p, t, alpha=1.0),
    "LogCoshLoss": sf.log_cosh_loss,
    "PoissonRegressionLoss": sf.poisson_regression_loss,
    "QuantileRegressionLoss": sf.quantile_regression_loss,
    "RectifierAsinhUnit": sf.rectifier_asinh_unit,
    "RelativeLoss": lambda p, t: sf.relative_loss(
        nn.L1Loss(reduction="none")(p, t), sf.classical_relative_indicator(p, t)
    ),
    "SafeExp": sf.safe_exp,
    "SafeLog": sf.safe_log,
    "SafePow": lambda x: sf.safe_pow(x, 0.5),
}
# The losses that wrap another criterion: they have no function.
NO_FUNCTION = {"TransformedLoss"}
CASES = [case for case in MODULE_CASES if case.name in PARITY]

##################################
#     Tests for the registry     #
##################################


def test_every_loss_has_a_function() -> None:
    losses = {case.name for case in MODULE_CASES if case.name.endswith("Loss")}
    assert losses - NO_FUNCTION - set(PARITY) == set()


def test_parity_names_are_registered() -> None:
    assert set(PARITY) <= {case.name for case in MODULE_CASES}


#############################################
#     Tests for the module and function     #
#############################################


@pytest.mark.parametrize("case", CASES, ids=repr)
def test_module_output_equals_function_output(case: ModuleCase) -> None:
    inputs = case.make_inputs()
    assert objects_are_equal(case.build()(*inputs), PARITY[case.name](*inputs))
