r"""Provide functional implementations of PyTorch modules and layers.

This subpackage contains pure function implementations of various neural
network operations including activation functions, error calculations,
and loss functions that can be used directly without instantiating
module objects.
"""

from __future__ import annotations

__all__ = [
    "Reduction",
    "absolute_error",
    "absolute_relative_error",
    "arithmetical_mean_indicator",
    "binary_focal_loss",
    "binary_focal_loss_with_logits",
    "binary_poly1_loss",
    "binary_poly1_loss_with_logits",
    "classical_relative_indicator",
    "general_robust_regression_loss",
    "geometric_mean_indicator",
    "log_cosh_loss",
    "maximum_mean_indicator",
    "minimum_mean_indicator",
    "moment_mean_indicator",
    "poisson_regression_loss",
    "quantile_regression_loss",
    "rectifier_asinh_unit",
    "reduce_loss",
    "relative_loss",
    "reversed_relative_indicator",
    "safe_exp",
    "safe_log",
    "safe_pow",
    "symmetric_absolute_relative_error",
]

from sonnix.functional.activation import rectifier_asinh_unit, safe_exp, safe_log
from sonnix.functional.error import (
    absolute_error,
    absolute_relative_error,
    symmetric_absolute_relative_error,
)
from sonnix.functional.loss.focal import (
    binary_focal_loss,
    binary_focal_loss_with_logits,
)
from sonnix.functional.loss.general_robust import general_robust_regression_loss
from sonnix.functional.loss.log import log_cosh_loss
from sonnix.functional.loss.poisson import poisson_regression_loss
from sonnix.functional.loss.poly import binary_poly1_loss, binary_poly1_loss_with_logits
from sonnix.functional.loss.quantile import quantile_regression_loss
from sonnix.functional.loss.relative import (
    arithmetical_mean_indicator,
    classical_relative_indicator,
    geometric_mean_indicator,
    maximum_mean_indicator,
    minimum_mean_indicator,
    moment_mean_indicator,
    relative_loss,
    reversed_relative_indicator,
)
from sonnix.functional.power import safe_pow
from sonnix.functional.reduction import Reduction, reduce_loss
