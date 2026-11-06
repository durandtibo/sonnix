r"""Contain functions to validate function/method arguments.

The validators follow the conventions of ``coola.validation``: they
return ``None`` when the value is valid, a value of the wrong type
raises ``TypeError``, a value of the right type that breaks the
constraint raises ``ValueError``, and the message names the arguments
and the values received.
"""

from __future__ import annotations

__all__ = [
    "VALID_REDUCTIONS",
    "Reduction",
    "validate_abs_range",
    "validate_focal_params",
    "validate_frequency",
    "validate_general_robust_params",
    "validate_loss_reduction_strategy",
    "validate_moment_order",
    "validate_same_shape",
]

from sonnix.validation.loss import (
    VALID_REDUCTIONS,
    Reduction,
    validate_focal_params,
    validate_general_robust_params,
    validate_loss_reduction_strategy,
    validate_moment_order,
)
from sonnix.validation.numerical import validate_abs_range, validate_frequency
from sonnix.validation.tensor import validate_same_shape
