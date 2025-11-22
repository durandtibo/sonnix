r"""Contain functional implementation of some modules."""

from __future__ import annotations

__all__ = [
    "absolute_error",
    "absolute_relative_error",
    "check_loss_reduction_strategy",
    "reduce_loss",
    "symmetric_absolute_relative_error",
]

from sonnix.functional.error import (
    absolute_error,
    absolute_relative_error,
    symmetric_absolute_relative_error,
)
from sonnix.functional.reduction import check_loss_reduction_strategy, reduce_loss
