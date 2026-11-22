r"""Contain functions to validate the arguments of the losses."""

from __future__ import annotations

__all__ = [
    "VALID_REDUCTIONS",
    "Reduction",
    "validate_focal_params",
    "validate_general_robust_params",
    "validate_loss_reduction_strategy",
]

import math
from typing import Literal, get_args

from coola.validation import (
    validate_finite,
    validate_ge,
    validate_in,
    validate_number,
    validate_range,
)

from sonnix.validation.number import validate_positive

Reduction = Literal["none", "mean", "sum", "batchmean"]
r"""The valid loss reduction strategies."""

VALID_REDUCTIONS: tuple[str, ...] = get_args(Reduction)
r"""The valid loss reduction strategies, as a tuple."""


def validate_focal_params(alpha: float | None, gamma: float) -> None:
    r"""Validate the parameters of the focal loss.

    Args:
        alpha: The weighting factor. It has to be in the range
            ``[0, 1]``, or ``None`` to disable the weighting.
        gamma: The focusing parameter. It has to be finite and
            non-negative.

    Raises:
        TypeError: if one of the parameters is not a real number.
        ValueError: if one of the parameters is invalid.

    Example:
        ```pycon
        >>> from sonnix.validation import validate_focal_params
        >>> validate_focal_params(alpha=0.25, gamma=2.0)

        ```
    """
    if alpha is not None:
        validate_number(alpha, name="alpha")
        validate_range(alpha, 0, 1, name="alpha")
    validate_finite(gamma, name="gamma")
    validate_ge(gamma, 0, name="gamma")


def validate_general_robust_params(alpha: float, scale: float, max_val: float | None) -> None:
    r"""Validate the parameters of the general robust regression loss.

    Args:
        alpha: The shape parameter. It cannot be ``NaN`` or ``+inf``
            (``-inf`` is the Welsch limit).
        scale: The scale parameter. It has to be finite and strictly
            positive.
        max_val: The max value to clip the loss, or ``None`` to
            disable the clipping. It has to be finite and strictly
            positive.

    Raises:
        TypeError: if one of the parameters is not a real number.
        ValueError: if one of the parameters is invalid.

    Example:
        ```pycon
        >>> from sonnix.validation import validate_general_robust_params
        >>> validate_general_robust_params(alpha=1.0, scale=1.0, max_val=None)

        ```
    """
    validate_number(alpha, name="alpha")
    if math.isnan(alpha) or alpha == math.inf:
        msg = f"alpha cannot be NaN or +inf (received: {alpha})"
        raise ValueError(msg)
    validate_positive(scale, name="scale")
    if max_val is not None:
        validate_positive(max_val, name="max_val")


def validate_loss_reduction_strategy(reduction: str) -> None:
    r"""Validate that the reduction is a valid loss reduction.

    The valid reduction values are ``'mean'``, ``'none'``,  ``'sum'``,
    and ``'batchmean'``.

    Args:
        reduction: The reduction strategy to validate.

    Raises:
        ValueError: if the provided reduction is not valid.

    Example:
        ```pycon
        >>> from sonnix.validation import validate_loss_reduction_strategy
        >>> validate_loss_reduction_strategy("mean")

        ```
    """
    validate_in(reduction, VALID_REDUCTIONS, name="reduction")
