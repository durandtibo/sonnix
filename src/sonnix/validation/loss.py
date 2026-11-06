r"""Contain functions to validate the arguments of the losses."""

from __future__ import annotations

__all__ = [
    "VALID_REDUCTIONS",
    "validate_focal_params",
    "validate_general_robust_params",
    "validate_loss_reduction_strategy",
    "validate_moment_order",
]

import math

from coola.validation import validate_ge, validate_gt, validate_in, validate_int, validate_range

VALID_REDUCTIONS: tuple[str, ...] = ("none", "mean", "sum", "batchmean")
r"""The valid loss reduction strategies."""


def validate_focal_params(alpha: float | None, gamma: float) -> None:
    r"""Validate the parameters of the focal loss.

    Args:
        alpha: The weighting factor. It has to be in the range
            ``[0, 1]``, or ``None`` to disable the weighting.
        gamma: The focusing parameter. It has to be non-negative.

    Raises:
        ValueError: if one of the parameters is invalid.

    Example:
        ```pycon
        >>> from sonnix.validation import validate_focal_params
        >>> validate_focal_params(alpha=0.25, gamma=2.0)

        ```
    """
    if alpha is not None:
        validate_range(alpha, 0, 1, name="alpha")
    validate_ge(gamma, 0, name="gamma")


def validate_general_robust_params(alpha: float, scale: float, max_val: float | None) -> None:
    r"""Validate the parameters of the general robust regression loss.

    Args:
        alpha: The shape parameter. It cannot be ``NaN`` or ``+inf``
            (``-inf`` is the Welsch limit).
        scale: The scale parameter. It has to be strictly positive.
        max_val: The max value to clip the loss, or ``None`` to
            disable the clipping. It has to be strictly positive.

    Raises:
        ValueError: if one of the parameters is invalid.

    Example:
        ```pycon
        >>> from sonnix.validation import validate_general_robust_params
        >>> validate_general_robust_params(alpha=1.0, scale=1.0, max_val=None)

        ```
    """
    if math.isnan(alpha) or alpha == math.inf:
        msg = f"alpha cannot be NaN or +inf (received: {alpha})"
        raise ValueError(msg)
    validate_gt(scale, 0, name="scale")
    if max_val is not None:
        validate_gt(max_val, 0, name="max_val")


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


def validate_moment_order(k: int) -> None:
    r"""Validate the order of the moment mean indicator.

    Args:
        k: The order.

    Raises:
        TypeError: if the order is not an integer.
        ValueError: if the order is lower than 1.

    Example:
        ```pycon
        >>> from sonnix.validation import validate_moment_order
        >>> validate_moment_order(2)

        ```
    """
    validate_int(k, name="k")
    validate_ge(k, 1, name="k")
