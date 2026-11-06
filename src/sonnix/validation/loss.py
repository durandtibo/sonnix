r"""Contain functions to validate the arguments of the losses."""

from __future__ import annotations

__all__ = [
    "VALID_REDUCTIONS",
    "validate_focal_params",
    "validate_loss_reduction_strategy",
    "validate_moment_order",
]

from coola.validation import validate_ge, validate_in, validate_range

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
        ValueError: if the order is not a positive integer.

    Example:
        ```pycon
        >>> from sonnix.validation import validate_moment_order
        >>> validate_moment_order(2)

        ```
    """
    validate_ge(k, 1, name="k")
