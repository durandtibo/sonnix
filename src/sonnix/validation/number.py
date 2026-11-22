r"""Contain functions to validate numbers."""

from __future__ import annotations

__all__ = ["validate_positive"]

from coola.validation import validate_finite, validate_gt


def validate_positive(value: float, *, name: str = "value") -> None:
    r"""Validate that a value is a finite number strictly greater than 0.

    Args:
        value: The value to validate.
        name: The name of the value, used in the error message.

    Raises:
        TypeError: If ``value`` is not a real number (a ``bool`` is
            rejected).
        ValueError: If ``value`` is ``NaN``, infinite, or not strictly
            positive.

    Example:
        ```pycon
        >>> from sonnix.validation import validate_positive
        >>> validate_positive(1e-8, name="eps")

        ```
    """
    validate_finite(value, name=name)
    validate_gt(value, 0, name=name)
