r"""Contain functions to validate the arguments of the numerical
encoders."""

from __future__ import annotations

__all__ = ["validate_abs_range", "validate_frequency"]

from coola.validation import validate_ge, validate_gt


def validate_abs_range(min_abs_value: float, max_abs_value: float) -> None:
    r"""Validate the value range parameters.

    Args:
        min_abs_value: The minimum absolute value to encode.
        max_abs_value: The maximum absolute value to encode.

    Raises:
        ValueError: if ``min_abs_value`` is not strictly positive or
            if ``max_abs_value`` is lower than ``min_abs_value``.

    Example:
        ```pycon
        >>> from sonnix.validation import validate_abs_range
        >>> validate_abs_range(min_abs_value=0.1, max_abs_value=10.0)

        ```
    """
    validate_gt(min_abs_value, 0, name="min_abs_value")
    if max_abs_value < min_abs_value:
        msg = (
            f"'max_abs_value' (received: {max_abs_value}) has to be greater than "
            f"'min_abs_value' (received: {min_abs_value})"
        )
        raise ValueError(msg)


def validate_frequency(num_frequencies: int, min_frequency: float, max_frequency: float) -> None:
    r"""Validate the frequency parameters.

    Args:
        num_frequencies: The number of frequencies.
        min_frequency: The minimum frequency.
        max_frequency: The maximum frequency.

    Raises:
        ValueError: if ``num_frequencies`` is lower than 1, if
            ``min_frequency`` is not strictly positive or if
            ``max_frequency`` is lower than ``min_frequency``.

    Example:
        ```pycon
        >>> from sonnix.validation import validate_frequency
        >>> validate_frequency(num_frequencies=3, min_frequency=0.1, max_frequency=1.0)

        ```
    """
    validate_ge(num_frequencies, 1, name="num_frequencies")
    validate_gt(min_frequency, 0, name="min_frequency")
    if max_frequency < min_frequency:
        msg = (
            f"'max_frequency' (received: {max_frequency}) has to be greater than "
            f"'min_frequency' (received: {min_frequency})"
        )
        raise ValueError(msg)
