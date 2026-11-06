r"""Contain helper functions shared by the numerical encoders."""

from __future__ import annotations

__all__ = ["check_abs_range", "check_frequency", "prepare_tensor_param"]

from typing import TYPE_CHECKING

from coola.validation import validate_ge, validate_gt

if TYPE_CHECKING:
    from torch import Tensor


def prepare_tensor_param(tensor: Tensor, name: str) -> Tensor:
    r"""Prepare a tensor parameter to be a 2d tensor.

    Args:
        tensor: The tensor to prepare.
        name: The name associated to the tensor.

    Returns:
        The prepared tensor.

    Raises:
        ValueError: if the input tensor is not a 1d or 2d tensor.

    Example:
        ```pycon
        >>> import torch
        >>> from sonnix.modules.numerical._utils import prepare_tensor_param
        >>> prepare_tensor_param(torch.tensor([1.0, 2.0, 4.0]), name="scale")
        tensor([[1., 2., 4.]])

        ```
    """
    if tensor.ndim == 1:
        tensor = tensor.view(1, -1)
    if tensor.ndim != 2:
        msg = f"Incorrect shape for '{name}': {tensor.shape}"
        raise ValueError(msg)
    return tensor


def check_abs_range(min_abs_value: float, max_abs_value: float) -> None:
    r"""Check the value range parameters.

    Args:
        min_abs_value: The minimum absolute value to encode.
        max_abs_value: The maximum absolute value to encode.

    Raises:
        ValueError: if ``min_abs_value`` is not strictly positive or
            if ``max_abs_value`` is lower than ``min_abs_value``.

    Example:
        ```pycon
        >>> from sonnix.modules.numerical._utils import check_abs_range
        >>> check_abs_range(min_abs_value=0.1, max_abs_value=10.0)

        ```
    """
    validate_gt(min_abs_value, 0, name="min_abs_value")
    if max_abs_value < min_abs_value:
        msg = (
            f"'max_abs_value' (received: {max_abs_value}) has to be greater than "
            f"'min_abs_value' (received: {min_abs_value})"
        )
        raise ValueError(msg)


def check_frequency(num_frequencies: int, min_frequency: float, max_frequency: float) -> None:
    r"""Check the frequency parameters.

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
        >>> from sonnix.modules.numerical._utils import check_frequency
        >>> check_frequency(num_frequencies=3, min_frequency=0.1, max_frequency=1.0)

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
