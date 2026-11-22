r"""Contain functional implementation of some activation layers built
with hyperbolic functions."""

from __future__ import annotations

__all__ = ["rectifier_asinh_unit"]

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


def rectifier_asinh_unit(input: torch.Tensor) -> torch.Tensor:  # noqa: A002
    r"""Compute the inverse hyperbolic sine (arcsinh) of the positive
    elements, and zero for the negative elements.

    Args:
        input: The input tensor.

    Returns:
        A tensor with inverse hyperbolic sine of the positive elements.

    Example:
        ```pycon
        >>> import torch
        >>> from sonnix.functional import rectifier_asinh_unit
        >>> output = rectifier_asinh_unit(torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0]))
        >>> output
        tensor([0.0000, 0.0000, 0.0000, 0.8814, 1.4436])

        ```
    """
    return input.clamp(min=0).asinh()
