r"""Contain numerically safe power functions."""

from __future__ import annotations

__all__ = ["safe_pow"]

import torch
from coola.validation import validate_finite, validate_ge


def safe_pow(input: torch.Tensor, exponent: float) -> torch.Tensor:  # noqa: A002
    r"""Compute ``input ** exponent`` for non-negative values with a finite
    gradient at zero.

    For an exponent in ``(0, 1)``, the derivative of the power is
    infinite at zero and ``inf * 0`` gives a ``NaN`` gradient. The
    zeros are masked before the power so the gradient is zero there.

    Args:
        input: The non-negative input tensor.
        exponent: The exponent, which must be non-negative.

    Returns:
        The tensor ``input ** exponent``, which is exactly ``0`` where
            ``input`` is ``0``, except for ``exponent=0`` where it is
            ``1`` everywhere.

    Raises:
        TypeError: if the exponent is not a number.
        ValueError: if the exponent is negative or not finite.

    Example:
        ```pycon
        >>> import torch
        >>> from sonnix.functional import safe_pow
        >>> x = torch.tensor([0.0, 4.0], requires_grad=True)
        >>> safe_pow(x, 0.5)
        tensor([0., 2.], grad_fn=<WhereBackward0>)
        >>> safe_pow(x, 0.5).sum().backward()
        >>> x.grad
        tensor([0.0000, 0.2500])

        ```
    """
    validate_finite(exponent, name="exponent")
    validate_ge(exponent, 0, name="exponent")
    if exponent == 0:
        return torch.ones_like(input)
    is_positive = input > 0
    return torch.where(is_positive, torch.where(is_positive, input, 1.0).pow(exponent), 0.0)
