r"""Contain numerically safe power functions."""

from __future__ import annotations

__all__ = ["safe_pow"]

import torch
from coola.validation import validate_ge


def safe_pow(x: torch.Tensor, exponent: float) -> torch.Tensor:
    r"""Compute ``x ** exponent`` for non-negative values with a finite
    gradient at zero.

    For an exponent in ``(0, 1)``, the derivative of the power is
    infinite at zero and ``inf * 0`` gives a ``NaN`` gradient. The
    zeros are masked before the power so the gradient is zero there.

    Args:
        x: The non-negative input tensor.
        exponent: The exponent, which must be non-negative.

    Returns:
        The tensor ``x ** exponent``, which is exactly ``0`` where
            ``x`` is ``0``, except for ``exponent=0`` where it is
            ``1`` everywhere.

    Raises:
        ValueError: if the exponent is negative.

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
    validate_ge(exponent, 0, name="exponent")
    if exponent == 0:
        return torch.ones_like(x)
    is_positive = x > 0
    return torch.where(is_positive, torch.where(is_positive, x, 1.0).pow(exponent), 0.0)
