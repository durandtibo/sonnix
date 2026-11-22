r"""Contain activation modules using hyperbolic functions."""

from __future__ import annotations

__all__ = ["RectifierAsinhUnit"]

import torch
from torch import nn

from sonnix.functional import rectifier_asinh_unit


class RectifierAsinhUnit(nn.Module):
    r"""Implement a ``torch.nn.Module`` to compute the inverse hyperbolic
    sine (arcsinh) of the positive elements, and zero for the negative
    elements.

    Shape:
        - Input: ``(*)``, where ``*`` means any number of dimensions.
        - Output: ``(*)``, same shape as the input.

    Example:
        ```pycon
        >>> import torch
        >>> from sonnix.modules import RectifierAsinhUnit
        >>> m = RectifierAsinhUnit()
        >>> m
        RectifierAsinhUnit()
        >>> out = m(torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 2.0, 4.0]]))
        >>> out
        tensor([[0.0000, 0.0000, 0.8814],
                [0.0000, 1.4436, 2.0947]])

        ```
    """

    def forward(self, input: torch.Tensor) -> torch.Tensor:  # noqa: A002
        return rectifier_asinh_unit(input)
