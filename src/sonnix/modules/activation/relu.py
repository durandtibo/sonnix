r"""Contain relu-like activation modules."""

from __future__ import annotations

__all__ = ["SquaredReLU"]

from typing import TYPE_CHECKING

from torch import nn
from torch.nn import functional

if TYPE_CHECKING:
    import torch


class SquaredReLU(nn.Module):
    r"""Implement the Squared ReLU.

    Squared ReLU is defined in the following paper:

        Primer: Searching for Efficient Transformers for Language Modeling.
        So DR., Mańke W., Liu H., Dai Z., Shazeer N., Le QV.
        NeurIPS, 2021. (https://arxiv.org/pdf/2109.08668.pdf)

    Shape:
        - Input: ``(*)``, where ``*`` means any number of dimensions.
        - Output: ``(*)``, same shape as the input.

    Example:
        ```pycon
        >>> import torch
        >>> from sonnix.modules import SquaredReLU
        >>> m = SquaredReLU()
        >>> m
        SquaredReLU()
        >>> out = m(torch.arange(8, dtype=torch.float).view(2, 4))
        >>> out
        tensor([[ 0.,  1.,  4.,  9.],
                [16., 25., 36., 49.]])

        ```
    """

    def forward(self, input: torch.Tensor) -> torch.Tensor:  # noqa: A002
        x = functional.relu(input)
        return x * x
