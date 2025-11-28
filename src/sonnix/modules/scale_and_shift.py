r"""Contain ``torch.nn.Module``s to compute scale and shift
transformations."""

from __future__ import annotations

__all__ = ["ScaleAndShift"]

import torch
from torch import nn
from torch.nn import init


class ScaleAndShift(nn.Module):
    r"""Applies a scale and shift transformation over a mini-batch of
    inputs.

    This layer implements the following operation:

    y = gamma * x + beta

    Args:
        normalized_shape: The input shape to normalize.
            If a single integer is used, it is treated as a singleton
            list, and this module willcnormalize over the last
            dimension which is expected to be of that specific size.

    Shape:
        - Input: ``(N, *)``, where ``*`` means any number of dimensions.
        - Output: ``(N, *)``, same shape as the input.

    Example usage:

    ```pycon

    >>> import torch
    >>> from sonnix.modules import ScaleAndShift
    >>> m = ScaleAndShift(normalized_shape=5)
    >>> m
    ScaleAndShift(normalized_shape=(5,))
    >>> out = m(torch.tensor([[-2, -1, 0, 1, 2], [3, 2, 1, 2, 3]]))
    >>> out
    tensor([[-2., -1.,  0.,  1.,  2.],
            [ 3.,  2.,  1.,  2.,  3.]], grad_fn=<AddBackward0>)

    ```
    """

    def __init__(self, normalized_shape: int | list[int] | tuple[int, ...]) -> None:
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self._normalized_shape = tuple(normalized_shape)

        self.weight = nn.Parameter(torch.ones(self._normalized_shape))
        self.bias = nn.Parameter(torch.zeros(self._normalized_shape))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.weight + self.bias

    def extra_repr(self) -> str:
        return f"normalized_shape={self._normalized_shape}"

    def reset_parameters(self) -> None:
        init.ones_(self.weight)
        init.zeros_(self.bias)
