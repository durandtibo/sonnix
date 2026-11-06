r"""Contain ``torch.nn.Module`` to clamp values."""

from __future__ import annotations

__all__ = ["Clamp"]

import torch


class Clamp(torch.nn.Module):
    r"""Implement a module to clamp all elements in input into the range
    ``[min, max]``.

    Args:
        min_val: The lower-bound of the range to be clamped to.
            ``None`` means there is no minimum value.
        max_val: The upper-bound of the range to be clamped to.
            ``None`` means there is no maximum value.

    Shape:
        - Input: ``(*)``, where ``*`` means any number of dimensions.
        - Output: ``(*)``, same shape as the input.

    Example:
        ```pycon
        >>> import torch
        >>> from sonnix.modules import Clamp
        >>> m = Clamp(min_val=-1, max_val=2)
        >>> m
        Clamp(min_val=-1, max_val=2)
        >>> out = m(torch.tensor([[-2.0, -1.0, 0.0], [1.0, 2.0, 3.0]]))
        >>> out
        tensor([[-1., -1.,  0.], [ 1.,  2.,  2.]])

        ```
    """

    def __init__(self, min_val: float | None = -1.0, max_val: float | None = 1.0) -> None:
        super().__init__()
        self._min = min_val
        self._max = max_val

    def extra_repr(self) -> str:
        return f"min_val={self._min}, max_val={self._max}"

    def forward(self, input: torch.Tensor) -> torch.Tensor:  # noqa: A002
        return input.clamp(min=self._min, max=self._max)
