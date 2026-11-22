r"""Contain ``torch.nn.Module`` to clamp values."""

from __future__ import annotations

__all__ = ["Clamp"]

import torch
from coola.validation import validate_finite


class Clamp(torch.nn.Module):
    r"""Implement a module to clamp all elements in input into the range
    ``[min, max]``.

    Args:
        min_val: The lower-bound of the range to be clamped to.
            ``None`` means there is no minimum value.
        max_val: The upper-bound of the range to be clamped to.
            ``None`` means there is no maximum value.

    Raises:
        TypeError: if ``min_val`` or ``max_val`` is not a number.
        ValueError: if ``min_val`` or ``max_val`` is not finite, or if
            ``min_val`` is greater than ``max_val``.

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
        if min_val is not None:
            validate_finite(min_val, name="min_val")
        if max_val is not None:
            validate_finite(max_val, name="max_val")
        if min_val is not None and max_val is not None and min_val > max_val:
            msg = f"min_val ({min_val}) must be lower than or equal to max_val ({max_val})"
            raise ValueError(msg)
        self.min_val = min_val
        self.max_val = max_val

    def extra_repr(self) -> str:
        return f"min_val={self.min_val}, max_val={self.max_val}"

    def forward(self, input: torch.Tensor) -> torch.Tensor:  # noqa: A002
        return input.clamp(min=self.min_val, max=self.max_val)
