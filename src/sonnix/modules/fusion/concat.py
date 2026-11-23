r"""Contain concatenation fusion modules."""

from __future__ import annotations

__all__ = ["ConcatFusion"]

import torch
from coola.validation import validate_int

from sonnix.modules.fusion.base import BaseFusion


class ConcatFusion(BaseFusion):
    r"""Implement a module to concatenate inputs.

    Args:
        dim: The fusion dimension. ``-1`` means the last dimension.

    Raises:
        TypeError: if ``dim`` is not an integer.

    Example:
        ```pycon
        >>> import torch
        >>> from sonnix.modules import ConcatFusion
        >>> module = ConcatFusion()
        >>> module
        ConcatFusion(dim=-1)
        >>> x1 = torch.tensor([[2.0, 3.0, 4.0], [5.0, 6.0, 7.0]], requires_grad=True)
        >>> x2 = torch.tensor([[12.0, 13.0, 14.0], [15.0, 16.0, 17.0]], requires_grad=True)
        >>> out = module(x1, x2)
        >>> out
        tensor([[ 2.,  3.,  4., 12., 13., 14.],
                [ 5.,  6.,  7., 15., 16., 17.]], grad_fn=<CatBackward0>)
        >>> out.mean().backward()

        ```
    """

    _same_shape = False

    def __init__(self, dim: int = -1) -> None:
        super().__init__()
        validate_int(dim, name="dim")
        self.dim = dim

    def extra_repr(self) -> str:
        return f"dim={self.dim}"

    def fuse(self, inputs: tuple[torch.Tensor, ...]) -> torch.Tensor:
        return torch.cat(inputs, dim=self.dim)
