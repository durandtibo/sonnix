r"""Contain ``torch.nn.Module``s to change tensor's shape."""

from __future__ import annotations

__all__ = ["Squeeze", "View"]


import torch
from coola.validation import validate_int
from torch import nn


class Squeeze(nn.Module):
    r"""Implement a ``torch.nn.Module`` to squeeze the input tensor.

    Args:
        dim: The dimension(s) to squeeze the input tensor. It can be
            an integer or a tuple of integers. If ``None``, all the
            dimensions of the input tensor of size 1 are removed.

    Raises:
        TypeError: If ``dim`` is not an integer, a tuple of integers or ``None``.

    Example:
        ```pycon
        >>> import torch
        >>> from sonnix.modules import Squeeze
        >>> m = Squeeze()
        >>> m
        Squeeze(dim=None)
        >>> out = m(torch.ones(2, 1, 3, 1))
        >>> out.shape
        torch.Size([2, 3])
        >>> Squeeze(dim=(1, 3))(torch.ones(2, 1, 3, 1)).shape
        torch.Size([2, 3])

        ```
    """

    def __init__(self, dim: int | tuple[int, ...] | list[int] | None = None) -> None:
        super().__init__()
        if dim is not None:
            for d in dim if isinstance(dim, (tuple, list)) else (dim,):
                validate_int(d, name="dim")
        self.dim: int | tuple[int, ...] | None = (
            tuple(dim) if isinstance(dim, (tuple, list)) else dim
        )

    def extra_repr(self) -> str:
        return f"dim={self.dim}"

    def forward(self, input: torch.Tensor) -> torch.Tensor:  # noqa: A002
        if self.dim is None:
            return input.squeeze()
        return input.squeeze(self.dim)


class View(nn.Module):
    r"""Implement a ``torch.nn.Module`` to return a new tensor with the
    same data as the input tensor but of a different shape.

    It uses ``torch.reshape``, so it also works with non-contiguous
    inputs (the data are copied if needed).

    Args:
        shape: The desired shape.

    Example:
        ```pycon
        >>> import torch
        >>> from sonnix.modules import View
        >>> m = View(shape=(-1, 2, 3))
        >>> m
        View(shape=(-1, 2, 3))
        >>> out = m(torch.ones(4, 5, 2, 3))
        >>> out.shape
        torch.Size([20, 2, 3])

        ```
    """

    def __init__(self, shape: tuple[int, ...] | list[int]) -> None:
        super().__init__()
        self.shape = tuple(shape)

    def extra_repr(self) -> str:
        return f"shape={self.shape}"

    def forward(self, input: torch.Tensor) -> torch.Tensor:  # noqa: A002
        return input.reshape(self.shape)
