r"""Contain ``torch.nn.Module``s to change tensor's data type."""

from __future__ import annotations

__all__ = ["ToDtype"]

import torch
from coola.validation import validate_in, validate_isinstance
from torch import nn

_DTYPE_NAMES = tuple(name for name in dir(torch) if isinstance(getattr(torch, name), torch.dtype))
r"""The names of the ``torch.dtype`` objects, as ``torch`` exposes them."""


class ToDtype(nn.Module):
    r"""Implement a ``torch.nn.Module`` to convert a tensor to a given
    data type.

    Args:
        dtype: The target data type, as a ``torch.dtype`` or as its
            name (for example ``"float32"`` or ``"int64"``). A name
            can be written in a configuration, which cannot contain a
            ``torch.dtype``. The attribute ``dtype`` is always a
            ``torch.dtype``.

    Shape:
        - Input: ``(*)``, where ``*`` means any number of dimensions.
        - Output: ``(*)``, same shape as the input.

    Raises:
        TypeError: if ``dtype`` is not a ``torch.dtype`` or a string.
        ValueError: if ``dtype`` is a string that is not the name of a
            ``torch.dtype``.

    Example:
        ```pycon
        >>> import torch
        >>> from sonnix.modules import ToDtype
        >>> m = ToDtype(torch.float64)
        >>> m
        ToDtype(dtype=torch.float64)
        >>> out = m(torch.tensor([[2, -1, 0], [1, 2, 3]]))
        >>> out
        tensor([[ 2., -1.,  0.],
                [ 1.,  2.,  3.]], dtype=torch.float64)
        >>> ToDtype("int64")
        ToDtype(dtype=torch.int64)

        ```
    """

    def __init__(self, dtype: torch.dtype | str) -> None:
        super().__init__()
        validate_isinstance(dtype, (torch.dtype, str), name="dtype")
        if isinstance(dtype, str):
            validate_in(dtype, _DTYPE_NAMES, name="dtype")
        self.dtype: torch.dtype = getattr(torch, dtype) if isinstance(dtype, str) else dtype

    def extra_repr(self) -> str:
        return f"dtype={self.dtype}"

    def forward(self, input: torch.Tensor) -> torch.Tensor:  # noqa: A002
        return input.to(dtype=self.dtype)
