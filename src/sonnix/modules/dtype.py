r"""Contain ``torch.nn.Module``s to change tensor's data type."""

from __future__ import annotations

__all__ = ["ToDtype", "ToFloat", "ToLong"]

import torch
from coola.validation import validate_isinstance
from torch import nn


class ToDtype(nn.Module):
    r"""Implement a ``torch.nn.Module`` to convert a tensor to a given
    data type.

    Args:
        dtype: The target data type.

    Shape:
        - Input: ``(*)``, where ``*`` means any number of dimensions.
        - Output: ``(*)``, same shape as the input.

    Raises:
        TypeError: if ``dtype`` is not a ``torch.dtype``.

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

        ```
    """

    def __init__(self, dtype: torch.dtype) -> None:
        super().__init__()
        validate_isinstance(dtype, torch.dtype, name="dtype")
        self.dtype = dtype

    def extra_repr(self) -> str:
        return f"dtype={self.dtype}"

    def forward(self, input: torch.Tensor) -> torch.Tensor:  # noqa: A002
        return input.to(dtype=self.dtype)


class ToFloat(ToDtype):
    r"""Implement a ``torch.nn.Module`` to convert a tensor to a float
    tensor (``torch.float32``).

    Shape:
        - Input: ``(*)``, where ``*`` means any number of dimensions.
        - Output: ``(*)``, same shape as the input.

    Example:
        ```pycon
        >>> import torch
        >>> from sonnix.modules import ToFloat
        >>> m = ToFloat()
        >>> m
        ToFloat(dtype=torch.float32)
        >>> out = m(torch.tensor([[2, -1, 0], [1, 2, 3]]))
        >>> out
        tensor([[ 2., -1.,  0.],
                [ 1.,  2.,  3.]])

        ```
    """

    def __init__(self) -> None:
        super().__init__(dtype=torch.float32)


class ToLong(ToDtype):
    r"""Implement a ``torch.nn.Module`` to convert a tensor to a long
    tensor (``torch.int64``).

    Shape:
        - Input: ``(*)``, where ``*`` means any number of dimensions.
        - Output: ``(*)``, same shape as the input.

    Example:
        ```pycon
        >>> import torch
        >>> from sonnix.modules import ToLong
        >>> m = ToLong()
        >>> m
        ToLong(dtype=torch.int64)
        >>> out = m(torch.tensor([[2.0, -1.0, 0.0], [1.0, 2.0, 3.0]]))
        >>> out
        tensor([[ 2, -1,  0],
                [ 1,  2,  3]])

        ```
    """

    def __init__(self) -> None:
        super().__init__(dtype=torch.int64)
