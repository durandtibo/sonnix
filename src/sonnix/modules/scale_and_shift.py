r"""Contain ``torch.nn.Module``s to compute scale and shift
transformations."""

from __future__ import annotations

__all__ = ["ScaleAndShift"]

import torch
from coola.validation import validate_ge, validate_int
from torch import nn
from torch.nn import init


class ScaleAndShift(nn.Module):
    r"""Applies a scale and shift transformation over a mini-batch of
    inputs.

    This layer implements the following operation:

    y = gamma * x + beta

    where ``beta`` is dropped if ``bias=False``.

    Args:
        normalized_shape: The input shape to normalize.
            If a single integer is used, it is treated as a singleton
            list, and this module will normalize over the last
            dimension which is expected to be of that specific size.
        bias: If ``False``, the module has no learnable shift
            ``beta``.
        device: The device of the parameters.
        dtype: The data type of the parameters. ``None`` uses the
            default floating point type.

    Raises:
        TypeError: if a dimension of ``normalized_shape`` is not an
            integer.
        ValueError: if a dimension of ``normalized_shape`` is lower
            than 1.

    Shape:
        - Input: ``(N, *)``, where ``*`` means any number of dimensions.
        - Output: ``(N, *)``, same shape as the input.

    Example:
        ```pycon
        >>> import torch
        >>> from sonnix.modules import ScaleAndShift
        >>> m = ScaleAndShift(normalized_shape=5)
        >>> m
        ScaleAndShift(normalized_shape=(5,), bias=True)
        >>> out = m(torch.tensor([[-2, -1, 0, 1, 2], [3, 2, 1, 2, 3]]))
        >>> out
        tensor([[-2., -1.,  0.,  1.,  2.],
                [ 3.,  2.,  1.,  2.,  3.]], grad_fn=<AddBackward0>)

        ```
    """

    def __init__(
        self,
        normalized_shape: int | list[int] | tuple[int, ...],
        *,
        bias: bool = True,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        for dim_size in self.normalized_shape:
            validate_int(dim_size, name="normalized_shape")
            validate_ge(dim_size, 1, name="normalized_shape")

        self.weight: nn.Parameter = nn.Parameter(
            torch.empty(self.normalized_shape, device=device, dtype=dtype)
        )
        self.bias: nn.Parameter | None
        if bias:
            self.bias = nn.Parameter(torch.empty(self.normalized_shape, device=device, dtype=dtype))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def forward(self, input: torch.Tensor) -> torch.Tensor:  # noqa: A002
        out = input * self.weight
        if self.bias is not None:
            out = out + self.bias
        return out

    def extra_repr(self) -> str:
        return f"normalized_shape={self.normalized_shape}, bias={self.bias is not None}"

    def reset_parameters(self) -> None:
        r"""Reset the parameters: ``gamma`` to 1 and ``beta`` to 0."""
        init.ones_(self.weight)
        if self.bias is not None:
            init.zeros_(self.bias)
