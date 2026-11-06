r"""Contain modules to compute N separate linear layers."""

from __future__ import annotations

__all__ = ["NLinear"]

import math
from typing import TYPE_CHECKING

import torch
from torch import nn
from torch.nn import init

if TYPE_CHECKING:
    from torch import Tensor


class NLinear(nn.Module):
    r"""Implement N separate linear layers.

    Technically, ``NLinear(n, in, out)`` is just a layout of ``n``
    linear layers ``torch.nn.Linear(in, out)``.

    Args:
        n: The number of separate linear layers.
        in_features: The size of each input sample.
        out_features: The size of each output sample.
        bias: If set to ``False``, the layer will not learn an
            additive bias.
        device: The device of the parameters.
        dtype: The data type of the parameters.

    Shape:
        - Input: ``(*, n, in_features)``, where ``*`` means any number of
            dimensions.
        - Output: ``(*, n, out_features)``,  where ``*`` has
            the same shape as the input.

    Example:
        ```pycon
        >>> import torch
        >>> from sonnix.modules import NLinear
        >>> m = NLinear(n=3, in_features=4, out_features=6)
        >>> m
        NLinear(n=3, in_features=4, out_features=6, bias=True)
        >>> out = m(torch.randn(2, 3, 4))
        >>> out.shape
        torch.Size([2, 3, 6])
        >>> out = m(torch.randn(2, 5, 3, 4))
        >>> out.shape
        torch.Size([2, 5, 3, 6])

        ```
    """

    def __init__(
        self,
        n: int,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.weight: nn.Parameter = nn.Parameter(
            torch.empty((n, in_features, out_features), device=device, dtype=dtype)
        )
        if bias:
            self.bias: nn.Parameter = nn.Parameter(
                torch.empty(n, out_features, device=device, dtype=dtype)
            )
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    @property
    def n(self) -> int:
        return self.weight.shape[0]

    @property
    def in_features(self) -> int:
        return self.weight.shape[1]

    @property
    def out_features(self) -> int:
        return self.weight.shape[2]

    def reset_parameters(self) -> None:
        # Same initialization as ``torch.nn.Linear``: uniform in
        # [-1/sqrt(in_features), 1/sqrt(in_features)] for both parameters.
        bound = 1 / math.sqrt(self.in_features) if self.in_features > 0 else 0
        init.uniform_(self.weight, -bound, bound)
        if self.bias is not None:
            init.uniform_(self.bias, -bound, bound)

    def forward(self, x: Tensor) -> Tensor:
        shape = x.shape
        x = x.reshape(-1, shape[-2], shape[-1]).transpose(0, 1)  # (n, batch, in)
        x = torch.bmm(x, self.weight).transpose(0, 1)  # (batch, n, out)
        if self.bias is not None:
            x = x + self.bias
        return x.reshape(*shape[:-1], self.out_features)

    def extra_repr(self) -> str:
        return (
            f"n={self.n}, in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}"
        )
