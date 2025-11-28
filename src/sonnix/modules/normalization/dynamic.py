r"""Contain ``torch.nn.Module``s that implements the Dynamic Tanh (DyT).

Paper:
Transformers without Normalization. CVPR 2025.
https://arxiv.org/pdf/2503.10622
"""

from __future__ import annotations

__all__ = ["DynamicAsinh", "DynamicTanh"]

import torch
from torch import nn
from torch.nn import init


class DynamicAsinh(nn.Module):
    r"""Applies the Dynamic Asinh normalization over a mini-batch of
    inputs.

    This layer implements the following operation:

    y = gamma * asinh(alpha * x) + beta

    Args:
        normalized_shape: The input shape to normalize.
            If a single integer is used, it is treated as a singleton
            list, and this module willcnormalize over the last
            dimension which is expected to be of that specific size.
        alpha_init_value: The initial value for alpha.

    Shape:
        - Input: ``(N, *)``, where ``*`` means any number of dimensions.
        - Output: ``(N, *)``, same shape as the input.

    Example usage:

    ```pycon

    >>> import torch
    >>> from sonnix.modules import DynamicAsinh
    >>> m = DynamicAsinh(normalized_shape=5)
    >>> m
    DynamicAsinh(normalized_shape=(5,))
    >>> out = m(torch.tensor([[-2, -1, 0, 1, 2], [3, 2, 1, 2, 3]]))
    >>> out
    tensor([[-1.1752, -0.5211,  0.0000,  0.5211,  1.1752],
            [ 2.1293,  1.1752,  0.5211,  1.1752,  2.1293]], grad_fn=<AddBackward0>)

    ```
    """

    def __init__(
        self, normalized_shape: int | list[int] | tuple[int, ...], alpha_init_value: float = 0.5
    ) -> None:
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self._normalized_shape = tuple(normalized_shape)
        self._alpha_init_value = alpha_init_value

        self.alpha = nn.Parameter(torch.empty(1))
        self.weight = nn.Parameter(torch.empty(self._normalized_shape))
        self.bias = nn.Parameter(torch.empty(self._normalized_shape))
        self.reset_parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.asinh(self.alpha * x) * self.weight + self.bias

    def extra_repr(self) -> str:
        return f"normalized_shape={self._normalized_shape}"

    def reset_parameters(self) -> None:
        init.constant_(self.alpha, self._alpha_init_value)
        init.ones_(self.weight)
        init.zeros_(self.bias)


class DynamicTanh(nn.Module):
    r"""Applies the Dynamic Tanh normalization over a mini-batch of
    inputs.

    This layer implements the following operation:

    y = gamma * tanh(alpha * x) + beta

    Paper:
        Transformers without Normalization. CVPR 2025.
        https://arxiv.org/pdf/2503.10622

    Args:
        normalized_shape: The input shape to normalize.
            If a single integer is used, it is treated as a singleton
            list, and this module willcnormalize over the last
            dimension which is expected to be of that specific size.
        alpha_init_value: The initial value for alpha.

    Shape:
        - Input: ``(N, *)``, where ``*`` means any number of dimensions.
        - Output: ``(N, *)``, same shape as the input.

    Example usage:

    ```pycon

    >>> import torch
    >>> from sonnix.modules import DynamicTanh
    >>> m = DynamicTanh(normalized_shape=5)
    >>> m
    DynamicTanh(normalized_shape=(5,))
    >>> out = m(torch.tensor([[-2, -1, 0, 1, 2], [3, 2, 1, 2, 3]]))
    >>> out
    tensor([[-0.7616, -0.4621,  0.0000,  0.4621,  0.7616],
            [ 0.9051,  0.7616,  0.4621,  0.7616,  0.9051]], grad_fn=<AddBackward0>)

    ```
    """

    def __init__(
        self, normalized_shape: int | list[int] | tuple[int, ...], alpha_init_value: float = 0.5
    ) -> None:
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self._normalized_shape = tuple(normalized_shape)
        self._alpha_init_value = alpha_init_value

        self.alpha = nn.Parameter(torch.empty(1))
        self.weight = nn.Parameter(torch.empty(self._normalized_shape))
        self.bias = nn.Parameter(torch.empty(self._normalized_shape))
        self.reset_parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.alpha * x) * self.weight + self.bias

    def extra_repr(self) -> str:
        return f"normalized_shape={self._normalized_shape}"

    def reset_parameters(self) -> None:
        init.constant_(self.alpha, self._alpha_init_value)
        init.ones_(self.weight)
        init.zeros_(self.bias)
