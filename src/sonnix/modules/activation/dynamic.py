r"""Contain ``torch.nn.Module``s that implements the Dynamic Tanh (DyT).

Paper:
Transformers without Normalization. CVPR 2025.
https://arxiv.org/pdf/2503.10622
"""

from __future__ import annotations

__all__ = ["BaseDynamicNorm", "DynamicAsinh", "DynamicTanh"]

from abc import ABC, abstractmethod

import torch
from torch import nn
from torch.nn import init

from sonnix.modules.scale_and_shift import ScaleAndShift


class BaseDynamicNorm(ScaleAndShift, ABC):
    r"""Defines the base class to implement a dynamic normalization layer
    like Dynamic Tanh (DyT).

    It is a ``ScaleAndShift`` layer with a learnable scalar ``alpha``
    and a nonlinearity applied to ``alpha * x``. The subclasses
    implement ``forward``.

    Args:
        normalized_shape: The input shape to normalize.
            If a single integer is used, it is treated as a singleton
            list, and this module will normalize over the last
            dimension which is expected to be of that specific size.
        alpha_init_value: The initial value for alpha.
        bias: If ``False``, the module has no learnable shift
            ``beta``.
        device: The device of the parameters.
        dtype: The data type of the parameters. ``None`` uses the
            default floating point type.

    Shape:
        - Input: ``(N, *)``, where ``*`` means any number of dimensions.
        - Output: ``(N, *)``, same shape as the input.

    Example:
        ```pycon
        >>> import torch
        >>> from sonnix.modules.activation.dynamic import BaseDynamicNorm
        >>> class MyDynamicNorm(BaseDynamicNorm):
        ...     def forward(self, input: torch.Tensor) -> torch.Tensor:
        ...         return self._affine(input.mul(self.alpha).tanh())
        ...
        >>> m = MyDynamicNorm(normalized_shape=5)
        >>> m
        MyDynamicNorm(normalized_shape=(5,), bias=True)

        ```
    """

    def __init__(
        self,
        normalized_shape: int | list[int] | tuple[int, ...],
        alpha_init_value: float = 0.5,
        *,
        bias: bool = True,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__(normalized_shape, bias=bias, device=device, dtype=dtype)
        self.alpha_init_value = alpha_init_value
        self.alpha: nn.Parameter = nn.Parameter(torch.empty(1, device=device, dtype=dtype))
        self.reset_parameters()

    @abstractmethod
    def forward(self, input: torch.Tensor) -> torch.Tensor:  # noqa: A002
        r"""Apply the dynamic normalization.

        Args:
            input: The input tensor.

        Returns:
            The output tensor, which has the same shape as the input.
        """

    def reset_parameters(self) -> None:
        r"""Reset the parameters: ``alpha`` to ``alpha_init_value``,
        ``gamma`` to 1 and ``beta`` to 0."""
        super().reset_parameters()
        init.constant_(self.alpha, self.alpha_init_value)


class DynamicAsinh(BaseDynamicNorm):
    r"""Applies the Dynamic Asinh normalization over a mini-batch of
    inputs.

    This layer implements the following operation:

    y = gamma * asinh(alpha * x) + beta

    Args:
        normalized_shape: The input shape to normalize.
            If a single integer is used, it is treated as a singleton
            list, and this module will normalize over the last
            dimension which is expected to be of that specific size.
        alpha_init_value: The initial value for alpha.
        bias: If ``False``, the module has no learnable shift
            ``beta``.
        device: The device of the parameters.
        dtype: The data type of the parameters. ``None`` uses the
            default floating point type.

    Shape:
        - Input: ``(N, *)``, where ``*`` means any number of dimensions.
        - Output: ``(N, *)``, same shape as the input.

    Example:
        ```pycon
        >>> import torch
        >>> from sonnix.modules import DynamicAsinh
        >>> m = DynamicAsinh(normalized_shape=5)
        >>> m
        DynamicAsinh(normalized_shape=(5,), bias=True)
        >>> out = m(torch.tensor([[-2, -1, 0, 1, 2], [3, 2, 1, 2, 3]]))
        >>> out
        tensor([[-0.8814, -0.4812,  0.0000,  0.4812,  0.8814],
                [ 1.1948,  0.8814,  0.4812,  0.8814,  1.1948]], grad_fn=<AddBackward0>)

        ```
    """

    def forward(self, input: torch.Tensor) -> torch.Tensor:  # noqa: A002
        return self._affine(input.mul(self.alpha).asinh())


class DynamicTanh(BaseDynamicNorm):
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
            list, and this module will normalize over the last
            dimension which is expected to be of that specific size.
        alpha_init_value: The initial value for alpha.
        bias: If ``False``, the module has no learnable shift
            ``beta``.
        device: The device of the parameters.
        dtype: The data type of the parameters. ``None`` uses the
            default floating point type.

    Shape:
        - Input: ``(N, *)``, where ``*`` means any number of dimensions.
        - Output: ``(N, *)``, same shape as the input.

    Example:
        ```pycon
        >>> import torch
        >>> from sonnix.modules import DynamicTanh
        >>> m = DynamicTanh(normalized_shape=5)
        >>> m
        DynamicTanh(normalized_shape=(5,), bias=True)
        >>> out = m(torch.tensor([[-2, -1, 0, 1, 2], [3, 2, 1, 2, 3]]))
        >>> out
        tensor([[-0.7616, -0.4621,  0.0000,  0.4621,  0.7616],
                [ 0.9051,  0.7616,  0.4621,  0.7616,  0.9051]], grad_fn=<AddBackward0>)

        ```
    """

    def forward(self, input: torch.Tensor) -> torch.Tensor:  # noqa: A002
        return self._affine(input.mul(self.alpha).tanh())
