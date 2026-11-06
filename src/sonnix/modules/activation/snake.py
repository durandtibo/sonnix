r"""Contain the Snake activation module."""

from __future__ import annotations

__all__ = ["Snake"]

from typing import TYPE_CHECKING

from sonnix.modules.activation.alpha import BaseAlphaActivation

if TYPE_CHECKING:
    import torch


class Snake(BaseAlphaActivation):
    r"""Implement the Snake activation layer.

    Formula: ``x + sin(alpha * x) ** 2 / alpha``

    Snake was proposed in the following paper:

        Neural Networks Fail to Learn Periodic Functions and How to Fix It.
        Ziyin L., Hartwig T., Ueda M.
        NeurIPS, 2020. (http://arxiv.org/pdf/2006.08195)

    Args:
        num_parameters: The number of learnable parameters. Although
            it takes an integer as input, only two values are
            legitimate: ``1``, or the number of channels at input.
        init: The initial value of the parameter(s) ``alpha``, which
            is the frequency of the periodic component. It has to be
            different from 0.
        learnable: If ``True``, the parameters are learned during the
            training, otherwise they are fixed.
        channel_dim: The dimension of the input that ``alpha`` is
            broadcast over when ``num_parameters`` is greater than 1.
            The default ``-1`` is the last dimension.
        device: The device of the parameters.
        dtype: The data type of the parameters. ``None`` uses the
            default floating point type.

    Shape:
        - Input: ``(*)``, where ``*`` means any number of dimensions.
        - Output: ``(*)``, same shape as the input.

    Raises:
        ValueError: if ``num_parameters`` is lower than 1 or ``init``
            is 0.

    Example:
        ```pycon
        >>> import torch
        >>> from sonnix.modules import Snake
        >>> m = Snake()
        >>> m
        Snake(num_parameters=1, learnable=True)
        >>> out = m(torch.arange(8, dtype=torch.float).view(2, 4))
        >>> out
        tensor([[0.0000, 1.7081, 2.8268, 3.0199],
                [4.5728, 5.9195, 6.0781, 7.4316]], grad_fn=<AddBackward0>)

        ```
    """

    allow_zero_init = False

    def forward(self, input: torch.Tensor) -> torch.Tensor:  # noqa: A002
        alpha = self.broadcast_alpha(input)
        return input + input.mul(alpha).sin().square().div(alpha)
