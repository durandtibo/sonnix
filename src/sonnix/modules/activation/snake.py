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

    The constructor arguments (``num_parameters``, ``init``, ``learnable``,
    ``channel_dim``, ``device`` and ``dtype``) are the ones of
    ``BaseAlphaActivation``. Here ``alpha`` is the frequency of the periodic component.

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

    def _validate_init(self, init: float) -> None:
        super()._validate_init(init)
        if init == 0:
            msg = f"'init' has to be different from 0 (received {init})"
            raise ValueError(msg)

    def forward(self, input: torch.Tensor) -> torch.Tensor:  # noqa: A002
        alpha = self._broadcast_alpha(input)
        return input + input.mul(alpha).sin().square().div(alpha)
