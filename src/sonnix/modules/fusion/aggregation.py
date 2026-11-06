r"""Contain aggregation fusion modules."""

from __future__ import annotations

__all__ = ["AverageFusion", "MultiplicationFusion", "SumFusion"]

from typing import TYPE_CHECKING

from sonnix.modules.fusion.base import BaseFusion

if TYPE_CHECKING:
    import torch


class MultiplicationFusion(BaseFusion):
    r"""Implement a fusion layer that multiplies the inputs.

    Example:
        ```pycon
        >>> import torch
        >>> from sonnix.modules import MultiplicationFusion
        >>> module = MultiplicationFusion()
        >>> module
        MultiplicationFusion()
        >>> x1 = torch.tensor([[2.0, 3.0, 4.0], [5.0, 6.0, 7.0]], requires_grad=True)
        >>> x2 = torch.tensor([[12.0, 13.0, 14.0], [15.0, 16.0, 17.0]], requires_grad=True)
        >>> out = module(x1, x2)
        >>> out
        tensor([[ 24.,  39.,  56.],
                [ 75.,  96., 119.]], grad_fn=<MulBackward0>)
        >>> out.mean().backward()

        ```
    """

    def fuse(self, inputs: tuple[torch.Tensor, ...]) -> torch.Tensor:
        output = inputs[0]
        for xi in inputs[1:]:
            output = output.mul(xi)
        return output


class SumFusion(BaseFusion):
    r"""Implement a layer to sum the inputs.

    Example:
        ```pycon
        >>> import torch
        >>> from sonnix.modules import SumFusion
        >>> module = SumFusion()
        >>> module
        SumFusion()
        >>> x1 = torch.tensor([[2.0, 3.0, 4.0], [5.0, 6.0, 7.0]], requires_grad=True)
        >>> x2 = torch.tensor([[12.0, 13.0, 14.0], [15.0, 16.0, 17.0]], requires_grad=True)
        >>> out = module(x1, x2)
        >>> out
        tensor([[14., 16., 18.],
                [20., 22., 24.]], grad_fn=<AddBackward0>)
        >>> out.mean().backward()

        ```
    """

    def fuse(self, inputs: tuple[torch.Tensor, ...]) -> torch.Tensor:
        output = inputs[0]
        for x in inputs[1:]:
            output = output.add(x)
        return output


class AverageFusion(BaseFusion):
    r"""Implement a layer to average the inputs.

    Example:
        ```pycon
        >>> import torch
        >>> from sonnix.modules import AverageFusion
        >>> module = AverageFusion()
        >>> module
        AverageFusion()
        >>> x1 = torch.tensor([[2.0, 3.0, 4.0], [5.0, 6.0, 7.0]], requires_grad=True)
        >>> x2 = torch.tensor([[12.0, 13.0, 14.0], [15.0, 16.0, 17.0]], requires_grad=True)
        >>> out = module(x1, x2)
        >>> out
        tensor([[ 7.,  8.,  9.],
                [10., 11., 12.]], grad_fn=<DivBackward0>)
        >>> out.mean().backward()

        ```
    """

    def fuse(self, inputs: tuple[torch.Tensor, ...]) -> torch.Tensor:
        output = inputs[0]
        for x in inputs[1:]:
            output = output.add(x)
        return output.div(len(inputs))
