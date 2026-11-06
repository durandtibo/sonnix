r"""Contain loss reduction functions."""

from __future__ import annotations

__all__ = ["Reduction", "reduce_loss"]

from typing import TYPE_CHECKING, Literal

from sonnix.validation.loss import validate_loss_reduction_strategy

if TYPE_CHECKING:
    from torch import Tensor

Reduction = Literal["none", "mean", "sum", "batchmean"]
r"""The valid loss reduction strategies."""


def reduce_loss(tensor: Tensor, reduction: Reduction) -> Tensor:
    r"""Return the reduced loss.

    This function is designed to be used with loss functions.

    Args:
        tensor: The input tensor to reduce.
        reduction: The reduction strategy. The valid values are
            ``'mean'``, ``'none'``,  ``'sum'``, and ``'batchmean'``.
            ``'none'``: no reduction will be applied, ``'mean'``: the
            sum will be divided by the number of elements in the
            input, ``'sum'``: the output will be summed,
            ``'batchmean'``: the sum will be divided by the size of the
            first dimension.

    Returns:
        The reduced tensor. The shape of the tensor depends on the
            reduction strategy.

    Raises:
        ValueError: if the reduction is not valid.

    Example:
        ```pycon
        >>> import torch
        >>> from sonnix.functional import reduce_loss
        >>> tensor = torch.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]])
        >>> reduce_loss(tensor, "none")
        tensor([[0., 1., 2.],
                [3., 4., 5.]])
        >>> reduce_loss(tensor, "sum")
        tensor(15.)
        >>> reduce_loss(tensor, "mean")
        tensor(2.5000)

        ```
    """
    validate_loss_reduction_strategy(reduction)
    if reduction == "mean":
        return tensor.mean()
    if reduction == "sum":
        return tensor.sum()
    if reduction == "none":
        return tensor
    # reduction == "batchmean"
    if tensor.ndim == 0:
        msg = "'batchmean' reduction requires a tensor with at least one dimension"
        raise ValueError(msg)
    return tensor.sum() / tensor.size(0)
