r"""Contain loss reduction functions."""

from __future__ import annotations

__all__ = ["Reduction", "reduce_loss"]

from typing import TYPE_CHECKING

from sonnix.validation.loss import Reduction, validate_loss_reduction_strategy

if TYPE_CHECKING:
    from torch import Tensor


def reduce_loss(loss: Tensor, *, reduction: Reduction = "mean") -> Tensor:
    r"""Return the reduced loss.

    This function is designed to be used with loss functions.

    Args:
        loss: The loss tensor to reduce.
        reduction: The reduction strategy. The default is ``'mean'``.
            The valid values are
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
        >>> reduce_loss(tensor, reduction="none")
        tensor([[0., 1., 2.],
                [3., 4., 5.]])
        >>> reduce_loss(tensor, reduction="sum")
        tensor(15.)
        >>> reduce_loss(tensor)
        tensor(2.5000)

        ```
    """
    validate_loss_reduction_strategy(reduction)
    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    if reduction == "none":
        return loss
    # reduction == "batchmean"
    if loss.ndim == 0:
        msg = "'batchmean' reduction requires a tensor with at least one dimension"
        raise ValueError(msg)
    return loss.sum() / loss.size(0)
