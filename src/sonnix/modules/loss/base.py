r"""Contain the base class of the loss modules."""

from __future__ import annotations

__all__ = ["BaseLoss"]

from typing import TYPE_CHECKING

from torch import nn

from sonnix.validation import validate_loss_reduction_strategy

if TYPE_CHECKING:
    from sonnix.functional.reduction import Reduction


class BaseLoss(nn.Module):
    r"""Define the base class of the loss modules that have a reduction.

    It validates the reduction strategy and stores it in the
    ``reduction`` attribute.

    Args:
        reduction: The reduction strategy: ``'none'``, ``'mean'``, ``'sum'`` or
            ``'batchmean'``. See `sonnix.functional.reduce_loss` for the details.

    Raises:
        ValueError: if the reduction is not valid.

    Example:
        ```pycon
        >>> import torch
        >>> from sonnix.modules.loss import BaseLoss
        >>> class MyLoss(BaseLoss):
        ...     def forward(self, prediction, target):
        ...         return prediction.sub(target).abs().mean()
        ...
        >>> criterion = MyLoss(reduction="sum")
        >>> criterion.reduction
        'sum'

        ```
    """

    def __init__(self, reduction: Reduction = "mean") -> None:
        super().__init__()
        validate_loss_reduction_strategy(reduction)
        self.reduction: Reduction = reduction

    def extra_repr(self) -> str:
        return f"reduction={self.reduction}"
