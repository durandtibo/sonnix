r"""Contain the loss functions that use the inverse hyperbolic sine
(asinh) to transform the predictions or targets."""

from __future__ import annotations

__all__ = ["AsinhMSELoss", "AsinhSmoothL1Loss"]

from typing import TYPE_CHECKING

from sonnix.functional.loss.asinh import asinh_mse_loss, asinh_smooth_l1_loss
from sonnix.modules.loss.base import BaseLoss

if TYPE_CHECKING:
    import torch

    from sonnix.functional.reduction import Reduction


class AsinhMSELoss(BaseLoss):
    r"""Implement a loss module that computes the mean squared error
    (MSE) on the inverse hyperbolic sine (asinh) transformed predictions
    and targets.

    It is a generalization of mean squared logarithmic error (MSLE)
    that works for real values. The ``asinh`` transformation is used
    instead of ``log1p`` because ``asinh`` works on negative values.

    Args:
        reduction: The reduction strategy. The valid values are
            ``'mean'``, ``'none'``, ``'sum'``,
            and ``'batchmean'``.
            ``'none'``: no reduction will be applied, ``'mean'``: the
            sum will be divided by the number of elements in the
            input, ``'sum'``: the output will be summed,
            ``'batchmean'``: the sum will be divided by the size of the
            first dimension.

    Example:
        ```pycon
        >>> import torch
        >>> from sonnix.modules import AsinhMSELoss
        >>> criterion = AsinhMSELoss()
        >>> criterion
        AsinhMSELoss(reduction=mean)
        >>> loss = criterion(torch.randn(2, 4, requires_grad=True), torch.randn(2, 4))
        >>> loss
        tensor(..., grad_fn=<MeanBackward0>)
        >>> loss.backward()

        ```
    """

    def __init__(self, *, reduction: Reduction = "mean") -> None:
        super().__init__(reduction=reduction)

    def extra_repr(self) -> str:
        return f"reduction={self.reduction}"

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return asinh_mse_loss(prediction, target, reduction=self.reduction)


class AsinhSmoothL1Loss(BaseLoss):
    r"""Implement a loss module that computes the smooth L1 loss on the
    inverse hyperbolic sine (asinh) transformed predictions and targets.

    It is a generalization of mean squared logarithmic error (MSLE)
    that works for real values. The ``asinh`` transformation is used
    instead of ``log1p`` because ``asinh`` works on negative values.

    Args:
        beta: The threshold at which to change between L1 and L2 loss.
            The value must be non-negative.
        reduction: The reduction strategy. The valid values are
            ``'mean'``, ``'none'``, ``'sum'``,
            and ``'batchmean'``.
            ``'none'``: no reduction will be applied, ``'mean'``: the
            sum will be divided by the number of elements in the
            input, ``'sum'``: the output will be summed,
            ``'batchmean'``: the sum will be divided by the size of the
            first dimension.

    Example:
        ```pycon
        >>> import torch
        >>> from sonnix.modules import AsinhSmoothL1Loss
        >>> criterion = AsinhSmoothL1Loss()
        >>> criterion
        AsinhSmoothL1Loss(reduction=mean, beta=1.0)
        >>> loss = criterion(torch.randn(2, 4, requires_grad=True), torch.randn(2, 4))
        >>> loss
        tensor(..., grad_fn=<MeanBackward0>)
        >>> loss.backward()

        ```
    """

    def __init__(self, beta: float = 1.0, *, reduction: Reduction = "mean") -> None:
        super().__init__(reduction=reduction)
        self._beta = float(beta)

    def extra_repr(self) -> str:
        return f"reduction={self.reduction}, beta={self._beta}"

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return asinh_smooth_l1_loss(prediction, target, reduction=self.reduction, beta=self._beta)
