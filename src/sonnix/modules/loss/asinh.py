r"""Contain the loss functions that use the inverse hyperbolic sine
(asinh) to transform the predictions or targets."""

from __future__ import annotations

from coola.validation import validate_ge

__all__ = ["AsinhMSELoss", "AsinhSmoothL1Loss"]

from typing import TYPE_CHECKING

from sonnix.functional import asinh_mse_loss, asinh_smooth_l1_loss
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
        reduction: The reduction strategy: ``'none'``, ``'mean'``, ``'sum'`` or
            ``'batchmean'``. See `sonnix.functional.reduce_loss` for the details.

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
        reduction: The reduction strategy: ``'none'``, ``'mean'``, ``'sum'`` or
            ``'batchmean'``. See `sonnix.functional.reduce_loss` for the details.

    Raises:
        ValueError: if ``beta`` is negative.

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
        validate_ge(beta, 0, name="beta")
        self.beta = float(beta)

    def extra_repr(self) -> str:
        return f"{super().extra_repr()}, beta={self.beta}"

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return asinh_smooth_l1_loss(prediction, target, reduction=self.reduction, beta=self.beta)
