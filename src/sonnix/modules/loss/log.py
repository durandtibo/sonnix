r"""Contain the loss modules that use the logarithmic function to
transform the predictions or targets."""

from __future__ import annotations

__all__ = ["LogCoshLoss", "MSLELoss"]

from typing import TYPE_CHECKING

from coola.validation import validate_gt

from sonnix.functional import log_cosh_loss, msle_loss
from sonnix.modules.loss.base import BaseLoss

if TYPE_CHECKING:
    import torch

    from sonnix.functional.reduction import Reduction


class LogCoshLoss(BaseLoss):
    r"""Implement a loss module that computes the logarithm of the
    hyperbolic cosine of the prediction error.

    The loss is ``scale * log(cosh(error / scale))``. It behaves like
    the MSE for small errors and like the MAE for large errors.

    Args:
        scale: The scale factor. The value must be greater than 0.
            See ``sonnix.functional.log_cosh_loss``.
        reduction: The reduction strategy: ``'none'``, ``'mean'``, ``'sum'`` or
            ``'batchmean'``. See `sonnix.functional.reduce_loss` for the details.

    Raises:
        ValueError: if ``scale`` is not greater than 0 or if the
            reduction is not valid.

    Example:
        ```pycon
        >>> import torch
        >>> from sonnix.modules import LogCoshLoss
        >>> criterion = LogCoshLoss()
        >>> criterion
        LogCoshLoss(reduction=mean, scale=1.0)
        >>> loss = criterion(torch.randn(2, 4, requires_grad=True), torch.randn(2, 4))
        >>> loss
        tensor(..., grad_fn=<MeanBackward0>)
        >>> loss.backward()

        ```
    """

    def __init__(self, scale: float = 1.0, *, reduction: Reduction = "mean") -> None:
        super().__init__(reduction=reduction)
        validate_gt(scale, 0, name="scale")
        self.scale = float(scale)

    def extra_repr(self) -> str:
        return f"{super().extra_repr()}, scale={self.scale}"

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return log_cosh_loss(prediction, target, scale=self.scale, reduction=self.reduction)


class MSLELoss(BaseLoss):
    r"""Implement a loss module that computes the mean squared error
    (MSE) on the logarithmic transformed predictions and targets.

    The inputs must be greater than or equal to ``-1`` (``log1p`` is
    undefined below, the loss is ``nan``). See
    ``sonnix.functional.msle_loss``.

    Args:
        reduction: The reduction strategy: ``'none'``, ``'mean'``, ``'sum'`` or
            ``'batchmean'``. See `sonnix.functional.reduce_loss` for the details.

    Raises:
        ValueError: if the reduction is not valid.

    Example:
        ```pycon
        >>> import torch
        >>> from sonnix.modules import MSLELoss
        >>> criterion = MSLELoss()
        >>> criterion
        MSLELoss(reduction=mean)
        >>> loss = criterion(torch.rand(2, 4, requires_grad=True), torch.rand(2, 4))
        >>> loss
        tensor(..., grad_fn=<MeanBackward0>)
        >>> loss.backward()

        ```
    """

    def __init__(self, *, reduction: Reduction = "mean") -> None:
        super().__init__(reduction=reduction)

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return msle_loss(prediction, target, reduction=self.reduction)
