r"""Contain the implementation of the quantile regression loss
function."""

from __future__ import annotations

__all__ = ["QuantileRegressionLoss"]

from typing import TYPE_CHECKING

from coola.validation import validate_range

from sonnix.functional.loss import quantile_regression_loss
from sonnix.modules.loss.base import BaseLoss

if TYPE_CHECKING:
    import torch

    from sonnix.functional.reduction import Reduction


class QuantileRegressionLoss(BaseLoss):
    r"""Implement a loss module that computes the quantile regression
    loss.

    Loss Functions and Metrics in Deep Learning
    https://arxiv.org/pdf/2307.02694

    Args:
        q: The quantile value. ``q=0.5`` gives half of the Mean
            Absolute Error (MAE), because the loss is
            ``max(q * d, (q - 1) * d)`` with ``d = target - prediction``.
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
        >>> from sonnix.modules import QuantileRegressionLoss
        >>> criterion = QuantileRegressionLoss()
        >>> criterion
        QuantileRegressionLoss(reduction=mean, q=0.5)
        >>> loss = criterion(torch.randn(2, 4, requires_grad=True), torch.randn(2, 4))
        >>> loss
        tensor(..., grad_fn=<MeanBackward0>)
        >>> loss.backward()

        ```
    """

    def __init__(self, q: float = 0.5, *, reduction: Reduction = "mean") -> None:
        super().__init__(reduction=reduction)
        validate_range(q, 0.0, 1.0, name="q")
        self.q = float(q)

    def extra_repr(self) -> str:
        return f"{super().extra_repr()}, q={self.q}"

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return quantile_regression_loss(prediction, target, reduction=self.reduction, q=self.q)
