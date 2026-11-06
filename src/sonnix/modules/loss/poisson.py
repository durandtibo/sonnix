r"""Contain the implementation of the poisson regression loss
function."""

from __future__ import annotations

from coola.validation import validate_gt

__all__ = ["PoissonRegressionLoss"]

from typing import TYPE_CHECKING

from sonnix.functional.loss import poisson_regression_loss
from sonnix.modules.loss.base import BaseLoss

if TYPE_CHECKING:
    import torch

    from sonnix.functional.reduction import Reduction


class PoissonRegressionLoss(BaseLoss):
    r"""Implement a loss module that computes the Poisson regression
    loss.

    Loss Functions and Metrics in Deep Learning
    https://arxiv.org/pdf/2307.02694

    Args:
        eps: An arbitrary small strictly positive number to avoid
            undefined results when the count is zero.
        reduction: The reduction strategy. The valid values are
            ``'mean'``, ``'none'``, ``'sum'``,
            and ``'batchmean'``.
            ``'none'``: no reduction will be applied, ``'mean'``: the
            sum will be divided by the number of elements in the
            input, ``'sum'``: the output will be summed,
            ``'batchmean'``: the sum will be divided by the size of the
            first dimension.

    Raises:
        ValueError: if ``eps`` is not strictly positive.

    Example:
        ```pycon
        >>> import torch
        >>> from sonnix.modules import PoissonRegressionLoss
        >>> criterion = PoissonRegressionLoss()
        >>> criterion
        PoissonRegressionLoss(reduction=mean, eps=1e-08)
        >>> loss = criterion(torch.randn(2, 4, requires_grad=True), torch.randint(0, 5, (2, 4)).float())
        >>> loss
        tensor(..., grad_fn=<MeanBackward0>)
        >>> loss.backward()

        ```
    """

    def __init__(self, eps: float = 1e-8, *, reduction: Reduction = "mean") -> None:
        super().__init__(reduction=reduction)
        validate_gt(eps, 0, name="eps")
        self.eps = float(eps)

    def extra_repr(self) -> str:
        return f"reduction={self.reduction}, eps={self.eps}"

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return poisson_regression_loss(prediction, target, reduction=self.reduction, eps=self.eps)
