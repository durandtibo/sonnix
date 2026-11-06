r"""Contain the implementation of the poisson regression loss
function."""

from __future__ import annotations

__all__ = ["PoissonRegressionLoss"]

from typing import TYPE_CHECKING

from coola.validation import validate_gt

from sonnix.functional import poisson_regression_loss
from sonnix.modules.loss.base import BaseLoss

if TYPE_CHECKING:
    import torch

    from sonnix.functional.reduction import Reduction


class PoissonRegressionLoss(BaseLoss):
    r"""Implement a loss module that computes the Poisson regression
    loss.

    Loss Functions and Metrics in Deep Learning
    https://arxiv.org/pdf/2307.02694

    The targets must be non-negative counts (zero is allowed). If
    ``log_input=False``, the predictions must be positive (negative
    values are clamped to ``eps``). If ``log_input=True``, the
    predictions are the logarithm of the rate.

    Args:
        eps: An arbitrary small strictly positive number to avoid
            undefined results when the count is zero.
        log_input: If ``True``, the predictions are the logarithm of
            the rate and ``eps`` is not used.
        reduction: The reduction strategy: ``'none'``, ``'mean'``, ``'sum'`` or
            ``'batchmean'``. See `sonnix.functional.reduce_loss` for the details.

    Raises:
        ValueError: if ``eps`` is not strictly positive.

    Example:
        ```pycon
        >>> import torch
        >>> from sonnix.modules import PoissonRegressionLoss
        >>> criterion = PoissonRegressionLoss()
        >>> criterion
        PoissonRegressionLoss(reduction=mean, eps=1e-08, log_input=False)
        >>> loss = criterion(
        ...     torch.rand(2, 4, requires_grad=True), torch.randint(0, 5, (2, 4)).float()
        ... )
        >>> loss
        tensor(..., grad_fn=<MeanBackward0>)
        >>> loss.backward()

        ```
    """

    def __init__(
        self, eps: float = 1e-8, *, log_input: bool = False, reduction: Reduction = "mean"
    ) -> None:
        super().__init__(reduction=reduction)
        validate_gt(eps, 0, name="eps")
        self.eps = float(eps)
        self.log_input = bool(log_input)

    def extra_repr(self) -> str:
        return f"{super().extra_repr()}, eps={self.eps}, log_input={self.log_input}"

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return poisson_regression_loss(
            prediction,
            target,
            eps=self.eps,
            log_input=self.log_input,
            reduction=self.reduction,
        )
