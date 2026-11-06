r"""Contain the general robust regression loss a.k.a the Barron robust
loss."""

from __future__ import annotations

__all__ = ["GeneralRobustRegressionLoss"]

from typing import TYPE_CHECKING

from coola.validation import validate_gt

from sonnix.functional import (
    general_robust_regression_loss,
)
from sonnix.modules.loss.base import BaseLoss

if TYPE_CHECKING:
    import torch

    from sonnix.functional.reduction import Reduction


class GeneralRobustRegressionLoss(BaseLoss):
    r"""Implement the general robust regression loss a.k.a. Barron robust
    loss.

    Based on the paper:

        A General and Adaptive Robust Loss Function
        Jonathan T. Barron
        CVPR 2019 (https://arxiv.org/abs/1701.03077)

    Note:
        The "adaptive" part of the loss is not implemented in this
        module.

    Args:
        alpha: The shape parameter that controls the robustness of the
            loss. It is a Python float, not a tensor. ``-inf`` is
            supported (Welsch loss). The limit cases ``alpha=2``
            (``x^2``) and ``alpha=0`` (``log(0.5 * x^2 + 1)``) are used
            when ``alpha`` is within ``1e-5`` of 2 or 0.
        scale: The scale parameter that controls the size of the loss's
            quadratic bowl near 0.
        max_val: The max value to clip the loss before to compute the
            reduction. ``None`` means no clipping is used.
        reduction: The reduction strategy: ``'none'``, ``'mean'``, ``'sum'`` or
            ``'batchmean'``. See `sonnix.functional.reduce_loss` for the details.

    Shape:
        - Input: ``(*)``, where ``*`` means any number of dimensions.
        - Target: ``(*)``, same shape as the input.
        - Output: scalar. If ``reduction`` is ``'none'``, then ``(*)``, same
          shape as input.

    Example:
        ```pycon
        >>> import torch
        >>> from sonnix.modules import GeneralRobustRegressionLoss
        >>> criterion = GeneralRobustRegressionLoss()
        >>> criterion
        GeneralRobustRegressionLoss(reduction=mean, alpha=2.0, scale=1.0, max_val=None)
        >>> input = torch.randn(3, 2, requires_grad=True)
        >>> target = torch.rand(3, 2, requires_grad=False)
        >>> loss = criterion(input, target)
        >>> loss
        tensor(..., grad_fn=<MeanBackward0>)
        >>> loss.backward()

        ```
    """

    def __init__(
        self,
        alpha: float = 2.0,
        scale: float = 1.0,
        max_val: float | None = None,
        *,
        reduction: Reduction = "mean",
    ) -> None:
        super().__init__(reduction=reduction)
        self.alpha = float(alpha)
        validate_gt(scale, 0, name="scale")
        self.scale = float(scale)
        self.max_val = max_val

    def extra_repr(self) -> str:
        return (
            f"{super().extra_repr()}, alpha={self.alpha}, scale={self.scale}, "
            f"max_val={self.max_val}"
        )

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return general_robust_regression_loss(
            prediction=prediction,
            target=target,
            alpha=self.alpha,
            scale=self.scale,
            max_val=self.max_val,
            reduction=self.reduction,
        )
