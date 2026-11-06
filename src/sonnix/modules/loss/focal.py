r"""Contain the focal loss implementation."""

from __future__ import annotations

__all__ = ["BinaryFocalLoss", "BinaryFocalLossWithLogits"]

from typing import TYPE_CHECKING

from sonnix.functional import (
    binary_focal_loss,
    binary_focal_loss_with_logits,
)
from sonnix.functional.loss.focal import validate_focal_params
from sonnix.modules.loss.base import BaseLoss

if TYPE_CHECKING:
    import torch

    from sonnix.functional.reduction import Reduction


class BinaryFocalLoss(BaseLoss):
    r"""Implementation of the binary focal loss.

    Based on "focal loss for Dense Object Detection"
    (https://arxiv.org/pdf/1708.02002.pdf)

    Args:
        alpha: The weighting factor, which must be in the range
            ``[0, 1]``. ``None`` disables the weighting.
        gamma: The focusing parameter, which must be positive
            (``>=0``).
        reduction: The reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'`` | ``'batchmean'``.
            ``'none'``: no reduction will be applied, ``'mean'``: the
            sum of the output will be divided by the number of
            elements in the output, ``'sum'``: the output will be
            summed.

    Shape:
        - Input: ``(*)``, where ``*`` means any number of dimensions.
        - Target: ``(*)``, same shape as the input.
        - Output: scalar. If ``reduction`` is ``'none'``, then ``(*)``,
            same shape as input.

    Example:
        ```pycon
        >>> import torch
        >>> from sonnix.modules import BinaryFocalLoss
        >>> criterion = BinaryFocalLoss()
        >>> criterion
        BinaryFocalLoss(alpha=0.25, gamma=2.0, reduction=mean)
        >>> prediction = torch.rand(2, 4, requires_grad=True)
        >>> target = torch.tensor([[1.0, 0.0, 0.0, 1.0], [1.0, 0.0, 1.0, 0.0]])
        >>> loss = criterion(prediction, target)
        >>> loss
        tensor(..., grad_fn=<MeanBackward0>)
        >>> loss.backward()

        ```
    """

    def __init__(
        self, alpha: float | None = 0.25, gamma: float = 2.0, *, reduction: Reduction = "mean"
    ) -> None:
        super().__init__(reduction=reduction)
        validate_focal_params(alpha=alpha, gamma=gamma)
        self._alpha = None if alpha is None else float(alpha)
        self._gamma = float(gamma)

    def extra_repr(self) -> str:
        return f"alpha={self._alpha}, gamma={self._gamma}, reduction={self.reduction}"

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        r"""Compute the binary focal loss.

        Args:
            prediction: The float tensor with predictions as
                probabilities for each example.
            target: A float tensor with the same shape as inputs.
                It stores the binary classification label for each
                element in inputs (0 for the negative class and 1 for
                the positive class).

        Returns:
            ``torch.Tensor`` of type float: The loss value(s). The
                shape of the tensor depends on the reduction. If the
                reduction is ``mean`` or ``sum``, the tensor has a
                single scalar value. If the reduction is ``none``,
                the shape of the tensor is the same that the inputs.
        """
        return binary_focal_loss(
            prediction=prediction,
            target=target,
            alpha=self._alpha,
            gamma=self._gamma,
            reduction=self.reduction,
        )


class BinaryFocalLossWithLogits(BaseLoss):
    r"""Implementation of the binary focal loss.

    Based on "focal loss for Dense Object Detection"
    (https://arxiv.org/pdf/1708.02002.pdf)

    Args:
        alpha: The weighting factor, which must be in the range
            ``[0, 1]``. ``None`` disables the weighting.
        gamma: The focusing parameter, which must be positive
            (``>=0``).
        reduction: The reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'`` | ``'batchmean'``.
            ``'none'``: no reduction will be applied, ``'mean'``: the
            sum of the output will be divided by the number of
            elements in the output, ``'sum'``: the output will be
            summed.

    Shape:
        - Input: ``(*)``, where ``*`` means any number of dimensions.
        - Target: ``(*)``, same shape as the input.
        - Output: scalar. If ``reduction`` is ``'none'``, then ``(*)``,
            same shape as input.

    Example:
        ```pycon
        >>> import torch
        >>> from sonnix.modules import BinaryFocalLossWithLogits
        >>> criterion = BinaryFocalLossWithLogits()
        >>> criterion
        BinaryFocalLossWithLogits(alpha=0.25, gamma=2.0, reduction=mean)
        >>> prediction = torch.randn(2, 4, requires_grad=True)
        >>> target = torch.tensor([[1.0, 0.0, 0.0, 1.0], [1.0, 0.0, 1.0, 0.0]])
        >>> loss = criterion(prediction, target)
        >>> loss
        tensor(..., grad_fn=<MeanBackward0>)
        >>> loss.backward()

        ```
    """

    def __init__(
        self, alpha: float | None = 0.25, gamma: float = 2.0, *, reduction: Reduction = "mean"
    ) -> None:
        super().__init__(reduction=reduction)
        validate_focal_params(alpha=alpha, gamma=gamma)
        self._alpha = None if alpha is None else float(alpha)
        self._gamma = float(gamma)

    def extra_repr(self) -> str:
        return f"alpha={self._alpha}, gamma={self._gamma}, reduction={self.reduction}"

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        r"""Compute the binary focal loss.

        Args:
            prediction: The float tensor with predictions as
                unnormalized scores (often referred to as logits) for
                each example.
            target: A float tensor with the same shape as inputs.
                It stores the binary classification label for each
                element in inputs (0 for the negative class and 1 for
                the positive class).

        Returns:
            ``torch.Tensor`` of type float: The loss value(s). The
                shape of the tensor depends on the reduction. If the
                reduction is ``mean`` or ``sum``, the tensor has a
                single scalar value. If the reduction is ``none``,
                the shape of the tensor is the same that the inputs.
        """
        return binary_focal_loss_with_logits(
            prediction=prediction,
            target=target,
            alpha=self._alpha,
            gamma=self._gamma,
            reduction=self.reduction,
        )
