r"""Contain the Poly1Loss implementation."""

from __future__ import annotations

__all__ = ["BaseBinaryPoly1Loss", "BinaryPoly1Loss", "BinaryPoly1LossWithLogits"]

from typing import TYPE_CHECKING

from sonnix.functional import (
    binary_poly1_loss,
    binary_poly1_loss_with_logits,
)
from sonnix.modules.loss.base import BaseLoss

if TYPE_CHECKING:
    import torch

    from sonnix.functional.reduction import Reduction


class BaseBinaryPoly1Loss(BaseLoss):
    r"""Define the base class of the binary Poly-1 losses.

    It stores the hyperparameters shared by ``BinaryPoly1Loss`` and
    ``BinaryPoly1LossWithLogits``.

    Args:
        alpha: The weight of the ``(1 - p_t)`` term, where ``p_t`` is the
            probability of the target class.
        reduction: The reduction strategy. The valid values are
            ``'mean'``, ``'none'``, ``'sum'``, and ``'batchmean'``.
    """

    def __init__(self, alpha: float = 1.0, *, reduction: Reduction = "mean") -> None:
        super().__init__(reduction=reduction)
        self._alpha = float(alpha)

    def extra_repr(self) -> str:
        return f"alpha={self._alpha}, reduction={self.reduction}"


class BinaryPoly1Loss(BaseBinaryPoly1Loss):
    r"""Implementation of the binary Poly-1 loss for binary targets.

    Based on "PolyLoss: A Polynomial Expansion Perspective of
    Classification Loss Functions"
    (https://arxiv.org/pdf/2204.12511)

    Args:
        alpha: The weight of the ``(1 - p_t)`` term, where ``p_t`` is the
            probability of the target class. The loss is
            ``cross_entropy + alpha * (1 - p_t)``.
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
        >>> from sonnix.modules import BinaryPoly1Loss
        >>> criterion = BinaryPoly1Loss()
        >>> criterion
        BinaryPoly1Loss(alpha=1.0, reduction=mean)
        >>> prediction = torch.rand(2, 4, requires_grad=True)
        >>> target = torch.tensor([[1.0, 0.0, 0.0, 1.0], [1.0, 0.0, 1.0, 0.0]])
        >>> loss = criterion(prediction, target)
        >>> loss
        tensor(..., grad_fn=<MeanBackward0>)
        >>> loss.backward()

        ```
    """

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        r"""Compute the binary Poly-1 loss for binary targets.

        Args:
            prediction: The float tensor with predictions as
                probabilities for each example.
            target: A float tensor with the same shape as inputs.
                It stores the binary classification label for each
                element in inputs (0 for the negative class and 1 for
                the positive class).

        Returns:
            The loss value(s). The shape of the tensor depends on the
                reduction. If the reduction is ``mean`` or ``sum``,
                the tensor has a single scalar value. If the reduction
                is ``none``, the shape of the tensor is the same as
                the inputs.
        """
        return binary_poly1_loss(
            prediction=prediction,
            target=target,
            alpha=self._alpha,
            reduction=self.reduction,
        )


class BinaryPoly1LossWithLogits(BaseBinaryPoly1Loss):
    r"""Implementation of the binary Poly-1 loss for binary targets.

    Based on "PolyLoss: A Polynomial Expansion Perspective of
    Classification Loss Functions"
    (https://arxiv.org/pdf/2204.12511)

    Args:
        alpha: The weight of the ``(1 - p_t)`` term, where ``p_t`` is the
            probability of the target class. The loss is
            ``cross_entropy + alpha * (1 - p_t)``.
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
        >>> from sonnix.modules import BinaryPoly1LossWithLogits
        >>> criterion = BinaryPoly1LossWithLogits()
        >>> criterion
        BinaryPoly1LossWithLogits(alpha=1.0, reduction=mean)
        >>> prediction = torch.randn(2, 4, requires_grad=True)
        >>> target = torch.tensor([[1.0, 0.0, 0.0, 1.0], [1.0, 0.0, 1.0, 0.0]])
        >>> loss = criterion(prediction, target)
        >>> loss
        tensor(..., grad_fn=<MeanBackward0>)
        >>> loss.backward()

        ```
    """

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        r"""Compute the binary Poly-1 loss for binary targets.

        Args:
            prediction: The float tensor with predictions as
                unnormalized scores (often referred to as logits) for
                each example.
            target: A float tensor with the same shape as inputs.
                It stores the binary classification label for each
                element in inputs (0 for the negative class and 1 for
                the positive class).

        Returns:
            The loss value(s). The shape of the tensor depends on the
                reduction. If the reduction is ``mean`` or ``sum``,
                the tensor has a single scalar value. If the reduction
                is ``none``, the shape of the tensor is the same as
                the inputs.
        """
        return binary_poly1_loss_with_logits(
            prediction=prediction,
            target=target,
            alpha=self._alpha,
            reduction=self.reduction,
        )
