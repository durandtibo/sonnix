r"""Contain the Poly1Loss implementation."""

from __future__ import annotations

__all__ = ["BaseBinaryPoly1Loss", "BinaryPoly1Loss", "BinaryPoly1LossWithLogits"]

from typing import TYPE_CHECKING

from coola.validation import validate_finite

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
        reduction: The reduction strategy: ``'none'``, ``'mean'``, ``'sum'`` or
            ``'batchmean'``. See `sonnix.functional.reduce_loss` for the details.
    """

    def __init__(self, alpha: float = 1.0, *, reduction: Reduction = "mean") -> None:
        super().__init__(reduction=reduction)
        validate_finite(alpha, name="alpha")
        self.alpha = float(alpha)

    def extra_repr(self) -> str:
        return f"{super().extra_repr()}, alpha={self.alpha}"


class BinaryPoly1Loss(BaseBinaryPoly1Loss):
    r"""Implementation of the binary Poly-1 loss for binary targets.

    Based on "PolyLoss: A Polynomial Expansion Perspective of
    Classification Loss Functions"
    (https://arxiv.org/pdf/2204.12511)

    Args:
        alpha: The weight of the ``(1 - p_t)`` term, where ``p_t`` is the
            probability of the target class. The loss is
            ``cross_entropy + alpha * (1 - p_t)``.
        reduction: The reduction strategy: ``'none'``, ``'mean'``, ``'sum'`` or
            ``'batchmean'``. See `sonnix.functional.reduce_loss` for the details.

    Shape:
        - Input: ``(*)``, where ``*`` means any number of dimensions.
        - Target: ``(*)``, same shape as the input.
        - Output: scalar. If ``reduction`` is ``'none'``, then ``(*)``,
            same shape as input.

    Raises:
        ValueError: If ``alpha`` is not finite.

    Example:
        ```pycon
        >>> import torch
        >>> from sonnix.modules import BinaryPoly1Loss
        >>> criterion = BinaryPoly1Loss()
        >>> criterion
        BinaryPoly1Loss(reduction=mean, alpha=1.0)
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
            alpha=self.alpha,
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
        reduction: The reduction strategy: ``'none'``, ``'mean'``, ``'sum'`` or
            ``'batchmean'``. See `sonnix.functional.reduce_loss` for the details.

    Shape:
        - Input: ``(*)``, where ``*`` means any number of dimensions.
        - Target: ``(*)``, same shape as the input.
        - Output: scalar. If ``reduction`` is ``'none'``, then ``(*)``,
            same shape as input.

    Raises:
        ValueError: If ``alpha`` is not finite.

    Example:
        ```pycon
        >>> import torch
        >>> from sonnix.modules import BinaryPoly1LossWithLogits
        >>> criterion = BinaryPoly1LossWithLogits()
        >>> criterion
        BinaryPoly1LossWithLogits(reduction=mean, alpha=1.0)
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
            alpha=self.alpha,
            reduction=self.reduction,
        )
