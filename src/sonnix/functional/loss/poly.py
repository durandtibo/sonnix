r"""Contain the implementation of the PolyLoss function."""

from __future__ import annotations

__all__ = ["binary_poly1_loss", "binary_poly1_loss_with_logits"]

from typing import TYPE_CHECKING

from coola.validation import validate_finite
from torch.nn.functional import binary_cross_entropy, binary_cross_entropy_with_logits

from sonnix.functional.reduction import reduce_loss
from sonnix.validation import validate_same_shape

if TYPE_CHECKING:
    import torch

    from sonnix.functional.reduction import Reduction


def binary_poly1_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    alpha: float = 1.0,
    *,
    reduction: Reduction = "mean",
) -> torch.Tensor:
    r"""Compute the Poly-1 loss for binary targets.

    Based on "PolyLoss: A Polynomial Expansion Perspective of
    Classification Loss Functions"
    (https://arxiv.org/pdf/2204.12511)

    Note:
        ``torch.nn.functional.binary_cross_entropy`` raises an error under
        ``torch.autocast``. Use the ``*_with_logits`` variant for mixed
        precision training.

    Args:
        prediction: The float tensor with predictions as probabilities
            for each example.
        target: A float tensor with the same shape as inputs. It stores
            the binary classification label for each element in inputs
            (0 for the negative class and 1 for the positive class).
        alpha: The weight of the ``(1 - p_t)`` term, where ``p_t`` is the
            probability of the target class. The loss is
            ``cross_entropy + alpha * (1 - p_t)``.
        reduction: The reduction strategy: ``'none'``, ``'mean'``, ``'sum'`` or
            ``'batchmean'``. See `sonnix.functional.reduce_loss` for the details.

    Returns:
        The computed Poly-1 loss value. The shape of the tensor depends
            on the reduction strategy.

    Raises:
        ValueError: If ``alpha`` is not finite.
        ValueError: If ``prediction`` and ``target`` do not have the same shape.

    Example:
        ```pycon
        >>> import torch
        >>> from sonnix.functional import binary_poly1_loss
        >>> loss = binary_poly1_loss(
        ...     torch.rand(2, 4, requires_grad=True),
        ...     torch.tensor([[1.0, 0.0, 0.0, 1.0], [1.0, 0.0, 1.0, 0.0]]),
        ... )
        >>> loss
        tensor(..., grad_fn=<MeanBackward0>)
        >>> loss.backward()

        ```
    """
    validate_finite(alpha, name="alpha")
    validate_same_shape(prediction, target, name1="prediction", name2="target")
    ce_loss = binary_cross_entropy(prediction, target, reduction="none")
    p_t = prediction * target + (1 - prediction) * (1 - target)
    loss = ce_loss + alpha * (1 - p_t)
    return reduce_loss(loss, reduction)


def binary_poly1_loss_with_logits(
    prediction: torch.Tensor,
    target: torch.Tensor,
    alpha: float = 1.0,
    *,
    reduction: Reduction = "mean",
) -> torch.Tensor:
    r"""Compute the Poly-1 loss.

    Based on "PolyLoss: A Polynomial Expansion Perspective of
    Classification Loss Functions"
    (https://arxiv.org/pdf/2204.12511)

    Note:
        This variant is safe to use with ``torch.autocast``. Soft targets
        in ``[0, 1]`` are supported and give the same value as the
        probability variant applied to ``sigmoid(prediction)``.

    Args:
        prediction: The float tensor with predictions as unnormalized
            scores (often referred to as logits) for each example.
        target: A float tensor with the same shape as inputs. It stores
            the binary classification label for each element in inputs
            (0 for the negative class and 1 for the positive class).
        alpha: The weight of the ``(1 - p_t)`` term, where ``p_t`` is the
            probability of the target class. The loss is
            ``cross_entropy + alpha * (1 - p_t)``.
        reduction: The reduction strategy: ``'none'``, ``'mean'``, ``'sum'`` or
            ``'batchmean'``. See `sonnix.functional.reduce_loss` for the details.

    Returns:
        The computed Poly-1 loss value. The shape of the tensor depends
            on the reduction strategy.

    Raises:
        ValueError: If ``alpha`` is not finite.
        ValueError: If ``prediction`` and ``target`` do not have the same shape.

    Example:
        ```pycon
        >>> import torch
        >>> from sonnix.functional import binary_poly1_loss_with_logits
        >>> loss = binary_poly1_loss_with_logits(
        ...     torch.randn(2, 4, requires_grad=True),
        ...     torch.tensor([[1.0, 0.0, 0.0, 1.0], [1.0, 0.0, 1.0, 0.0]]),
        ... )
        >>> loss
        tensor(..., grad_fn=<MeanBackward0>)
        >>> loss.backward()

        ```
    """
    validate_finite(alpha, name="alpha")
    validate_same_shape(prediction, target, name1="prediction", name2="target")
    ce_loss = binary_cross_entropy_with_logits(prediction, target, reduction="none")
    # ``exp(-ce) = p_t`` only holds for binary targets, so ``p_t`` is computed
    # from the probabilities to support soft targets (label smoothing, mixup).
    probs = prediction.sigmoid()
    p_t = probs * target + (1 - probs) * (1 - target)
    loss = ce_loss + alpha * (1 - p_t)
    return reduce_loss(loss, reduction)
