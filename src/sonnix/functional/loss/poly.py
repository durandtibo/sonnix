r"""Contain the implementation of the PolyLoss function."""

from __future__ import annotations

__all__ = ["binary_poly1_loss", "binary_poly1_loss_with_logits"]


from typing import TYPE_CHECKING

from torch.nn.functional import binary_cross_entropy, binary_cross_entropy_with_logits

from sonnix.functional.reduction import reduce_loss

if TYPE_CHECKING:
    import torch


def binary_poly1_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    alpha: float = 1.0,
    reduction: str = "mean",
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
        reduction: The reduction strategy. The valid values are
            ``'mean'``, ``'none'``, ``'sum'``,
            and ``'batchmean'``.
            ``'none'``: no reduction will be applied, ``'mean'``: the
            sum will be divided by the number of elements in the
            input, ``'sum'``: the output will be summed,
            ``'batchmean'``: the sum will be divided by the size of the
            first dimension.

    Returns:
        The computed Poly-1 loss value. The shape of the tensor depends
            on the reduction strategy.

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
    ce_loss = binary_cross_entropy(prediction, target, reduction="none")
    p_t = prediction * target + (1 - prediction) * (1 - target)
    loss = ce_loss + alpha * (1 - p_t)
    return reduce_loss(loss, reduction)


def binary_poly1_loss_with_logits(
    prediction: torch.Tensor,
    target: torch.Tensor,
    alpha: float = 1.0,
    reduction: str = "mean",
) -> torch.Tensor:
    r"""Compute the Poly-1 loss.

    Based on "PolyLoss: A Polynomial Expansion Perspective of
    Classification Loss Functions"
    (https://arxiv.org/pdf/2204.12511)

    Note:
        Targets are expected to be binary (0 or 1). This variant is safe to
        use with ``torch.autocast``.

    Args:
        prediction: The float tensor with predictions as unnormalized
            scores (often referred to as logits) for each example.
        target: A float tensor with the same shape as inputs. It stores
            the binary classification label for each element in inputs
            (0 for the negative class and 1 for the positive class).
        alpha: The weight of the ``(1 - p_t)`` term, where ``p_t`` is the
            probability of the target class. The loss is
            ``cross_entropy + alpha * (1 - p_t)``.
        reduction: The reduction strategy. The valid values are
            ``'mean'``, ``'none'``, ``'sum'``,
            and ``'batchmean'``.
            ``'none'``: no reduction will be applied, ``'mean'``: the
            sum will be divided by the number of elements in the
            input, ``'sum'``: the output will be summed,
            ``'batchmean'``: the sum will be divided by the size of the
            first dimension.

    Returns:
        The computed Poly-1 loss value. The shape of the tensor depends
            on the reduction strategy.

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
    ce_loss = binary_cross_entropy_with_logits(prediction, target, reduction="none")
    # For binary targets, ``exp(-ce) = p_t``: no need to compute the sigmoid.
    p_t = ce_loss.neg().exp()
    loss = ce_loss + alpha * (1 - p_t)
    return reduce_loss(loss, reduction)
