r"""Contain the implementation of the focal loss function."""

from __future__ import annotations

__all__ = ["binary_focal_loss", "binary_focal_loss_with_logits"]

from typing import TYPE_CHECKING

from coola.validation import validate_ge, validate_range
from torch.nn.functional import binary_cross_entropy, binary_cross_entropy_with_logits

from sonnix.functional.reduction import reduce_loss

if TYPE_CHECKING:
    import torch

    from sonnix.functional.reduction import Reduction


def validate_focal_params(alpha: float | None, gamma: float) -> None:
    r"""Check the parameters of the focal loss.

    Args:
        alpha: The weighting factor. It has to be in the range
            ``[0, 1]``, or ``None`` to disable the weighting.
        gamma: The focusing parameter. It has to be non-negative.

    Raises:
        ValueError: if one of the parameters is invalid.
    """
    if alpha is not None:
        validate_range(alpha, 0, 1, name="alpha")
    validate_ge(gamma, 0, name="gamma")


def binary_focal_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    alpha: float | None = 0.25,
    gamma: float = 2.0,
    *,
    reduction: Reduction = "mean",
) -> torch.Tensor:
    r"""Compute the binary focal loss.

    Based on "Focal Loss for Dense Object Detection"
    (https://arxiv.org/pdf/1708.02002.pdf)
    Implementation is based on
    https://pytorch.org/vision/main/_modules/torchvision/ops/focal_loss.html

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
        alpha: The weighting factor, which must be in the range
            ``[0, 1]``. ``None`` disables the weighting.
        gamma: The focusing parameter, which must be positive
            (``>=0``).
        reduction: The reduction strategy. The valid values are
            ``'mean'``, ``'none'``, ``'sum'``,
            and ``'batchmean'``.
            ``'none'``: no reduction will be applied, ``'mean'``: the
            sum will be divided by the number of elements in the
            input, ``'sum'``: the output will be summed,
            ``'batchmean'``: the sum will be divided by the size of the
            first dimension.

    Returns:
        The computed binary focal loss. The shape of the tensor depends
            on the reduction strategy.

    Example:
        ```pycon
        >>> import torch
        >>> from sonnix.functional import binary_focal_loss
        >>> loss = binary_focal_loss(
        ...     torch.rand(2, 4, requires_grad=True),
        ...     torch.tensor([[1.0, 0.0, 0.0, 1.0], [1.0, 0.0, 1.0, 0.0]]),
        ... )
        >>> loss
        tensor(..., grad_fn=<MeanBackward0>)
        >>> loss.backward()

        ```
    """
    validate_focal_params(alpha=alpha, gamma=gamma)
    ce_loss = binary_cross_entropy(prediction, target, reduction="none")
    p_t = prediction * target + (1 - prediction) * (1 - target)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha is not None:
        alpha_t = alpha * target + (1 - alpha) * (1 - target)
        loss = alpha_t * loss
    return reduce_loss(loss, reduction)


def binary_focal_loss_with_logits(
    prediction: torch.Tensor,
    target: torch.Tensor,
    alpha: float | None = 0.25,
    gamma: float = 2.0,
    *,
    reduction: Reduction = "mean",
) -> torch.Tensor:
    r"""Compute the binary focal loss.

    Based on "Focal Loss for Dense Object Detection"
    (https://arxiv.org/pdf/1708.02002.pdf)
    Implementation is based on
    https://pytorch.org/vision/main/_modules/torchvision/ops/focal_loss.html

    Note:
        Targets are expected to be binary (0 or 1). This variant is safe to
        use with ``torch.autocast``.

    Args:
        prediction: The float tensor with predictions as unnormalized
            scores (often referred to as logits) for each example.
        target: A float tensor with the same shape as inputs. It stores
            the binary classification label for each element in inputs
            (0 for the negative class and 1 for the positive class).
        alpha: The weighting factor, which must be in the range
            ``[0, 1]``. ``None`` disables the weighting.
        gamma: The focusing parameter, which must be positive
            (``>=0``).
        reduction: The reduction strategy. The valid values are
            ``'mean'``, ``'none'``, ``'sum'``,
            and ``'batchmean'``.
            ``'none'``: no reduction will be applied, ``'mean'``: the
            sum will be divided by the number of elements in the
            input, ``'sum'``: the output will be summed,
            ``'batchmean'``: the sum will be divided by the size of the
            first dimension.

    Returns:
        The computed binary focal loss. The shape of the tensor depends
            on the reduction strategy.

    Example:
        ```pycon
        >>> import torch
        >>> from sonnix.functional import binary_focal_loss_with_logits
        >>> loss = binary_focal_loss_with_logits(
        ...     torch.randn(2, 4, requires_grad=True),
        ...     torch.tensor([[1.0, 0.0, 0.0, 1.0], [1.0, 0.0, 1.0, 0.0]]),
        ... )
        >>> loss
        tensor(..., grad_fn=<MeanBackward0>)
        >>> loss.backward()

        ```
    """
    validate_focal_params(alpha=alpha, gamma=gamma)
    ce_loss = binary_cross_entropy_with_logits(prediction, target, reduction="none")
    # For binary targets, ``exp(-ce) = p_t``: no need to compute the sigmoid.
    p_t = ce_loss.neg().exp()
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha is not None:
        alpha_t = alpha * target + (1 - alpha) * (1 - target)
        loss = alpha_t * loss
    return reduce_loss(loss, reduction)
