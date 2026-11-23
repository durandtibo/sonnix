r"""Contain the implementation of the focal loss function."""

from __future__ import annotations

__all__ = ["binary_focal_loss", "binary_focal_loss_with_logits"]

from typing import TYPE_CHECKING

from torch.nn.functional import binary_cross_entropy, binary_cross_entropy_with_logits

from sonnix.functional.loss._binary import cast_target, class_weight, probability_of_target
from sonnix.functional.power import safe_pow
from sonnix.functional.reduction import reduce_loss
from sonnix.validation import validate_focal_params, validate_same_shape

if TYPE_CHECKING:
    import torch

    from sonnix.functional.reduction import Reduction


def binary_focal_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    alpha: float | None = 0.25,
    gamma: float = 2.0,
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
        target: A tensor with the same shape as ``prediction``. It stores
            the binary classification label for each element in inputs
            (0 for the negative class and 1 for the positive class). An
            integer or ``bool`` tensor (for example the labels of a
            ``DataLoader``) is converted to the dtype of ``prediction``.
        alpha: The weighting factor, which must be in the range
            ``[0, 1]``. ``None`` disables the weighting.
        gamma: The focusing parameter, which must be positive
            (``>=0``).
        reduction: The reduction strategy: ``'none'``, ``'mean'``, ``'sum'`` or
            ``'batchmean'``. See `sonnix.functional.reduce_loss` for the details.

    Returns:
        The computed binary focal loss. The shape of the tensor depends
            on the reduction strategy.

    Raises:
        ValueError: If ``prediction`` and ``target`` do not have the same shape.

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
    validate_same_shape(prediction, target, name1="prediction", name2="target")
    target = cast_target(target, prediction)
    ce_loss = binary_cross_entropy(prediction, target, reduction="none")
    p_t = probability_of_target(prediction, target)
    return _focal(ce_loss, p_t, target, alpha=alpha, gamma=gamma, reduction=reduction)


def binary_focal_loss_with_logits(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    alpha: float | None = 0.25,
    gamma: float = 2.0,
    reduction: Reduction = "mean",
) -> torch.Tensor:
    r"""Compute the binary focal loss.

    Based on "Focal Loss for Dense Object Detection"
    (https://arxiv.org/pdf/1708.02002.pdf)
    Implementation is based on
    https://pytorch.org/vision/main/_modules/torchvision/ops/focal_loss.html

    Note:
        This variant is safe to use with ``torch.autocast``. Soft targets
        in ``[0, 1]`` are supported and give the same value as the
        probability variant applied to ``sigmoid(prediction)``.

    Args:
        prediction: The float tensor with predictions as unnormalized
            scores (often referred to as logits) for each example.
        target: A tensor with the same shape as ``prediction``. It stores
            the binary classification label for each element in inputs
            (0 for the negative class and 1 for the positive class). An
            integer or ``bool`` tensor (for example the labels of a
            ``DataLoader``) is converted to the dtype of ``prediction``.
        alpha: The weighting factor, which must be in the range
            ``[0, 1]``. ``None`` disables the weighting.
        gamma: The focusing parameter, which must be positive
            (``>=0``).
        reduction: The reduction strategy: ``'none'``, ``'mean'``, ``'sum'`` or
            ``'batchmean'``. See `sonnix.functional.reduce_loss` for the details.

    Returns:
        The computed binary focal loss. The shape of the tensor depends
            on the reduction strategy.

    Raises:
        ValueError: If ``prediction`` and ``target`` do not have the same shape.

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
    validate_same_shape(prediction, target, name1="prediction", name2="target")
    target = cast_target(target, prediction)
    ce_loss = binary_cross_entropy_with_logits(prediction, target, reduction="none")
    p_t = probability_of_target(prediction.sigmoid(), target)
    return _focal(ce_loss, p_t, target, alpha=alpha, gamma=gamma, reduction=reduction)


def _focal(
    ce_loss: torch.Tensor,
    p_t: torch.Tensor,
    target: torch.Tensor,
    *,
    alpha: float | None,
    gamma: float,
    reduction: Reduction,
) -> torch.Tensor:
    r"""Compute the focal loss from the cross entropy.

    It is the part that the probability variant and the logits variant
    of the binary focal loss have in common: the two variants only
    differ by the way they compute ``ce_loss`` and ``p_t``.

    Args:
        ce_loss: The element-wise binary cross entropy.
        p_t: The probability of the target class.
        target: The targets, with the same shape as ``ce_loss``.
        alpha: The weighting factor, or ``None`` to disable the
            weighting.
        gamma: The focusing parameter.
        reduction: The reduction strategy.

    Returns:
        The focal loss, reduced with ``reduction``.

    Example:
        ```pycon
        >>> import torch
        >>> from sonnix.functional.loss.focal import _focal
        >>> _focal(
        ...     torch.tensor([0.5, 2.0]),
        ...     torch.tensor([0.5, 0.5]),
        ...     torch.tensor([1.0, 0.0]),
        ...     alpha=None,
        ...     gamma=2.0,
        ...     reduction="none",
        ... )
        tensor([0.1250, 0.5000])

        ```
    """
    loss = ce_loss * safe_pow(1 - p_t, gamma)
    if alpha is not None:
        loss = class_weight(alpha, target) * loss
    return reduce_loss(loss, reduction=reduction)
