r"""Contain the loss functions that use the inverse hyperbolic sine
(asinh) to transform the predictions or targets."""

from __future__ import annotations

__all__ = ["asinh_mse_loss", "asinh_smooth_l1_loss"]

from typing import TYPE_CHECKING

from torch.nn.functional import mse_loss, smooth_l1_loss

from sonnix.functional.reduction import reduce_loss

if TYPE_CHECKING:
    import torch

    from sonnix.functional.reduction import Reduction


def asinh_mse_loss(
    prediction: torch.Tensor, target: torch.Tensor, *, reduction: Reduction = "mean"
) -> torch.Tensor:
    r"""Compute the mean squared error (MSE) on the inverse hyperbolic
    sine (asinh) transformed predictions and targets.

    It is a generalization of mean squared logarithmic error (MSLE)
    that works for real values. The ``asinh`` transformation is used
    instead of ``log1p`` because ``asinh`` works on negative values.

    Args:
        prediction: The predictions.
        target: The target values.
        reduction: The reduction strategy. The valid values are
            ``'mean'``, ``'none'``, ``'sum'``,
            and ``'batchmean'``.
            ``'none'``: no reduction will be applied, ``'mean'``: the
            sum will be divided by the number of elements in the
            input, ``'sum'``: the output will be summed,
            ``'batchmean'``: the sum will be divided by the size of the
            first dimension.

    Returns:
        The mean squared error (MSE) on the inverse hyperbolic sine
            (asinh) transformed predictions and targets. The shape of
            the tensor depends on the reduction strategy.

    Example:
        ```pycon
        >>> import torch
        >>> from sonnix.functional import asinh_mse_loss
        >>> loss = asinh_mse_loss(torch.randn(2, 4, requires_grad=True), torch.randn(2, 4))
        >>> loss
        tensor(..., grad_fn=<MeanBackward0>)
        >>> loss.backward()

        ```
    """
    return reduce_loss(mse_loss(prediction.asinh(), target.asinh(), reduction="none"), reduction)


def asinh_smooth_l1_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    beta: float = 1.0,
    *,
    reduction: Reduction = "mean",
) -> torch.Tensor:
    r"""Compute the smooth L1 loss on the inverse hyperbolic sine (asinh)
    transformed predictions and targets.

    It is a generalization of mean squared logarithmic error (MSLE)
    that works for real values. The ``asinh`` transformation is used
    instead of ``log1p`` because ``asinh`` works on negative values.

    Args:
        prediction: The predictions.
        target: The target values.
        beta: The threshold at which to change between L1 and L2 loss.
            The value must be non-negative.
        reduction: The reduction strategy. The valid values are
            ``'mean'``, ``'none'``, ``'sum'``,
            and ``'batchmean'``.
            ``'none'``: no reduction will be applied, ``'mean'``: the
            sum will be divided by the number of elements in the
            input, ``'sum'``: the output will be summed,
            ``'batchmean'``: the sum will be divided by the size of the
            first dimension.

    Returns:
        The smooth L1 loss on the inverse hyperbolic sine (asinh)
            transformed predictions and targets. The shape of
            the tensor depends on the reduction strategy.

    Example:
        ```pycon
        >>> import torch
        >>> from sonnix.functional import asinh_smooth_l1_loss
        >>> loss = asinh_smooth_l1_loss(torch.randn(2, 4, requires_grad=True), torch.randn(2, 4))
        >>> loss
        tensor(..., grad_fn=<MeanBackward0>)
        >>> loss.backward()

        ```
    """
    return reduce_loss(
        smooth_l1_loss(prediction.asinh(), target.asinh(), reduction="none", beta=beta),
        reduction,
    )
