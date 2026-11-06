r"""Contain the loss functions that use the logarithmic function to
transform the predictions or targets."""

from __future__ import annotations

__all__ = ["log_cosh_loss", "msle_loss"]

import math
from typing import TYPE_CHECKING

from coola.validation import validate_gt
from torch.nn.functional import mse_loss, softplus

from sonnix.functional.reduction import reduce_loss
from sonnix.validation import validate_same_shape

if TYPE_CHECKING:
    import torch

    from sonnix.functional.reduction import Reduction


def log_cosh_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    scale: float = 1.0,
    reduction: Reduction = "mean",
) -> torch.Tensor:
    r"""Compute the logarithm of the hyperbolic cosine of the prediction
    error.

    Args:
        prediction: The predictions.
        target: The target values.
        scale: The scale factor. The error is divided by ``scale``
            and the loss is multiplied by ``scale``, so the loss is
            ``scale * log(cosh(error / scale))``. Its magnitude is
            comparable to the Huber loss for any ``scale``.
        reduction: The reduction strategy: ``'none'``, ``'mean'``, ``'sum'`` or
            ``'batchmean'``. See `sonnix.functional.reduce_loss` for the details.

    Returns:
        The logarithm of the hyperbolic cosine of the prediction error.

    Raises:
        ValueError: if ``scale`` is not strictly positive or if the
            reduction is not valid.
        ValueError: If ``prediction`` and ``target`` do not have the same shape.

    Example:
        ```pycon
        >>> import torch
        >>> from sonnix.functional import log_cosh_loss
        >>> loss = log_cosh_loss(torch.randn(3, 5, requires_grad=True), torch.randn(3, 5))
        >>> loss
        tensor(..., grad_fn=<MeanBackward0>)
        >>> loss.backward()

        ```
    """
    validate_gt(scale, 0, name="scale")
    validate_same_shape(prediction, target, name1="prediction", name2="target")
    error = target.sub(prediction).div(scale).abs()
    # Numerically stable form of log(cosh(x)): it does not overflow for large |x|.
    loss = error + softplus(-2.0 * error) - math.log(2.0)
    return reduce_loss(loss.mul(scale), reduction=reduction)


def msle_loss(
    prediction: torch.Tensor, target: torch.Tensor, *, reduction: Reduction = "mean"
) -> torch.Tensor:
    r"""Compute the mean squared error (MSE) on the logarithmic
    transformed predictions and targets.

    This loss is best to use when targets having exponential growth,
    such as population counts, average sales of a commodity over a
    span of years etc. Note that this loss penalizes an
    under-predicted estimate greater than an over-predicted estimate.

    Note: this loss only works with values greater or equal to ``-1``
    (``log1p`` is undefined below). For inputs lower than ``-1``, the
    result is ``nan`` and no error is raised. Clamp or shift the inputs
    if they can be negative.

    Args:
        prediction: The predictions.
        target: The target values.
        reduction: The reduction strategy: ``'none'``, ``'mean'``, ``'sum'`` or
            ``'batchmean'``. See `sonnix.functional.reduce_loss` for the details.

    Returns:
        The mean squared logarithmic error. The shape of
            the tensor depends on the reduction strategy.

    Raises:
        ValueError: If ``prediction`` and ``target`` do not have the same shape.

    Example:
        ```pycon
        >>> import torch
        >>> from sonnix.functional import msle_loss
        >>> loss = msle_loss(torch.randn(2, 4, requires_grad=True), torch.randn(2, 4))
        >>> loss
        tensor(..., grad_fn=<MeanBackward0>)
        >>> loss.backward()

        ```
    """
    validate_same_shape(prediction, target, name1="prediction", name2="target")
    return reduce_loss(mse_loss(prediction.log1p(), target.log1p(), reduction="none"), reduction)
