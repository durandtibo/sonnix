r"""Contain the implementation of the quantile regression loss
function."""

from __future__ import annotations

__all__ = ["quantile_regression_loss"]

from typing import TYPE_CHECKING

from coola.validation import validate_range

from sonnix.functional.reduction import reduce_loss
from sonnix.validation import validate_same_shape

if TYPE_CHECKING:
    import torch

    from sonnix.functional.reduction import Reduction


def quantile_regression_loss(
    prediction: torch.Tensor, target: torch.Tensor, q: float = 0.5, *, reduction: Reduction = "mean"
) -> torch.Tensor:
    r"""Compute the quantile regression loss.

    Loss Functions and Metrics in Deep Learning
    https://arxiv.org/pdf/2307.02694

    Args:
        prediction: The predictions.
        target: The target values.
        q: The quantile value. ``q=0.5`` gives half of the Mean
            Absolute Error (MAE), because the loss is
            ``max(q * d, (q - 1) * d)`` with ``d = target - prediction``.
        reduction: The reduction strategy: ``'none'``, ``'mean'``, ``'sum'`` or
            ``'batchmean'``. See `sonnix.functional.reduce_loss` for the details.

    Returns:
        The quantile regression loss. The shape of the tensor depends
            on the reduction strategy.

    Raises:
        ValueError: If ``prediction`` and ``target`` do not have the same shape.

    Example:
        ```pycon
        >>> import torch
        >>> from sonnix.functional import quantile_regression_loss
        >>> loss = quantile_regression_loss(
        ...     torch.randn(2, 4, requires_grad=True), torch.randn(2, 4)
        ... )
        >>> loss
        tensor(..., grad_fn=<MeanBackward0>)
        >>> loss.backward()

        ```
    """
    validate_range(q, 0.0, 1.0, name="q")
    validate_same_shape(prediction, target, name1="prediction", name2="target")
    diff = target - prediction
    loss = q * diff.clamp_min(0.0) + (1.0 - q) * (diff.neg().clamp_min(0.0))
    return reduce_loss(loss, reduction)
