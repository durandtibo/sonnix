r"""Contain the general robust regression loss a.k.a the Barron robust
loss."""

from __future__ import annotations

__all__ = ["general_robust_regression_loss"]

from typing import TYPE_CHECKING

import torch

from sonnix.functional.reduction import reduce_loss
from sonnix.validation import validate_general_robust_params, validate_same_shape

# ``alpha`` values closer than ``_ALPHA_EPS`` to 0 or 2 use the limit formulas.
_ALPHA_EPS = 1e-5


if TYPE_CHECKING:
    from sonnix.functional.reduction import Reduction


def general_robust_regression_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    alpha: float = 2.0,
    scale: float = 1.0,
    max_val: float | None = None,
    reduction: Reduction = "mean",
) -> torch.Tensor:
    r"""Compute the general robust regression loss a.k.a. Barron robust
    loss.

    Based on the paper:

        A General and Adaptive Robust Loss Function
        Jonathan T. Barron
        CVPR 2019 (https://arxiv.org/abs/1701.03077)

    Note:
        The "adaptive" part of the loss is not implemented in this
        function.

    Note:
        If ``max_val`` is set, the loss is clamped, so the gradient is
        zero for the elements whose loss is larger than ``max_val``.

    Args:
        prediction: The predictions.
        target: The target values.
        alpha: The shape parameter that controls the robustness of the
            loss. It is a Python float, not a tensor. ``-inf`` is supported (Welsch loss),
            ``NaN`` and ``+inf`` are not. The limit
            cases ``alpha=2``
            (``x^2``), ``alpha=0`` (``log(0.5 * x^2 + 1)``) are
            used when ``alpha`` is within ``1e-5`` of 0 or 2.
        scale: The scale parameter that controls the size of the loss's
            quadratic bowl near 0.
        max_val: The max value to clip the loss before to compute the
            reduction. It has to be strictly positive.
            ``None`` means no clipping is used.
        reduction: The reduction strategy: ``'none'``, ``'mean'``, ``'sum'`` or
            ``'batchmean'``. See `sonnix.functional.reduce_loss` for the details.

    Returns:
        The loss. The shape of the tensor depends on the reduction
            strategy.

    Raises:
        ValueError: if the reduction is not valid.
        ValueError: if ``alpha`` is ``NaN`` or ``+inf``, or if ``scale`` or
            ``max_val`` is not strictly positive.
        ValueError: If ``prediction`` and ``target`` do not have the same shape.

    Example:
        ```pycon
        >>> import torch
        >>> from sonnix.functional import general_robust_regression_loss
        >>> loss = general_robust_regression_loss(
        ...     torch.randn(2, 4, requires_grad=True), torch.randn(2, 4)
        ... )
        >>> loss
        tensor(..., grad_fn=<MeanBackward0>)
        >>> loss.backward()

        ```
    """
    validate_general_robust_params(alpha, scale, max_val)
    validate_same_shape(prediction, target, name1="prediction", name2="target")
    squared_error = prediction.sub(target).div(scale).pow(2)
    if abs(alpha - 2) < _ALPHA_EPS:
        loss = squared_error
    elif abs(alpha) < _ALPHA_EPS:
        loss = squared_error.mul(0.5).log1p()
    elif alpha == float("-inf"):
        loss = -torch.expm1(squared_error.mul(-0.5))
    else:
        # ``log1p``/``expm1`` keep the computation stable near ``alpha=0`` and ``alpha=2``.
        b = abs(alpha - 2)
        loss = (alpha / 2 * squared_error.div(b).log1p()).expm1().mul(b / alpha)
    if max_val is not None:
        loss = loss.clamp(max=max_val)
    return reduce_loss(loss, reduction)
