r"""Contain the general robust regression loss a.k.a the Barron robust
loss."""

from __future__ import annotations

__all__ = ["general_robust_regression_loss"]

import torch

from sonnix.functional.reduction import reduce_loss

# ``alpha`` values closer than ``ALPHA_EPS`` to 0 or 2 use the limit formulas.
ALPHA_EPS = 1e-5


def general_robust_regression_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    alpha: float = 2.0,
    scale: float = 1.0,
    max: float | None = None,  # noqa: A002
    reduction: str = "mean",
) -> torch.Tensor:
    r"""Compute the general robust regression loss a.k.a. Barron robust
    loss.

    Based on the paper:

        A General and Adaptive Robust Loss Function
        Jonathan T. Barron
        CVPR 2019 (https://arxiv.org/abs/1701.03077)

    Note:
        The "adaptative" part of the loss is not implemented in this
            function.

    Note:
        If ``max`` is set, the loss is clamped, so the gradient is zero
        for the elements whose loss is larger than ``max``.

    Args:
        prediction: The predictions.
        target: The target values.
        alpha: The shape parameter that controls the robustness of the
            loss. ``-inf`` is supported (Welsch loss). The limit
            cases ``alpha=2``
            (``x^2``), ``alpha=0`` (``log(0.5 * x^2 + 1)``) are
            used when ``alpha`` is within ``1e-5`` of 0 or 2.
        scale: The scale parameter that controls the size of the loss's
            quadratic bowl near 0.
        max: The max value to clip the loss before to compute the
            reduction. ``None`` means no clipping is used.
        reduction: The reduction strategy. The valid values are
            ``'mean'``, ``'none'``, ``'sum'``,
            and ``'batchmean'``.
            ``'none'``: no reduction will be applied, ``'mean'``: the
            sum will be divided by the number of elements in the
            input, ``'sum'``: the output will be summed,
            ``'batchmean'``: the sum will be divided by the size of the
            first dimension.

    Returns:
        The loss. The shape of the tensor depends on the reduction
            strategy.

    Raises:
        ValueError: if the reduction is not valid.

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
    squared_error = prediction.sub(target).div(scale).pow(2)
    if abs(alpha - 2) < ALPHA_EPS:
        loss = squared_error
    elif abs(alpha) < ALPHA_EPS:
        loss = squared_error.mul(0.5).log1p()
    elif alpha == float("-inf"):
        loss = -torch.expm1(squared_error.mul(-0.5))
    else:
        # ``log1p``/``expm1`` keep the computation stable near ``alpha=0`` and ``alpha=2``.
        b = abs(alpha - 2)
        loss = (alpha / 2 * squared_error.div(b).log1p()).expm1().mul(b / alpha)
    if max is not None:
        loss = loss.clamp(max=max)
    return reduce_loss(loss, reduction)
