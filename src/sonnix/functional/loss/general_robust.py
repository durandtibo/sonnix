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
    alpha: float | torch.Tensor = 2.0,
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

    Args:
        prediction: The predictions.
        target: The target values.
        alpha: The shape parameter that controls the robustness of the
            loss. It can be a float or a tensor that is broadcastable
            with the error (e.g. a learnable parameter). ``-inf`` is
            supported (Welsch loss). The limit cases ``alpha=2``
            (``x^2``), ``alpha=0`` (``log(0.5 * x^2 + 1)``) are
            used when ``alpha`` is within ``1e-5`` of 0 or 2.
        scale: The scale parameter that controls the size of the loss's
            quadratic bowl near 0.
        max: The max value to clip the loss before to compute the
            reduction. ``None`` means no clipping is used.
        reduction: The reduction strategy. The valid values are
            ``'mean'``, ``'none'``,  and ``'sum'``.
            ``'none'``: no reduction will be applied, ``'mean'``: the
            sum will be divided by the number of elements in the
            input, ``'sum'``: the output will be summed.

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
    alpha_t = torch.as_tensor(alpha, dtype=squared_error.dtype, device=squared_error.device)
    is_two = (alpha_t - 2).abs() < ALPHA_EPS
    is_zero = alpha_t.abs() < ALPHA_EPS
    is_neg_inf = torch.isneginf(alpha_t)
    is_special = is_two | is_zero | is_neg_inf

    # Generic case, computed with ``log1p``/``expm1`` for numerical stability near
    # ``alpha=0`` and ``alpha=2``. The special cases are replaced by a dummy value
    # (``alpha=1``) to avoid NaN values in the gradient of the unused branches.
    safe_alpha = torch.where(is_special, torch.ones_like(alpha_t), alpha_t)
    b = (safe_alpha - 2).abs()
    generic = (safe_alpha / 2 * squared_error.div(b).log1p()).expm1().mul(b / safe_alpha)

    loss = torch.where(is_two, squared_error, generic)
    loss = torch.where(is_zero, squared_error.mul(0.5).log1p(), loss)
    loss = torch.where(is_neg_inf, -torch.expm1(squared_error.mul(-0.5)), loss)
    if max is not None:
        loss = loss.clamp(max=max)
    return reduce_loss(loss, reduction)
