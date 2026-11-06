r"""Contain the implementation of the Poisson regression loss
function."""

from __future__ import annotations

__all__ = ["poisson_regression_loss"]

from typing import TYPE_CHECKING

from sonnix.functional.reduction import reduce_loss

if TYPE_CHECKING:
    import torch

    from sonnix.functional.reduction import Reduction


def poisson_regression_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-8,
    *,
    reduction: Reduction = "mean",
) -> torch.Tensor:
    r"""Compute the Poisson regression loss.

    Loss Functions and Metrics in Deep Learning
    https://arxiv.org/pdf/2307.02694

    Note:
        This loss is similar to
        ``torch.nn.functional.poisson_nll_loss(log_input=False, full=False)``
        with two differences: the predictions are clamped to ``eps``
        (``poisson_nll_loss`` adds ``eps`` instead), and there is no
        Stirling term. The predictions are clamped to ``eps``, so the gradient is zero
        for the predictions lower than ``eps``.

    Args:
        prediction: The count predictions. The values must be
            positive: the loss is not meaningful for negative values.
        target: The count target values. The values must be positive
            (zero is allowed).
        eps: An arbitrary small strictly positive number to avoid
            undefined results when the count is zero.
        reduction: The reduction strategy. The valid values are
            ``'mean'``, ``'none'``, ``'sum'``,
            and ``'batchmean'``.
            ``'none'``: no reduction will be applied, ``'mean'``: the
            sum will be divided by the number of elements in the
            input, ``'sum'``: the output will be summed,
            ``'batchmean'``: the sum will be divided by the size of the
            first dimension.

    Returns:
        The Poisson regression loss. The shape of the tensor depends
            on the reduction strategy.

    Example:
        ```pycon
        >>> import torch
        >>> from sonnix.functional import poisson_regression_loss
        >>> loss = poisson_regression_loss(
        ...     torch.rand(2, 4, requires_grad=True), torch.randint(0, 5, (2, 4)).float()
        ... )
        >>> loss
        tensor(..., grad_fn=<MeanBackward0>)
        >>> loss.backward()

        ```
    """
    loss = prediction - target.mul(prediction.clamp(min=eps).log())
    return reduce_loss(loss, reduction)
