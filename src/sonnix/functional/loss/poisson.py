r"""Contain the implementation of the Poisson regression loss
function."""

from __future__ import annotations

__all__ = ["poisson_regression_loss"]

from typing import TYPE_CHECKING

from coola.validation import validate_gt

from sonnix.functional.reduction import reduce_loss

if TYPE_CHECKING:
    import torch

    from sonnix.functional.reduction import Reduction


def poisson_regression_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-8,
    *,
    log_input: bool = False,
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
        Stirling term. The predictions are clamped to ``eps`` in the
        whole loss, so the loss is bounded below and the gradient is
        zero for the predictions lower than ``eps``.

    Note:
        If ``log_input=True``, the predictions are the logarithm of the
        rate, so they have no positivity constraint, and the loss is
        ``exp(prediction) - target * prediction`` (as
        ``torch.nn.functional.poisson_nll_loss(log_input=True)``
        without the Stirling term). ``eps`` is not used.

    Args:
        prediction: The count predictions. If ``log_input=False``, the
            values must be positive: negative values are clamped to
            ``eps``. If ``log_input=True``, the values are the
            logarithm of the predicted rate.
        target: The count target values. The values must be
            non-negative (zero is allowed).
        eps: An arbitrary small strictly positive number to avoid
            undefined results when the count is zero.
        log_input: If ``True``, the predictions are the logarithm of
            the rate.
        reduction: The reduction strategy: ``'none'``, ``'mean'``, ``'sum'`` or
            ``'batchmean'``. See `sonnix.functional.reduce_loss` for the details.

    Returns:
        The Poisson regression loss. The shape of the tensor depends
            on the reduction strategy.

    Raises:
        ValueError: if ``eps`` is not strictly positive.

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
    validate_gt(eps, 0, name="eps")
    if log_input:
        loss = prediction.exp() - target.mul(prediction)
    else:
        rate = prediction.clamp(min=eps)
        loss = rate - target.mul(rate.log())
    return reduce_loss(loss, reduction)
