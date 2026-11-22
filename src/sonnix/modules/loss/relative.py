r"""Contain relative loss implementations."""

from __future__ import annotations

__all__ = ["RelativeLoss"]

from typing import TYPE_CHECKING, Any

from coola.validation import validate_gt

from sonnix.factory import resolve_module
from sonnix.functional import classical_relative_indicator, relative_loss
from sonnix.modules.loss.base import BaseLoss

if TYPE_CHECKING:
    from collections.abc import Callable

    import torch
    from torch import nn

    from sonnix.functional.reduction import Reduction


class RelativeLoss(BaseLoss):
    r"""Implement a "generic" relative loss that takes as input a
    criterion.

    Warning:
        A configuration (a ``dict`` with a ``_target_`` key) calls its
        ``_target_`` before the type of the result is checked, so it
        must come from a trusted source. See ``sonnix.factory``.

    Args:
        criterion: The criterion or its configuration. This criterion
            should not have reduction to be compatible with the shapes
            of the prediction and targets.
        indicator: The indicator function. It takes the prediction
            and the target and returns a non-negative tensor of the
            same shape (see the ``*_indicator`` functions of
            ``sonnix.functional``). Use ``functools.partial`` to set
            its options. If ``None``,
            ``classical_relative_indicator`` is used.
        eps: An arbitrary small strictly positive number to avoid
            undefined results when the indicator is zero.
        detach_indicator: If ``True``, the indicator is detached from
            the graph so no gradient flows through the denominator.
            By default (``False``) the gradients flow through the
            indicator, which can reward shrinking it (e.g. with
            ``reversed_relative_indicator``).
        reduction: The reduction strategy: ``'none'``, ``'mean'``, ``'sum'`` or
            ``'batchmean'``. See `sonnix.functional.reduce_loss` for the details.

    Raises:
        TypeError: if ``indicator`` is not callable.
        ValueError: if ``eps`` is not strictly positive.

    Example:
        ```pycon
        >>> import torch
        >>> from sonnix.modules import RelativeLoss
        >>> from sonnix.functional import classical_relative_indicator
        >>> criterion = RelativeLoss(
        ...     criterion=torch.nn.MSELoss(reduction="none"),
        ...     indicator=classical_relative_indicator,
        ... )
        >>> criterion
        RelativeLoss(
          reduction=mean, eps=1e-08, detach_indicator=False
          (criterion): MSELoss()
        )
        >>> prediction = torch.randn(3, 5, requires_grad=True)
        >>> target = torch.randn(3, 5)
        >>> loss = criterion(prediction, target)
        >>> loss
        tensor(..., grad_fn=<MeanBackward0>)
        >>> loss.backward()

        ```
    """

    def __init__(
        self,
        criterion: nn.Module | dict[Any, Any],
        indicator: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
        eps: float = 1e-8,
        *,
        reduction: Reduction = "mean",
        detach_indicator: bool = False,
    ) -> None:
        super().__init__(reduction=reduction)
        self.criterion: nn.Module = resolve_module(criterion)
        if indicator is None:
            indicator = classical_relative_indicator
        if not callable(indicator):
            msg = f"indicator must be callable but received {indicator!r}"
            raise TypeError(msg)
        self.indicator = indicator

        validate_gt(eps, 0, name="eps")
        self.eps = float(eps)
        self.detach_indicator = bool(detach_indicator)

    def extra_repr(self) -> str:
        return f"{super().extra_repr()}, eps={self.eps}, detach_indicator={self.detach_indicator}"

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = self.criterion(prediction, target)
        indicator = self.indicator(prediction, target)
        if self.detach_indicator:
            indicator = indicator.detach()
        return relative_loss(
            loss=loss,
            indicator=indicator,
            reduction=self.reduction,
            eps=self.eps,
        )
