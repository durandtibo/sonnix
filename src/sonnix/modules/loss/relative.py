r"""Contain relative loss implementations."""

from __future__ import annotations

__all__ = ["RelativeLoss", "RelativeMSELoss", "RelativeSmoothL1Loss"]

from typing import TYPE_CHECKING, Any

import torch
from torch import nn

from sonnix.functional import relative_loss
from sonnix.modules.loss.base import BaseLoss
from sonnix.modules.loss.indicators import ClassicalRelativeIndicator
from sonnix.utils.factory import resolve_module

if TYPE_CHECKING:
    from sonnix.functional.reduction import Reduction
    from sonnix.modules.loss.indicators import BaseRelativeIndicator


class RelativeLoss(BaseLoss):
    r"""Implement a "generic" relative loss that takes as input a
    criterion.

    Args:
        criterion: The criterion or its configuration. This criterion
            should not have reduction to be compatible with the shapes
            of the prediction and targets.
        indicator: The indicator module or its configuration. If
            ``None``, ``ClassicalRelativeIndicator`` is instantiated.
        eps: An arbitrary small strictly positive number to avoid
            undefined results when the indicator is zero.
        detach_indicator: If ``True``, the indicator is detached from
            the graph so no gradient flows through the denominator.
            By default (``False``) the gradients flow through the
            indicator, which can reward shrinking it (e.g. with
            ``ReversedRelativeIndicator``).
        reduction: The reduction strategy. The valid values are
            ``'mean'``, ``'none'``, ``'sum'``,
            and ``'batchmean'``.
            ``'none'``: no reduction will be applied, ``'mean'``: the
            sum will be divided by the number of elements in the
            input, ``'sum'``: the output will be summed,
            ``'batchmean'``: the sum will be divided by the size of the
            first dimension.

    Example:
        ```pycon
        >>> import torch
        >>> from sonnix.modules import RelativeLoss
        >>> from sonnix.modules.loss import ClassicalRelativeIndicator
        >>> criterion = RelativeLoss(
        ...     criterion=torch.nn.MSELoss(reduction="none"),
        ...     indicator=ClassicalRelativeIndicator(),
        ... )
        >>> criterion
        RelativeLoss(
          eps=1e-08, reduction=mean
          (criterion): MSELoss()
          (indicator): ClassicalRelativeIndicator()
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
        indicator: BaseRelativeIndicator | dict[Any, Any] | None = None,
        eps: float = 1e-8,
        *,
        reduction: Reduction = "mean",
        detach_indicator: bool = False,
    ) -> None:
        super().__init__(reduction=reduction)
        self.criterion: nn.Module = resolve_module(criterion)
        self.indicator: nn.Module = resolve_module(
            ClassicalRelativeIndicator() if indicator is None else indicator
        )

        self._eps = eps
        self._detach_indicator = bool(detach_indicator)

    def extra_repr(self) -> str:
        return f"eps={self._eps}, reduction={self.reduction}, detach_indicator={self._detach_indicator}"

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = self.criterion(prediction, target)
        indicator = self.indicator(prediction, target)
        if self._detach_indicator:
            indicator = indicator.detach()
        return relative_loss(
            loss=loss,
            indicator=indicator,
            reduction=self.reduction,
            eps=self._eps,
        )


class RelativeMSELoss(RelativeLoss):
    r"""Implement the relative MSE loss.

    Args:
        indicator: The indicator module or its configuration. If
            ``None``, ``ClassicalRelativeIndicator`` is instantiated.
        eps: An arbitrary small strictly positive number to avoid
            undefined results when the indicator is zero.
        detach_indicator: If ``True``, the indicator is detached from
            the graph so no gradient flows through the denominator.
            By default (``False``) the gradients flow through the
            indicator, which can reward shrinking it (e.g. with
            ``ReversedRelativeIndicator``).
        reduction: The reduction strategy. The valid values are
            ``'mean'``, ``'none'``, ``'sum'``,
            and ``'batchmean'``.
            ``'none'``: no reduction will be applied, ``'mean'``: the
            sum will be divided by the number of elements in the
            input, ``'sum'``: the output will be summed,
            ``'batchmean'``: the sum will be divided by the size of the
            first dimension.

    Example:
        ```pycon
        >>> import torch
        >>> from sonnix.modules import RelativeMSELoss
        >>> from sonnix.modules.loss import ClassicalRelativeIndicator
        >>> criterion = RelativeMSELoss(indicator=ClassicalRelativeIndicator())
        >>> criterion
        RelativeMSELoss(
          eps=1e-08, reduction=mean
          (criterion): MSELoss()
          (indicator): ClassicalRelativeIndicator()
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
        indicator: BaseRelativeIndicator | dict[Any, Any] | None = None,
        eps: float = 1e-8,
        *,
        reduction: Reduction = "mean",
        detach_indicator: bool = False,
    ) -> None:
        super().__init__(
            criterion=nn.MSELoss(reduction="none"),
            indicator=indicator,
            reduction=reduction,
            eps=eps,
            detach_indicator=detach_indicator,
        )


class RelativeSmoothL1Loss(RelativeLoss):
    r"""Implement the relative smooth L1 loss.

    Args:
        indicator: The indicator module or its configuration. If
            ``None``, ``ClassicalRelativeIndicator`` is instantiated.
        eps: An arbitrary small strictly positive number to avoid
            undefined results when the indicator is zero.
        detach_indicator: If ``True``, the indicator is detached from
            the graph so no gradient flows through the denominator.
            By default (``False``) the gradients flow through the
            indicator, which can reward shrinking it (e.g. with
            ``ReversedRelativeIndicator``).
        beta: The threshold at which to change between L1 and L2
            loss. The value must be non-negative.
        reduction: The reduction strategy. The valid values are
            ``'mean'``, ``'none'``, ``'sum'``,
            and ``'batchmean'``.
            ``'none'``: no reduction will be applied, ``'mean'``: the
            sum will be divided by the number of elements in the
            input, ``'sum'``: the output will be summed,
            ``'batchmean'``: the sum will be divided by the size of the
            first dimension.

    Example:
        ```pycon
        >>> import torch
        >>> from sonnix.modules import RelativeSmoothL1Loss
        >>> from sonnix.modules.loss import ClassicalRelativeIndicator
        >>> criterion = RelativeSmoothL1Loss(indicator=ClassicalRelativeIndicator())
        >>> criterion
        RelativeSmoothL1Loss(
          eps=1e-08, reduction=mean
          (criterion): SmoothL1Loss()
          (indicator): ClassicalRelativeIndicator()
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
        indicator: BaseRelativeIndicator | dict[Any, Any] | None = None,
        eps: float = 1e-8,
        *,
        reduction: Reduction = "mean",
        detach_indicator: bool = False,
        beta: float = 1.0,
    ) -> None:
        super().__init__(
            criterion=nn.SmoothL1Loss(reduction="none", beta=beta),
            indicator=indicator,
            reduction=reduction,
            eps=eps,
            detach_indicator=detach_indicator,
        )
