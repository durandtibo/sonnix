r"""Contain relative loss implementations."""

from __future__ import annotations

__all__ = ["RelativeLoss"]

import functools
import inspect
from typing import TYPE_CHECKING, Any

from coola.validation import validate_isinstance

from sonnix.factory import resolve_module
from sonnix.functional import classical_relative_indicator, relative_loss
from sonnix.modules.loss.base import BaseLoss
from sonnix.validation import validate_positive, validate_same_shape

if TYPE_CHECKING:
    from collections.abc import Callable

    import torch
    from torch import nn

    from sonnix.functional import Reduction


def _callable_repr(function: Callable) -> str:
    r"""Return a short representation of a callable.

    The ``repr`` of a function has the address of the function, which
    changes at each run. A function is shown by its name, and a
    ``functools.partial`` by its function and its arguments.

    Args:
        function: The callable to represent.

    Returns:
        The representation, for example ``classical_relative_indicator``
            or ``functools.partial(moment_mean_indicator, k=2)``.

    Example:
        ```pycon
        >>> from functools import partial
        >>> from sonnix.functional import moment_mean_indicator
        >>> from sonnix.modules.loss.relative import _callable_repr
        >>> _callable_repr(moment_mean_indicator)
        'moment_mean_indicator'
        >>> _callable_repr(partial(moment_mean_indicator, k=2))
        'functools.partial(moment_mean_indicator, k=2)'

        ```
    """
    if isinstance(function, functools.partial):
        arguments = [
            _callable_repr(function.func),
            *(repr(arg) for arg in function.args),
            *(f"{key}={value!r}" for key, value in function.keywords.items()),
        ]
        return f"functools.partial({', '.join(arguments)})"
    return function.__name__ if inspect.isroutine(function) else repr(function)


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
        eps: An arbitrary small finite, strictly positive number to avoid
            undefined results when the indicator is zero. It is raised to
            the smallest normal number of the dtype of the indicator.
        reduction: The reduction strategy: ``'none'``, ``'mean'``, ``'sum'`` or
            ``'batchmean'``. See `sonnix.functional.reduce_loss` for the details.
        detach_indicator: If ``True``, the indicator is detached from
            the graph so no gradient flows through the denominator.
            By default (``False``) the gradients flow through the
            indicator, which can reward shrinking it (e.g. with
            ``reversed_relative_indicator``).

    Raises:
        TypeError: if ``indicator`` is not callable.
        TypeError: if ``eps`` is not a number or ``detach_indicator`` is not a ``bool``.
        ValueError: if ``eps`` is not finite and strictly positive, or if
            ``prediction`` and ``target`` do not have the same shape
            (checked before the criterion is called).

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
          reduction=mean, indicator=classical_relative_indicator, eps=1e-08, detach_indicator=False
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
        *,
        indicator: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
        eps: float = 1e-8,
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

        validate_positive(eps, name="eps")
        validate_isinstance(detach_indicator, bool, name="detach_indicator")
        self.eps = float(eps)
        self.detach_indicator = detach_indicator

    def extra_repr(self) -> str:
        return (
            f"{super().extra_repr()}, indicator={_callable_repr(self.indicator)}, "
            f"eps={self.eps}, detach_indicator={self.detach_indicator}"
        )

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        validate_same_shape(prediction, target, name1="prediction", name2="target")
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
