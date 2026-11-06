r"""Contain a wrapper to flatten the multiclass inputs of a loss."""

from __future__ import annotations

__all__ = ["MulticlassFlatten"]

from typing import Any

import torch
from torch import nn

from sonnix.factory import resolve_module


class MulticlassFlatten(nn.Module):
    r"""Implement a wrapper to flat the multiclass inputs of a
    ``torch.nn.Module``.

    The input prediction tensor shape is ``(d1, d2, ..., dn, C)``
    and is reshaped to ``(d1 * d2 * ... * dn, C)``.
    The input target tensor shape is ``(d1, d2, ..., dn)``
    and is reshaped to ``(d1 * d2 * ... * dn,)``.
    ``reshape`` is used, so non-contiguous inputs are supported.

    Warning:
        A configuration (a ``dict`` with a ``_target_`` key) calls its
        ``_target_`` before the type of the result is checked, so it
        must come from a trusted source. See ``sonnix.factory``.

    Args:
        module: The module (or its configuration) to wrap. It takes
            the flattened prediction and target as inputs and returns
            a tensor.

    Example:
        ```pycon
        >>> import torch
        >>> from sonnix.modules import MulticlassFlatten
        >>> m = MulticlassFlatten(torch.nn.CrossEntropyLoss())
        >>> m
        MulticlassFlatten(
          (module): CrossEntropyLoss()
        )
        >>> out = m(torch.ones(6, 2, 4, requires_grad=True), torch.zeros(6, 2, dtype=torch.long))
        >>> out
        tensor(1.3863, grad_fn=<NllLossBackward0>)

        ```
    """

    def __init__(self, module: nn.Module | dict[Any, Any]) -> None:
        super().__init__()
        self.module: nn.Module = resolve_module(module)

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target = torch.flatten(target)
        return self.module(prediction.reshape(target.numel(), prediction.shape[-1]), target)
