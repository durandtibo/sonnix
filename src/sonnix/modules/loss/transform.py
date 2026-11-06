r"""Contain a loss function where the predictions and targets are
transformed before to be fed to the loss function."""

from __future__ import annotations

__all__ = ["TransformedLoss"]

from typing import Any

import torch
from torch import nn

from sonnix.factory import resolve_module


class TransformedLoss(nn.Module):
    r"""Implement a loss function where the predictions and targets are
    transformed before to be fed to the loss function.

    Args:
        criterion: The criterion or its configuration. The loss has
            two inputs: predictions and targets.
        prediction_transform: The transformation for the predictions or
            its configuration. If ``None``, the identity transformation
            is used.
        target_transform: The transformation for the targets or its
            configuration. If ``None``, the identity transformation
            is used.

    Example:
        ```pycon
        >>> import torch
        >>> from sonnix.modules import TransformedLoss, Asinh
        >>> criterion = TransformedLoss(
        ...     criterion=torch.nn.SmoothL1Loss(),
        ...     prediction_transform=Asinh(),
        ...     target_transform=Asinh(),
        ... )
        >>> loss = criterion(torch.randn(2, 4, requires_grad=True), torch.randn(2, 4))
        >>> loss
        tensor(..., grad_fn=<SmoothL1LossBackward0>)
        >>> loss.backward()

        ```
    """

    def __init__(
        self,
        criterion: nn.Module | dict[Any, Any],
        prediction_transform: nn.Module | dict[Any, Any] | None = None,
        target_transform: nn.Module | dict[Any, Any] | None = None,
    ) -> None:
        super().__init__()
        self.criterion: nn.Module = resolve_module(criterion)
        self.prediction_transform: nn.Module = resolve_module(
            nn.Identity() if prediction_transform is None else prediction_transform
        )
        self.target_transform: nn.Module = resolve_module(
            nn.Identity() if target_transform is None else target_transform
        )

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        prediction = self.prediction_transform(prediction)
        target = self.target_transform(target)
        return self.criterion(prediction, target)
