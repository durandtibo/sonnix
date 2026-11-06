r"""Contain the base class of the fusion modules."""

from __future__ import annotations

__all__ = ["BaseFusion"]

from abc import ABC, abstractmethod

import torch
from torch import nn


class BaseFusion(nn.Module, ABC):
    r"""Define the base class of the modules that fuse several tensors
    into one tensor.

    It checks that there is at least one input, then calls ``fuse``.
    The subclasses implement ``fuse``.

    Raises:
        ValueError: if ``forward`` is called without input.

    Example:
        ```pycon
        >>> import torch
        >>> from sonnix.modules.fusion import BaseFusion
        >>> class MaxFusion(BaseFusion):
        ...     def fuse(self, inputs):
        ...         return torch.stack(inputs).amax(dim=0)
        ...
        >>> module = MaxFusion()
        >>> module(torch.tensor([1.0, 5.0]), torch.tensor([3.0, 2.0]))
        tensor([3., 5.])

        ```
    """

    def forward(self, *inputs: torch.Tensor) -> torch.Tensor:
        if not inputs:
            msg = f"{self.__class__.__qualname__} needs at least one tensor as input"
            raise ValueError(msg)
        return self.fuse(inputs)

    @abstractmethod
    def fuse(self, inputs: tuple[torch.Tensor, ...]) -> torch.Tensor:
        r"""Fuse the inputs.

        Args:
            inputs: The tensors to fuse. There is at least one tensor.

        Returns:
            The fused tensor.
        """
