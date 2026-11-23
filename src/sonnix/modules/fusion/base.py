r"""Contain the base class of the fusion modules."""

from __future__ import annotations

__all__ = ["BaseFusion"]

from abc import ABC, abstractmethod

import torch
from torch import nn

from sonnix.validation import validate_same_shape


class BaseFusion(nn.Module, ABC):
    r"""Define the base class of the modules that fuse several tensors
    into one tensor.

    It checks that there is at least one input, and that all the
    inputs have the same shape (nothing broadcasts), then calls
    ``fuse``. The subclasses implement ``fuse``. A subclass that
    accepts inputs of different shapes (``ConcatFusion``) sets the
    class attribute ``_same_shape`` to ``False``.

    Raises:
        ValueError: if ``forward`` is called without input, or if an
            input does not have the shape of the first input.

    Example:
        ```pycon
        >>> import torch
        >>> from sonnix.modules import BaseFusion
        >>> class MaxFusion(BaseFusion):
        ...     def fuse(self, inputs):
        ...         return torch.stack(inputs).amax(dim=0)
        ...
        >>> module = MaxFusion()
        >>> module(torch.tensor([1.0, 5.0]), torch.tensor([3.0, 2.0]))
        tensor([3., 5.])

        ```
    """

    _same_shape: bool = True

    def forward(self, *inputs: torch.Tensor) -> torch.Tensor:
        if not inputs:
            msg = f"{self.__class__.__qualname__} needs at least one tensor as input"
            raise ValueError(msg)
        if self._same_shape:
            for i, x in enumerate(inputs[1:], start=1):
                validate_same_shape(inputs[0], x, name1="inputs[0]", name2=f"inputs[{i}]")
        return self.fuse(inputs)

    @staticmethod
    def _sum(inputs: tuple[torch.Tensor, ...]) -> torch.Tensor:
        r"""Sum the inputs.

        It is shared by the fusions that are built on a sum
        (``SumFusion`` and ``AverageFusion``).

        Args:
            inputs: The tensors to sum. There is at least one tensor.

        Returns:
            The sum of the tensors.

        Example:
            ```pycon
            >>> import torch
            >>> from sonnix.modules import BaseFusion
            >>> BaseFusion._sum((torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0])))
            tensor([4., 6.])

            ```
        """
        output = inputs[0]
        for x in inputs[1:]:
            output = output.add(x)
        return output

    @abstractmethod
    def fuse(self, inputs: tuple[torch.Tensor, ...]) -> torch.Tensor:
        r"""Fuse the inputs.

        Args:
            inputs: The tensors to fuse. There is at least one tensor.

        Returns:
            The fused tensor.
        """
