r"""Contain the base class of the numerical encoders."""

from __future__ import annotations

__all__ = ["BaseNumericalEncoder"]

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from torch import nn

if TYPE_CHECKING:
    from torch import Tensor


class BaseNumericalEncoder(nn.Module, ABC):
    r"""Define the base class of the numerical encoders.

    A numerical encoder embeds each of the ``n_features`` numerical
    features of the input into a vector. The input has the shape
    ``(*, n_features)``. The last dimension has to be ``n_features``: a
    size of 1 is not broadcast. The subclasses implement the property
    ``n_features`` and the method ``_encode``, which receives an input
    that was already checked.

    Raises:
        ValueError: if the input has no dimension or if its last
            dimension is not ``n_features``.

    Example:
        ```pycon
        >>> import torch
        >>> from sonnix.modules import BaseNumericalEncoder
        >>> class SquareEncoder(BaseNumericalEncoder):
        ...     @property
        ...     def n_features(self):
        ...         return 2
        ...
        ...     def _encode(self, input):
        ...         return torch.stack([input, input.square()], dim=-1)
        ...
        >>> m = SquareEncoder()
        >>> m(torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
        tensor([[[ 1.,  1.],
                 [ 2.,  4.]],
                [[ 3.,  9.],
                 [ 4., 16.]]])

        ```
    """

    @property
    @abstractmethod
    def n_features(self) -> int:
        r"""Return the input feature size."""

    @abstractmethod
    def _encode(self, input: Tensor) -> Tensor:  # noqa: A002
        r"""Encode an input of shape ``(*, n_features)``."""

    def _check(self, input: Tensor) -> Tensor:  # noqa: A002
        r"""Check the last dimension of the input.

        Args:
            input: The input tensor.

        Returns:
            The input tensor.

        Raises:
            ValueError: if the input has no dimension or if its last
                dimension is not ``n_features``.
        """
        if input.ndim < 1 or input.shape[-1] != self.n_features:
            msg = (
                f"Expected an input of shape (*, n_features={self.n_features}) "
                f"but received {tuple(input.shape)}"
            )
            raise ValueError(msg)
        return input

    def forward(self, input: Tensor) -> Tensor:  # noqa: A002
        return self._encode(self._check(input))
