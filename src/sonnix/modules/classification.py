r"""Contain the classification specific layers."""

from __future__ import annotations

__all__ = ["ToBinaryLabel", "ToCategoricalLabel"]

import torch
from coola.validation import validate_finite
from torch import nn


class ToBinaryLabel(nn.Module):
    r"""Implement a ``torch.nn.Module`` to compute binary labels from
    scores by thresholding.

    The output label is ``1`` if the value is greater than the
    threshold, and ``0`` otherwise.

    Args:
        threshold: The threshold value used to compute the binary
            labels.

    Raises:
        ValueError: If ``threshold`` is not finite.

    Example:
        ```pycon
        >>> import torch
        >>> from sonnix.modules import ToBinaryLabel
        >>> transform = ToBinaryLabel()
        >>> transform
        ToBinaryLabel(threshold=0.0)
        >>> out = transform(torch.tensor([-1.0, 1.0, -2.0, 1.0]))
        >>> out
        tensor([0, 1, 0, 1])

        ```
    """

    def __init__(self, threshold: float = 0.0) -> None:
        super().__init__()
        validate_finite(threshold, name="threshold")
        self.threshold = float(threshold)

    def extra_repr(self) -> str:
        return f"threshold={self.threshold}"

    def forward(self, input: torch.Tensor) -> torch.Tensor:  # noqa: A002
        r"""Compute binary labels from scores.

        Args:
            input: The scores used to compute the binary labels.
                This input must be a ``torch.Tensor`` of type float
                and shape ``(d0, d1, ..., dn)``

        Returns:
            The computed binary labels where the values are ``0`` and
                ``1``. The output is a ``torch.Tensor`` of type long
                and shape ``(d0, d1, ..., dn)``.

        Example:
            ```pycon
            >>> import torch
            >>> from sonnix.modules import ToBinaryLabel
            >>> transform = ToBinaryLabel()
            >>> out = transform(torch.tensor([-1.0, 1.0, -2.0, 1.0]))
            >>> out
            tensor([0, 1, 0, 1])

            ```
        """
        return (input > self.threshold).long()


class ToCategoricalLabel(nn.Module):
    r"""Implement a ``torch.nn.Module`` to compute categorical labels
    from scores.

    Args:
        dim: The dimension of the classes.

    Example:
        ```pycon
        >>> import torch
        >>> from sonnix.modules import ToCategoricalLabel
        >>> transform = ToCategoricalLabel()
        >>> transform
        ToCategoricalLabel(dim=-1)
        >>> out = transform(torch.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 3.0, 2.0, 2.0]]))
        >>> out
        tensor([3, 0])

        ```
    """

    def __init__(self, dim: int = -1) -> None:
        super().__init__()
        self.dim = dim

    def extra_repr(self) -> str:
        return f"dim={self.dim}"

    def forward(self, input: torch.Tensor) -> torch.Tensor:  # noqa: A002
        r"""Compute categorical labels from scores.

        Args:
            input: The scores used to compute the categorical labels.
                This input must be a ``torch.Tensor`` of type float
                where the dimension ``dim`` has size ``num_classes``,
                for example ``(d0, d1, ..., dn, num_classes)`` with
                the default ``dim``.

        Returns:
            The computed categorical labels where the values are in
                ``{0, 1, ..., num_classes-1}``. The output is a
                ``torch.Tensor`` of type long and shape
                ``(d0, d1, ..., dn)``.

        Example:
            ```pycon
            >>> import torch
            >>> from sonnix.modules import ToCategoricalLabel
            >>> transform = ToCategoricalLabel()
            >>> out = transform(torch.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 3.0, 2.0, 2.0]]))
            >>> out
            tensor([3, 0])

            ```
        """
        return input.argmax(dim=self.dim, keepdim=False)
