r"""Contain functions to validate tensors."""

from __future__ import annotations

__all__ = ["validate_same_shape"]

import torch
from coola.validation import validate_isinstance


def validate_same_shape(
    tensor1: torch.Tensor,
    tensor2: torch.Tensor,
    *,
    name1: str = "tensor1",
    name2: str = "tensor2",
) -> None:
    r"""Validate that two tensors have the same shape.

    Args:
        tensor1: The first tensor to validate.
        tensor2: The second tensor to validate.
        name1: The name of the first tensor, used in the error message.
        name2: The name of the second tensor, used in the error message.

    Raises:
        TypeError: If ``tensor1`` or ``tensor2`` is not a tensor.
        ValueError: If the shapes of ``tensor1`` and ``tensor2`` differ.

    Example:
        ```pycon
        >>> import torch
        >>> from sonnix.validation import validate_same_shape
        >>> validate_same_shape(torch.ones(2, 3), torch.zeros(2, 3))

        ```
    """
    validate_isinstance(tensor1, torch.Tensor, name=name1)
    validate_isinstance(tensor2, torch.Tensor, name=name2)
    if tensor1.shape != tensor2.shape:
        msg = (
            f"{name1} and {name2} must have the same shape, "
            f"got {tuple(tensor1.shape)} and {tuple(tensor2.shape)}"
        )
        raise ValueError(msg)
