r"""Contain helper functions shared by the numerical encoders."""

from __future__ import annotations

__all__ = ["prepare_tensor_param"]

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor


def prepare_tensor_param(tensor: Tensor, name: str) -> Tensor:
    r"""Prepare a tensor parameter to be a 2d tensor.

    Args:
        tensor: The tensor to prepare.
        name: The name associated to the tensor.

    Returns:
        The prepared tensor.

    Raises:
        ValueError: if the input tensor is not a 1d or 2d tensor.

    Example:
        ```pycon
        >>> import torch
        >>> from sonnix.modules.numerical._utils import prepare_tensor_param
        >>> prepare_tensor_param(torch.tensor([1.0, 2.0, 4.0]), name="scale")
        tensor([[1., 2., 4.]])

        ```
    """
    if tensor.ndim == 1:
        tensor = tensor.view(1, -1)
    if tensor.ndim != 2:
        msg = f"Incorrect shape for '{name}': {tensor.shape}"
        raise ValueError(msg)
    return tensor
