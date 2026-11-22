r"""Contain utility functions to register the tensors of a
``torch.nn.Module``."""

from __future__ import annotations

__all__ = ["register_tensor"]

from typing import TYPE_CHECKING

from torch import nn

if TYPE_CHECKING:
    from torch import Tensor


def register_tensor(module: nn.Module, name: str, tensor: Tensor, *, learnable: bool) -> None:
    r"""Register a tensor as a parameter or as a buffer.

    A learnable tensor is a parameter and a frozen tensor is a buffer
    with the same name, so both are saved in the state dict under the
    same key and moved with the module. The module owns a copy of
    ``tensor``, detached from its graph.

    Args:
        module: The module that owns the tensor.
        name: The name of the parameter or of the buffer.
        tensor: The initial value.
        learnable: If ``True``, the tensor is registered as a
            parameter, otherwise as a buffer.

    Example:
        ```pycon
        >>> import torch
        >>> from torch import nn
        >>> from sonnix.utils.tensor import register_tensor
        >>> module = nn.Module()
        >>> register_tensor(module, "alpha", torch.ones(2), learnable=True)
        >>> [name for name, _ in module.named_parameters()]
        ['alpha']
        >>> register_tensor(module, "beta", torch.zeros(2), learnable=False)
        >>> [name for name, _ in module.named_buffers()]
        ['beta']

        ```
    """
    if learnable:
        module.register_parameter(name, nn.Parameter(tensor.detach().clone()))
    else:
        module.register_buffer(name, tensor.detach().clone())
