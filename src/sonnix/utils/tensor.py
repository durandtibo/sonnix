r"""Contain utility functions to register the tensors of a
``torch.nn.Module`` and to adapt the numerical guards to the dtype of a
tensor."""

from __future__ import annotations

__all__ = ["register_tensor", "resolve_eps", "resolve_max_exponent"]

import math
from typing import TYPE_CHECKING

import torch
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
        >>> from sonnix.utils import register_tensor
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


def resolve_eps(eps: float, tensor: Tensor) -> float:
    r"""Raise a guard to the smallest positive normal number of the
    dtype of a tensor.

    A small guard such as ``1e-8`` is ``0`` in ``float16`` (the
    smallest normal number is ``6.1e-5``), so a division or a
    logarithm guarded by it is not protected. The guard is raised to
    the smallest normal number of the dtype when it is lower. A
    tensor that is not floating point uses the default dtype.

    Args:
        eps: The guard, a small strictly positive number.
        tensor: The tensor whose dtype is used.

    Returns:
        The maximum of ``eps`` and the smallest positive normal number
            of the dtype of ``tensor``.

    Example:
        ```pycon
        >>> import torch
        >>> from sonnix.utils import resolve_eps
        >>> resolve_eps(1e-8, torch.zeros(1, dtype=torch.float16))
        6.103515625e-05
        >>> resolve_eps(1e-8, torch.zeros(1))
        1e-08

        ```
    """
    return max(eps, torch.finfo(_get_float_dtype(tensor)).tiny)


def resolve_max_exponent(max_val: float, tensor: Tensor) -> float:
    r"""Lower an exponent bound so that ``exp`` of it is finite in the
    dtype of a tensor.

    ``exp(20)`` is ``inf`` in ``float16`` (the largest number is
    ``65504``). The bound is lowered to ``log(finfo.max) - 1`` when it
    is higher: the margin of ``1`` is needed because ``log(65504)``
    rounds up in ``float16``. A tensor that is not floating point
    uses the default dtype.

    Args:
        max_val: The upper bound of the exponent.
        tensor: The tensor whose dtype is used.

    Returns:
        The minimum of ``max_val`` and ``log(finfo.max) - 1``, where
            ``finfo`` is the one of the dtype of ``tensor``.

    Example:
        ```pycon
        >>> import torch
        >>> from sonnix.utils import resolve_max_exponent
        >>> resolve_max_exponent(20.0, torch.zeros(1, dtype=torch.float16))
        10.089866488461016
        >>> resolve_max_exponent(20.0, torch.zeros(1))
        20.0

        ```
    """
    return min(max_val, math.log(torch.finfo(_get_float_dtype(tensor)).max) - 1.0)


def _get_float_dtype(tensor: Tensor) -> torch.dtype:
    r"""Return the floating point dtype of a tensor.

    Args:
        tensor: The tensor.

    Returns:
        The dtype of ``tensor`` if it is floating point, otherwise the
            default dtype of ``torch``.
    """
    return tensor.dtype if tensor.is_floating_point() else torch.get_default_dtype()
