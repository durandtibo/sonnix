r"""Contain utility functions to manage the device(s) of a
``torch.nn.Module``."""

from __future__ import annotations

__all__ = ["get_module_device", "get_module_devices", "is_module_on_device"]

from itertools import chain
from typing import TYPE_CHECKING

import torch
from torch import nn

if TYPE_CHECKING:
    from collections.abc import Iterator


def get_module_device(module: nn.Module) -> torch.device:
    r"""Get the device used by this module.

    This function assumes the module uses a single device. If the
    module uses several devices, you should use
    ``get_module_devices``. The parameters and the buffers are
    taken into account. It returns ``torch.device('cpu')`` if the
    module does not have parameters and buffers.

    Args:
        module: The module.

    Returns:
        The device

    Example:
        ```pycon
        >>> import torch
        >>> from sonnix.utils.device import get_module_device
        >>> get_module_device(torch.nn.Linear(4, 6))
        device(type='cpu')

        ```
    """
    tensor = next(_iter_tensors(module), None)
    if tensor is None:
        return torch.device("cpu")
    return tensor.device


def get_module_devices(module: nn.Module) -> list[torch.device]:
    r"""Get the devices used in a module.

    The parameters and the buffers are taken into account.

    Args:
        module: The module.

    Returns:
        The list of unique ``torch.device``s used in the module,
            sorted by their string representation so the order is
            deterministic.

    Example:
        ```pycon
        >>> import torch
        >>> from sonnix.utils.device import get_module_devices
        >>> get_module_devices(torch.nn.Linear(4, 6))
        [device(type='cpu')]

        ```
    """
    return sorted({tensor.device for tensor in _iter_tensors(module)}, key=str)


def is_module_on_device(module: nn.Module, device: torch.device) -> bool:
    r"""Indicate if all the parameters and buffers of a module are on the
    specified device.

    Args:
        module: The module.
        device: The device.

    Returns:
        ``True`` if all the parameters and buffers of the module are
            on the specified device, otherwise ``False``.

    Example:
        ```pycon
        >>> import torch
        >>> from sonnix.utils.device import is_module_on_device
        >>> is_module_on_device(torch.nn.Linear(4, 6), torch.device("cpu"))
        True

        ```
    """
    return all(tensor.device == device for tensor in _iter_tensors(module))


def _iter_tensors(module: nn.Module) -> Iterator[torch.Tensor]:
    r"""Iterate over the parameters and the buffers of a module.

    Args:
        module: The module.

    Returns:
        An iterator over the parameters, followed by the buffers.
    """
    return chain(module.parameters(), module.buffers())
