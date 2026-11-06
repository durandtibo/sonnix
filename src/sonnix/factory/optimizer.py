r"""Contain functions to instantiate a ``torch.optim.Optimizer`` object
from its configuration."""

from __future__ import annotations

__all__ = [
    "is_optimizer_config",
    "resolve_optimizer",
]

from typing import Any, TypeVar

from coola.factory import is_object_config, resolve_object
from torch.optim import Optimizer

T = TypeVar("T")


def is_optimizer_config(config: dict[Any, Any]) -> bool:
    r"""Indicate if the input configuration is a configuration for a
    ``torch.optim.Optimizer``.

    This function only checks if the value of the key  ``_target_``
    is valid. It does not check the other values. If ``_target_``
    indicates a function, the returned type hint is used to check
    the class.

    Args:
        config: The configuration to check.

    Returns:
        ``True`` if the input configuration is a configuration
            for a ``torch.optim.Optimizer`` object, otherwise ``False``.

    Example:
        ```pycon
        >>> from torch import nn
        >>> from sonnix.factory import is_optimizer_config
        >>> linear = nn.Linear(4, 6)
        >>> is_optimizer_config({"_target_": "torch.optim.SGD", "params": linear.parameters()})
        True

        ```
    """
    return is_object_config(config, Optimizer, import_target=True)


def resolve_optimizer(optimizer: Optimizer | dict[Any, Any]) -> Optimizer:
    r"""Resolve a ``torch.optim.Optimizer`` object.

    Warning:
        A configuration (a ``dict`` with a ``_target_`` key) calls its
        ``_target_`` before the type of the result is checked, so it
        must come from a trusted source. See ``sonnix.factory``.

    Args:
        optimizer: The optimizer or its configuration.

    Returns:
        The instantiated ``torch.optim.Optimizer`` object.

    Example:
        ```pycon
        >>> from torch import nn
        >>> from sonnix.factory import resolve_optimizer
        >>> linear = nn.Linear(4, 6)
        >>> sgd = resolve_optimizer({"_target_": "torch.optim.SGD", "params": linear.parameters()})
        >>> sgd
        SGD (
        Parameter Group 0
            dampening: 0
            differentiable: False
            foreach: None
            fused: None
            lr: 0.001
            maximize: False
            momentum: 0
            nesterov: False
            weight_decay: 0
        )

        ```
    """
    return resolve_object(optimizer, Optimizer)
