r"""Contain functions to instantiate a ``torch.nn.Module`` object from
its configuration."""

from __future__ import annotations

__all__ = [
    "create_sequential",
    "is_module_config",
    "resolve_module",
]

from typing import TYPE_CHECKING, Any, TypeVar

from coola.factory import is_object_config, resolve_object
from torch.nn import Module, Sequential

if TYPE_CHECKING:
    from collections.abc import Sequence

T = TypeVar("T")


def is_module_config(config: dict[Any, Any]) -> bool:
    r"""Indicate if the input configuration is a configuration for a
    ``torch.nn.Module``.

    This function only checks if the value of the key  ``_target_``
    is valid. It does not check the other values. If ``_target_``
    indicates a function, the returned type hint is used to check
    the class.

    Args:
        config: The configuration to check.

    Returns:
        ``True`` if the input configuration is a configuration
            for a ``torch.nn.Module`` object, otherwise ``False``.

    Example:
        ```pycon
        >>> from sonnix.factory import is_module_config
        >>> is_module_config({"_target_": "torch.nn.Identity"})
        True

        ```
    """
    return is_object_config(config, Module, import_target=True)


def resolve_module(module: Module | dict[Any, Any]) -> Module:
    r"""Resolve a ``torch.nn.Module`` object.

    Warning:
        A configuration (a ``dict`` with a ``_target_`` key) calls its
        ``_target_`` before the type of the result is checked, so it
        must come from a trusted source. See ``sonnix.factory``.

    Args:
        module: The module or its configuration.

    Returns:
        The instantiated ``torch.nn.Module`` object.

    Example:
        ```pycon
        >>> from sonnix.factory import resolve_module
        >>> linear = resolve_module(
        ...     {"_target_": "torch.nn.Linear", "in_features": 4, "out_features": 6}
        ... )
        >>> linear
        Linear(in_features=4, out_features=6, bias=True)

        ```
    """
    return resolve_object(module, Module)


def create_sequential(modules: Sequence[Module | dict[Any, Any]]) -> Sequential:
    r"""Create a ``torch.nn.Sequential`` from a sequence of modules.

    Warning:
        A configuration (a ``dict`` with a ``_target_`` key) calls its
        ``_target_`` before the type of the result is checked, so it
        must come from a trusted source. See ``sonnix.factory``.

    Args:
        modules: The sequence of modules or their configuration.

    Returns:
        The instantiated ``torch.nn.Sequential`` object.

    Example:
        ```pycon
        >>> from sonnix.factory import create_sequential
        >>> seq = create_sequential(
        ...     [
        ...         {"_target_": "torch.nn.Linear", "in_features": 4, "out_features": 6},
        ...         {"_target_": "torch.nn.ReLU"},
        ...         {"_target_": "torch.nn.Linear", "in_features": 6, "out_features": 6},
        ...     ]
        ... )
        >>> seq
        Sequential(
          (0): Linear(in_features=4, out_features=6, bias=True)
          (1): ReLU()
          (2): Linear(in_features=6, out_features=6, bias=True)
        )

        ```
    """
    return Sequential(*[resolve_module(module) for module in modules])
