r"""Contain functions to instantiate a ``torch.nn.Module`` object from
its configuration."""

from __future__ import annotations

__all__ = [
    "create_sequential",
    "is_dataset_config",
    "is_module_config",
    "is_optimizer_config",
    "resolve_dataset",
    "resolve_module",
    "resolve_optimizer",
]

from typing import TYPE_CHECKING, Any, TypeVar

from coola.factory import is_object_config, resolve_object
from torch.nn import Module, Sequential
from torch.optim import Optimizer
from torch.utils.data import Dataset

if TYPE_CHECKING:
    from collections.abc import Sequence

T = TypeVar("T")


def is_dataset_config(config: dict[Any, Any]) -> bool:
    r"""Indicate if the input configuration is a configuration for a
    ``torch.utils.data.Dataset``.

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
        >>> from sonnix.utils.factory import is_dataset_config
        >>> is_dataset_config(
        ...     {
        ...         "_target_": "sonnix.testing.dummy.DummyDataset",
        ...         "num_examples": 10,
        ...         "feature_size": 4,
        ...     }
        ... )
        True

        ```
    """
    return is_object_config(config, Dataset, import_target=True)


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
        >>> from sonnix.utils.factory import is_module_config
        >>> is_module_config({"_target_": "torch.nn.Identity"})
        True

        ```
    """
    return is_object_config(config, Module, import_target=True)


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
        >>> from sonnix.utils.factory import is_optimizer_config
        >>> linear = nn.Linear(4, 6)
        >>> is_optimizer_config({"_target_": "torch.optim.SGD", "params": linear.parameters()})
        True

        ```
    """
    return is_object_config(config, Optimizer, import_target=True)


def resolve_dataset(dataset: Dataset[T] | dict[Any, Any]) -> Dataset[T]:
    r"""Resolve a ``torch.utils.data.Dataset`` object.

    Args:
        dataset: The dataset or its configuration.

    Returns:
        The instantiated ``torch.utils.data.Dataset`` object.

    Example:
        ```pycon
        >>> from sonnix.utils.factory import resolve_dataset
        >>> dataset = resolve_dataset(
        ...     {
        ...         "_target_": "sonnix.testing.dummy.DummyDataset",
        ...         "num_examples": 10,
        ...         "feature_size": 4,
        ...     }
        ... )
        >>> dataset
        DummyDataset(num_examples=10, feature_size=4, rng_seed=5476923051063844987)

        ```
    """
    return resolve_object(dataset, Dataset)


def resolve_module(module: Module | dict[Any, Any]) -> Module:
    r"""Resolve a ``torch.nn.Module`` object.

    Args:
        module: The module or its configuration.

    Returns:
        The instantiated ``torch.nn.Module`` object.

    Example:
        ```pycon
        >>> from sonnix.utils.factory import resolve_module
        >>> linear = resolve_module(
        ...     {"_target_": "torch.nn.Linear", "in_features": 4, "out_features": 6}
        ... )
        >>> linear
        Linear(in_features=4, out_features=6, bias=True)

        ```
    """
    return resolve_object(module, Module)


def resolve_optimizer(optimizer: Optimizer | dict[Any, Any]) -> Optimizer:
    r"""Resolve a ``torch.optim.Optimizer`` object.

    Args:
        optimizer: The optimizer or its configuration.

    Returns:
        The instantiated ``torch.optim.Optimizer`` object.

    Example:
        ```pycon
        >>> from torch import nn
        >>> from sonnix.utils.factory import resolve_optimizer
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


def create_sequential(modules: Sequence[Module | dict[Any, Any]]) -> Sequential:
    r"""Create a ``torch.nn.Sequential`` from a sequence of modules.

    Args:
        modules: The sequence of modules or their configuration.

    Returns:
        The instantiated ``torch.nn.Sequential`` object.

    Example:
        ```pycon
        >>> from sonnix.utils.factory import create_sequential
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
