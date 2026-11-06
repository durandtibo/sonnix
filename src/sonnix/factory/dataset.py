r"""Contain functions to instantiate a ``torch.utils.data.Dataset``
object from its configuration."""

from __future__ import annotations

__all__ = [
    "is_dataset_config",
    "resolve_dataset",
]

from typing import Any, TypeVar

from coola.factory import is_object_config, resolve_object
from torch.utils.data import Dataset

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
        >>> from sonnix.factory import is_dataset_config
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


def resolve_dataset(dataset: Dataset[T] | dict[Any, Any]) -> Dataset[T]:
    r"""Resolve a ``torch.utils.data.Dataset`` object.

    Warning:
        A configuration (a ``dict`` with a ``_target_`` key) calls its
        ``_target_`` before the type of the result is checked, so it
        must come from a trusted source. See ``sonnix.factory``.

    Args:
        dataset: The dataset or its configuration.

    Returns:
        The instantiated ``torch.utils.data.Dataset`` object.

    Example:
        ```pycon
        >>> from sonnix.factory import resolve_dataset
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
