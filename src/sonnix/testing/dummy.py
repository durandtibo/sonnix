r"""Provide dummy modules and datasets for testing and debugging
purposes.

This module contains lightweight implementations that can be used for
unit tests and debugging without requiring real data or complex model
initialization.
"""

from __future__ import annotations

__all__ = ["DummyDataset"]

import torch
from coola.random import make_torch_generator
from coola.validation import validate_ge, validate_int
from torch.utils.data import Dataset


class DummyDataset(Dataset[dict[str, torch.Tensor]]):
    r"""Implement a dummy map-style dataset for testing purpose.

    Args:
        feature_size: The feature size.
        n_examples: The number of examples.
        rng_seed: The random seed used to generate the features.

    Raises:
        TypeError: If ``feature_size``, ``n_examples`` or ``rng_seed``
            is not an integer (a ``bool`` is rejected).
        ValueError: If ``feature_size`` or ``n_examples`` is lower
            than 1.

    Example:
        ```pycon
        >>> from sonnix.testing import DummyDataset
        >>> dataset = DummyDataset(n_examples=10, feature_size=7)
        >>> dataset[0]
        {'feature': tensor([...]), 'target': tensor([...])}

        ```
    """

    def __init__(
        self, feature_size: int = 4, n_examples: int = 8, rng_seed: int = 5476923051063844987
    ) -> None:
        for name, value in (("feature_size", feature_size), ("n_examples", n_examples)):
            validate_int(value, name=name)
            validate_ge(value, 1, name=name)
        validate_int(rng_seed, name="rng_seed")
        self._feature_size = feature_size
        self._n_examples = n_examples
        self._rng_seed = rng_seed

        self._features = torch.randn(
            self._n_examples, self._feature_size, generator=make_torch_generator(rng_seed)
        )
        self._target = 0.6 * self._features[:, :1] + 0.4 * self._features[:, -1:]

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {"feature": self._features[index], "target": self._target[index]}

    def __len__(self) -> int:
        return self._n_examples

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__qualname__}(n_examples={self._n_examples:,}, "
            f"feature_size={self._feature_size:,}, rng_seed={self._rng_seed})"
        )
