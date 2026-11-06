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
from torch.utils.data import Dataset


class DummyDataset(Dataset[dict[str, torch.Tensor]]):
    r"""Implement a dummy map-style dataset for testing purpose.

    Args:
        feature_size: The feature size.
        num_examples: The number of examples.
        rng_seed: The random seed used to generate the features.

    Example:
        ```pycon
        >>> from sonnix.testing.dummy import DummyDataset
        >>> dataset = DummyDataset(num_examples=10, feature_size=7)
        >>> dataset[0]
        {'feature': tensor([...]), 'target': tensor([...])}

        ```
    """

    def __init__(
        self, feature_size: int = 4, num_examples: int = 8, rng_seed: int = 5476923051063844987
    ) -> None:
        self._feature_size = int(feature_size)
        self._num_examples = int(num_examples)
        self._rng_seed = rng_seed

        self._features = torch.randn(
            self._num_examples, self._feature_size, generator=make_torch_generator(rng_seed)
        )
        self._target = 0.6 * self._features[:, :1] + 0.4 * self._features[:, -1:]

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {"feature": self._features[index], "target": self._target[index]}

    def __len__(self) -> int:
        return self._num_examples

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__qualname__}(num_examples={self._num_examples:,}, "
            f"feature_size={self._feature_size:,}, rng_seed={self._rng_seed})"
        )
