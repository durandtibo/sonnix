r"""Define some utility functions/classes."""

from __future__ import annotations

__all__ = [
    "DEVICES_WITHOUT_MPS",
    "ExamplePair",
    "make_generator",
    "normal_pair",
    "positive_pair",
    "probability_pair",
]

from dataclasses import dataclass
from typing import Any

import torch
from coola.utils.tensor import get_available_devices

DEVICES_WITHOUT_MPS = tuple(d for d in get_available_devices() if not d.startswith("mps:"))


@dataclass
class ExamplePair:
    actual: Any
    expected: Any
    expected_message: str | None = None
    atol: float = 0.0
    rtol: float = 0.0


def make_generator(seed: int = 42) -> torch.Generator:
    r"""Create a seeded random number generator.

    Args:
        seed: The seed of the generator.

    Returns:
        A new CPU generator seeded with ``seed``.
    """
    return torch.Generator().manual_seed(seed)


def normal_pair(
    *shape: int, dtype: torch.dtype = torch.float32, seed: int = 42
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Create two seeded tensors sampled from a standard normal distribution."""
    generator = make_generator(seed)
    return (
        torch.randn(*shape, generator=generator, dtype=dtype),
        torch.randn(*shape, generator=generator, dtype=dtype),
    )


def positive_pair(
    *shape: int, dtype: torch.dtype = torch.float32, seed: int = 42
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Create two seeded tensors with values in ``[0.5, 1.5)``."""
    generator = make_generator(seed)
    return (
        torch.rand(*shape, generator=generator, dtype=dtype) + 0.5,
        torch.rand(*shape, generator=generator, dtype=dtype) + 0.5,
    )


def probability_pair(
    *shape: int, dtype: torch.dtype = torch.float32, seed: int = 42
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Create a seeded probability in ``[0.1, 0.9)`` and a tensor in ``[0, 1)``."""
    generator = make_generator(seed)
    return (
        torch.rand(*shape, generator=generator, dtype=dtype) * 0.8 + 0.1,
        torch.rand(*shape, generator=generator, dtype=dtype),
    )
