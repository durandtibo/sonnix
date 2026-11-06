r"""Contain utility functions to manage randomness."""

from __future__ import annotations

__all__ = ["resolve_torch_generator"]

import torch
from coola.random import make_torch_generator


def resolve_torch_generator(generator_or_seed: int | torch.Generator) -> torch.Generator:
    r"""Resolve a ``torch.Generator`` object.

    Args:
        generator_or_seed: A ``torch.Generator`` object or
            a random seed.

    Returns:
        A ``torch.Generator`` object.

    Example:
        ```pycon
        >>> from sonnix.utils.random import resolve_torch_generator
        >>> generator = resolve_torch_generator(42)
        >>> generator
        <torch._C.Generator object at 0x...>

        ```
    """
    if isinstance(generator_or_seed, torch.Generator):
        return generator_or_seed
    return make_torch_generator(generator_or_seed)
