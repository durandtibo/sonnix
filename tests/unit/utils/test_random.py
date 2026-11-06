from __future__ import annotations

from coola.equality import objects_are_equal
from coola.random import make_torch_generator

from sonnix.utils.random import resolve_torch_generator

#############################################
#     Tests for resolve_torch_generator     #
#############################################


def test_resolve_torch_generator() -> None:
    generator = make_torch_generator(1)
    assert resolve_torch_generator(generator) is generator


def test_resolve_torch_generator_seed() -> None:
    assert objects_are_equal(
        resolve_torch_generator(1).get_state(), make_torch_generator(1).get_state()
    )
