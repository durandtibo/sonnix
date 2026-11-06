from __future__ import annotations

import torch
from coola.equality import objects_are_equal

from tests.helpers import make_generator, normal_pair, positive_pair, probability_pair

#################################
#     Tests for make_generator  #
#################################


def test_make_generator_is_seeded() -> None:
    assert torch.rand(3, generator=make_generator(1)).equal(
        torch.rand(3, generator=make_generator(1))
    )


def test_make_generator_different_seeds() -> None:
    assert not torch.rand(3, generator=make_generator(1)).equal(
        torch.rand(3, generator=make_generator(2))
    )


def test_make_generator_default_seed() -> None:
    assert torch.rand(3, generator=make_generator()).equal(
        torch.rand(3, generator=make_generator(42))
    )


##############################
#     Tests for the pairs    #
##############################


def test_normal_pair_shape_and_dtype() -> None:
    first, second = normal_pair(2, 3, dtype=torch.float64)
    assert first.shape == second.shape == (2, 3)
    assert first.dtype == second.dtype == torch.float64


def test_normal_pair_elements_differ() -> None:
    first, second = normal_pair(2, 3)
    assert not first.equal(second)


def test_normal_pair_deterministic() -> None:
    assert objects_are_equal(normal_pair(2, 3), normal_pair(2, 3))


def test_positive_pair_range() -> None:
    first, second = positive_pair(50)
    assert torch.cat([first, second]).min() >= 0.5
    assert torch.cat([first, second]).max() < 1.5


def test_positive_pair_deterministic() -> None:
    assert objects_are_equal(positive_pair(2, 3), positive_pair(2, 3))


def test_probability_pair_prediction_range() -> None:
    prediction, _ = probability_pair(50)
    assert prediction.min() >= 0.1
    assert prediction.max() < 0.9


def test_probability_pair_target_range() -> None:
    _, target = probability_pair(50)
    assert target.min() >= 0.0
    assert target.max() < 1.0
