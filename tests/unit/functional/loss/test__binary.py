from __future__ import annotations

import pytest
import torch
from coola.equality import objects_are_allclose, objects_are_equal

from sonnix.functional.loss._binary import cast_target, class_weight, probability_of_target

###########################################
#     Tests for probability_of_target     #
###########################################


def test_probability_of_target_binary_targets() -> None:
    assert objects_are_allclose(
        probability_of_target(torch.tensor([0.8, 0.8]), torch.tensor([1.0, 0.0])),
        torch.tensor([0.8, 0.2]),
    )


def test_probability_of_target_soft_targets() -> None:
    # 0.8 * 0.5 + 0.2 * 0.5
    assert objects_are_allclose(
        probability_of_target(torch.tensor([0.8]), torch.tensor([0.5])), torch.tensor([0.5])
    )


def test_probability_of_target_keeps_shape() -> None:
    assert probability_of_target(torch.rand(2, 3), torch.ones(2, 3)).shape == (2, 3)


def test_probability_of_target_does_not_mutate_inputs() -> None:
    probability, target = torch.tensor([0.8]), torch.tensor([1.0])
    probability_of_target(probability, target)
    assert probability.equal(torch.tensor([0.8]))


##################################
#     Tests for class_weight     #
##################################


def test_class_weight_binary_targets() -> None:
    assert objects_are_allclose(
        class_weight(0.25, torch.tensor([1.0, 0.0])), torch.tensor([0.25, 0.75])
    )


def test_class_weight_soft_targets() -> None:
    # 0.25 * 0.5 + 0.75 * 0.5
    assert objects_are_allclose(class_weight(0.25, torch.tensor([0.5])), torch.tensor([0.5]))


def test_class_weight_alpha_0() -> None:
    assert objects_are_allclose(
        class_weight(0.0, torch.tensor([1.0, 0.0])), torch.tensor([0.0, 1.0])
    )


def test_class_weight_alpha_1() -> None:
    assert objects_are_allclose(
        class_weight(1.0, torch.tensor([1.0, 0.0])), torch.tensor([1.0, 0.0])
    )


#################################
#     Tests for cast_target     #
#################################


@pytest.mark.parametrize("dtype", [torch.long, torch.int32, torch.uint8, torch.bool])
def test_cast_target_converts_integer_and_bool_targets(dtype: torch.dtype) -> None:
    assert objects_are_equal(
        cast_target(torch.tensor([1, 0, 1], dtype=dtype), torch.rand(3)),
        torch.tensor([1.0, 0.0, 1.0]),
    )


def test_cast_target_uses_the_prediction_dtype() -> None:
    assert (
        cast_target(torch.tensor([1, 0]), torch.rand(2, dtype=torch.float64)).dtype == torch.float64
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.float16])
def test_cast_target_keeps_a_float_target(dtype: torch.dtype) -> None:
    assert cast_target(torch.tensor([1.0, 0.5], dtype=dtype), torch.rand(2)).dtype == dtype


def test_cast_target_keeps_the_same_float_tensor() -> None:
    target = torch.tensor([1.0, 0.5])
    assert cast_target(target, torch.rand(2)) is target


def test_cast_target_does_not_modify_the_input() -> None:
    target = torch.tensor([1, 0])
    cast_target(target, torch.rand(2))
    assert objects_are_equal(target, torch.tensor([1, 0]))
