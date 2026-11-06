from __future__ import annotations

import torch
from coola.equality import objects_are_allclose

from sonnix.functional.loss._binary import class_weight, probability_of_target

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
