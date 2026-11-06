from __future__ import annotations

import pytest
import torch
from coola.equality import objects_are_equal

from sonnix.modules.numerical._utils import check_abs_range, check_frequency, prepare_tensor_param

##########################################
#     Tests for prepare_tensor_param     #
##########################################


@pytest.mark.parametrize("tensor", [torch.tensor([1.0, 2.0, 4.0]), torch.tensor([[1.0, 2.0, 4.0]])])
def test_prepare_tensor_param_1d(tensor: torch.Tensor) -> None:
    assert objects_are_equal(
        prepare_tensor_param(tensor, name="scale"), torch.tensor([[1.0, 2.0, 4.0]])
    )


def test_prepare_tensor_param_2d() -> None:
    assert objects_are_equal(
        prepare_tensor_param(torch.tensor([[1.0, 2.0, 4.0], [2.0, 3.0, 4.0]]), name="scale"),
        torch.tensor([[1.0, 2.0, 4.0], [2.0, 3.0, 4.0]]),
    )


def test_prepare_tensor_param_incorrect_shape() -> None:
    with pytest.raises(ValueError, match=r"Incorrect shape for 'scale':"):
        prepare_tensor_param(torch.ones(2, 3, 4), name="scale")


#####################################
#     Tests for check_abs_range     #
#####################################


def test_check_abs_range_valid() -> None:
    check_abs_range(min_abs_value=1, max_abs_value=5)


def test_check_abs_range_invalid_min_abs_value() -> None:
    with pytest.raises(ValueError, match=r"min_abs_value must be greater than 0"):
        check_abs_range(min_abs_value=-1, max_abs_value=5)


def test_check_abs_range_invalid_max_abs_value() -> None:
    with pytest.raises(
        ValueError, match=r"'max_abs_value' .* has to be greater than 'min_abs_value'"
    ):
        check_abs_range(min_abs_value=5, max_abs_value=1)


#####################################
#     Tests for check_frequency     #
#####################################


def test_check_frequency_valid() -> None:
    check_frequency(num_frequencies=3, min_frequency=0.2, max_frequency=1)


def test_check_frequency_invalid_num_frequencies() -> None:
    with pytest.raises(ValueError, match=r"num_frequencies must be greater than or equal to 1"):
        check_frequency(num_frequencies=0, min_frequency=0.2, max_frequency=1)


def test_check_frequency_invalid_min_frequency() -> None:
    with pytest.raises(ValueError, match=r"min_frequency must be greater than 0"):
        check_frequency(num_frequencies=3, min_frequency=-2, max_frequency=1)


def test_check_frequency_invalid_max_frequency() -> None:
    with pytest.raises(
        ValueError, match=r"'max_frequency' .* has to be greater than 'min_frequency'"
    ):
        check_frequency(num_frequencies=3, min_frequency=2, max_frequency=1)
