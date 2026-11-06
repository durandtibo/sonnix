from __future__ import annotations

import pytest
import torch
from coola.equality import objects_are_equal

from sonnix.modules.numerical._utils import prepare_tensor_param

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
