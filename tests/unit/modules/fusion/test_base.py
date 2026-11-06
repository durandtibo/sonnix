from __future__ import annotations

import pytest
import torch
from coola.equality import objects_are_equal

from sonnix.modules.fusion import BaseFusion


class MyFusion(BaseFusion):
    def fuse(self, inputs: tuple[torch.Tensor, ...]) -> torch.Tensor:
        return torch.stack(inputs).amax(dim=0)


class RecordFusion(BaseFusion):
    def fuse(self, inputs: tuple[torch.Tensor, ...]) -> torch.Tensor:
        self.received = inputs
        return inputs[0]


################################
#     Tests for BaseFusion     #
################################


def test_base_fusion_abstract() -> None:
    with pytest.raises(TypeError, match=r"abstract"):
        BaseFusion()


def test_base_fusion_forward_0_input() -> None:
    with pytest.raises(ValueError, match=r"MyFusion needs at least one tensor as input"):
        MyFusion()()


def test_base_fusion_forward_1_input() -> None:
    assert objects_are_equal(MyFusion()(torch.tensor([1.0, 2.0])), torch.tensor([1.0, 2.0]))


def test_base_fusion_forward_2_inputs() -> None:
    assert objects_are_equal(
        MyFusion()(torch.tensor([1.0, 5.0]), torch.tensor([3.0, 2.0])), torch.tensor([3.0, 5.0])
    )


def test_base_fusion_fuse_receives_the_inputs() -> None:
    module = RecordFusion()
    x1, x2 = torch.ones(2), torch.zeros(2)
    module(x1, x2)
    assert objects_are_equal(module.received, (x1, x2))
