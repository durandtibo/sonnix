from __future__ import annotations

import pytest
import torch
from coola.equality import objects_are_equal

from sonnix.modules import BaseFusion


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


#########################
#     Tests for _sum    #
#########################


def test_sum_one_input() -> None:
    assert objects_are_equal(BaseFusion._sum((torch.tensor([1.0, 2.0]),)), torch.tensor([1.0, 2.0]))


def test_sum_several_inputs() -> None:
    assert objects_are_equal(
        BaseFusion._sum(
            (torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0]), torch.tensor([5.0, 6.0]))
        ),
        torch.tensor([9.0, 12.0]),
    )


def test_sum_does_not_modify_the_inputs() -> None:
    x1, x2 = torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0])
    BaseFusion._sum((x1, x2))
    assert objects_are_equal(x1, torch.tensor([1.0, 2.0]))


def test_sum_keeps_the_gradient() -> None:
    x1 = torch.tensor([1.0, 2.0], requires_grad=True)
    BaseFusion._sum((x1, torch.tensor([3.0, 4.0]))).sum().backward()
    assert objects_are_equal(x1.grad, torch.ones(2))


class NoCheckFusion(BaseFusion):
    _same_shape = False

    def fuse(self, inputs: tuple[torch.Tensor, ...]) -> torch.Tensor:
        return torch.cat(inputs, dim=-1)


def test_base_fusion_same_shape_default() -> None:
    assert BaseFusion._same_shape


def test_base_fusion_forward_different_shapes() -> None:
    with pytest.raises(ValueError, match=r"inputs\[0\] and inputs\[1\] must have the same shape"):
        MyFusion()(torch.ones(2, 3), torch.ones(2, 1))


def test_base_fusion_forward_different_shapes_allowed_by_the_subclass() -> None:
    assert NoCheckFusion()(torch.ones(2, 3), torch.ones(2, 1)).shape == (2, 4)


def test_base_fusion_forward_does_not_call_fuse_on_different_shapes() -> None:
    module = RecordFusion()
    with pytest.raises(ValueError, match=r"same shape"):
        module(torch.ones(2), torch.ones(3))
    assert not hasattr(module, "received")
