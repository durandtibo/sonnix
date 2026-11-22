from __future__ import annotations

import torch
from coola.equality import objects_are_equal
from torch import nn

from sonnix.utils.tensor import register_tensor

#######################################
#     Tests for register_tensor     #
#######################################


def test_register_tensor_learnable_is_a_parameter() -> None:
    module = nn.Module()
    register_tensor(module, "alpha", torch.ones(2), learnable=True)
    assert isinstance(module.alpha, nn.Parameter)


def test_register_tensor_learnable_requires_grad() -> None:
    module = nn.Module()
    register_tensor(module, "alpha", torch.ones(2), learnable=True)
    assert module.alpha.requires_grad


def test_register_tensor_frozen_is_a_buffer() -> None:
    module = nn.Module()
    register_tensor(module, "alpha", torch.ones(2), learnable=False)
    assert [name for name, _ in module.named_buffers()] == ["alpha"]


def test_register_tensor_frozen_is_not_a_parameter() -> None:
    module = nn.Module()
    register_tensor(module, "alpha", torch.ones(2), learnable=False)
    assert list(module.parameters()) == []


def test_register_tensor_frozen_is_not_a_tensor_with_grad() -> None:
    module = nn.Module()
    register_tensor(module, "alpha", torch.ones(2), learnable=False)
    assert not module.alpha.requires_grad


def test_register_tensor_state_dict_key_does_not_depend_on_learnable() -> None:
    learnable, frozen = nn.Module(), nn.Module()
    register_tensor(learnable, "alpha", torch.ones(2), learnable=True)
    register_tensor(frozen, "alpha", torch.ones(2), learnable=False)
    assert list(learnable.state_dict()) == list(frozen.state_dict())


def test_register_tensor_keeps_the_value() -> None:
    module = nn.Module()
    register_tensor(module, "alpha", torch.tensor([1.0, 2.0]), learnable=True)
    assert objects_are_equal(module.alpha.data, torch.tensor([1.0, 2.0]))


def test_register_tensor_keeps_the_dtype() -> None:
    module = nn.Module()
    register_tensor(module, "alpha", torch.ones(2, dtype=torch.float64), learnable=False)
    assert module.alpha.dtype == torch.float64


def test_register_tensor_copies_the_tensor() -> None:
    tensor = torch.ones(2)
    module = nn.Module()
    register_tensor(module, "alpha", tensor, learnable=False)
    tensor.add_(1.0)
    assert objects_are_equal(module.alpha, torch.ones(2))


def test_register_tensor_detaches_the_tensor() -> None:
    module = nn.Module()
    register_tensor(module, "alpha", torch.ones(2, requires_grad=True) * 2, learnable=False)
    assert module.alpha.grad_fn is None


def test_register_tensor_does_not_modify_the_tensor() -> None:
    tensor = torch.ones(2)
    register_tensor(nn.Module(), "alpha", tensor, learnable=True)
    assert not tensor.requires_grad
