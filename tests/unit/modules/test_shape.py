from __future__ import annotations

from unittest.mock import Mock

import pytest
import torch
from coola.equality import objects_are_allclose, objects_are_equal
from coola.utils.tensor import get_available_devices
from torch import nn

from sonnix.modules import MulticlassFlatten, Squeeze, View

#######################################
#     Tests for MulticlassFlatten     #
#######################################


def test_flatten_multiclass_forward_args() -> None:
    module = Mock(spec=torch.nn.Module, return_value=torch.tensor(1.0))
    net = MulticlassFlatten(module)
    out = net(torch.ones(6, 2, 4), torch.zeros(6, 2))
    assert objects_are_equal(out, torch.tensor(1.0))
    assert objects_are_equal(module.call_args.args, (torch.ones(12, 4), torch.zeros(12)))


def test_flatten_multiclass_forward_kwargs() -> None:
    module = Mock(spec=torch.nn.Module, return_value=torch.tensor(1.0))
    net = MulticlassFlatten(module)
    out = net(prediction=torch.ones(6, 2, 4), target=torch.zeros(6, 2))
    assert objects_are_equal(out, torch.tensor(1.0))
    assert objects_are_equal(module.call_args.args, (torch.ones(12, 4), torch.zeros(12)))


def test_flatten_multiclass_forward_args_and_kwargs() -> None:
    module = Mock(spec=torch.nn.Module, return_value=torch.tensor(1.0))
    net = MulticlassFlatten(module)
    out = net(torch.ones(6, 2, 4), target=torch.zeros(6, 2))
    assert objects_are_equal(out, torch.tensor(1.0))
    assert objects_are_equal(module.call_args.args, (torch.ones(12, 4), torch.zeros(12)))


def test_flatten_multiclass_cross_entropy() -> None:
    net = MulticlassFlatten(nn.CrossEntropyLoss())
    loss = net(torch.ones(6, 2, 4, requires_grad=True), torch.zeros(6, 2, dtype=torch.long))
    loss.backward()
    assert objects_are_allclose(loss, torch.tensor(1.3862943649291992), atol=1e-6)


#############################
#     Tests for Squeeze     #
#############################


def test_squeeze_str() -> None:
    assert str(Squeeze()).startswith("Squeeze(")


@pytest.mark.parametrize("device", get_available_devices())
def test_squeeze_forward_dim_none(device: str) -> None:
    device = torch.device(device)
    module = Squeeze().to(device=device)
    assert objects_are_equal(
        module(torch.ones(2, 1, 3, 1, 4, device=device)),
        torch.ones(2, 3, 4, device=device),
    )


@pytest.mark.parametrize("device", get_available_devices())
def test_squeeze_forward_dim_1(device: str) -> None:
    device = torch.device(device)
    module = Squeeze(dim=1).to(device=device)
    assert objects_are_equal(
        module(torch.ones(2, 1, 3, 1, 4, device=device)),
        torch.ones(2, 3, 1, 4, device=device),
    )


##########################
#     Tests for View     #
##########################


def test_view_str() -> None:
    assert str(View(shape=(-1, 3))).startswith("View(")


@pytest.mark.parametrize("device", get_available_devices())
def test_view_forward_2d(device: str) -> None:
    device = torch.device(device)
    module = View(shape=(-1, 3)).to(device=device)
    assert objects_are_equal(
        module(torch.ones(2, 3, device=device)), torch.ones(2, 3, device=device)
    )


@pytest.mark.parametrize("device", get_available_devices())
def test_view_forward_3d(device: str) -> None:
    device = torch.device(device)
    module = View(shape=(-1, 3)).to(device=device)
    assert objects_are_equal(
        module(torch.ones(4, 2, 3, device=device)), torch.ones(8, 3, device=device)
    )
