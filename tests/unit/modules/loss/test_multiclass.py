from __future__ import annotations

from unittest.mock import Mock

import torch
from coola.equality import objects_are_allclose, objects_are_equal
from torch import nn

from sonnix.modules import MulticlassFlatten

#######################################
#     Tests for MulticlassFlatten     #
#######################################


def test_flatten_multiclass_forward_args_check1() -> None:
    module = Mock(spec=torch.nn.Module, return_value=torch.tensor(1.0))
    net = MulticlassFlatten(module)
    out = net(torch.ones(6, 2, 4), torch.zeros(6, 2))
    assert objects_are_equal(out, torch.tensor(1.0))


def test_flatten_multiclass_forward_args_call_args_args() -> None:
    module = Mock(spec=torch.nn.Module, return_value=torch.tensor(1.0))
    net = MulticlassFlatten(module)
    net(torch.ones(6, 2, 4), torch.zeros(6, 2))
    assert objects_are_equal(module.call_args.args, (torch.ones(12, 4), torch.zeros(12)))


def test_flatten_multiclass_forward_non_contiguous_prediction() -> None:
    module = Mock(spec=torch.nn.Module, return_value=torch.tensor(1.0))
    net = MulticlassFlatten(module)
    prediction = torch.arange(48, dtype=torch.float).view(4, 6, 2).transpose(0, 1)
    net(prediction, torch.zeros(6, 4))
    assert objects_are_equal(module.call_args.args[0], prediction.reshape(24, 2))


def test_flatten_multiclass_forward_kwargs_check1() -> None:
    module = Mock(spec=torch.nn.Module, return_value=torch.tensor(1.0))
    net = MulticlassFlatten(module)
    out = net(prediction=torch.ones(6, 2, 4), target=torch.zeros(6, 2))
    assert objects_are_equal(out, torch.tensor(1.0))


def test_flatten_multiclass_forward_kwargs_call_args_args() -> None:
    module = Mock(spec=torch.nn.Module, return_value=torch.tensor(1.0))
    net = MulticlassFlatten(module)
    net(prediction=torch.ones(6, 2, 4), target=torch.zeros(6, 2))
    assert objects_are_equal(module.call_args.args, (torch.ones(12, 4), torch.zeros(12)))


def test_flatten_multiclass_forward_args_and_kwargs_check1() -> None:
    module = Mock(spec=torch.nn.Module, return_value=torch.tensor(1.0))
    net = MulticlassFlatten(module)
    out = net(torch.ones(6, 2, 4), target=torch.zeros(6, 2))
    assert objects_are_equal(out, torch.tensor(1.0))


def test_flatten_multiclass_forward_args_and_kwargs_call_args_args() -> None:
    module = Mock(spec=torch.nn.Module, return_value=torch.tensor(1.0))
    net = MulticlassFlatten(module)
    net(torch.ones(6, 2, 4), target=torch.zeros(6, 2))
    assert objects_are_equal(module.call_args.args, (torch.ones(12, 4), torch.zeros(12)))


def test_flatten_multiclass_cross_entropy() -> None:
    net = MulticlassFlatten(nn.CrossEntropyLoss())
    loss = net(torch.ones(6, 2, 4, requires_grad=True), torch.zeros(6, 2, dtype=torch.long))
    loss.backward()
    assert objects_are_allclose(loss, torch.tensor(1.3862943649291992), atol=1e-6)
