from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import torch
from coola.factory import OBJECT_TARGET
from torch import nn

from sonnix.testing.dummy import DummyDataset
from sonnix.utils.factory import (
    create_sequential,
    is_dataset_config,
    is_module_config,
    is_optimizer_config,
    resolve_dataset,
    resolve_module,
    resolve_optimizer,
)

if TYPE_CHECKING:
    from collections.abc import Sequence


#######################################
#     Tests for is_dataset_config     #
#######################################


def test_is_dataset_config_true() -> None:
    assert is_dataset_config(
        {
            OBJECT_TARGET: "sonnix.testing.dummy.DummyDataset",
            "num_examples": 10,
            "feature_size": 4,
        }
    )


def test_is_dataset_config_false() -> None:
    assert not is_dataset_config({OBJECT_TARGET: "torch.nn.Identity"})


######################################
#     Tests for is_module_config     #
######################################


def test_is_module_config_true() -> None:
    assert is_module_config({OBJECT_TARGET: "torch.nn.Identity"})


def test_is_module_config_false() -> None:
    assert not is_module_config({OBJECT_TARGET: "torch.device"})


#########################################
#     Tests for is_optimizer_config     #
#########################################


def test_is_optimizer_config_true() -> None:
    assert is_optimizer_config(
        {OBJECT_TARGET: "torch.optim.SGD", "params": [torch.ones(2, 4, requires_grad=True)]}
    )


def test_is_optimizer_config_false() -> None:
    assert not is_optimizer_config({OBJECT_TARGET: "torch.nn.Identity"})


###################################
#     Tests for resolve_dataset     #
###################################


@pytest.mark.parametrize(
    "dataset",
    [
        DummyDataset(num_examples=10, feature_size=4),
        {
            OBJECT_TARGET: "sonnix.testing.dummy.DummyDataset",
            "num_examples": 10,
            "feature_size": 4,
        },
    ],
)
def test_resolve_dataset(dataset: DummyDataset | dict) -> None:
    assert isinstance(resolve_dataset(dataset), DummyDataset)


def test_resolve_dataset_object() -> None:
    dataset = DummyDataset(num_examples=10, feature_size=4)
    assert resolve_dataset(dataset) is dataset


def test_resolve_dataset_incorrect_type() -> None:
    with pytest.raises(TypeError, match=r"Dataset"):
        resolve_dataset({OBJECT_TARGET: "torch.nn.Identity"})


##################################
#     Tests for resolve_module     #
##################################


@pytest.mark.parametrize("module", [torch.nn.ReLU(), {OBJECT_TARGET: "torch.nn.ReLU"}])
def test_resolve_module(module: torch.nn.Module | dict) -> None:
    assert isinstance(resolve_module(module), torch.nn.ReLU)


def test_resolve_module_object() -> None:
    module = torch.nn.ReLU()
    assert resolve_module(module) is module


def test_resolve_module_incorrect_type() -> None:
    with pytest.raises(TypeError, match=r"Module"):
        resolve_module({OBJECT_TARGET: "torch.device", "type": "cpu"})


#####################################
#     Tests for resolve_optimizer     #
#####################################


@pytest.mark.parametrize(
    "optimizer",
    [
        torch.optim.SGD([torch.ones(2, 4, requires_grad=True)]),
        {OBJECT_TARGET: "torch.optim.SGD", "params": [torch.ones(2, 4, requires_grad=True)]},
    ],
)
def test_resolve_optimizer(optimizer: torch.optim.Optimizer | dict) -> None:
    assert isinstance(resolve_optimizer(optimizer), torch.optim.SGD)


def test_resolve_optimizer_object() -> None:
    optimizer = torch.optim.SGD([torch.ones(2, 4, requires_grad=True)])
    assert resolve_optimizer(optimizer) is optimizer


def test_resolve_optimizer_incorrect_type() -> None:
    with pytest.raises(TypeError, match=r"Optimizer"):
        resolve_optimizer({OBJECT_TARGET: "torch.nn.ReLU"})


#######################################
#     Tests for create_sequential     #
#######################################


@pytest.mark.parametrize(
    "modules",
    [
        [{OBJECT_TARGET: "torch.nn.Linear", "in_features": 4, "out_features": 6}, nn.ReLU()],
        ({OBJECT_TARGET: "torch.nn.Linear", "in_features": 4, "out_features": 6}, nn.ReLU()),
    ],
)
def test_create_sequential(modules: Sequence) -> None:
    module = create_sequential(modules)
    assert isinstance(module, nn.Sequential)
    assert len(module) == 2
    assert isinstance(module[0], nn.Linear)
    assert isinstance(module[1], nn.ReLU)


def test_create_sequential_empty() -> None:
    module = create_sequential([])
    assert isinstance(module, nn.Sequential)
    assert len(module) == 0


#######
