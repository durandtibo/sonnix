from __future__ import annotations

import logging
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest
import torch
from torch import nn
from torch.nn import Module, ReLU

from sonnix.testing import objectory_available
from sonnix.testing.dummy import DummyDataset
from sonnix.utils.factory import (
    create_sequential,
    is_dataset_config,
    is_module_config,
    is_optimizer_config,
    setup_dataset,
    setup_module,
    setup_object,
    setup_object_typed,
    setup_optimizer,
    str_target_object,
)
from sonnix.utils.imports import is_objectory_available

if TYPE_CHECKING:
    from collections.abc import Sequence

if is_objectory_available():
    from objectory import OBJECT_TARGET
else:  # pragma: no cover
    from sonnix.utils.fallback.objectory import OBJECT_TARGET


#######################################
#     Tests for is_dataset_config     #
#######################################


@objectory_available
def test_is_dataset_config_true() -> None:
    assert is_dataset_config(
        {
            OBJECT_TARGET: "sonnix.testing.dummy.DummyDataset",
            "num_examples": 10,
            "feature_size": 4,
        }
    )


@objectory_available
def test_is_dataset_config_false() -> None:
    assert not is_dataset_config({OBJECT_TARGET: "torch.nn.Identity"})


######################################
#     Tests for is_module_config     #
######################################


@objectory_available
def test_is_module_config_true() -> None:
    assert is_module_config({OBJECT_TARGET: "torch.nn.Identity"})


@objectory_available
def test_is_module_config_false() -> None:
    assert not is_module_config({OBJECT_TARGET: "torch.device"})


#########################################
#     Tests for is_optimizer_config     #
#########################################


@objectory_available
def test_is_optimizer_config_true() -> None:
    assert is_optimizer_config(
        {OBJECT_TARGET: "torch.optim.SGD", "params": [torch.ones(2, 4, requires_grad=True)]}
    )


@objectory_available
def test_is_optimizer_config_false() -> None:
    assert not is_optimizer_config({OBJECT_TARGET: "torch.nn.Identity"})


###################################
#     Tests for setup_dataset     #
###################################


@objectory_available
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
def test_setup_dataset(dataset: DummyDataset | dict) -> None:
    assert isinstance(setup_dataset(dataset), DummyDataset)


@objectory_available
def test_setup_dataset_object() -> None:
    dataset = DummyDataset(num_examples=10, feature_size=4)
    assert setup_dataset(dataset) is dataset


@objectory_available
def test_setup_dataset_incorrect_type(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(level=logging.WARNING):
        assert isinstance(setup_dataset({OBJECT_TARGET: "torch.nn.Identity"}), nn.Identity)
        assert caplog.messages


def test_setup_dataset_object_no_objectory() -> None:
    with (
        patch("sonnix.utils.imports.is_objectory_available", lambda: False),
        pytest.raises(RuntimeError, match=r"'objectory' package is required but not installed."),
    ):
        setup_dataset(
            {
                OBJECT_TARGET: "sonnix.testing.dummy.DummyDataset",
                "num_examples": 10,
                "feature_size": 4,
            }
        )


##################################
#     Tests for setup_module     #
##################################


@objectory_available
@pytest.mark.parametrize("module", [torch.nn.ReLU(), {OBJECT_TARGET: "torch.nn.ReLU"}])
def test_setup_module(module: torch.nn.Module | dict) -> None:
    assert isinstance(setup_module(module), torch.nn.ReLU)


@objectory_available
def test_setup_module_object() -> None:
    module = torch.nn.ReLU()
    assert setup_module(module) is module


@objectory_available
def test_setup_module_incorrect_type(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(level=logging.WARNING):
        assert isinstance(
            setup_module({OBJECT_TARGET: "torch.device", "type": "cpu"}), torch.device
        )
        assert caplog.messages


def test_setup_module_object_no_objectory() -> None:
    with (
        patch("sonnix.utils.imports.is_objectory_available", lambda: False),
        pytest.raises(RuntimeError, match=r"'objectory' package is required but not installed."),
    ):
        setup_module({OBJECT_TARGET: "torch.nn.ReLU"})


#####################################
#     Tests for setup_optimizer     #
#####################################


@objectory_available
@pytest.mark.parametrize(
    "optimizer",
    [
        torch.optim.SGD([torch.ones(2, 4, requires_grad=True)]),
        {OBJECT_TARGET: "torch.optim.SGD", "params": [torch.ones(2, 4, requires_grad=True)]},
    ],
)
def test_setup_optimizer(optimizer: torch.optim.Optimizer | dict) -> None:
    assert isinstance(setup_optimizer(optimizer), torch.optim.SGD)


@objectory_available
def test_setup_optimizer_object() -> None:
    optimizer = torch.optim.SGD([torch.ones(2, 4, requires_grad=True)])
    assert setup_optimizer(optimizer) is optimizer


@objectory_available
def test_setup_optimizer_incorrect_type(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(level=logging.WARNING):
        assert isinstance(setup_optimizer({OBJECT_TARGET: "torch.nn.ReLU"}), torch.nn.ReLU)
        assert caplog.messages


def test_setup_optimizer_object_no_objectory() -> None:
    with (
        patch("sonnix.utils.imports.is_objectory_available", lambda: False),
        pytest.raises(RuntimeError, match=r"'objectory' package is required but not installed."),
    ):
        setup_optimizer(
            {OBJECT_TARGET: "torch.optim.SGD", "params": [torch.ones(2, 4, requires_grad=True)]}
        )


#######################################
#     Tests for create_sequential     #
#######################################


@objectory_available
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


##################################
#     Tests for setup_object     #
##################################


@objectory_available
@pytest.mark.parametrize("module", [ReLU(), {OBJECT_TARGET: "torch.nn.ReLU"}])
def test_setup_object(module: Module | dict) -> None:
    assert isinstance(setup_object(module), ReLU)


@objectory_available
def test_setup_object_object() -> None:
    module = ReLU()
    assert setup_object(module) is module


def test_setup_object_object_no_objectory() -> None:
    with (
        patch("sonnix.utils.imports.is_objectory_available", lambda: False),
        pytest.raises(RuntimeError, match=r"'objectory' package is required but not installed."),
    ):
        setup_object({OBJECT_TARGET: "torch.nn.ReLU"})


########################################
#     Tests for setup_object_typed     #
########################################


@objectory_available
@pytest.mark.parametrize("module", [ReLU(), {OBJECT_TARGET: "torch.nn.ReLU"}])
def test_setup_object_typed(module: Module | dict) -> None:
    assert isinstance(setup_object_typed(module, cls=torch.nn.Module), ReLU)


@objectory_available
@pytest.mark.parametrize("module", [ReLU(), {OBJECT_TARGET: "torch.nn.ReLU"}])
def test_setup_object_typed_with_name(module: Module | dict) -> None:
    assert isinstance(setup_object_typed(module, cls=torch.nn.Module, name="torch.nn.Module"), ReLU)


@objectory_available
def test_setup_object_typed_object() -> None:
    module = ReLU()
    assert setup_object_typed(module, cls=torch.nn.Module) is module


@objectory_available
def test_setup_object_typed_incorrect_type(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(level=logging.WARNING):
        assert isinstance(
            setup_object_typed({OBJECT_TARGET: "torch.device", "type": "cpu"}, cls=torch.nn.Module),
            torch.device,
        )
        assert caplog.messages


def test_setup_object_typed_object_no_objectory() -> None:
    with (
        patch("sonnix.utils.imports.is_objectory_available", lambda: False),
        pytest.raises(RuntimeError, match=r"'objectory' package is required but not installed."),
    ):
        setup_object_typed({OBJECT_TARGET: "torch.nn.ReLU"}, cls=torch.nn.Module)


#######################################
#     Tests for str_target_object     #
#######################################


@objectory_available
def test_str_target_object_with_target() -> None:
    assert str_target_object({OBJECT_TARGET: "something.MyClass"}) == "something.MyClass"


@objectory_available
def test_str_target_object_without_target() -> None:
    assert str_target_object({}) == "N/A"
