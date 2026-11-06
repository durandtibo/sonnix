from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import torch
from coola.factory import OBJECT_TARGET
from torch import nn

from sonnix.factory.module import (
    create_sequential,
    is_module_config,
    resolve_module,
)

if TYPE_CHECKING:
    from collections.abc import Sequence


######################################
#     Tests for is_module_config     #
######################################


def test_is_module_config_true() -> None:
    assert is_module_config({OBJECT_TARGET: "torch.nn.Identity"})


def test_is_module_config_false() -> None:
    assert not is_module_config({OBJECT_TARGET: "torch.device"})


####################################
#     Tests for resolve_module     #
####################################


@pytest.mark.parametrize("module", [torch.nn.ReLU(), {OBJECT_TARGET: "torch.nn.ReLU"}])
def test_resolve_module(module: torch.nn.Module | dict) -> None:
    assert isinstance(resolve_module(module), torch.nn.ReLU)


def test_resolve_module_object() -> None:
    module = torch.nn.ReLU()
    assert resolve_module(module) is module


def test_resolve_module_incorrect_type() -> None:
    with pytest.raises(TypeError, match=r"Module"):
        resolve_module({OBJECT_TARGET: "torch.device", "type": "cpu"})


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
