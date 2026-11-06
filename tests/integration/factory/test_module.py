from __future__ import annotations

import torch
from coola.factory import OBJECT_TARGET

from sonnix.factory.module import (
    is_module_config,
    resolve_module,
)

######################################
#     Tests for is_module_config     #
######################################


def test_is_module_config() -> None:
    assert is_module_config({OBJECT_TARGET: "torch.nn.Identity"})


####################################
#     Tests for resolve_module     #
####################################


def test_resolve_module() -> None:
    assert isinstance(resolve_module({OBJECT_TARGET: "torch.nn.ReLU"}), torch.nn.ReLU)
