from __future__ import annotations

import pytest
import torch
from coola.factory import OBJECT_TARGET

from sonnix.factory.optimizer import (
    is_optimizer_config,
    resolve_optimizer,
)

#########################################
#     Tests for is_optimizer_config     #
#########################################


def test_is_optimizer_config_true() -> None:
    assert is_optimizer_config(
        {OBJECT_TARGET: "torch.optim.SGD", "params": [torch.ones(2, 4, requires_grad=True)]}
    )


def test_is_optimizer_config_false() -> None:
    assert not is_optimizer_config({OBJECT_TARGET: "torch.nn.Identity"})


#######################################
#     Tests for resolve_optimizer     #
#######################################


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
