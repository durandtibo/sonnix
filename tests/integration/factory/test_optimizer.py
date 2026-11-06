from __future__ import annotations

import torch
from coola.factory import OBJECT_TARGET

from sonnix.factory.optimizer import (
    is_optimizer_config,
    resolve_optimizer,
)

#########################################
#     Tests for is_optimizer_config     #
#########################################


def test_is_optimizer_config() -> None:
    assert is_optimizer_config(
        {OBJECT_TARGET: "torch.optim.SGD", "params": [torch.ones(2, 4, requires_grad=True)]}
    )


#######################################
#     Tests for resolve_optimizer     #
#######################################


def test_resolve_optimizer() -> None:
    assert isinstance(
        resolve_optimizer(
            {OBJECT_TARGET: "torch.optim.SGD", "params": [torch.ones(2, 4, requires_grad=True)]}
        ),
        torch.optim.SGD,
    )
