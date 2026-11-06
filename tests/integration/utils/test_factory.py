from __future__ import annotations

import torch
from coola.factory import OBJECT_TARGET

from sonnix.testing.dummy import DummyDataset
from sonnix.utils.factory import (
    is_dataset_config,
    is_module_config,
    is_optimizer_config,
    resolve_dataset,
    resolve_module,
    resolve_optimizer,
)

#######################################
#     Tests for is_dataset_config     #
#######################################


def test_is_dataset_config() -> None:
    assert is_dataset_config(
        {
            OBJECT_TARGET: "sonnix.testing.dummy.DummyDataset",
            "num_examples": 10,
            "feature_size": 4,
        }
    )


######################################
#     Tests for is_module_config     #
######################################


def test_is_module_config() -> None:
    assert is_module_config({OBJECT_TARGET: "torch.nn.Identity"})


#########################################
#     Tests for is_optimizer_config     #
#########################################


def test_is_optimizer_config() -> None:
    assert is_optimizer_config(
        {OBJECT_TARGET: "torch.optim.SGD", "params": [torch.ones(2, 4, requires_grad=True)]}
    )


###################################
#     Tests for resolve_dataset     #
###################################


def test_resolve_dataset() -> None:
    assert isinstance(
        resolve_dataset(
            {
                OBJECT_TARGET: "sonnix.testing.dummy.DummyDataset",
                "num_examples": 10,
                "feature_size": 4,
            }
        ),
        DummyDataset,
    )


##################################
#     Tests for resolve_module     #
##################################


def test_resolve_module() -> None:
    assert isinstance(resolve_module({OBJECT_TARGET: "torch.nn.ReLU"}), torch.nn.ReLU)


#####################################
#     Tests for resolve_optimizer     #
#####################################


def test_resolve_optimizer() -> None:
    assert isinstance(
        resolve_optimizer(
            {OBJECT_TARGET: "torch.optim.SGD", "params": [torch.ones(2, 4, requires_grad=True)]}
        ),
        torch.optim.SGD,
    )
