from __future__ import annotations

import torch
from coola.factory import OBJECT_TARGET
from torch.nn import ReLU

from sonnix.testing.dummy import DummyDataset
from sonnix.utils.factory import (
    is_dataset_config,
    is_module_config,
    is_optimizer_config,
    setup_dataset,
    setup_module,
    setup_object,
    setup_object_typed,
    setup_optimizer,
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
#     Tests for setup_dataset     #
###################################


def test_setup_dataset() -> None:
    assert isinstance(
        setup_dataset(
            {
                OBJECT_TARGET: "sonnix.testing.dummy.DummyDataset",
                "num_examples": 10,
                "feature_size": 4,
            }
        ),
        DummyDataset,
    )


##################################
#     Tests for setup_module     #
##################################


def test_setup_module() -> None:
    assert isinstance(setup_module({OBJECT_TARGET: "torch.nn.ReLU"}), torch.nn.ReLU)


#####################################
#     Tests for setup_optimizer     #
#####################################


def test_setup_optimizer() -> None:
    assert isinstance(
        setup_optimizer(
            {OBJECT_TARGET: "torch.optim.SGD", "params": [torch.ones(2, 4, requires_grad=True)]}
        ),
        torch.optim.SGD,
    )


##################################
#     Tests for setup_object     #
##################################


def test_setup_object() -> None:
    assert isinstance(setup_object({OBJECT_TARGET: "torch.nn.ReLU"}), ReLU)


########################################
#     Tests for setup_object_typed     #
########################################


def test_setup_object_typed() -> None:
    assert isinstance(
        setup_object_typed({OBJECT_TARGET: "torch.nn.ReLU"}, cls=torch.nn.Module), ReLU
    )
