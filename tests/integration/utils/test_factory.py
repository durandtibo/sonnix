from __future__ import annotations

import pytest
import torch
from torch.nn import ReLU

from sonnix.testing import objectory_available, objectory_not_available
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
from sonnix.utils.imports import is_objectory_available

if is_objectory_available():
    from objectory import OBJECT_TARGET
else:  # pragma: no cover
    from sonnix.utils.fallback.objectory import OBJECT_TARGET


#######################################
#     Tests for is_dataset_config     #
#######################################


@objectory_available
def test_is_dataset_config_with_objectory() -> None:
    assert is_dataset_config(
        {
            OBJECT_TARGET: "sonnix.testing.dummy.DummyDataset",
            "num_examples": 10,
            "feature_size": 4,
        }
    )


@objectory_not_available
def test_is_dataset_config_without_objectory() -> None:
    with pytest.raises(RuntimeError, match=r"'objectory' package is required but not installed."):
        is_dataset_config(
            {
                OBJECT_TARGET: "sonnix.testing.dummy.DummyDataset",
                "num_examples": 10,
                "feature_size": 4,
            }
        )


######################################
#     Tests for is_module_config     #
######################################


@objectory_available
def test_is_module_config_with_objectory() -> None:
    assert is_module_config({OBJECT_TARGET: "torch.nn.Identity"})


@objectory_not_available
def test_is_module_config_without_objectory() -> None:
    with pytest.raises(RuntimeError, match=r"'objectory' package is required but not installed."):
        is_module_config({OBJECT_TARGET: "torch.nn.Identity"})


#########################################
#     Tests for is_optimizer_config     #
#########################################


@objectory_available
def test_is_optimizer_config_with_objectory() -> None:
    assert is_optimizer_config(
        {OBJECT_TARGET: "torch.optim.SGD", "params": [torch.ones(2, 4, requires_grad=True)]}
    )


@objectory_not_available
def test_is_optimizer_config_without_objectory() -> None:
    with pytest.raises(RuntimeError, match=r"'objectory' package is required but not installed."):
        is_optimizer_config(
            {OBJECT_TARGET: "torch.optim.SGD", "params": [torch.ones(2, 4, requires_grad=True)]}
        )


###################################
#     Tests for setup_dataset     #
###################################


@objectory_available
def test_setup_dataset_with_objectory() -> None:
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


@objectory_not_available
def test_setup_dataset_without_objectory() -> None:
    with pytest.raises(RuntimeError, match=r"'objectory' package is required but not installed."):
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
def test_setup_module_with_objectory() -> None:
    assert isinstance(setup_module({OBJECT_TARGET: "torch.nn.ReLU"}), torch.nn.ReLU)


@objectory_not_available
def test_setup_module_without_objectory() -> None:
    with pytest.raises(RuntimeError, match=r"'objectory' package is required but not installed."):
        setup_module({OBJECT_TARGET: "torch.nn.ReLU"})


#####################################
#     Tests for setup_optimizer     #
#####################################


@objectory_available
def test_setup_optimizer_with_objectory() -> None:
    assert isinstance(
        setup_optimizer(
            {OBJECT_TARGET: "torch.optim.SGD", "params": [torch.ones(2, 4, requires_grad=True)]}
        ),
        torch.optim.SGD,
    )


@objectory_not_available
def test_setup_optimizer_without_objectory() -> None:
    with pytest.raises(RuntimeError, match=r"'objectory' package is required but not installed."):
        setup_optimizer(
            {OBJECT_TARGET: "torch.optim.SGD", "params": [torch.ones(2, 4, requires_grad=True)]}
        )


##################################
#     Tests for setup_object     #
##################################


@objectory_available
def test_setup_object_with_objectory() -> None:
    assert isinstance(setup_object({OBJECT_TARGET: "torch.nn.ReLU"}), ReLU)


@objectory_not_available
def test_setup_object_without_objectory() -> None:
    with pytest.raises(RuntimeError, match=r"'objectory' package is required but not installed."):
        setup_object({OBJECT_TARGET: "torch.nn.ReLU"})


########################################
#     Tests for setup_object_typed     #
########################################


@objectory_available
def test_setup_object_typed_with_objectory() -> None:
    assert isinstance(
        setup_object_typed({OBJECT_TARGET: "torch.nn.ReLU"}, cls=torch.nn.Module), ReLU
    )


@objectory_not_available
def test_setup_object_typed_without_objectory() -> None:
    with pytest.raises(RuntimeError, match=r"'objectory' package is required but not installed."):
        setup_object_typed({OBJECT_TARGET: "torch.nn.ReLU"}, cls=torch.nn.Module)
