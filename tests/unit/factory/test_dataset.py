from __future__ import annotations

import pytest
from coola.factory import OBJECT_TARGET

from sonnix.factory.dataset import (
    is_dataset_config,
    resolve_dataset,
)
from sonnix.testing.dummy import DummyDataset

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


#####################################
#     Tests for resolve_dataset     #
#####################################


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
