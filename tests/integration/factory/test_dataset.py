from __future__ import annotations

from coola.factory import OBJECT_TARGET

from sonnix.factory.dataset import (
    is_dataset_config,
    resolve_dataset,
)
from sonnix.testing.dummy import DummyDataset

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


#####################################
#     Tests for resolve_dataset     #
#####################################


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
