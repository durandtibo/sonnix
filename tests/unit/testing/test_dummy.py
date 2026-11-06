from __future__ import annotations

import pytest
import torch

from sonnix.testing.dummy import DummyDataset

##################################
#     Tests for DummyDataset     #
##################################


def test_dummy_dataset_len() -> None:
    assert len(DummyDataset(feature_size=3, num_examples=5)) == 5


def test_dummy_dataset_getitem_shapes() -> None:
    example = DummyDataset(feature_size=3, num_examples=5)[0]
    assert example["feature"].shape == (3,)
    assert example["target"].shape == (1,)


def test_dummy_dataset_getitem_keys() -> None:
    assert list(DummyDataset()[0]) == ["feature", "target"]


def test_dummy_dataset_target_formula() -> None:
    example = DummyDataset(feature_size=3, num_examples=5)[2]
    expected = 0.6 * example["feature"][:1] + 0.4 * example["feature"][-1:]
    assert example["target"].allclose(expected)


def test_dummy_dataset_same_seed_is_deterministic() -> None:
    assert DummyDataset(rng_seed=1)[0]["feature"].equal(DummyDataset(rng_seed=1)[0]["feature"])


def test_dummy_dataset_different_seed_differs() -> None:
    assert not DummyDataset(rng_seed=1)[0]["feature"].equal(DummyDataset(rng_seed=2)[0]["feature"])


@pytest.mark.parametrize(("feature_size", "num_examples"), [(3.0, 5), (3, 5.0), ("3", "5")])
def test_dummy_dataset_numeric_arguments_are_converted_to_int(
    feature_size: object, num_examples: object
) -> None:
    dataset = DummyDataset(feature_size=feature_size, num_examples=num_examples)
    assert dataset[0]["feature"].shape == torch.Size([3])


def test_dummy_dataset_repr() -> None:
    assert repr(DummyDataset(feature_size=3, num_examples=5, rng_seed=7)) == (
        "DummyDataset(num_examples=5, feature_size=3, rng_seed=7)"
    )
