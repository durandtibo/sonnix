from __future__ import annotations

import pytest

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


@pytest.mark.parametrize("name", ["feature_size", "num_examples", "rng_seed"])
@pytest.mark.parametrize("value", [2.5, 3.0, True, "3"])
def test_dummy_dataset_rejects_a_non_integer(name: str, value: object) -> None:
    with pytest.raises(TypeError, match=rf"{name} must be an int"):
        DummyDataset(**{name: value})


@pytest.mark.parametrize("name", ["feature_size", "num_examples"])
@pytest.mark.parametrize("value", [0, -1])
def test_dummy_dataset_rejects_a_size_lower_than_1(name: str, value: int) -> None:
    with pytest.raises(ValueError, match=rf"{name} must be greater than or equal to 1"):
        DummyDataset(**{name: value})


def test_dummy_dataset_accepts_a_size_of_1() -> None:
    assert len(DummyDataset(feature_size=1, num_examples=1)) == 1


def test_dummy_dataset_repr() -> None:
    assert repr(DummyDataset(feature_size=3, num_examples=5, rng_seed=7)) == (
        "DummyDataset(num_examples=5, feature_size=3, rng_seed=7)"
    )
