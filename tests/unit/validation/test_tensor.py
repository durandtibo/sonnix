from __future__ import annotations

import pytest
import torch

from sonnix.validation import validate_same_shape

#########################################
#     Tests for validate_same_shape     #
#########################################


def test_validate_same_shape() -> None:
    assert validate_same_shape(torch.ones(2, 3), torch.zeros(2, 3)) is None


def test_validate_same_shape_different_dtype() -> None:
    assert validate_same_shape(torch.ones(2, 3), torch.zeros(2, 3, dtype=torch.long)) is None


def test_validate_same_shape_scalar() -> None:
    assert validate_same_shape(torch.tensor(1.0), torch.tensor(2.0)) is None


def test_validate_same_shape_empty() -> None:
    assert validate_same_shape(torch.zeros(0, 3), torch.zeros(0, 3)) is None


def test_validate_same_shape_different_ndim() -> None:
    with pytest.raises(ValueError, match=r"tensor1 and tensor2 must have the same shape"):
        validate_same_shape(torch.ones(2, 3), torch.ones(2, 3, 1))


def test_validate_same_shape_different_size() -> None:
    with pytest.raises(ValueError, match=r"got \(2, 3\) and \(2, 4\)"):
        validate_same_shape(torch.ones(2, 3), torch.ones(2, 4))


def test_validate_same_shape_transposed() -> None:
    with pytest.raises(ValueError, match=r"got \(2, 3\) and \(3, 2\)"):
        validate_same_shape(torch.ones(2, 3), torch.ones(3, 2))


def test_validate_same_shape_names() -> None:
    with pytest.raises(ValueError, match=r"^pred and target must have the same shape"):
        validate_same_shape(torch.ones(2), torch.ones(3), name1="pred", name2="target")


def test_validate_same_shape_tensor1_not_a_tensor() -> None:
    with pytest.raises(TypeError, match=r"tensor1"):
        validate_same_shape([1.0, 2.0], torch.ones(2))


def test_validate_same_shape_tensor2_not_a_tensor() -> None:
    with pytest.raises(TypeError, match=r"target"):
        validate_same_shape(torch.ones(2), None, name2="target")


def test_validate_same_shape_does_not_mutate() -> None:
    tensor1 = torch.arange(6.0).reshape(2, 3)
    tensor2 = torch.arange(6.0).reshape(2, 3)
    validate_same_shape(tensor1, tensor2)
    assert tensor1.equal(torch.arange(6.0).reshape(2, 3))
