from __future__ import annotations

import pytest
import torch
from coola.equality import objects_are_equal
from coola.utils.tensor import get_available_devices

from sonnix.modules import ToDtype

#############################
#     Tests for ToDtype     #
#############################


@pytest.mark.parametrize("dtype", [torch.float64, torch.int32, torch.bfloat16])
def test_to_dtype_forward_dtype(dtype: torch.dtype) -> None:
    assert ToDtype(dtype)(torch.tensor([1, 2, 3])).dtype == dtype


def test_to_dtype_forward_values() -> None:
    assert objects_are_equal(
        ToDtype(torch.float64)(torch.tensor([1, 2, 3])),
        torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64),
    )


def test_to_dtype_forward_same_dtype() -> None:
    assert objects_are_equal(ToDtype(torch.long)(torch.tensor([1, 2, 3])), torch.tensor([1, 2, 3]))


def test_to_dtype_repr() -> None:
    assert repr(ToDtype(torch.float64)) == "ToDtype(dtype=torch.float64)"


def test_to_dtype_dtype_attribute() -> None:
    assert ToDtype(torch.float16).dtype == torch.float16


@pytest.mark.parametrize("dtype", [1, 1.5, None, float])
def test_to_dtype_incorrect_type(dtype: object) -> None:
    with pytest.raises(TypeError, match=r"dtype must be an instance of"):
        ToDtype(dtype)


@pytest.mark.parametrize(
    ("name", "dtype"),
    [
        ("float32", torch.float32),
        ("float64", torch.float64),
        ("int64", torch.int64),
        ("bfloat16", torch.bfloat16),
        ("bool", torch.bool),
    ],
)
def test_to_dtype_name(name: str, dtype: torch.dtype) -> None:
    assert ToDtype(name).dtype == dtype


def test_to_dtype_name_repr() -> None:
    assert repr(ToDtype("int64")) == "ToDtype(dtype=torch.int64)"


@pytest.mark.parametrize("device", get_available_devices())
def test_to_dtype_name_forward(device: str) -> None:
    device = torch.device(device)
    assert objects_are_equal(
        ToDtype("float32")(torch.tensor([1, 2, 3, 4], dtype=torch.long, device=device)),
        torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float, device=device),
    )


@pytest.mark.parametrize("device", get_available_devices())
def test_to_dtype_name_forward_long(device: str) -> None:
    device = torch.device(device)
    assert objects_are_equal(
        ToDtype("int64")(torch.tensor([1.0, 2.0, 3.0, 4.0], device=device)),
        torch.tensor([1, 2, 3, 4], dtype=torch.long, device=device),
    )


@pytest.mark.parametrize("name", ["float33", "Tensor", "torch.float32", "", "device"])
def test_to_dtype_unknown_name(name: str) -> None:
    with pytest.raises(ValueError, match=r"dtype must be one of"):
        ToDtype(name)
