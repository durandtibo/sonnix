from __future__ import annotations

import pytest
import torch
from coola.equality import objects_are_equal
from coola.utils.tensor import get_available_devices

from sonnix.modules import ToDtype, ToFloat, ToLong

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


def test_to_dtype_invalid_dtype() -> None:
    with pytest.raises(TypeError, match=r"dtype"):
        ToDtype("float32")


#############################
#     Tests for ToFloat     #
#############################


def test_to_float_dtype() -> None:
    assert ToFloat().dtype == torch.float32


def test_to_float_is_to_dtype() -> None:
    assert isinstance(ToFloat(), ToDtype)


def test_to_float_repr() -> None:
    assert repr(ToFloat()) == "ToFloat(dtype=torch.float32)"


@pytest.mark.parametrize("device", get_available_devices())
def test_to_float_forward(device: str) -> None:
    device = torch.device(device)
    module = ToFloat().to(device=device)
    output = module(torch.tensor([1, 2, 3, 4], dtype=torch.long, device=device))
    assert objects_are_equal(
        output, torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float, device=device)
    )


############################
#     Tests for ToLong     #
############################


@pytest.mark.parametrize("device", get_available_devices())
def test_to_long_forward(device: str) -> None:
    device = torch.device(device)
    module = ToLong().to(device=device)
    output = module(torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float, device=device))
    assert objects_are_equal(output, torch.tensor([1, 2, 3, 4], dtype=torch.long, device=device))


def test_to_long_dtype() -> None:
    assert ToLong().dtype == torch.int64


def test_to_long_is_to_dtype() -> None:
    assert isinstance(ToLong(), ToDtype)


def test_to_long_repr() -> None:
    assert repr(ToLong()) == "ToLong(dtype=torch.int64)"
