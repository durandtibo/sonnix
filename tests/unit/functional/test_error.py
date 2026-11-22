from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import torch
from coola.equality import objects_are_equal
from coola.utils.tensor import get_available_devices

from sonnix.functional import (
    absolute_error,
    absolute_relative_error,
    symmetric_absolute_relative_error,
)
from tests.helpers import DEVICES_WITHOUT_MPS

if TYPE_CHECKING:
    from collections.abc import Callable

DTYPES = (torch.long, torch.float)


####################################
#     Tests for absolute_error     #
####################################


@pytest.mark.parametrize("device", get_available_devices())
def test_absolute_error_correct(device: str) -> None:
    assert objects_are_equal(
        absolute_error(torch.ones(2, 3, device=device), torch.ones(2, 3, device=device)),
        torch.zeros(2, 3, device=device),
    )


@pytest.mark.parametrize("device", get_available_devices())
def test_absolute_error_incorrect(device: str) -> None:
    assert objects_are_equal(
        absolute_error(
            torch.ones(2, 3, device=device),
            torch.tensor([[2.0, 2.0, 2.0], [-2.0, -2.0, -2.0]], device=device),
        ),
        torch.tensor([[1.0, 1.0, 1.0], [3.0, 3.0, 3.0]], device=device),
    )


@pytest.mark.parametrize("device", get_available_devices())
def test_absolute_error_partially_correct(device: str) -> None:
    assert objects_are_equal(
        absolute_error(torch.eye(2, device=device), torch.ones(2, 2, device=device)),
        torch.tensor([[0.0, 1.0], [1.0, 0.0]], device=device),
    )


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("shape", [(2,), (2, 3), (2, 3, 4)])
def test_absolute_error_shape(device: str, shape: tuple[int, ...]) -> None:
    assert objects_are_equal(
        absolute_error(torch.ones(*shape, device=device), torch.ones(*shape, device=device)),
        torch.zeros(*shape, device=device),
    )


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("dtype", DTYPES)
def test_absolute_error_dtypes(device: str, dtype: torch.dtype) -> None:
    assert objects_are_equal(
        absolute_error(
            torch.ones(2, 3, dtype=dtype, device=device),
            torch.ones(2, 3, dtype=dtype, device=device),
        ),
        torch.zeros(2, 3, dtype=dtype, device=device),
    )


#############################################
#     Tests for absolute_relative_error     #
#############################################


@pytest.mark.parametrize("device", get_available_devices())
def test_absolute_relative_error_correct(device: str) -> None:
    assert objects_are_equal(
        absolute_relative_error(torch.ones(2, 3, device=device), torch.ones(2, 3, device=device)),
        torch.zeros(2, 3, device=device),
    )


@pytest.mark.parametrize("device", DEVICES_WITHOUT_MPS)
def test_absolute_relative_error_correct_zero(device: str) -> None:
    assert objects_are_equal(
        absolute_relative_error(torch.zeros(2, 3, device=device), torch.zeros(2, 3, device=device)),
        torch.zeros(2, 3, device=device),
    )


@pytest.mark.parametrize("device", get_available_devices())
def test_absolute_relative_error_incorrect(device: str) -> None:
    assert objects_are_equal(
        absolute_relative_error(
            torch.ones(2, 3, device=device),
            torch.tensor([[2.0, 2.0, 2.0], [-2.0, -2.0, -2.0]], device=device),
        ),
        torch.tensor([[0.5, 0.5, 0.5], [1.5, 1.5, 1.5]], device=device),
    )


@pytest.mark.parametrize("device", DEVICES_WITHOUT_MPS)
def test_absolute_relative_error_incorrect_zero_target(device: str) -> None:
    assert objects_are_equal(
        absolute_relative_error(torch.ones(2, 3, device=device), torch.zeros(2, 3, device=device)),
        torch.full((2, 3), 1e8, device=device),
    )


@pytest.mark.parametrize("device", get_available_devices())
def test_absolute_relative_error_incorrect_zero_target_eps(device: str) -> None:
    assert objects_are_equal(
        absolute_relative_error(
            torch.ones(2, 3, device=device), torch.zeros(2, 3, device=device), eps=1e-4
        ),
        torch.full((2, 3), 1e4, device=device),
    )


@pytest.mark.parametrize("device", get_available_devices())
def test_absolute_relative_error_partially_correct(device: str) -> None:
    assert objects_are_equal(
        absolute_relative_error(torch.eye(2, device=device), torch.ones(2, 2, device=device)),
        torch.tensor([[0.0, 1.0], [1.0, 0.0]], device=device),
    )


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("shape", [(2,), (2, 3), (2, 3, 4)])
def test_absolute_relative_error_shape(device: str, shape: tuple[int, ...]) -> None:
    assert objects_are_equal(
        absolute_relative_error(
            torch.ones(*shape, device=device), torch.ones(*shape, device=device)
        ),
        torch.zeros(*shape, device=device),
    )


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("dtype", DTYPES)
def test_absolute_relative_error_dtypes(device: str, dtype: torch.dtype) -> None:
    assert objects_are_equal(
        absolute_relative_error(
            torch.ones(2, 3, dtype=dtype, device=device),
            torch.ones(2, 3, dtype=dtype, device=device),
            eps=1,
        ),
        torch.zeros(2, 3, dtype=torch.float, device=device),
    )


#######################################################
#     Tests for symmetric_absolute_relative_error     #
#######################################################


@pytest.mark.parametrize("device", get_available_devices())
def test_symmetric_absolute_relative_error_correct(device: str) -> None:
    assert objects_are_equal(
        symmetric_absolute_relative_error(
            torch.ones(2, 3, device=device), torch.ones(2, 3, device=device)
        ),
        torch.zeros(2, 3, device=device),
    )


@pytest.mark.parametrize("device", DEVICES_WITHOUT_MPS)
def test_symmetric_absolute_relative_error_correct_zero(device: str) -> None:
    assert objects_are_equal(
        symmetric_absolute_relative_error(
            torch.zeros(2, 3, device=device), torch.zeros(2, 3, device=device)
        ),
        torch.zeros(2, 3, device=device),
    )


@pytest.mark.parametrize("device", get_available_devices())
def test_symmetric_absolute_relative_error_incorrect(device: str) -> None:
    assert objects_are_equal(
        symmetric_absolute_relative_error(
            torch.ones(2, 3, device=device),
            torch.tensor([[3.0, 3.0, 3.0], [-3.0, -3.0, -3.0]], device=device),
        ),
        torch.tensor([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]], device=device),
    )


@pytest.mark.parametrize("device", get_available_devices())
def test_symmetric_absolute_relative_error_incorrect_zero_prediction(device: str) -> None:
    assert objects_are_equal(
        symmetric_absolute_relative_error(
            torch.zeros(2, 3, device=device), torch.ones(2, 3, device=device)
        ),
        torch.full((2, 3), 2.0, device=device),
    )


@pytest.mark.parametrize("device", get_available_devices())
def test_symmetric_absolute_relative_error_incorrect_zero_target(device: str) -> None:
    assert objects_are_equal(
        symmetric_absolute_relative_error(
            torch.ones(2, 3, device=device), torch.zeros(2, 3, device=device)
        ),
        torch.full((2, 3), 2.0, device=device),
    )


@pytest.mark.parametrize("device", get_available_devices())
def test_symmetric_absolute_relative_error_partially_correct(device: str) -> None:
    assert objects_are_equal(
        symmetric_absolute_relative_error(
            torch.eye(2, device=device), torch.ones(2, 2, device=device)
        ),
        torch.tensor([[0.0, 2.0], [2.0, 0.0]], device=device),
    )


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("shape", [(2,), (2, 3), (2, 3, 4)])
def test_symmetric_absolute_relative_error_shape(device: str, shape: tuple[int, ...]) -> None:
    assert objects_are_equal(
        symmetric_absolute_relative_error(
            torch.ones(*shape, device=device), torch.ones(*shape, device=device)
        ),
        torch.zeros(*shape, device=device),
    )


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("dtype", DTYPES)
def test_symmetric_absolute_relative_error_dtypes(device: str, dtype: torch.dtype) -> None:
    assert objects_are_equal(
        symmetric_absolute_relative_error(
            torch.ones(2, 3, dtype=dtype, device=device),
            torch.ones(2, 3, dtype=dtype, device=device),
        ),
        torch.zeros(2, 3, dtype=torch.float, device=device),
    )


@pytest.mark.parametrize("eps", [0.0, -1.0])
def test_absolute_relative_error_invalid_eps(eps: float) -> None:
    with pytest.raises(ValueError, match=r"eps must be greater than 0"):
        absolute_relative_error(torch.ones(2, 3), torch.ones(2, 3), eps=eps)


@pytest.mark.parametrize("eps", [0.0, -1.0])
def test_symmetric_absolute_relative_error_invalid_eps(eps: float) -> None:
    with pytest.raises(ValueError, match=r"eps must be greater than 0"):
        symmetric_absolute_relative_error(torch.ones(2, 3), torch.ones(2, 3), eps=eps)


@pytest.mark.parametrize(
    "func", [absolute_error, absolute_relative_error, symmetric_absolute_relative_error]
)
def test_error_functions_broadcast_shapes_are_rejected(func: Callable[..., torch.Tensor]) -> None:
    with pytest.raises(
        ValueError,
        match=r"prediction and target must have the same shape, got \(3, 1\) and \(1, 4\)",
    ):
        func(torch.ones(3, 1), torch.ones(1, 4))


@pytest.mark.parametrize(
    "func", [absolute_error, absolute_relative_error, symmetric_absolute_relative_error]
)
def test_error_functions_scalar_target_is_rejected(func: Callable[..., torch.Tensor]) -> None:
    with pytest.raises(ValueError, match=r"prediction and target must have the same shape"):
        func(torch.ones(3), torch.ones(()))
