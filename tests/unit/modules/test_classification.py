from __future__ import annotations

import pytest
import torch
from coola.equality import objects_are_equal
from coola.utils.tensor import get_available_devices

from sonnix.modules import ToBinaryLabel, ToCategoricalLabel

###################################
#     Tests for ToBinaryLabel     #
###################################


def test_to_binary_label_str() -> None:
    assert str(ToBinaryLabel()).startswith("ToBinaryLabel(")


@pytest.mark.parametrize("threshold", [0.0, 0.5])
def test_to_binary_label_threshold(threshold: float) -> None:
    assert ToBinaryLabel(threshold=threshold).threshold == threshold


def test_to_binary_label_threshold_is_a_plain_attribute() -> None:
    module = ToBinaryLabel()
    module.threshold = 1.5
    assert module(torch.tensor([1.0, 2.0])).equal(torch.tensor([0, 1]))


def test_to_binary_label_threshold_default() -> None:
    assert ToBinaryLabel().threshold == 0.0


@pytest.mark.parametrize("device", get_available_devices())
def test_to_binary_label_forward(device: str) -> None:
    device = torch.device(device)
    module = ToBinaryLabel().to(device=device)
    assert objects_are_equal(
        module(torch.tensor([-1.0, 1.0, -2.0, 1.0], dtype=torch.float, device=device)),
        torch.tensor([0, 1, 0, 1], dtype=torch.long, device=device),
    )


@pytest.mark.parametrize("device", get_available_devices())
def test_to_binary_label_forward_threshold_0_5(device: str) -> None:
    device = torch.device(device)
    module = ToBinaryLabel(threshold=0.5).to(device=device)
    assert objects_are_equal(
        module(torch.tensor([0.1, 0.6, 0.4, 1.0], dtype=torch.float, device=device)),
        torch.tensor([0, 1, 0, 1], dtype=torch.long, device=device),
    )


@pytest.mark.parametrize("device", get_available_devices())
def test_to_binary_label_forward_2d(device: str) -> None:
    device = torch.device(device)
    module = ToBinaryLabel().to(device=device)
    assert objects_are_equal(
        module(
            torch.tensor(
                [[-1.0, 1.0, -2.0, 1.0], [0.0, 1.0, 2.0, -1.0]], dtype=torch.float, device=device
            )
        ),
        torch.tensor([[0, 1, 0, 1], [0, 1, 1, 0]], dtype=torch.long, device=device),
    )


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("dtype", [torch.float, torch.long, torch.bool])
def test_to_binary_label_forward_dtype(device: str, dtype: torch.dtype) -> None:
    device = torch.device(device)
    module = ToBinaryLabel().to(device=device)
    assert objects_are_equal(
        module(torch.tensor([0, 1, 0, 1], dtype=dtype, device=device)),
        torch.tensor([0, 1, 0, 1], dtype=torch.long, device=device),
    )


########################################
#     Tests for ToCategoricalLabel     #
########################################


@pytest.mark.parametrize("device", get_available_devices())
def test_to_categorical_label_forward_1d(device: str) -> None:
    device = torch.device(device)
    module = ToCategoricalLabel().to(device=device)
    assert objects_are_equal(
        module(torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float, device=device)),
        torch.tensor(3, dtype=torch.long, device=device),
    )


@pytest.mark.parametrize("device", get_available_devices())
def test_to_categorical_label_forward_2d(device: str) -> None:
    device = torch.device(device)
    module = ToCategoricalLabel().to(device=device)
    assert objects_are_equal(
        module(
            torch.tensor(
                [[1.0, 2.0, 3.0, 4.0], [5.0, 3.0, 2.0, 2.0]], dtype=torch.float, device=device
            )
        ),
        torch.tensor([3, 0], dtype=torch.long, device=device),
    )


def test_to_categorical_label_dim_default() -> None:
    assert ToCategoricalLabel().dim == -1


def test_to_categorical_label_forward_dim_0() -> None:
    module = ToCategoricalLabel(dim=0)
    assert objects_are_equal(
        module(torch.tensor([[1.0, 5.0], [3.0, 2.0], [2.0, 4.0]])),
        torch.tensor([1, 0]),
    )


def test_to_categorical_label_forward_dim_1_3d() -> None:
    module = ToCategoricalLabel(dim=1)
    assert module(torch.tensor([[[1.0, 0.0], [0.0, 3.0]]])).shape == (1, 2)


def test_to_categorical_label_extra_repr() -> None:
    assert ToCategoricalLabel(dim=1).extra_repr() == "dim=1"


@pytest.mark.parametrize("threshold", [float("nan"), float("inf")])
def test_to_binary_label_invalid_threshold(threshold: float) -> None:
    with pytest.raises(ValueError, match=r"threshold must be finite"):
        ToBinaryLabel(threshold=threshold)
