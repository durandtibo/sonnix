from __future__ import annotations

import pytest
import torch
from coola import objects_are_allclose, objects_are_equal
from coola.utils.tensor import get_available_devices

from sonnix.modules.loss import (
    ArithmeticalMeanIndicator,
    ClassicalRelativeIndicator,
    GeometricMeanIndicator,
    MaximumMeanIndicator,
    MinimumMeanIndicator,
    MomentMeanIndicator,
    ReversedRelativeIndicator,
)

###############################################
#     Tests for ArithmeticalMeanIndicator     #
###############################################


def test_arithmetical_mean_indicator_str() -> None:
    assert str(ArithmeticalMeanIndicator()).startswith("ArithmeticalMeanIndicator(")


@pytest.mark.parametrize("device", get_available_devices())
def test_arithmetical_mean_indicator_forward(device: str) -> None:
    device = torch.device(device)
    indicator = ArithmeticalMeanIndicator()
    assert objects_are_equal(
        indicator(
            torch.tensor([[0.0, 1.0, -1.0], [3.0, 1.0, -1.0]], device=device),
            torch.tensor([[-2.0, 1.0, 0.0], [-3.0, 5.0, -1.0]], device=device),
        ),
        torch.tensor([[1.0, 1.0, 0.5], [3.0, 3.0, 1.0]], device=device),
    )


################################################
#     Tests for ClassicalRelativeIndicator     #
################################################


def test_classical_relative_indicator_str() -> None:
    assert str(ClassicalRelativeIndicator()).startswith("ClassicalRelativeIndicator(")


@pytest.mark.parametrize("device", get_available_devices())
def test_classical_relative_indicator_forward(device: str) -> None:
    device = torch.device(device)
    indicator = ClassicalRelativeIndicator()
    assert objects_are_equal(
        indicator(
            torch.ones(2, 3, device=device),
            torch.tensor([[-2.0, 1.0, 0.0], [-3.0, 5.0, -1.0]], device=device),
        ),
        torch.tensor([[2.0, 1.0, 0.0], [3.0, 5.0, 1.0]], device=device),
    )


################################################
#     Tests for GeometricRelativeIndicator     #
################################################


def test_geometric_mean_indicator_str() -> None:
    assert str(GeometricMeanIndicator()).startswith("GeometricMeanIndicator(")


@pytest.mark.parametrize("device", get_available_devices())
def test_geometric_mean_indicator_forward(device: str) -> None:
    device = torch.device(device)
    indicator = GeometricMeanIndicator()
    assert objects_are_allclose(
        indicator(
            torch.tensor([[0.0, 1.0, -1.0], [3.0, 1.0, -1.0]], device=device),
            torch.tensor([[-2.0, 1.0, 0.0], [-3.0, 5.0, -1.0]], device=device),
        ),
        torch.tensor([[0.0, 1.0, 1e-8], [3.0, 2.23606797749979, 1.0]], device=device),
    )


##########################################
#     Tests for MaximumMeanIndicator     #
##########################################


def test_maximum_mean_indicator_str() -> None:
    assert str(MaximumMeanIndicator()).startswith("MaximumMeanIndicator(")


@pytest.mark.parametrize("device", get_available_devices())
def test_maximum_mean_indicator_forward(device: str) -> None:
    device = torch.device(device)
    indicator = MaximumMeanIndicator()
    assert objects_are_allclose(
        indicator(
            torch.tensor([[0.0, 1.0, -1.0], [3.0, 1.0, -1.0]], device=device),
            torch.tensor([[-2.0, 1.0, 0.0], [-3.0, 5.0, -1.0]], device=device),
        ),
        torch.tensor([[2.0, 1.0, 1.0], [3.0, 5.0, 1.0]], device=device),
    )


##########################################
#     Tests for MinimumMeanIndicator     #
##########################################


def test_minimum_mean_indicator_str() -> None:
    assert str(MinimumMeanIndicator()).startswith("MinimumMeanIndicator(")


@pytest.mark.parametrize("device", get_available_devices())
def test_minimum_mean_indicator_forward(device: str) -> None:
    device = torch.device(device)
    indicator = MinimumMeanIndicator()
    assert objects_are_allclose(
        indicator(
            torch.tensor([[0.0, 1.0, -1.0], [3.0, 1.0, -1.0]], device=device),
            torch.tensor([[-2.0, 1.0, 0.0], [-3.0, 5.0, -1.0]], device=device),
        ),
        torch.tensor([[0.0, 1.0, 0.0], [3.0, 1.0, 1.0]], device=device),
    )


#########################################
#     Tests for MomentMeanIndicator     #
#########################################


def test_moment_mean_indicator_str() -> None:
    assert str(MomentMeanIndicator()).startswith("MomentMeanIndicator(")


@pytest.mark.parametrize("device", get_available_devices())
def test_moment_mean_indicator_forward_oder_1(device: str) -> None:
    device = torch.device(device)
    indicator = MomentMeanIndicator()
    assert objects_are_allclose(
        indicator(
            torch.tensor([[0.0, 1.0, -1.0], [3.0, 1.0, -1.0]], device=device),
            torch.tensor([[-2.0, 1.0, 0.0], [-3.0, 5.0, -1.0]], device=device),
        ),
        torch.tensor([[1.0, 1.0, 0.5], [3.0, 3.0, 1.0]], device=device),
    )


@pytest.mark.parametrize("device", get_available_devices())
def test_moment_mean_indicator_forward_oder_2(device: str) -> None:
    device = torch.device(device)
    indicator = MomentMeanIndicator(k=2)
    assert objects_are_allclose(
        indicator(
            torch.tensor([[0.0, 1.0, -1.0], [3.0, 1.0, -1.0]], device=device),
            torch.tensor([[-2.0, 1.0, 0.0], [-3.0, 5.0, -1.0]], device=device),
        ),
        torch.tensor(
            [[1.4142135623730951, 1.0, 0.7071067811865476], [3.0, 3.605551275463989, 1.0]],
            device=device,
        ),
    )


@pytest.mark.parametrize("device", get_available_devices())
def test_moment_mean_indicator_forward_oder_3(device: str) -> None:
    device = torch.device(device)
    indicator = MomentMeanIndicator(k=3)
    assert objects_are_allclose(
        indicator(
            torch.tensor([[0.0, 1.0, -1.0], [3.0, 1.0, -1.0]], device=device),
            torch.tensor([[-2.0, 1.0, 0.0], [-3.0, 5.0, -1.0]], device=device),
        ),
        torch.tensor(
            [[1.5874010519681994, 1.0, 0.7937005259840998], [3.0, 3.9790572078963917, 1.0]],
            device=device,
        ),
    )


###############################################
#     Tests for ReversedRelativeIndicator     #
###############################################


def test_reversed_relative_indicator_str() -> None:
    assert str(ReversedRelativeIndicator()).startswith("ReversedRelativeIndicator(")


@pytest.mark.parametrize("device", get_available_devices())
def test_reversed_relative_indicator_forward(device: str) -> None:
    device = torch.device(device)
    indicator = ReversedRelativeIndicator()
    assert objects_are_equal(
        indicator(
            torch.tensor([[-2.0, 1.0, 0.0], [-3.0, 5.0, -1.0]], device=device),
            torch.ones(2, 3, device=device),
        ),
        torch.tensor([[2.0, 1.0, 0.0], [3.0, 5.0, 1.0]], device=device),
    )
