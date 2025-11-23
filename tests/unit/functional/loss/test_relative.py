from __future__ import annotations

import pytest
import torch
from coola import objects_are_allclose, objects_are_equal
from coola.utils.tensor import get_available_devices

from sonnix.functional import relative_loss
from sonnix.functional.loss.relative import (
    arithmetical_mean_indicator,
    classical_relative_indicator,
    geometric_mean_indicator,
    maximum_mean_indicator,
    minimum_mean_indicator,
    moment_mean_indicator,
    reversed_relative_indicator,
)

###################################
#     Tests for relative_loss     #
###################################


@pytest.mark.parametrize("device", get_available_devices())
def test_relative_loss_reduction_mean(device: str) -> None:
    prediction = torch.tensor(
        [[-2.0, 1.0, 0.0], [-3.0, 5.0, -1.0]], device=device, requires_grad=True
    )
    target = torch.tensor([[0.0, 1.0, -1.0], [3.0, 1.0, -1.0]], device=device)
    loss = relative_loss(
        loss=torch.nn.functional.mse_loss(prediction, target, reduction="none"),
        indicator=classical_relative_indicator(prediction, target),
        eps=1e-5,
    )
    loss.backward()
    assert objects_are_equal(loss, torch.tensor(66671.5, device=device))


@pytest.mark.parametrize("device", get_available_devices())
def test_relative_loss_reduction_sum(device: str) -> None:
    prediction = torch.tensor(
        [[-2.0, 1.0, 0.0], [-3.0, 5.0, -1.0]], device=device, requires_grad=True
    )
    target = torch.tensor([[0.0, 1.0, -1.0], [3.0, 1.0, -1.0]], device=device)
    loss = relative_loss(
        loss=torch.nn.functional.mse_loss(prediction, target, reduction="none"),
        indicator=classical_relative_indicator(prediction, target),
        reduction="sum",
        eps=1e-5,
    )
    loss.backward()
    assert objects_are_equal(loss, torch.tensor(400029.0, device=device))


@pytest.mark.parametrize("device", get_available_devices())
def test_relative_loss_reduction_none(device: str) -> None:
    prediction = torch.tensor(
        [[-2.0, 1.0, 0.0], [-3.0, 5.0, -1.0]], device=device, requires_grad=True
    )
    target = torch.tensor([[0.0, 1.0, -1.0], [3.0, 1.0, -1.0]], device=device)
    loss = relative_loss(
        loss=torch.nn.functional.mse_loss(prediction, target, reduction="none"),
        indicator=classical_relative_indicator(prediction, target),
        reduction="none",
    )
    assert objects_are_equal(
        loss, torch.tensor([[4e8, 0.0, 1.0], [12.0, 16.0, 0.0]], device=device)
    )


def test_relative_loss_reduction_incorrect() -> None:
    prediction = torch.tensor([[-2.0, 1.0, 0.0], [-3.0, 5.0, -1.0]], requires_grad=True)
    target = torch.tensor([[0.0, 1.0, -1.0], [3.0, 1.0, -1.0]])
    with pytest.raises(ValueError, match=r"Incorrect reduction:"):
        relative_loss(
            loss=torch.nn.functional.mse_loss(prediction, target, reduction="none"),
            indicator=classical_relative_indicator(prediction, target),
            reduction="incorrect",
        )


@pytest.mark.parametrize("device", get_available_devices())
def test_relative_loss_indicator_arithmetical_mean(device: str) -> None:
    prediction = torch.tensor(
        [[-2.0, 1.0, 0.0], [-3.0, 5.0, -1.0]], device=device, requires_grad=True
    )
    target = torch.tensor([[0.0, 1.0, -1.0], [3.0, 1.0, -1.0]], device=device)
    loss = relative_loss(
        loss=torch.nn.functional.mse_loss(prediction, target, reduction="none"),
        indicator=arithmetical_mean_indicator(prediction, target),
        reduction="none",
    )
    assert objects_are_equal(
        loss, torch.tensor([[4.0, 0.0, 2.0], [12.0, 5.333333333333333, 0.0]], device=device)
    )


@pytest.mark.parametrize("device", get_available_devices())
def test_relative_loss_indicator_classical_relative(device: str) -> None:
    prediction = torch.tensor(
        [[-2.0, 1.0, 0.0], [-3.0, 5.0, -1.0]], device=device, requires_grad=True
    )
    target = torch.tensor([[0.0, 1.0, -1.0], [3.0, 1.0, -1.0]], device=device)
    loss = relative_loss(
        loss=torch.nn.functional.mse_loss(prediction, target, reduction="none"),
        indicator=classical_relative_indicator(prediction, target),
        reduction="none",
    )
    assert objects_are_equal(
        loss, torch.tensor([[4e8, 0.0, 1.0], [12.0, 16.0, 0.0]], device=device)
    )


@pytest.mark.parametrize("device", get_available_devices())
def test_relative_loss_indicator_reversed_relative(device: str) -> None:
    prediction = torch.tensor(
        [[-2.0, 1.0, 0.0], [-3.0, 5.0, -1.0]], device=device, requires_grad=True
    )
    target = torch.tensor([[0.0, 1.0, -1.0], [3.0, 1.0, -1.0]], device=device)
    loss = relative_loss(
        loss=torch.nn.functional.mse_loss(prediction, target, reduction="none"),
        indicator=reversed_relative_indicator(prediction, target),
        reduction="none",
    )
    assert objects_are_equal(loss, torch.tensor([[2.0, 0.0, 1e8], [12.0, 3.2, 0.0]], device=device))


def test_relative_loss_incorrect_shapes() -> None:
    prediction = torch.tensor([[-2.0, 1.0, 0.0], [-3.0, 5.0, -1.0]], requires_grad=True)
    target = torch.tensor([[0.0, 1.0, -1.0], [3.0, 1.0, -1.0]])
    with pytest.raises(RuntimeError, match=r"loss .* and indicator .* shapes do not match"):
        relative_loss(
            loss=torch.nn.functional.mse_loss(prediction, target),
            indicator=classical_relative_indicator(prediction, target),
        )


#################################################
#     Tests for arithmetical_mean_indicator     #
#################################################


@pytest.mark.parametrize("device", get_available_devices())
def test_arithmetical_mean_indicator(device: str) -> None:
    device = torch.device(device)
    assert objects_are_equal(
        arithmetical_mean_indicator(
            torch.tensor([[0.0, 1.0, -1.0], [3.0, 1.0, -1.0]], device=device),
            torch.tensor([[-2.0, 1.0, 0.0], [-3.0, 5.0, -1.0]], device=device),
        ),
        torch.tensor([[1.0, 1.0, 0.5], [3.0, 3.0, 1.0]], device=device),
    )


##################################################
#     Tests for classical_relative_indicator     #
##################################################


@pytest.mark.parametrize("device", get_available_devices())
def test_classical_relative_indicator(device: str) -> None:
    device = torch.device(device)
    assert objects_are_equal(
        classical_relative_indicator(
            torch.ones(2, 3, device=device),
            torch.tensor([[-2.0, 1.0, 0.0], [-3.0, 5.0, -1.0]], device=device),
        ),
        torch.tensor([[2.0, 1.0, 0.0], [3.0, 5.0, 1.0]], device=device),
    )


##############################################
#     Tests for geometric_mean_indicator     #
##############################################


@pytest.mark.parametrize("device", get_available_devices())
def test_geometric_mean_indicator(device: str) -> None:
    device = torch.device(device)
    assert objects_are_allclose(
        geometric_mean_indicator(
            torch.tensor([[0.0, 1.0, -1.0], [3.0, 1.0, -1.0]], device=device),
            torch.tensor([[-2.0, 1.0, 0.0], [-3.0, 5.0, -1.0]], device=device),
        ),
        torch.tensor([[1e-8, 1.0, 1e-8], [3.0, 2.23606797749979, 1.0]], device=device),
        atol=1e-5,
    )


############################################
#     Tests for maximum_mean_indicator     #
############################################


@pytest.mark.parametrize("device", get_available_devices())
def test_maximum_mean_indicator(device: str) -> None:
    device = torch.device(device)
    assert objects_are_equal(
        maximum_mean_indicator(
            torch.tensor([[0.0, 1.0, -1.0], [3.0, 1.0, -1.0]], device=device),
            torch.tensor([[-2.0, 1.0, 0.0], [-3.0, 5.0, -1.0]], device=device),
        ),
        torch.tensor([[2.0, 1.0, 1.0], [3.0, 5.0, 1.0]], device=device),
    )


############################################
#     Tests for minimum_mean_indicator     #
############################################


@pytest.mark.parametrize("device", get_available_devices())
def test_minimum_mean_indicator(device: str) -> None:
    device = torch.device(device)
    assert objects_are_equal(
        minimum_mean_indicator(
            torch.tensor([[0.0, 1.0, -1.0], [3.0, 1.0, -1.0]], device=device),
            torch.tensor([[-2.0, 1.0, 0.0], [-3.0, 5.0, -1.0]], device=device),
        ),
        torch.tensor([[0.0, 1.0, 0.0], [3.0, 1.0, 1.0]], device=device),
    )


###########################################
#     Tests for moment_mean_indicator     #
###########################################


@pytest.mark.parametrize("device", get_available_devices())
def test_moment_mean_indicator_order_1(device: str) -> None:
    device = torch.device(device)
    assert objects_are_equal(
        moment_mean_indicator(
            torch.tensor([[0.0, 1.0, -1.0], [3.0, 1.0, -1.0]], device=device),
            torch.tensor([[-2.0, 1.0, 0.0], [-3.0, 5.0, -1.0]], device=device),
        ),
        torch.tensor([[1.0, 1.0, 0.5], [3.0, 3.0, 1.0]], device=device),
    )


@pytest.mark.parametrize("device", get_available_devices())
def test_moment_mean_indicator_order_2(device: str) -> None:
    device = torch.device(device)
    assert objects_are_allclose(
        moment_mean_indicator(
            torch.tensor([[0.0, 1.0, -1.0], [3.0, 1.0, -1.0]], device=device),
            torch.tensor([[-2.0, 1.0, 0.0], [-3.0, 5.0, -1.0]], device=device),
            k=2,
        ),
        torch.tensor(
            [[1.4142135623730951, 1.0, 0.7071067811865476], [3.0, 3.605551275463989, 1.0]],
            device=device,
        ),
    )


@pytest.mark.parametrize("device", get_available_devices())
def test_moment_mean_indicator_order_3(device: str) -> None:
    device = torch.device(device)
    assert objects_are_allclose(
        moment_mean_indicator(
            torch.tensor([[0.0, 1.0, -1.0], [3.0, 1.0, -1.0]], device=device),
            torch.tensor([[-2.0, 1.0, 0.0], [-3.0, 5.0, -1.0]], device=device),
            k=3,
        ),
        torch.tensor(
            [[1.5874010519681994, 1.0, 0.7937005259840998], [3.0, 3.9790572078963917, 1.0]],
            device=device,
        ),
    )


#################################################
#     Tests for reversed_relative_indicator     #
#################################################


@pytest.mark.parametrize("device", get_available_devices())
def test_reversed_relative_indicator(device: str) -> None:
    device = torch.device(device)
    assert objects_are_equal(
        reversed_relative_indicator(
            torch.tensor([[-2.0, 1.0, 0.0], [-3.0, 5.0, -1.0]], device=device),
            torch.ones(2, 3, device=device),
        ),
        torch.tensor([[2.0, 1.0, 0.0], [3.0, 5.0, 1.0]], device=device),
    )
