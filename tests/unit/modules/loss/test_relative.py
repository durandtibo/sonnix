from __future__ import annotations

import pytest
import torch
from coola.equality import objects_are_allclose, objects_are_equal
from coola.utils.tensor import get_available_devices
from torch import nn

from sonnix.modules import RelativeLoss, RelativeMSELoss, RelativeSmoothL1Loss
from sonnix.modules.loss import (
    ArithmeticalMeanIndicator,
    ClassicalRelativeIndicator,
    ReversedRelativeIndicator,
)

##################################
#     Tests for RelativeLoss     #
##################################


def test_relative_loss_str() -> None:
    assert str(RelativeLoss(criterion=nn.MSELoss(reduction="none"))).startswith("RelativeLoss(")


@pytest.mark.parametrize("device", get_available_devices())
def test_relative_loss_reduction_mean(device: str) -> None:
    prediction = torch.tensor(
        [[-2.0, 1.0, 0.0], [-3.0, 5.0, -1.0]], device=device, requires_grad=True
    )
    target = torch.tensor([[0.0, 1.0, -1.0], [3.0, 1.0, -1.0]], device=device)
    criterion = RelativeLoss(criterion=nn.MSELoss(reduction="none"), eps=1e-5)
    loss = criterion(prediction=prediction, target=target)
    loss.backward()
    assert objects_are_equal(loss, torch.tensor(66671.5, device=device))


@pytest.mark.parametrize("device", get_available_devices())
def test_relative_loss_reduction_sum(device: str) -> None:
    prediction = torch.tensor(
        [[-2.0, 1.0, 0.0], [-3.0, 5.0, -1.0]], device=device, requires_grad=True
    )
    target = torch.tensor([[0.0, 1.0, -1.0], [3.0, 1.0, -1.0]], device=device)
    criterion = RelativeLoss(criterion=nn.MSELoss(reduction="none"), reduction="sum", eps=1e-5)
    loss = criterion(prediction=prediction, target=target)
    loss.backward()
    assert objects_are_equal(loss, torch.tensor(400029.0, device=device))


@pytest.mark.parametrize("device", get_available_devices())
def test_relative_loss_reduction_none(device: str) -> None:
    prediction = torch.tensor(
        [[-2.0, 1.0, 0.0], [-3.0, 5.0, -1.0]], device=device, requires_grad=True
    )
    target = torch.tensor([[0.0, 1.0, -1.0], [3.0, 1.0, -1.0]], device=device)
    criterion = RelativeLoss(criterion=nn.MSELoss(reduction="none"), reduction="none", eps=0.01)
    loss = criterion(prediction=prediction, target=target)
    assert objects_are_equal(
        loss, torch.tensor([[400.0, 0.0, 1.0], [12.0, 16.0, 0.0]], device=device)
    )


def test_relative_loss_reduction_incorrect() -> None:
    with pytest.raises(ValueError, match=r"Incorrect reduction:"):
        RelativeLoss(criterion=nn.MSELoss(reduction="none"), reduction="incorrect")


@pytest.mark.parametrize("device", get_available_devices())
def test_relative_loss_indicator_arithmetical_mean(device: str) -> None:
    prediction = torch.tensor(
        [[-2.0, 1.0, 0.0], [-3.0, 5.0, -1.0]], device=device, requires_grad=True
    )
    target = torch.tensor([[0.0, 1.0, -1.0], [3.0, 1.0, -1.0]], device=device)
    criterion = RelativeLoss(
        criterion=nn.MSELoss(reduction="none"),
        indicator=ArithmeticalMeanIndicator(),
        reduction="none",
    )
    loss = criterion(prediction=prediction, target=target)
    assert objects_are_equal(
        loss, torch.tensor([[4.0, 0.0, 2.0], [12.0, 5.333333333333333, 0.0]], device=device)
    )


@pytest.mark.parametrize("device", get_available_devices())
def test_relative_loss_indicator_classical_relative(device: str) -> None:
    prediction = torch.tensor(
        [[-2.0, 1.0, 0.0], [-3.0, 5.0, -1.0]], device=device, requires_grad=True
    )
    target = torch.tensor([[0.0, 1.0, -1.0], [3.0, 1.0, -1.0]], device=device)
    criterion = RelativeLoss(
        criterion=nn.MSELoss(reduction="none"),
        indicator=ClassicalRelativeIndicator(),
        reduction="none",
        eps=0.01,
    )
    loss = criterion(prediction=prediction, target=target)
    assert objects_are_equal(
        loss, torch.tensor([[400.0, 0.0, 1.0], [12.0, 16.0, 0.0]], device=device)
    )


@pytest.mark.parametrize("device", get_available_devices())
def test_relative_loss_indicator_reversed_relative(device: str) -> None:
    prediction = torch.tensor(
        [[-2.0, 1.0, 0.0], [-3.0, 5.0, -1.0]], device=device, requires_grad=True
    )
    target = torch.tensor([[0.0, 1.0, -1.0], [3.0, 1.0, -1.0]], device=device)
    criterion = RelativeLoss(
        criterion=nn.MSELoss(reduction="none"),
        indicator=ReversedRelativeIndicator(),
        reduction="none",
        eps=0.01,
    )
    loss = criterion(prediction=prediction, target=target)
    assert objects_are_equal(
        loss, torch.tensor([[2.0, 0.0, 100.0], [12.0, 3.2, 0.0]], device=device)
    )


def test_relative_loss_incorrect_shapes() -> None:
    prediction = torch.tensor([[-2.0, 1.0, 0.0], [-3.0, 5.0, -1.0]], requires_grad=True)
    target = torch.tensor([[0.0, 1.0, -1.0], [3.0, 1.0, -1.0]])
    criterion = RelativeLoss(criterion=nn.MSELoss())
    with pytest.raises(RuntimeError, match=r"loss .* and indicator .* shapes do not match"):
        criterion(prediction=prediction, target=target)


#####################################
#     Tests for RelativeMSELoss     #
#####################################


def test_relative_mse_loss_str() -> None:
    assert str(RelativeMSELoss()).startswith("RelativeMSELoss(")


@pytest.mark.parametrize("device", get_available_devices())
def test_relative_mse_loss_reduction_mean(device: str) -> None:
    prediction = torch.tensor(
        [[-2.0, 1.0, 0.0], [-3.0, 5.0, -1.0]], device=device, requires_grad=True
    )
    target = torch.tensor([[0.0, 1.0, -1.0], [3.0, 1.0, -1.0]], device=device)
    criterion = RelativeMSELoss(eps=1e-5)
    loss = criterion(prediction=prediction, target=target)
    loss.backward()
    assert objects_are_equal(loss, torch.tensor(66671.5, device=device))


@pytest.mark.parametrize("device", get_available_devices())
def test_relative_mse_loss_reduction_sum(device: str) -> None:
    prediction = torch.tensor(
        [[-2.0, 1.0, 0.0], [-3.0, 5.0, -1.0]], device=device, requires_grad=True
    )
    target = torch.tensor([[0.0, 1.0, -1.0], [3.0, 1.0, -1.0]], device=device)
    criterion = RelativeMSELoss(reduction="sum", eps=1e-5)
    loss = criterion(prediction=prediction, target=target)
    loss.backward()
    assert objects_are_equal(loss, torch.tensor(400029.0, device=device))


@pytest.mark.parametrize("device", get_available_devices())
def test_relative_mse_loss_reduction_none(device: str) -> None:
    prediction = torch.tensor(
        [[-2.0, 1.0, 0.0], [-3.0, 5.0, -1.0]], device=device, requires_grad=True
    )
    target = torch.tensor([[0.0, 1.0, -1.0], [3.0, 1.0, -1.0]], device=device)
    criterion = RelativeMSELoss(reduction="none", eps=0.01)
    loss = criterion(prediction=prediction, target=target)
    assert objects_are_equal(
        loss, torch.tensor([[400.0, 0.0, 1.0], [12.0, 16.0, 0.0]], device=device)
    )


def test_relative_mse_loss_reduction_incorrect() -> None:
    with pytest.raises(ValueError, match=r"Incorrect reduction:"):
        RelativeMSELoss(reduction="incorrect")


@pytest.mark.parametrize("device", get_available_devices())
def test_relative_mse_loss_indicator_arithmetical_mean(device: str) -> None:
    prediction = torch.tensor(
        [[-2.0, 1.0, 0.0], [-3.0, 5.0, -1.0]], device=device, requires_grad=True
    )
    target = torch.tensor([[0.0, 1.0, -1.0], [3.0, 1.0, -1.0]], device=device)
    criterion = RelativeMSELoss(indicator=ArithmeticalMeanIndicator(), reduction="none")
    loss = criterion(prediction=prediction, target=target)
    assert objects_are_equal(
        loss, torch.tensor([[4.0, 0.0, 2.0], [12.0, 5.333333333333333, 0.0]], device=device)
    )


@pytest.mark.parametrize("device", get_available_devices())
def test_relative_mse_loss_indicator_classical_relative(device: str) -> None:
    prediction = torch.tensor(
        [[-2.0, 1.0, 0.0], [-3.0, 5.0, -1.0]], device=device, requires_grad=True
    )
    target = torch.tensor([[0.0, 1.0, -1.0], [3.0, 1.0, -1.0]], device=device)
    criterion = RelativeMSELoss(indicator=ClassicalRelativeIndicator(), reduction="none", eps=0.01)
    loss = criterion(prediction=prediction, target=target)
    assert objects_are_equal(
        loss, torch.tensor([[400.0, 0.0, 1.0], [12.0, 16.0, 0.0]], device=device)
    )


@pytest.mark.parametrize("device", get_available_devices())
def test_relative_mse_loss_indicator_reversed_relative(device: str) -> None:
    prediction = torch.tensor(
        [[-2.0, 1.0, 0.0], [-3.0, 5.0, -1.0]], device=device, requires_grad=True
    )
    target = torch.tensor([[0.0, 1.0, -1.0], [3.0, 1.0, -1.0]], device=device)
    criterion = RelativeMSELoss(indicator=ReversedRelativeIndicator(), reduction="none", eps=0.01)
    loss = criterion(prediction=prediction, target=target)
    assert objects_are_equal(
        loss, torch.tensor([[2.0, 0.0, 100.0], [12.0, 3.2, 0.0]], device=device)
    )


##########################################
#     Tests for RelativeSmoothL1Loss     #
##########################################


def test_relative_smooth_l1_loss_str() -> None:
    assert str(RelativeSmoothL1Loss()).startswith("RelativeSmoothL1Loss(")


@pytest.mark.parametrize("device", get_available_devices())
def test_relative_smooth_l1_loss_reduction_mean(device: str) -> None:
    prediction = torch.tensor(
        [[-2.0, 1.0, 0.0], [-3.0, 5.0, -1.0]], device=device, requires_grad=True
    )
    target = torch.tensor([[0.0, 1.0, -1.0], [3.0, 1.0, -1.0]], device=device)
    criterion = RelativeSmoothL1Loss(eps=1e-5)
    loss = criterion(prediction=prediction, target=target)
    loss.backward()
    assert objects_are_allclose(loss, torch.tensor(25000.970703125, device=device), rtol=1e-5)


@pytest.mark.parametrize("device", get_available_devices())
def test_relative_smooth_l1_loss_reduction_sum(device: str) -> None:
    prediction = torch.tensor(
        [[-2.0, 1.0, 0.0], [-3.0, 5.0, -1.0]], device=device, requires_grad=True
    )
    target = torch.tensor([[0.0, 1.0, -1.0], [3.0, 1.0, -1.0]], device=device)
    criterion = RelativeSmoothL1Loss(reduction="sum", eps=1e-5)
    loss = criterion(prediction=prediction, target=target)
    loss.backward()
    assert objects_are_equal(loss, torch.tensor(150005.828125, device=device))


@pytest.mark.parametrize("device", get_available_devices())
def test_relative_smooth_l1_loss_reduction_none(device: str) -> None:
    prediction = torch.tensor(
        [[-2.0, 1.0, 0.0], [-3.0, 5.0, -1.0]], device=device, requires_grad=True
    )
    target = torch.tensor([[0.0, 1.0, -1.0], [3.0, 1.0, -1.0]], device=device)
    criterion = RelativeSmoothL1Loss(reduction="none", eps=0.01)
    loss = criterion(prediction=prediction, target=target)
    assert objects_are_equal(
        loss, torch.tensor([[150.0, 0.0, 0.5], [1.8333333333333333, 3.5, 0.0]], device=device)
    )


def test_relative_smooth_l1_loss_reduction_incorrect() -> None:
    with pytest.raises(ValueError, match=r"Incorrect reduction:"):
        RelativeSmoothL1Loss(reduction="incorrect")


@pytest.mark.parametrize("device", get_available_devices())
def test_relative_smooth_l1_loss_indicator_arithmetical_mean(device: str) -> None:
    prediction = torch.tensor(
        [[-2.0, 1.0, 0.0], [-3.0, 5.0, -1.0]], device=device, requires_grad=True
    )
    target = torch.tensor([[0.0, 1.0, -1.0], [3.0, 1.0, -1.0]], device=device)
    criterion = RelativeSmoothL1Loss(indicator=ArithmeticalMeanIndicator(), reduction="none")
    loss = criterion(prediction=prediction, target=target)
    assert objects_are_equal(
        loss,
        torch.tensor(
            [[1.5, 0.0, 1.0], [1.8333333333333333, 1.1666666666666667, 0.0]], device=device
        ),
    )


@pytest.mark.parametrize("device", get_available_devices())
def test_relative_smooth_l1_loss_indicator_classical_relative(device: str) -> None:
    prediction = torch.tensor(
        [[-2.0, 1.0, 0.0], [-3.0, 5.0, -1.0]], device=device, requires_grad=True
    )
    target = torch.tensor([[0.0, 1.0, -1.0], [3.0, 1.0, -1.0]], device=device)
    criterion = RelativeSmoothL1Loss(
        indicator=ClassicalRelativeIndicator(), reduction="none", eps=0.01
    )
    loss = criterion(prediction=prediction, target=target)
    assert objects_are_equal(
        loss, torch.tensor([[150.0, 0.0, 0.5], [1.8333333333333333, 3.5, 0.0]], device=device)
    )


@pytest.mark.parametrize("device", get_available_devices())
def test_relative_smooth_l1_loss_indicator_reversed_relative(device: str) -> None:
    prediction = torch.tensor(
        [[-2.0, 1.0, 0.0], [-3.0, 5.0, -1.0]], device=device, requires_grad=True
    )
    target = torch.tensor([[0.0, 1.0, -1.0], [3.0, 1.0, -1.0]], device=device)
    criterion = RelativeSmoothL1Loss(
        indicator=ReversedRelativeIndicator(), reduction="none", eps=0.01
    )
    loss = criterion(prediction=prediction, target=target)
    assert objects_are_equal(
        loss, torch.tensor([[0.75, 0.0, 50.0], [1.8333333333333333, 0.7, 0.0]], device=device)
    )
