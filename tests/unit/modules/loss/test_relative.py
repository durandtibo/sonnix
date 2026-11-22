from __future__ import annotations

from functools import partial

import pytest
import torch
from coola.equality import objects_are_equal
from coola.utils.tensor import get_available_devices
from torch import nn

from sonnix.functional import (
    arithmetical_mean_indicator,
    classical_relative_indicator,
    moment_mean_indicator,
    reversed_relative_indicator,
)
from sonnix.modules import RelativeLoss

##################################
#     Tests for RelativeLoss     #
##################################


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


@pytest.mark.parametrize("device", get_available_devices())
def test_relative_loss_indicator_arithmetical_mean(device: str) -> None:
    prediction = torch.tensor(
        [[-2.0, 1.0, 0.0], [-3.0, 5.0, -1.0]], device=device, requires_grad=True
    )
    target = torch.tensor([[0.0, 1.0, -1.0], [3.0, 1.0, -1.0]], device=device)
    criterion = RelativeLoss(
        criterion=nn.MSELoss(reduction="none"),
        indicator=arithmetical_mean_indicator,
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
        indicator=classical_relative_indicator,
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
        indicator=reversed_relative_indicator,
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
    with pytest.raises(ValueError, match=r"loss and indicator must have the same shape"):
        criterion(prediction=prediction, target=target)


def test_relative_loss_default_indicator() -> None:
    assert RelativeLoss(criterion=nn.MSELoss()).indicator is classical_relative_indicator


def test_relative_loss_indicator_is_kept() -> None:
    assert (
        RelativeLoss(criterion=nn.MSELoss(), indicator=reversed_relative_indicator).indicator
        is reversed_relative_indicator
    )


def test_relative_loss_indicator_partial() -> None:
    criterion = RelativeLoss(
        criterion=nn.L1Loss(reduction="none"),
        indicator=partial(moment_mean_indicator, k=2),
        reduction="none",
    )
    prediction = torch.tensor([[3.0, 1.0]])
    target = torch.tensor([[1.0, 1.0]])
    # The moment mean indicator of order 2 is the quadratic mean.
    assert criterion(prediction, target).allclose(torch.tensor([[0.8944271909999159, 0.0]]))


def test_relative_loss_indicator_incorrect() -> None:
    with pytest.raises(TypeError, match=r"indicator must be callable but received 1.0"):
        RelativeLoss(criterion=nn.MSELoss(), indicator=1.0)


#######################################
#     Tests for detach_indicator     #
#######################################


def test_relative_loss_detach_indicator_false_gradient_flows_through_indicator() -> None:
    prediction = torch.tensor([[2.0, 4.0]], requires_grad=True)
    target = torch.zeros(1, 2)
    module = RelativeLoss(
        criterion=nn.L1Loss(reduction="none"),
        indicator=reversed_relative_indicator,
        reduction="sum",
    )
    module(prediction, target).backward()
    # loss / indicator is constant (1) so the gradient is zero when the indicator is not detached.
    assert prediction.grad.allclose(torch.zeros(1, 2))


def test_relative_loss_detach_indicator_true_gradient_ignores_indicator() -> None:
    prediction = torch.tensor([[2.0, 4.0]], requires_grad=True)
    target = torch.zeros(1, 2)
    module = RelativeLoss(
        criterion=nn.L1Loss(reduction="none"),
        indicator=reversed_relative_indicator,
        reduction="sum",
        detach_indicator=True,
    )
    module(prediction, target).backward()
    assert prediction.grad.allclose(torch.tensor([[0.5, 0.25]]))


@pytest.mark.parametrize("eps", [0.0, -1.0])
def test_relative_loss_invalid_eps(eps: float) -> None:
    with pytest.raises(ValueError, match=r"eps must be greater than 0"):
        RelativeLoss(criterion=nn.MSELoss(reduction="none"), eps=eps)


def test_relative_loss_extra_repr() -> None:
    assert (
        RelativeLoss(nn.MSELoss(), eps=1e-3, reduction="sum", detach_indicator=True).extra_repr()
        == "reduction=sum, eps=0.001, detach_indicator=True"
    )
