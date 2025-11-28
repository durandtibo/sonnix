from __future__ import annotations

import pytest
import torch
from coola import objects_are_allclose, objects_are_equal
from coola.utils.tensor import get_available_devices
from torch import nn

from sonnix.modules import Asinh, TransformedLoss

SHAPES = [(2,), (2, 3), (2, 3, 4)]


#####################################
#     Tests for TransformedLoss     #
#####################################


def test_transformed_loss_str() -> None:
    assert str(TransformedLoss(criterion=nn.MSELoss())).startswith("TransformedLoss(")


def test_transformed_loss_default() -> None:
    criterion = TransformedLoss(criterion=nn.MSELoss())
    assert isinstance(criterion.prediction, nn.Identity)
    assert isinstance(criterion.target, nn.Identity)


@pytest.mark.parametrize("device", get_available_devices())
def test_transformed_loss_forward_correct(device: str) -> None:
    criterion = TransformedLoss(criterion=nn.MSELoss(), prediction=Asinh(), target=Asinh())
    loss = criterion(torch.ones(2, 3, device=device), torch.ones(2, 3, device=device))
    assert objects_are_equal(loss, torch.tensor(0.0, device=device))


@pytest.mark.parametrize("device", get_available_devices())
def test_transformed_loss_forward_incorrect(device: str) -> None:
    criterion = TransformedLoss(criterion=nn.MSELoss(), prediction=Asinh(), target=Asinh())
    loss = criterion(torch.ones(2, 3, device=device), -torch.ones(2, 3, device=device))
    assert objects_are_allclose(loss, torch.tensor(3.107277599582784, device=device))


@pytest.mark.parametrize("device", get_available_devices())
def test_transformed_loss_forward_partially_correct(device: str) -> None:
    criterion = TransformedLoss(criterion=nn.MSELoss(), prediction=Asinh(), target=Asinh())
    loss = criterion(torch.ones(2, 2, device=device), 2 * torch.eye(2, device=device) - 1)
    assert objects_are_allclose(loss, torch.tensor(1.553638799791392, device=device))


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("shape", SHAPES)
def test_transformed_loss_shape(device: str, shape: tuple[int, ...]) -> None:
    device = torch.device(device)
    criterion = TransformedLoss(criterion=nn.MSELoss(), prediction=Asinh(), target=Asinh())
    loss = criterion(torch.ones(*shape, device=device), torch.ones(*shape, device=device))
    assert objects_are_equal(loss, torch.tensor(0.0, device=device))
