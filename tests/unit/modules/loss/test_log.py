from __future__ import annotations

import pytest
import torch
from coola.equality import objects_are_allclose
from coola.utils.tensor import get_available_devices

from sonnix.functional import log_cosh_loss
from sonnix.modules import LogCoshLoss
from tests.helpers import make_generator

#################################
#     Tests for LogCoshLoss     #
#################################


def test_log_cosh_loss_repr_contains_scale() -> None:
    assert repr(LogCoshLoss(scale=2.0, reduction="sum")) == "LogCoshLoss(reduction=sum, scale=2.0)"


def test_log_cosh_loss_default_scale() -> None:
    assert LogCoshLoss().scale == 1.0


@pytest.mark.parametrize("device", get_available_devices())
def test_log_cosh_loss_correct(device: str) -> None:
    device = torch.device(device)
    criterion = LogCoshLoss().to(device=device)
    loss = criterion(torch.ones(2, 3, device=device), torch.ones(2, 3, device=device))
    assert objects_are_allclose(loss, torch.tensor(0.0, device=device))


@pytest.mark.parametrize("device", get_available_devices())
def test_log_cosh_loss_incorrect(device: str) -> None:
    device = torch.device(device)
    criterion = LogCoshLoss().to(device=device)
    loss = criterion(torch.ones(2, 3, device=device), -torch.ones(2, 3, device=device))
    assert objects_are_allclose(loss, torch.tensor(1.3250027473578645, device=device))


@pytest.mark.parametrize("device", get_available_devices())
def test_log_cosh_loss_scale(device: str) -> None:
    device = torch.device(device)
    criterion = LogCoshLoss(scale=2.0).to(device=device)
    loss = criterion(torch.ones(2, 3, device=device), -torch.ones(2, 3, device=device))
    assert objects_are_allclose(loss, torch.tensor(0.8675616609660542, device=device))


@pytest.mark.parametrize("device", get_available_devices())
def test_log_cosh_loss_reduction_none(device: str) -> None:
    device = torch.device(device)
    criterion = LogCoshLoss(reduction="none").to(device=device)
    loss = criterion(torch.ones(2, 3, device=device), torch.ones(2, 3, device=device))
    assert objects_are_allclose(loss, torch.zeros(2, 3, device=device))


def test_log_cosh_loss_same_as_function() -> None:
    generator = make_generator(1)
    prediction = torch.randn(4, 5, generator=generator)
    target = torch.randn(4, 5, generator=generator)
    assert LogCoshLoss(scale=0.5)(prediction, target).equal(
        log_cosh_loss(prediction, target, scale=0.5)
    )


def test_log_cosh_loss_backward_finite() -> None:
    prediction = torch.zeros(2, 3, requires_grad=True)
    LogCoshLoss()(prediction, torch.zeros(2, 3)).backward()
    assert prediction.grad.isfinite().all()


def test_log_cosh_loss_large_error_finite() -> None:
    assert LogCoshLoss()(torch.tensor([1e4]), torch.tensor([-1e4])).isfinite()


@pytest.mark.parametrize("scale", [0.0, -1.0])
def test_log_cosh_loss_scale_incorrect(scale: float) -> None:
    with pytest.raises(ValueError, match=r"scale must be greater than 0"):
        LogCoshLoss(scale=scale)


def test_log_cosh_loss_scale_keyword_reduction() -> None:
    with pytest.raises(TypeError, match=r"positional argument"):
        LogCoshLoss(1.0, "sum")
