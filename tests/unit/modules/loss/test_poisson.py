from __future__ import annotations

import pytest
import torch
from coola.equality import objects_are_allclose, objects_are_equal
from coola.utils.tensor import get_available_devices

from sonnix.modules import PoissonRegressionLoss

###########################################
#     Tests for PoissonRegressionLoss     #
###########################################


@pytest.mark.parametrize("device", get_available_devices())
def test_poisson_regression_loss_correct(device: str) -> None:
    device = torch.device(device)
    criterion = PoissonRegressionLoss().to(device=device)
    loss = criterion(torch.ones(2, 3, device=device), torch.ones(2, 3, device=device))
    assert objects_are_equal(loss, torch.tensor(1.0, device=device))


@pytest.mark.parametrize("device", get_available_devices())
def test_poisson_regression_loss_incorrect(device: str) -> None:
    device = torch.device(device)
    criterion = PoissonRegressionLoss().to(device=device)
    loss = criterion(2 * torch.ones(2, 3, device=device), torch.ones(2, 3, device=device))
    assert objects_are_allclose(loss, torch.tensor(1.3068528194400546, device=device))


@pytest.mark.parametrize("device", get_available_devices())
def test_poisson_regression_loss_partially_correct(device: str) -> None:
    device = torch.device(device)
    criterion = PoissonRegressionLoss().to(device=device)
    loss = criterion(torch.eye(2, device=device) + 1, torch.ones(2, 2, device=device))
    assert objects_are_allclose(loss, torch.tensor(1.1534264097200273, device=device))


@pytest.mark.parametrize("device", get_available_devices())
def test_poisson_regression_loss_reduction_none(device: str) -> None:
    device = torch.device(device)
    criterion = PoissonRegressionLoss(reduction="none").to(device=device)
    loss = criterion(torch.eye(2, device=device) + 1, torch.ones(2, 2, device=device))
    assert objects_are_allclose(
        loss, torch.tensor([[1.3068528194400546, 1.0], [1.0, 1.3068528194400546]], device=device)
    )


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("shape", [(2,), (2, 3), (2, 3, 4)])
def test_poisson_regression_loss_shape(device: str, shape: tuple[int, ...]) -> None:
    device = torch.device(device)
    criterion = PoissonRegressionLoss().to(device=device)
    loss = criterion(torch.ones(*shape, device=device), torch.ones(*shape, device=device))
    assert objects_are_equal(loss, torch.tensor(1.0, device=device))


@pytest.mark.parametrize("device", get_available_devices())
def test_poisson_regression_loss_zero(device: str) -> None:
    device = torch.device(device)
    criterion = PoissonRegressionLoss().to(device=device)
    loss = criterion(torch.zeros(2, 2, device=device), torch.zeros(2, 2, device=device))
    assert loss.allclose(torch.tensor(1e-8, device=device))


@pytest.mark.parametrize("device", get_available_devices())
def test_poisson_regression_loss_pred_zero(device: str) -> None:
    device = torch.device(device)
    criterion = PoissonRegressionLoss().to(device=device)
    loss = criterion(torch.zeros(2, 2, device=device), torch.ones(2, 2, device=device))
    assert loss.allclose(torch.tensor(18.420680743952367, device=device))


@pytest.mark.parametrize("device", get_available_devices())
def test_poisson_regression_loss_target_zero(device: str) -> None:
    device = torch.device(device)
    criterion = PoissonRegressionLoss().to(device=device)
    loss = criterion(torch.ones(2, 2, device=device), torch.zeros(2, 2, device=device))
    assert loss.equal(torch.tensor(1.0, device=device))


@pytest.mark.parametrize("eps", [0.0, -1.0])
def test_poisson_regression_loss_invalid_eps(eps: float) -> None:
    with pytest.raises(ValueError, match=r"eps must be greater than 0"):
        PoissonRegressionLoss(eps=eps)


def test_poisson_regression_loss_extra_repr() -> None:
    assert PoissonRegressionLoss(eps=0.001).extra_repr() == (
        "reduction=mean, eps=0.001, log_input=False"
    )


def test_poisson_regression_loss_extra_repr_log_input() -> None:
    assert PoissonRegressionLoss(log_input=True).extra_repr() == (
        "reduction=mean, eps=1e-08, log_input=True"
    )


def test_poisson_regression_loss_log_input_default() -> None:
    assert not PoissonRegressionLoss().log_input


def test_poisson_regression_loss_log_input_forward() -> None:
    criterion = PoissonRegressionLoss(log_input=True)
    assert criterion(torch.tensor([0.5]), torch.tensor([2.0])).allclose(
        torch.tensor(0.6487212707001282)
    )
