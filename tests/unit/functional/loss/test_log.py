from __future__ import annotations

import pytest
import torch
from coola.utils.tensor import get_available_devices

from sonnix.functional import log_cosh_loss, msle_loss

DTYPES = (torch.long, torch.float)

###################################
#     Tests for log_cosh_loss     #
###################################


@pytest.mark.parametrize("device", get_available_devices())
def test_log_cosh_loss_correct(device: str) -> None:
    device = torch.device(device)
    assert log_cosh_loss(torch.ones(2, 3, device=device), torch.ones(2, 3, device=device)).equal(
        torch.tensor(0.0, device=device)
    )


@pytest.mark.parametrize("device", get_available_devices())
def test_log_cosh_loss_correct_zeros(device: str) -> None:
    device = torch.device(device)
    assert log_cosh_loss(torch.zeros(2, 3, device=device), torch.zeros(2, 3, device=device)).equal(
        torch.tensor(0.0, device=device)
    )


@pytest.mark.parametrize("device", get_available_devices())
def test_log_cosh_loss_incorrect(device: str) -> None:
    device = torch.device(device)
    assert log_cosh_loss(
        torch.ones(2, 3, device=device), -torch.ones(2, 3, device=device)
    ).allclose(torch.tensor(1.3250027473578645, device=device))


@pytest.mark.parametrize("device", get_available_devices())
def test_log_cosh_loss_partially_correct(device: str) -> None:
    device = torch.device(device)
    assert log_cosh_loss(torch.eye(2, device=device), torch.ones(2, 2, device=device)).allclose(
        torch.tensor(0.21689041524151356, device=device)
    )


@pytest.mark.parametrize("device", get_available_devices())
def test_log_cosh_loss_scale_0_5(device: str) -> None:
    device = torch.device(device)
    assert log_cosh_loss(
        torch.eye(2, device=device), torch.ones(2, 2, device=device), scale=0.5
    ).allclose(torch.tensor(0.6625013736789322, device=device))


@pytest.mark.parametrize("device", get_available_devices())
def test_log_cosh_loss_scale_2(device: str) -> None:
    device = torch.device(device)
    assert log_cosh_loss(
        torch.eye(2, device=device), torch.ones(2, 2, device=device), scale=2.0
    ).allclose(torch.tensor(0.06005725347913873, device=device))


@pytest.mark.parametrize("device", get_available_devices())
def test_log_cosh_loss_reduction_sum(device: str) -> None:
    device = torch.device(device)
    assert log_cosh_loss(
        torch.eye(2, device=device),
        torch.ones(2, 2, device=device),
        reduction="sum",
    ).allclose(torch.tensor(0.8675616609660542, device=device))


@pytest.mark.parametrize("device", get_available_devices())
def test_log_cosh_loss_reduction_none(device: str) -> None:
    device = torch.device(device)
    assert log_cosh_loss(
        torch.eye(2, device=device),
        torch.ones(2, 2, device=device),
        reduction="none",
    ).allclose(torch.tensor([[0.0, 0.4337808304830271], [0.4337808304830271, 0.0]], device=device))


def test_log_cosh_loss_reduction_incorrect() -> None:
    with pytest.raises(ValueError, match=r"Incorrect reduction: incorrect."):
        log_cosh_loss(torch.ones(2, 2), torch.ones(2, 2), reduction="incorrect")


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("shape", [(2,), (2, 3), (2, 3, 4)])
def test_log_cosh_loss_shape(device: str, shape: tuple[int, ...]) -> None:
    device = torch.device(device)
    assert log_cosh_loss(
        torch.ones(*shape, device=device), torch.ones(*shape, device=device)
    ).equal(torch.tensor(0.0, device=device))


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("dtype_prediction", DTYPES)
@pytest.mark.parametrize("dtype_target", DTYPES)
def test_log_cosh_loss_dtype(
    device: str, dtype_prediction: torch.dtype, dtype_target: torch.dtype
) -> None:
    device = torch.device(device)
    assert log_cosh_loss(
        torch.ones(2, 3, device=device, dtype=dtype_prediction),
        torch.ones(2, 3, device=device, dtype=dtype_target),
    ).equal(torch.tensor(0.0, device=device, dtype=torch.float))


###############################
#     Tests for msle_loss     #
###############################


@pytest.mark.parametrize("device", get_available_devices())
def test_msle_loss_correct(device: str) -> None:
    device = torch.device(device)
    assert msle_loss(torch.ones(2, 3, device=device), torch.ones(2, 3, device=device)).equal(
        torch.tensor(0.0, device=device)
    )


@pytest.mark.parametrize("device", get_available_devices())
def test_msle_loss_incorrect(device: str) -> None:
    device = torch.device(device)
    assert msle_loss(torch.ones(2, 3, device=device), torch.zeros(2, 3, device=device)).allclose(
        torch.tensor(0.4804530139182014, device=device)
    )


@pytest.mark.parametrize("device", get_available_devices())
def test_msle_loss_partially_correct(device: str) -> None:
    device = torch.device(device)
    assert msle_loss(torch.ones(2, 2, device=device), torch.eye(2, device=device)).allclose(
        torch.tensor(0.2402265069591007, device=device)
    )


@pytest.mark.parametrize("device", get_available_devices())
def test_msle_loss_reduction_sum(device: str) -> None:
    device = torch.device(device)
    assert msle_loss(
        torch.ones(2, 2, device=device), torch.eye(2, device=device), reduction="sum"
    ).allclose(torch.tensor(0.9609060278364028, device=device))


@pytest.mark.parametrize("device", get_available_devices())
def test_msle_loss_reduction_none(device: str) -> None:
    device = torch.device(device)
    assert msle_loss(
        torch.ones(2, 2, device=device), torch.eye(2, device=device), reduction="none"
    ).allclose(
        torch.tensor([[0.0, 0.4804530139182014], [0.4804530139182014, 0.0]], device=device),
    )


def test_msle_loss_reduction_incorrect() -> None:
    with pytest.raises(ValueError, match=r"incorrect is not a valid value for reduction"):
        msle_loss(torch.ones(2, 2), torch.eye(2), reduction="incorrect")


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("shape", [(2,), (2, 3), (2, 3, 4)])
def test_msle_loss_shape(device: str, shape: tuple[int, ...]) -> None:
    device = torch.device(device)
    assert msle_loss(torch.ones(*shape, device=device), torch.ones(*shape, device=device)).equal(
        torch.tensor(0.0, device=device)
    )
