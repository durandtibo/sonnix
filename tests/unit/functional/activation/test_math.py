from __future__ import annotations

import pytest
import torch
from coola.equality import objects_are_allclose
from coola.utils.tensor import get_available_devices

from sonnix.functional import safe_exp, safe_log

SHAPES = [(2,), (2, 3), (2, 3, 4), (2, 3, 4, 5)]

##############################
#     Tests for safe_exp     #
##############################


@pytest.mark.parametrize("device", get_available_devices())
def test_safe_exp_max_value_default(device: str) -> None:
    assert objects_are_allclose(
        safe_exp(torch.tensor([-1.0, 0.0, 1.0, 10.0, 100.0], device=device)),
        torch.tensor(
            [0.3678794503211975, 1.0, 2.7182817459106445, 22026.46484375, 485165184.0],
            device=device,
        ),
    )


@pytest.mark.parametrize("device", get_available_devices())
def test_safe_exp_max_value_1(device: str) -> None:
    assert objects_are_allclose(
        safe_exp(torch.tensor([-1.0, 0.0, 1.0, 10.0, 100.0], device=device), max_val=1.0),
        torch.tensor(
            [0.3678794503211975, 1.0, 2.7182817459106445, 2.7182817459106445, 2.7182817459106445],
            device=device,
        ),
    )


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("shape", SHAPES)
def test_safe_exp_shape(device: str, shape: tuple[int, ...]) -> None:
    device = torch.device(device)
    assert objects_are_allclose(
        safe_exp(torch.zeros(*shape, device=device)), torch.ones(*shape, device=device)
    )


##############################
#     Tests for safe_log     #
##############################


@pytest.mark.parametrize("device", get_available_devices())
def test_safe_log_min_value_default(device: str) -> None:
    assert safe_log(torch.tensor([-1.0, 0.0, 1.0, 2.0], device=device)).allclose(
        torch.tensor(
            [-18.420680743952367, -18.420680743952367, 0.0, 0.6931471805599453], device=device
        ),
    )


@pytest.mark.parametrize("device", get_available_devices())
def test_safe_log_min_value_1(device: str) -> None:
    assert safe_log(torch.tensor([-1.0, 0.0, 1.0, 2.0], device=device), min_val=1.0).equal(
        torch.tensor([0.0, 0.0, 0.0, 0.6931471805599453], device=device),
    )


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("shape", SHAPES)
def test_safe_log_shape(device: str, shape: tuple[int, ...]) -> None:
    device = torch.device(device)
    assert objects_are_allclose(
        safe_log(torch.ones(*shape, device=device)), torch.zeros(*shape, device=device)
    )


def test_safe_exp_clamp_has_zero_gradient() -> None:
    x = torch.tensor([100.0], requires_grad=True)
    safe_exp(x, max_val=20.0).backward()
    assert x.grad.item() == 0.0


def test_safe_exp_preserve_gradient() -> None:
    x = torch.tensor([100.0], requires_grad=True)
    out = safe_exp(x, max_val=20.0, preserve_gradient=True)
    out.backward()
    assert out.allclose(torch.tensor(20.0).exp())
    assert x.grad.allclose(torch.tensor(20.0).exp())


def test_safe_log_clamp_has_zero_gradient() -> None:
    x = torch.tensor([1e-10], requires_grad=True)
    safe_log(x, min_val=1e-8).backward()
    assert x.grad.item() == 0.0


def test_safe_log_preserve_gradient() -> None:
    x = torch.tensor([1e-10], requires_grad=True)
    out = safe_log(x, min_val=1e-8, preserve_gradient=True)
    out.backward()
    assert out.allclose(torch.tensor(1e-8).log())
    assert x.grad.allclose(torch.tensor(1e8))


@pytest.mark.parametrize("min_val", [0.0, -1.0])
def test_safe_log_invalid_min_val(min_val: float) -> None:
    with pytest.raises(ValueError, match=r"min_val must be greater than 0"):
        safe_log(torch.ones(2, 3), min_val=min_val)


def test_safe_log_preserve_gradient_below_min_val_is_inverse_of_min_val() -> None:
    x = torch.tensor([0.0], requires_grad=True)
    safe_log(x, min_val=1e-2, preserve_gradient=True).backward()
    assert x.grad.allclose(torch.tensor([100.0]))


def test_safe_log_nan_stays_nan() -> None:
    assert safe_log(torch.tensor([float("nan")])).isnan().all()


def test_safe_exp_nan_stays_nan() -> None:
    assert safe_exp(torch.tensor([float("nan")])).isnan().all()
