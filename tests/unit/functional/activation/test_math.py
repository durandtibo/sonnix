from __future__ import annotations

import pytest
import torch
from coola.equality import objects_are_allclose, objects_are_equal
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


def test_safe_exp_preserve_gradient_check1() -> None:
    x = torch.tensor([100.0], requires_grad=True)
    out = safe_exp(x, max_val=20.0, preserve_gradient=True)
    out.backward()
    assert out.allclose(torch.tensor(20.0).exp())


def test_safe_exp_preserve_gradient_grad() -> None:
    x = torch.tensor([100.0], requires_grad=True)
    out = safe_exp(x, max_val=20.0, preserve_gradient=True)
    out.backward()
    assert x.grad.allclose(torch.tensor(20.0).exp())


def test_safe_log_clamp_has_zero_gradient() -> None:
    x = torch.tensor([1e-10], requires_grad=True)
    safe_log(x, min_val=1e-8).backward()
    assert x.grad.item() == 0.0


def test_safe_log_preserve_gradient_check1() -> None:
    x = torch.tensor([1e-10], requires_grad=True)
    out = safe_log(x, min_val=1e-8, preserve_gradient=True)
    out.backward()
    assert out.allclose(torch.tensor(1e-8).log())


def test_safe_log_preserve_gradient_grad() -> None:
    x = torch.tensor([1e-10], requires_grad=True)
    out = safe_log(x, min_val=1e-8, preserve_gradient=True)
    out.backward()
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


@pytest.mark.parametrize("max_val", [float("nan"), float("inf")])
def test_safe_exp_invalid_max_val(max_val: float) -> None:
    with pytest.raises(ValueError, match=r"max_val must be finite"):
        safe_exp(torch.ones(2, 3), max_val=max_val)


#########################################
#     Tests for the dtype of guards     #
#########################################


def test_safe_exp_float16_is_finite_for_a_large_input() -> None:
    assert safe_exp(torch.tensor([1000.0], dtype=torch.float16)).isfinite().all()


def test_safe_exp_float16_large_input_value() -> None:
    # exp(log(65504) - 1), rounded to float16
    assert objects_are_equal(
        safe_exp(torch.tensor([1000.0], dtype=torch.float16)),
        torch.tensor([24192.0], dtype=torch.float16),
    )


def test_safe_exp_float16_keeps_a_small_max_val() -> None:
    assert objects_are_allclose(
        safe_exp(torch.tensor([1000.0], dtype=torch.float16), max_val=1.0),
        torch.tensor([2.71875], dtype=torch.float16),
    )


def test_safe_exp_float32_large_max_val_is_finite() -> None:
    assert safe_exp(torch.tensor([1000.0]), max_val=100.0).isfinite().all()


def test_safe_exp_float32_large_max_val_value() -> None:
    # exp(log(float32 max) - 1)
    assert objects_are_allclose(
        safe_exp(torch.tensor([1000.0]), max_val=100.0), torch.tensor([1.2518291037928049e38])
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_safe_exp_is_finite_in_every_dtype(dtype: torch.dtype) -> None:
    assert safe_exp(torch.tensor([1e4], dtype=dtype), max_val=1e4).isfinite().all()


def test_safe_exp_float16_gradient_is_finite() -> None:
    x = torch.tensor([1000.0], dtype=torch.float16, requires_grad=True)
    safe_exp(x, preserve_gradient=True).backward()
    assert x.grad.isfinite().all()


def test_safe_log_float16_zero_is_finite() -> None:
    # log(6.1035e-05), the smallest normal number of float16
    assert objects_are_equal(
        safe_log(torch.zeros(1, dtype=torch.float16)),
        torch.tensor([-9.703125], dtype=torch.float16),
    )


def test_safe_log_float16_keeps_a_large_min_val() -> None:
    assert objects_are_allclose(
        safe_log(torch.zeros(1, dtype=torch.float16), min_val=1e-2),
        torch.tensor([-4.60546875], dtype=torch.float16),
    )


def test_safe_log_bfloat16_keeps_the_default_min_val() -> None:
    assert objects_are_equal(
        safe_log(torch.zeros(1, dtype=torch.bfloat16)),
        torch.tensor([-18.375], dtype=torch.bfloat16),
    )


def test_safe_log_float32_keeps_the_default_min_val() -> None:
    assert objects_are_allclose(safe_log(torch.zeros(1)), torch.tensor([-18.420680743952367]))


def test_safe_log_float16_gradient_is_finite() -> None:
    x = torch.zeros(1, dtype=torch.float16, requires_grad=True)
    safe_log(x, preserve_gradient=True).backward()
    assert x.grad.isfinite().all()
