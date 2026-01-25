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
        safe_exp(torch.tensor([-1.0, 0.0, 1.0, 10.0, 100.0], device=device), max=1.0),
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
    assert safe_log(torch.tensor([-1.0, 0.0, 1.0, 2.0], device=device), min=1.0).equal(
        torch.tensor([0.0, 0.0, 0.0, 0.6931471805599453], device=device),
    )


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("shape", SHAPES)
def test_safe_log_shape(device: str, shape: tuple[int, ...]) -> None:
    device = torch.device(device)
    assert objects_are_allclose(
        safe_log(torch.ones(*shape, device=device)), torch.zeros(*shape, device=device)
    )
