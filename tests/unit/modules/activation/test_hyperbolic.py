from __future__ import annotations

import pytest
import torch
from coola import objects_are_allclose
from coola.utils.tensor import get_available_devices

from sonnix.modules import (
    RectifierAsinhUnit,
)

SHAPES = [(2,), (2, 3), (2, 3, 4), (2, 3, 4, 5)]

########################################
#     Tests for RectifierAsinhUnit     #
########################################


@pytest.mark.parametrize("device", get_available_devices())
def test_rectifier_asinh_unit_forward(device: str) -> None:
    device = torch.device(device)
    module = RectifierAsinhUnit().to(device=device)
    assert objects_are_allclose(
        module(torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0], device=device)),
        torch.tensor(
            [0.0, 0.0, 0.0, 0.8813735842704773, 1.4436354637145996],
            dtype=torch.float,
            device=device,
        ),
    )


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("shape", SHAPES)
def test_rectifier_asinh_unit_forward_shape(device: str, shape: tuple[int, ...]) -> None:
    device = torch.device(device)
    module = RectifierAsinhUnit().to(device=device)
    assert objects_are_allclose(
        module(torch.ones(*shape, device=device)),
        torch.full(shape, fill_value=0.8813735842704773, device=device),
    )
