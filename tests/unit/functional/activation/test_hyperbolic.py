from __future__ import annotations

import pytest
import torch
from coola import objects_are_allclose
from coola.utils.tensor import get_available_devices

from sonnix.functional import rectifier_asinh_unit

SHAPES = [(2,), (2, 3), (2, 3, 4), (2, 3, 4, 5)]

##########################################
#     Tests for rectifier_asinh_unit     #
##########################################


@pytest.mark.parametrize("device", get_available_devices())
def test_rectifier_asinh_unit_max_value_default(device: str) -> None:
    assert objects_are_allclose(
        rectifier_asinh_unit(torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0], device=device)),
        torch.tensor(
            [0.0, 0.0, 0.0, 0.8813735842704773, 1.4436354637145996],
            device=device,
        ),
    )


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("shape", SHAPES)
def test_rectifier_asinh_unit_shape(device: str, shape: tuple[int, ...]) -> None:
    device = torch.device(device)
    assert objects_are_allclose(
        rectifier_asinh_unit(torch.zeros(*shape, device=device)), torch.ones(*shape, device=device)
    )
