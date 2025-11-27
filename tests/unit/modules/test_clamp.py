from __future__ import annotations

import pytest
import torch
from coola import objects_are_allclose, objects_are_equal
from coola.utils.tensor import get_available_devices

from sonnix.modules import Clamp

SHAPES = [(2,), (2, 3), (2, 3, 4), (2, 3, 4, 5)]

###########################
#     Tests for Clamp     #
###########################


def test_clamp_str() -> None:
    assert str(Clamp()).startswith("Clamp(")


@pytest.mark.parametrize("device", get_available_devices())
def test_clamp_forward(device: str) -> None:
    device = torch.device(device)
    module = Clamp().to(device=device)
    assert objects_are_equal(
        module(torch.arange(-3, 4, dtype=torch.float, device=device)),
        torch.tensor([-1.0, -1.0, -1.0, 0.0, 1.0, 1.0, 1.0], dtype=torch.float, device=device),
    )


@pytest.mark.parametrize("device", get_available_devices())
def test_clamp_forward_min_0(device: str) -> None:
    device = torch.device(device)
    module = Clamp(min=0.0).to(device=device)
    assert objects_are_equal(
        module(torch.arange(-3, 4, dtype=torch.float, device=device)),
        torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0], dtype=torch.float, device=device),
    )


@pytest.mark.parametrize("device", get_available_devices())
def test_clamp_forward_max_2(device: str) -> None:
    device = torch.device(device)
    module = Clamp(max=2).to(device=device)
    assert objects_are_equal(
        module(torch.arange(-3, 4, dtype=torch.float, device=device)),
        torch.tensor([-1.0, -1.0, -1.0, 0.0, 1.0, 2.0, 2.0], dtype=torch.float, device=device),
    )


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("shape", SHAPES)
def test_clamp_forward_shape(device: str, shape: tuple[int, ...]) -> None:
    device = torch.device(device)
    module = Clamp().to(device=device)
    assert objects_are_allclose(
        module(torch.ones(*shape, device=device)), torch.ones(*shape, device=device)
    )
