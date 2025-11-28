from __future__ import annotations

import pytest
import torch
from coola.utils.tensor import get_available_devices

from sonnix.modules import Snake

SHAPES = [(2,), (2, 3), (2, 3, 4), (2, 3, 4, 5)]


###########################
#     Tests for Snake     #
###########################


def test_snake_str() -> None:
    assert str(Snake()).startswith("Snake(")


def test_snake_forward_frequency_default() -> None:
    module = Snake()
    assert torch.allclose(
        module(torch.tensor([[1.0, 0.0, -1.0], [-2.0, 0.0, 2.0]])),
        torch.tensor(
            [
                [1.708073418273571, 0.0, -0.2919265817264288],
                [-1.173178189568194, 0.0, 2.826821810431806],
            ]
        ),
    )


def test_snake_forward_frequency_2() -> None:
    module = Snake(frequency=2)
    assert torch.allclose(
        module(torch.tensor([[1.0, 0.0, -1.0], [-2.0, 0.0, 2.0]])),
        torch.tensor(
            [
                [1.413410905215903, 0.0, -0.586589094784097],
                [-1.7136249915478468, 0.0, 2.2863750084521532],
            ]
        ),
    )


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("shape", SHAPES)
def test_snake_forward_shape(shape: tuple[int, ...], device: str) -> None:
    device = torch.device(device)
    module = Snake().to(device=device)
    out = module(torch.randn(*shape, device=device))
    assert out.shape == shape
    assert out.dtype == torch.float
    assert out.device == device
