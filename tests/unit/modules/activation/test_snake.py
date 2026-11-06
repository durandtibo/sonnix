from __future__ import annotations

import math

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
    module = Snake(init=2.0)
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


def test_snake_zero_init() -> None:
    with pytest.raises(ValueError, match=r"'init' has to be different from 0 \(received 0\)"):
        Snake(init=0)


def test_snake_extra_repr() -> None:
    assert (
        Snake(num_parameters=3, learnable=False).extra_repr() == "num_parameters=3, learnable=False"
    )


def test_snake_learnable_by_default() -> None:
    assert Snake().alpha.requires_grad


def test_snake_frozen_alpha_is_buffer() -> None:
    assert [n for n, _ in Snake(learnable=False).named_parameters()] == []


def test_snake_forward_per_channel_alpha() -> None:
    module = Snake(num_parameters=2, init=2.0)
    assert module(torch.tensor([[1.0, 0.0], [-1.0, 2.0]])).allclose(
        torch.tensor([[1.413410905215903, 0.0], [-0.586589094784097, 2.2863750084521532]])
    )


def test_snake_forward_alpha_gradient_is_finite() -> None:
    module = Snake()
    module(torch.randn(4, 3)).sum().backward()
    assert module.alpha.grad.isfinite().all()


@pytest.mark.parametrize("x", [-1e3, -1.0, 1.0, 1e3])
def test_snake_forward_equals_paper_formula(x: float) -> None:
    module = Snake(init=1.5)
    expected = x + math.sin(1.5 * x) ** 2 / 1.5
    assert module(torch.tensor([x], dtype=torch.float64)).item() == pytest.approx(expected)
