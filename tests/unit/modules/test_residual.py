from __future__ import annotations

import pytest
import torch
from coola.utils.tensor import get_available_devices
from torch import nn

from sonnix.modules import ResidualBlock

SIZES = (1, 2)


###################################
#     Tests for ResidualBlock     #
###################################


def test_residual_block_residual() -> None:
    assert isinstance(ResidualBlock(residual=nn.Linear(4, 4)).residual, nn.Linear)


def test_residual_block_skip_default() -> None:
    assert isinstance(ResidualBlock(residual=nn.Linear(4, 4)).skip, nn.Identity)


def test_residual_block_skip() -> None:
    assert isinstance(ResidualBlock(residual=nn.Linear(4, 4), skip=nn.Linear(4, 4)).skip, nn.Linear)


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("batch_size", SIZES)
@pytest.mark.parametrize("mode", [True, False])
def test_residual_block_forward(device: str, batch_size: int, mode: bool) -> None:
    device = torch.device(device)
    module = ResidualBlock(residual=nn.Linear(4, 4)).to(device=device)
    module.train(mode)
    out = module(torch.rand(batch_size, 4, device=device))
    assert out.shape == (batch_size, 4)
    assert out.device == device
    assert out.dtype == torch.float


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("batch_size", SIZES)
@pytest.mark.parametrize("mode", [True, False])
def test_residual_block_forward_skip(device: str, batch_size: int, mode: bool) -> None:
    device = torch.device(device)
    module = ResidualBlock(
        residual=nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 4)), skip=nn.Linear(4, 4)
    ).to(device=device)
    module.train(mode)
    out = module(torch.rand(batch_size, 4, device=device))
    assert out.shape == (batch_size, 4)
    assert out.device == device
    assert out.dtype == torch.float


def test_residual_block_empty_sequential_skip_is_kept() -> None:
    skip = nn.Sequential()
    module = ResidualBlock(residual=nn.Linear(4, 4), skip=skip)
    assert module.skip is skip


def test_residual_block_branches_with_different_shapes() -> None:
    with pytest.raises(
        ValueError,
        match=r"skip and residual must have the same shape, got \(4, 3\) and \(4, 1\)",
    ):
        ResidualBlock(residual=nn.Linear(3, 1))(torch.ones(4, 3))


def test_residual_block_branches_with_different_number_of_dimensions() -> None:
    with pytest.raises(ValueError, match=r"got \(4, 3\) and \(12,\)"):
        ResidualBlock(residual=nn.Flatten(0))(torch.ones(4, 3))


def test_residual_block_custom_skip_with_a_different_shape() -> None:
    with pytest.raises(ValueError, match=r"got \(4, 2\) and \(4, 3\)"):
        ResidualBlock(residual=nn.Identity(), skip=nn.Linear(3, 2))(torch.ones(4, 3))
