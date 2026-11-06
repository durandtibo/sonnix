from __future__ import annotations

import pytest
import torch
from torch import nn
from torch.nn import MSELoss

from sonnix.modules import NLinear
from sonnix.utils.loss import is_loss_decreasing_with_adam


def test_nlinear_loss() -> None:
    assert is_loss_decreasing_with_adam(
        module=nn.Sequential(
            NLinear(n=3, in_features=4, out_features=8),
            nn.ReLU(),
            NLinear(n=3, in_features=8, out_features=6),
        ),
        criterion=MSELoss(),
        feature=torch.randn(8, 3, 4),
        target=torch.randn(8, 3, 6),
    )


def _nlinear_and_linears(bias: bool) -> tuple[NLinear, nn.ModuleList]:
    module = NLinear(n=3, in_features=4, out_features=5, bias=bias)
    linears = nn.ModuleList([nn.Linear(4, 5, bias=bias) for _ in range(3)])
    with torch.no_grad():
        for i, linear in enumerate(linears):
            linear.weight.copy_(module.weight[i].t())
            if bias:
                linear.bias.copy_(module.bias[i])
    return module, linears


@pytest.mark.parametrize("bias", [True, False])
def test_nlinear_equals_n_linear_modules(bias: bool) -> None:
    module, linears = _nlinear_and_linears(bias)
    x = torch.randn(6, 3, 4, generator=torch.Generator().manual_seed(1))
    expected = torch.stack([linear(x[:, i]) for i, linear in enumerate(linears)], dim=1)
    assert module(x).allclose(expected, atol=1e-6)


def test_nlinear_equals_n_linear_modules_extra_dimensions() -> None:
    module, linears = _nlinear_and_linears(bias=True)
    x = torch.randn(2, 6, 3, 4, generator=torch.Generator().manual_seed(1))
    expected = torch.stack([linear(x[..., i, :]) for i, linear in enumerate(linears)], dim=-2)
    assert module(x).allclose(expected, atol=1e-6)


def test_nlinear_gradient_equals_n_linear_modules() -> None:
    module, linears = _nlinear_and_linears(bias=True)
    x = torch.randn(6, 3, 4, generator=torch.Generator().manual_seed(1))
    module(x).sum().backward()
    torch.stack([linear(x[:, i]) for i, linear in enumerate(linears)], dim=1).sum().backward()
    expected = torch.stack([linear.weight.grad.t() for linear in linears])
    assert module.weight.grad.allclose(expected, atol=1e-6)
