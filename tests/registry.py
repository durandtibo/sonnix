r"""Define a registry of the modules of ``sonnix.modules`` with a minimal
configuration and inputs.

A new module is covered by the registry-driven tests (``export``,
``compile``, dtypes, ...) by adding one entry to ``MODULE_CASES``.
"""

from __future__ import annotations

__all__ = ["MODULE_CASES", "ModuleCase"]

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch import nn

from sonnix import modules as sm

if TYPE_CHECKING:
    from collections.abc import Callable


@dataclass(frozen=True)
class ModuleCase:
    r"""Describe how to build a module and what to feed it.

    Attributes:
        name: The identifier of the case (used as the test id).
        build: A function that returns a new module.
        make_inputs: A function that returns the (seeded) positional
            inputs of the module in ``float32``.
        differentiable: ``True`` if the output is a floating point tensor
            that depends differentiably on the inputs or parameters.
    """

    name: str
    build: Callable[[], nn.Module]
    make_inputs: Callable[[], tuple[torch.Tensor, ...]]
    differentiable: bool = True

    def __repr__(self) -> str:
        return self.name


def _generator() -> torch.Generator:
    return torch.Generator().manual_seed(42)


def _positive(*shape: int) -> Callable[[], tuple[torch.Tensor, ...]]:
    def make() -> tuple[torch.Tensor, ...]:
        return (torch.rand(*shape, generator=_generator()) + 0.5,)

    return make


def _normal(*shape: int) -> Callable[[], tuple[torch.Tensor, ...]]:
    def make() -> tuple[torch.Tensor, ...]:
        return (torch.randn(*shape, generator=_generator()),)

    return make


def _pair_positive() -> tuple[torch.Tensor, ...]:
    generator = _generator()
    return (
        torch.rand(4, 3, generator=generator) + 0.5,
        torch.rand(4, 3, generator=generator) + 0.5,
    )


def _pair_normal() -> tuple[torch.Tensor, ...]:
    generator = _generator()
    return torch.randn(4, 3, generator=generator), torch.randn(4, 3, generator=generator)


def _pair_probability() -> tuple[torch.Tensor, ...]:
    generator = _generator()
    return (
        torch.rand(4, 3, generator=generator) * 0.8 + 0.1,
        torch.randint(0, 2, (4, 3), generator=generator).float(),
    )


def _pair_logits() -> tuple[torch.Tensor, ...]:
    generator = _generator()
    return (
        torch.randn(4, 3, generator=generator),
        torch.randint(0, 2, (4, 3), generator=generator).float(),
    )


def _triple_normal() -> tuple[torch.Tensor, ...]:
    generator = _generator()
    return tuple(torch.randn(4, 3, generator=generator) for _ in range(2))


def _case(
    cls: type[nn.Module],
    make_inputs: Callable[[], tuple[torch.Tensor, ...]],
    *args: object,
    differentiable: bool = True,
    **kwargs: object,
) -> ModuleCase:
    return ModuleCase(
        name=cls.__name__,
        build=lambda: cls(*args, **kwargs),
        make_inputs=make_inputs,
        differentiable=differentiable,
    )


_BINS = torch.tensor([[0.0, 1.0, 2.0, 3.0], [-1.0, 0.0, 1.0, 5.0], [0.0, 2.0, 4.0, 6.0]])

MODULE_CASES: tuple[ModuleCase, ...] = (
    # activations
    _case(sm.Asinh, _normal(4, 3)),
    _case(sm.Clamp, _normal(4, 3)),
    _case(sm.DynamicAsinh, _normal(4, 3), 3),
    _case(sm.DynamicTanh, _normal(4, 3), 3),
    _case(sm.Exp, _normal(4, 3)),
    _case(sm.ExpSin, _normal(4, 3)),
    _case(sm.Expm1, _normal(4, 3)),
    _case(sm.Gaussian, _normal(4, 3)),
    _case(sm.Laplacian, _normal(4, 3)),
    _case(sm.Log, _positive(4, 3)),
    _case(sm.Log1p, _positive(4, 3)),
    _case(sm.MultiQuadratic, _normal(4, 3)),
    _case(sm.Pow, _positive(4, 3), 2.0),
    _case(sm.Quadratic, _normal(4, 3)),
    _case(sm.ReLUn, _normal(4, 3)),
    _case(sm.RectifierAsinhUnit, _normal(4, 3)),
    _case(sm.SafeExp, _normal(4, 3)),
    _case(sm.SafeLog, _positive(4, 3)),
    _case(sm.Sin, _normal(4, 3)),
    _case(sm.Sinh, _normal(4, 3)),
    _case(sm.Snake, _normal(4, 3)),
    _case(sm.Square, _normal(4, 3)),
    _case(sm.SquaredReLU, _normal(4, 3)),
    # layers
    _case(sm.ExU, _normal(5, 3), 3, 4),
    _case(sm.NLinear, _normal(5, 2, 3), 2, 3, 4),
    _case(sm.ResidualBlock, _normal(4, 3), nn.Linear(3, 3)),
    _case(sm.ScaleAndShift, _normal(4, 3), 3),
    _case(sm.Squeeze, _normal(4, 1, 3), 1),
    _case(sm.View, _normal(4, 3), (2, 6)),
    # label and dtype helpers (the output is not differentiable)
    _case(sm.ToBinaryLabel, _normal(4, 3), differentiable=False),
    _case(sm.ToCategoricalLabel, _normal(4, 3), differentiable=False),
    _case(sm.ToDtype, _normal(4, 3), torch.float32, differentiable=False),
    _case(sm.ToFloat, _normal(4, 3), differentiable=False),
    _case(sm.ToLong, _normal(4, 3), differentiable=False),
    # fusion
    _case(sm.AverageFusion, _triple_normal),
    _case(sm.ConcatFusion, _triple_normal),
    _case(sm.MultiplicationFusion, _triple_normal),
    _case(sm.SumFusion, _triple_normal),
    # numerical encoders
    _case(sm.AsinhNumericalEncoder, _normal(4, 3), torch.ones(3)),
    _case(
        sm.CosSinNumericalEncoder,
        _normal(4, 3),
        torch.ones(3, 2),
        torch.zeros(3, 2),
    ),
    _case(
        sm.AsinhCosSinNumericalEncoder,
        _normal(4, 3),
        torch.ones(3, 2),
        torch.zeros(3, 2),
    ),
    _case(sm.PiecewiseLinearNumericalEncoder, _normal(4, 3), _BINS),
    # indicators
    _case(sm.ArithmeticalMeanIndicator, _pair_positive),
    _case(sm.ClassicalRelativeIndicator, _pair_positive),
    _case(sm.GeometricMeanIndicator, _pair_positive),
    _case(sm.MaximumMeanIndicator, _pair_positive),
    _case(sm.MinimumMeanIndicator, _pair_positive),
    _case(sm.MomentMeanIndicator, _pair_positive, 2),
    _case(sm.ReversedRelativeIndicator, _pair_positive),
    # losses
    _case(sm.AsinhMSELoss, _pair_normal),
    _case(sm.AsinhSmoothL1Loss, _pair_normal),
    _case(sm.BinaryFocalLoss, _pair_probability),
    _case(sm.BinaryFocalLossWithLogits, _pair_logits),
    _case(sm.BinaryPoly1Loss, _pair_probability),
    _case(sm.BinaryPoly1LossWithLogits, _pair_logits),
    _case(sm.GeneralRobustRegressionLoss, _pair_normal),
    _case(sm.LogCoshLoss, _pair_normal),
    _case(sm.MSLELoss, _pair_positive),
    _case(sm.PoissonRegressionLoss, _pair_positive),
    _case(sm.QuantileRegressionLoss, _pair_normal),
    _case(sm.RelativeLoss, _pair_positive, nn.MSELoss(reduction="none")),
    _case(sm.RelativeMSELoss, _pair_positive),
    _case(sm.RelativeSmoothL1Loss, _pair_positive),
    _case(sm.TransformedLoss, _pair_normal, nn.MSELoss(), sm.Asinh(), sm.Asinh()),
)
