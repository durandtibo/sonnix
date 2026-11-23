r"""Define a registry of the modules of ``sonnix.modules`` with a minimal
configuration and inputs.

A new module is covered by the registry-driven tests (``export``,
``compile``, dtypes, ...) by adding one entry to ``MODULE_CASES``.
"""

from __future__ import annotations

__all__ = [
    "ACCEPTS_NON_FINITE",
    "CALL_CASES",
    "EDGE_VALUES",
    "EXEMPT_FROM_SCALAR_CHECKS",
    "FUNCTION_CASES",
    "MODULE_CASES",
    "CallCase",
    "ModuleCase",
]

import inspect
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Any

import torch
from torch import nn

from sonnix import functional as sf, modules as sm, testing as st, utils as su
from tests.helpers import make_generator, normal_pair, positive_pair

if TYPE_CHECKING:
    from collections.abc import Callable

# The edge inputs of the numerical tests (property P6): the value of every floating input.
EDGE_VALUES: dict[str, float] = {
    "zeros": 0.0,
    "tiny": 1e-10,
    "large": 1e3,
    "negative_large": -1e3,
}


@dataclass(frozen=True)
class ModuleCase:
    r"""Describe how to build a module and what to feed it.

    Attributes:
        name: The identifier of the case (used as the test id).
        build: A function that returns a new module. The keyword
            arguments are passed to the constructor.
        make_inputs: A function that returns the (seeded) positional
            inputs of the module in ``float32``.
        differentiable: ``True`` if the output is a floating point tensor
            that depends differentiably on the inputs or parameters.
        other: A function that returns a second module of the same
            class with different values (it is used by the state
            round trip test). ``None`` if two builds already differ
            (random initialization) or if the module has no state.
        has_learnable_option: ``True`` if the constructor has the
            ``learnable`` option (a frozen tensor is a buffer with the
            name of the parameter).
        derived: The names of the tensors that are derived from other
            state. They must not be in the ``state_dict``.
        positional: The names of the constructor arguments that may be
            positional: the structural arguments (sizes, wrapped
            modules, tensors) and at most one hyperparameter, the
            defining one. The other arguments are keyword-only.
        fixed: The indices of the inputs that the edge tests keep as
            they are (the labels of a binary loss).
        wrong_inputs: A function that returns inputs with a wrong shape
            (property P4): the module must raise a ``ValueError`` that
            names the shapes or the argument. ``None`` if the module
            accepts any shape or has no shape of its own (the
            exceptions are listed with a reason in
            ``tests/integration/modules/test_wrong_inputs.py``).
        edges: The names of the edge inputs (keys of ``EDGE_VALUES``)
            that are valid for the module. A probability is not valid
            at ``1000``, for example.
    """

    name: str
    build: Callable[..., nn.Module]
    make_inputs: Callable[[], tuple[torch.Tensor, ...]]
    differentiable: bool = True
    other: Callable[..., nn.Module] | None = None
    has_learnable_option: bool = False
    derived: tuple[str, ...] = ()
    positional: tuple[str, ...] = ()
    fixed: tuple[int, ...] = ()
    wrong_inputs: Callable[[], tuple[torch.Tensor, ...]] | None = None
    edges: tuple[str, ...] = tuple(EDGE_VALUES)

    def __repr__(self) -> str:
        return self.name


def _positive(*shape: int) -> Callable[[], tuple[torch.Tensor, ...]]:
    def make() -> tuple[torch.Tensor, ...]:
        return (torch.rand(*shape, generator=make_generator()) + 0.5,)

    return make


def _normal(*shape: int) -> Callable[[], tuple[torch.Tensor, ...]]:
    def make() -> tuple[torch.Tensor, ...]:
        return (torch.randn(*shape, generator=make_generator()),)

    return make


def _pair_positive() -> tuple[torch.Tensor, ...]:
    return positive_pair(4, 3)


def _pair_normal() -> tuple[torch.Tensor, ...]:
    return normal_pair(4, 3)


def _pair_probability() -> tuple[torch.Tensor, ...]:
    generator = make_generator()
    return (
        torch.rand(4, 3, generator=generator) * 0.8 + 0.1,
        torch.randint(0, 2, (4, 3), generator=generator).float(),
    )


def _pair_logits() -> tuple[torch.Tensor, ...]:
    generator = make_generator()
    return (
        torch.randn(4, 3, generator=generator),
        torch.randint(0, 2, (4, 3), generator=generator).float(),
    )


def _wrong_pair() -> tuple[torch.Tensor, ...]:
    # Two tensors of the same role with different shapes (a size-1 dimension is not expanded).
    return torch.zeros(4, 3), torch.zeros(4, 1)


def _triple_normal() -> tuple[torch.Tensor, ...]:
    generator = make_generator()
    return tuple(torch.randn(4, 3, generator=generator) for _ in range(2))


# The constructor arguments that may be positional (property P3): the structural arguments
# (sizes, wrapped modules, tensors) and at most one hyperparameter, the defining one. The
# modules that are not listed have keyword-only arguments only. ``Clamp`` is the one
# exception with two hyperparameters, like ``torch.nn.Hardtanh(min_val, max_val)``.
_ALPHA_ACTIVATION = ("num_parameters", "init")
POSITIONAL: dict[str, tuple[str, ...]] = {
    "AsinhCosSinNumericalEncoder": ("frequency", "phase_shift"),
    "AsinhNumericalEncoder": ("scale",),
    "BinaryFocalLoss": ("alpha",),
    "BinaryFocalLossWithLogits": ("alpha",),
    "BinaryPoly1Loss": ("alpha",),
    "BinaryPoly1LossWithLogits": ("alpha",),
    "Clamp": ("min_val", "max_val"),
    "ConcatFusion": ("dim",),
    "CosSinNumericalEncoder": ("frequency", "phase_shift"),
    "DynamicAsinh": ("normalized_shape", "alpha_init_value"),
    "DynamicTanh": ("normalized_shape", "alpha_init_value"),
    "ExU": ("in_features", "out_features"),
    "ExpSin": _ALPHA_ACTIVATION,
    "Gaussian": _ALPHA_ACTIVATION,
    "GeneralRobustRegressionLoss": ("alpha",),
    "Laplacian": _ALPHA_ACTIVATION,
    "LogCoshLoss": ("scale",),
    "MultiQuadratic": _ALPHA_ACTIVATION,
    "NLinear": ("n", "in_features", "out_features"),
    "PiecewiseLinearNumericalEncoder": ("edges",),
    "PoissonRegressionLoss": ("eps",),
    "Pow": ("exponent",),
    "Quadratic": _ALPHA_ACTIVATION,
    "QuantileRegressionLoss": ("q",),
    "ReLUn": ("max_val",),
    "RelativeLoss": ("criterion",),
    "ResidualBlock": ("residual", "skip"),
    "SafeExp": ("max_val",),
    "SafeLog": ("min_val",),
    "SafePow": ("exponent",),
    "ScaleAndShift": ("normalized_shape",),
    "Snake": _ALPHA_ACTIVATION,
    "Squeeze": ("dim",),
    "ToBinaryLabel": ("threshold",),
    "ToCategoricalLabel": ("dim",),
    "ToDtype": ("dtype",),
    "TransformedLoss": ("criterion", "prediction_transform", "target_transform"),
}


def _builder(cls: type[nn.Module], *args: object, **kwargs: object) -> Callable[..., nn.Module]:
    # The positional arguments are bound to their names, so that a keyword argument of the call
    # replaces the base value of the same argument (``Pow(2.0)`` then ``build(exponent=nan)``).
    base = {**dict(zip(inspect.signature(cls).parameters, args, strict=False)), **kwargs}
    return lambda **extra: cls(**{**base, **extra})


def _case(
    cls: type[nn.Module],
    make_inputs: Callable[[], tuple[torch.Tensor, ...]],
    *args: object,
    differentiable: bool = True,
    other: Callable[..., nn.Module] | None = None,
    has_learnable_option: bool = False,
    derived: tuple[str, ...] = (),
    fixed: tuple[int, ...] = (),
    wrong_inputs: Callable[[], tuple[torch.Tensor, ...]] | None = None,
    edges: tuple[str, ...] = tuple(EDGE_VALUES),
    **kwargs: object,
) -> ModuleCase:
    return ModuleCase(
        name=cls.__name__,
        build=_builder(cls, *args, **kwargs),
        make_inputs=make_inputs,
        differentiable=differentiable,
        other=other,
        has_learnable_option=has_learnable_option,
        derived=derived,
        positional=POSITIONAL.get(cls.__name__, ()),
        fixed=fixed,
        wrong_inputs=wrong_inputs,
        edges=edges,
    )


def _lazy_case(
    cls: type[nn.Module],
    make_inputs: Callable[[], tuple[torch.Tensor, ...]],
    make_args: Callable[[], tuple[object, ...]],
    *,
    wrong_inputs: Callable[[], tuple[torch.Tensor, ...]] | None = None,
) -> ModuleCase:
    # The arguments are built for every module: a submodule shared between
    # two modules would be modified in place (for example by ``.to(dtype)``).
    return ModuleCase(
        name=cls.__name__,
        build=lambda **extra: cls(*make_args(), **extra),
        make_inputs=make_inputs,
        positional=POSITIONAL.get(cls.__name__, ()),
        wrong_inputs=wrong_inputs,
    )


def _piecewise_inputs() -> tuple[torch.Tensor, ...]:
    # The first column has values inside the step segment (between 1 and 2).
    return (torch.tensor([[1.5, 0.5, 3.0], [-3.0, 2.0, 5.0], [1.2, 1.5, 4.5], [0.0, 0.0, 0.0]]),)


# The first feature has a duplicated edge (a step segment).
_EDGES = torch.tensor([[0.0, 1.0, 1.0, 3.0], [-1.0, 0.0, 1.0, 5.0], [0.0, 2.0, 4.0, 6.0]])
_OTHER_EDGES = torch.tensor([[0.0, 1.0, 2.0, 3.0], [-2.0, 0.0, 2.0, 4.0], [1.0, 2.0, 3.0, 6.0]])

# A probability is between 0 and 1, so ``large`` and ``negative_large`` are not valid. At ``0``
# (and at ``tiny``, which is ``0`` in ``float16``) the ``binary_cross_entropy`` of ``torch``
# divides by ``0`` in ``float16`` and its gradient is ``nan``: it is documented as not guaranteed
# (the ``*_with_logits`` variants are tested instead).
_PROBABILITY_EDGES = ()

MODULE_CASES: tuple[ModuleCase, ...] = (
    # activations
    _case(sm.Asinh, _normal(4, 3)),
    _case(sm.Clamp, _normal(4, 3)),
    _case(
        sm.DynamicAsinh,
        _normal(4, 3),
        3,
        other=_builder(sm.DynamicAsinh, 3, alpha_init_value=2.0),
        wrong_inputs=_normal(4, 2),
    ),
    _case(
        sm.DynamicTanh,
        _normal(4, 3),
        3,
        other=_builder(sm.DynamicTanh, 3, alpha_init_value=2.0),
        wrong_inputs=_normal(4, 2),
    ),
    _case(sm.Exp, _normal(4, 3)),
    _case(
        sm.ExpSin,
        _normal(4, 3),
        other=_builder(sm.ExpSin, init=2.0),
        has_learnable_option=True,
    ),
    _case(sm.Expm1, _normal(4, 3)),
    _case(
        sm.Gaussian,
        _normal(4, 3),
        other=_builder(sm.Gaussian, init=2.0),
        has_learnable_option=True,
    ),
    _case(
        sm.Laplacian,
        _normal(4, 3),
        other=_builder(sm.Laplacian, init=2.0),
        has_learnable_option=True,
    ),
    _case(sm.Log, _positive(4, 3)),
    _case(sm.Log1p, _positive(4, 3)),
    _case(
        sm.MultiQuadratic,
        _normal(4, 3),
        other=_builder(sm.MultiQuadratic, init=2.0),
        has_learnable_option=True,
    ),
    _case(sm.Pow, _positive(4, 3), 2.0),
    _case(
        sm.Quadratic,
        _normal(4, 3),
        other=_builder(sm.Quadratic, init=2.0),
        has_learnable_option=True,
    ),
    _case(sm.ReLUn, _normal(4, 3)),
    _case(sm.RectifierAsinhUnit, _normal(4, 3)),
    _case(sm.SafeExp, _normal(4, 3)),
    _case(sm.SafeLog, _positive(4, 3)),
    _case(sm.SafePow, _positive(4, 3), 0.5),
    _case(sm.Sin, _normal(4, 3)),
    _case(sm.Sinh, _normal(4, 3)),
    _case(
        sm.Snake,
        _normal(4, 3),
        other=_builder(sm.Snake, init=2.0),
        has_learnable_option=True,
    ),
    _case(sm.Square, _normal(4, 3)),
    _case(sm.SquaredReLU, _normal(4, 3)),
    # layers
    _case(sm.ExU, _normal(5, 3), 3, 4),
    _case(sm.NLinear, _normal(5, 2, 3), 2, 3, 4, wrong_inputs=_normal(5, 3, 3)),
    _lazy_case(sm.ResidualBlock, _normal(4, 3), lambda: (nn.Linear(3, 3),)),
    _case(sm.ScaleAndShift, _normal(4, 3), 3, wrong_inputs=_normal(4, 2)),
    _case(sm.Squeeze, _normal(4, 1, 3), 1),
    # label and dtype helpers (the output is not differentiable)
    _case(sm.ToBinaryLabel, _normal(4, 3), differentiable=False),
    _case(sm.ToCategoricalLabel, _normal(4, 3), differentiable=False),
    _case(sm.ToDtype, _normal(4, 3), torch.float32, differentiable=False),
    _case(sm.ToFloat, _normal(4, 3), differentiable=False),
    _case(sm.ToLong, _normal(4, 3), differentiable=False),
    # fusion
    _case(sm.AverageFusion, _triple_normal, wrong_inputs=_wrong_pair),
    _case(sm.ConcatFusion, _triple_normal),
    _case(sm.MultiplicationFusion, _triple_normal, wrong_inputs=_wrong_pair),
    _case(sm.SumFusion, _triple_normal, wrong_inputs=_wrong_pair),
    # numerical encoders
    _case(
        sm.AsinhNumericalEncoder,
        _normal(4, 3),
        torch.ones(3, 2),
        other=_builder(sm.AsinhNumericalEncoder, torch.full((3, 2), 2.0)),
        has_learnable_option=True,
        wrong_inputs=_normal(4, 2),
    ),
    _case(
        sm.CosSinNumericalEncoder,
        _normal(4, 3),
        torch.ones(3, 2),
        torch.zeros(3, 2),
        other=_builder(sm.CosSinNumericalEncoder, torch.full((3, 2), 2.0), torch.ones(3, 2)),
        has_learnable_option=True,
        wrong_inputs=_normal(4, 2),
    ),
    _case(
        sm.AsinhCosSinNumericalEncoder,
        _normal(4, 3),
        torch.ones(3, 2),
        torch.zeros(3, 2),
        other=_builder(sm.AsinhCosSinNumericalEncoder, torch.full((3, 2), 2.0), torch.ones(3, 2)),
        has_learnable_option=True,
        wrong_inputs=_normal(4, 2),
    ),
    _case(
        sm.PiecewiseLinearNumericalEncoder,
        _piecewise_inputs,
        _EDGES,
        on_duplicate="ignore",
        other=_builder(sm.PiecewiseLinearNumericalEncoder, _OTHER_EDGES),
        derived=("start", "width", "is_step", "lower", "upper"),
        wrong_inputs=_normal(4, 2),
    ),
    # losses
    _case(
        sm.BinaryFocalLoss,
        _pair_probability,
        fixed=(1,),
        edges=_PROBABILITY_EDGES,
        wrong_inputs=_wrong_pair,
    ),
    _case(sm.BinaryFocalLossWithLogits, _pair_logits, fixed=(1,), wrong_inputs=_wrong_pair),
    _case(
        sm.BinaryPoly1Loss,
        _pair_probability,
        fixed=(1,),
        edges=_PROBABILITY_EDGES,
        wrong_inputs=_wrong_pair,
    ),
    _case(sm.BinaryPoly1LossWithLogits, _pair_logits, fixed=(1,), wrong_inputs=_wrong_pair),
    _case(sm.GeneralRobustRegressionLoss, _pair_normal, alpha=1.0, wrong_inputs=_wrong_pair),
    _case(sm.LogCoshLoss, _pair_normal, wrong_inputs=_wrong_pair),
    _case(sm.PoissonRegressionLoss, _pair_positive, wrong_inputs=_wrong_pair),
    _case(sm.QuantileRegressionLoss, _pair_normal, wrong_inputs=_wrong_pair),
    _lazy_case(
        sm.RelativeLoss,
        _pair_positive,
        lambda: (nn.L1Loss(reduction="none"),),
        wrong_inputs=_wrong_pair,
    ),
    _lazy_case(sm.TransformedLoss, _pair_normal, lambda: (nn.MSELoss(), sm.Asinh(), sm.Asinh())),
)


class _FunctionModule(nn.Module):
    r"""Wrap a function of ``sonnix.functional`` in a module, to run it with
    the tests of the registry."""

    def __init__(self, function: Callable[..., torch.Tensor]) -> None:
        super().__init__()
        self.function = function

    def forward(self, *inputs: torch.Tensor) -> torch.Tensor:
        return self.function(*inputs)


def _function_case(name: str, function: Callable[..., torch.Tensor]) -> ModuleCase:
    return ModuleCase(
        name=name,
        build=lambda: _FunctionModule(function),
        make_inputs=_pair_normal,
    )


# The functions of ``sonnix.functional`` without a module: the errors, the indicators and
# ``safe_pow``. They go through the numerical tests of the registry (property P6).
FUNCTION_CASES: tuple[ModuleCase, ...] = (
    _function_case("absolute_error", sf.absolute_error),
    _function_case("absolute_relative_error", sf.absolute_relative_error),
    _function_case("symmetric_absolute_relative_error", sf.symmetric_absolute_relative_error),
    _function_case("arithmetical_mean_indicator", sf.arithmetical_mean_indicator),
    _function_case("classical_relative_indicator", sf.classical_relative_indicator),
    _function_case("geometric_mean_indicator", sf.geometric_mean_indicator),
    _function_case("maximum_mean_indicator", sf.maximum_mean_indicator),
    _function_case("minimum_mean_indicator", sf.minimum_mean_indicator),
    _function_case("moment_mean_indicator", partial(sf.moment_mean_indicator, k=2)),
    _function_case("reversed_relative_indicator", sf.reversed_relative_indicator),
    _function_case("safe_pow", lambda x, y: sf.safe_pow(x, 2.0).add(sf.safe_pow(y, 0.5))),
)


@dataclass(frozen=True)
class CallCase:
    r"""Describe a valid call of a public function, class or class method.

    The test of the scalar arguments walks the signature of ``target`` and
    calls ``call`` with one invalid argument at a time (property P2).

    Attributes:
        name: The public name (``Class.method`` for a class method), used as the
            test id.
        target: The object whose signature lists the arguments: a function, a
            class (its ``__init__``) or a class method.
        call: A function that makes a valid call. The keyword arguments replace
            the argument of the same name.
    """

    name: str
    target: Callable[..., Any]
    call: Callable[..., Any]

    def __repr__(self) -> str:
        return self.name


def _two(function: Callable[..., Any]) -> Callable[..., Any]:
    # A loss or an indicator: ``function(prediction, target, **kwargs)``.
    return lambda **kwargs: function(torch.ones(2, 3), torch.ones(2, 3), **kwargs)


def _one(function: Callable[..., Any]) -> Callable[..., Any]:
    # A single-tensor operation: ``function(input, **kwargs)``.
    return lambda **kwargs: function(torch.ones(2, 3), **kwargs)


def _case_of(name: str, target: Callable[..., Any], call: Callable[..., Any]) -> CallCase:
    return CallCase(name=name, target=target, call=call)


def _register_tensor(**kwargs: object) -> None:
    su.register_tensor(nn.Module(), "alpha", torch.ones(2), **{"learnable": True, **kwargs})


def _load_state_dict_to_module(**kwargs: object) -> None:
    module = nn.Linear(2, 3)
    su.load_state_dict_to_module(module.state_dict(), module, **kwargs)


def _is_loss_decreasing(**kwargs: object) -> bool:
    module = nn.Linear(2, 1)
    return st.is_loss_decreasing(
        module,
        nn.MSELoss(),
        torch.optim.SGD(module.parameters(), lr=0.01),
        torch.ones(4, 2),
        torch.ones(4, 1),
        **kwargs,
    )


def _is_loss_decreasing_with(function: Callable[..., bool]) -> Callable[..., bool]:
    return lambda **kwargs: function(
        nn.Linear(2, 1), nn.MSELoss(), torch.ones(4, 2), torch.ones(4, 1), **kwargs
    )


_ENCODER_INPUT = torch.arange(6, dtype=torch.float).view(6, 1)

# The functions, the utilities, the helpers of ``testing`` and the ``create`` class methods that
# take a scalar argument, with a valid call. The modules are added from ``MODULE_CASES``.
_OTHER_CALL_CASES: tuple[CallCase, ...] = (
    _case_of(
        "absolute_relative_error", sf.absolute_relative_error, _two(sf.absolute_relative_error)
    ),
    _case_of("binary_focal_loss", sf.binary_focal_loss, _two(sf.binary_focal_loss)),
    _case_of(
        "binary_focal_loss_with_logits",
        sf.binary_focal_loss_with_logits,
        _two(sf.binary_focal_loss_with_logits),
    ),
    _case_of("binary_poly1_loss", sf.binary_poly1_loss, _two(sf.binary_poly1_loss)),
    _case_of(
        "binary_poly1_loss_with_logits",
        sf.binary_poly1_loss_with_logits,
        _two(sf.binary_poly1_loss_with_logits),
    ),
    _case_of(
        "general_robust_regression_loss",
        sf.general_robust_regression_loss,
        _two(sf.general_robust_regression_loss),
    ),
    _case_of("log_cosh_loss", sf.log_cosh_loss, _two(sf.log_cosh_loss)),
    _case_of("moment_mean_indicator", sf.moment_mean_indicator, _two(sf.moment_mean_indicator)),
    _case_of(
        "poisson_regression_loss", sf.poisson_regression_loss, _two(sf.poisson_regression_loss)
    ),
    _case_of(
        "quantile_regression_loss", sf.quantile_regression_loss, _two(sf.quantile_regression_loss)
    ),
    _case_of("relative_loss", sf.relative_loss, _two(sf.relative_loss)),
    _case_of("safe_exp", sf.safe_exp, _one(sf.safe_exp)),
    _case_of("safe_log", sf.safe_log, _one(sf.safe_log)),
    _case_of(
        "safe_pow",
        sf.safe_pow,
        lambda **kwargs: sf.safe_pow(torch.ones(2, 3), **{"exponent": 2.0, **kwargs}),
    ),
    _case_of(
        "symmetric_absolute_relative_error",
        sf.symmetric_absolute_relative_error,
        _two(sf.symmetric_absolute_relative_error),
    ),
    _case_of("register_tensor", su.register_tensor, _register_tensor),
    _case_of("load_state_dict_to_module", su.load_state_dict_to_module, _load_state_dict_to_module),
    _case_of("DummyDataset", st.DummyDataset, st.DummyDataset),
    _case_of("is_loss_decreasing", st.is_loss_decreasing, _is_loss_decreasing),
    _case_of(
        "is_loss_decreasing_with_adam",
        st.is_loss_decreasing_with_adam,
        _is_loss_decreasing_with(st.is_loss_decreasing_with_adam),
    ),
    _case_of(
        "is_loss_decreasing_with_sgd",
        st.is_loss_decreasing_with_sgd,
        _is_loss_decreasing_with(st.is_loss_decreasing_with_sgd),
    ),
    _case_of(
        "AsinhNumericalEncoder.create",
        sm.AsinhNumericalEncoder.create,
        lambda **kwargs: sm.AsinhNumericalEncoder.create(
            **{"spacing": "linspace", "feature_size": 3, "scale_range": (1.0, 2.0), **kwargs}
        ),
    ),
    *[
        _case_of(
            f"{cls.__name__}.create",
            cls.create,
            lambda cls=cls, **kwargs: cls.create(
                **{
                    "spacing": "linspace",
                    "n_frequencies": 3,
                    "frequency_range": (1.0, 2.0),
                    **kwargs,
                }
            ),
        )
        for cls in (sm.CosSinNumericalEncoder, sm.AsinhCosSinNumericalEncoder)
    ],
    _case_of(
        "PiecewiseLinearNumericalEncoder.from_quantiles",
        sm.PiecewiseLinearNumericalEncoder.from_quantiles,
        lambda **kwargs: sm.PiecewiseLinearNumericalEncoder.from_quantiles(
            **{"x": _ENCODER_INPUT, "n_bins": 2, **kwargs}
        ),
    ),
)

# Every public module of the registry, built with its valid arguments.
CALL_CASES: tuple[CallCase, ...] = (
    *[_case_of(case.name, getattr(sm, case.name), case.build) for case in MODULE_CASES],
    *_OTHER_CALL_CASES,
)

# The ``float`` arguments that accept a non-finite value, with the reason. The key is
# ``<name>.<argument>``.
ACCEPTS_NON_FINITE: dict[str, str] = {
    "general_robust_regression_loss.alpha": "-inf is the Welsch loss",
    "GeneralRobustRegressionLoss.alpha": "-inf is the Welsch loss",
}

# The public names with a scalar argument that the test of the scalar arguments does not call,
# with the reason.
EXEMPT_FROM_SCALAR_CHECKS: dict[str, str] = {
    "BaseAlphaActivation": "abstract; its subclasses are called",
    "BaseDynamicNorm": "abstract; its subclasses are called",
    "validate_focal_params": "a validator: it is the check",
    "validate_general_robust_params": "a validator: it is the check",
    "validate_positive": "a validator: it is the check",
    "resolve_eps": "the caller validated the argument (documented precondition)",
    "resolve_max_exponent": "the caller validated the argument (documented precondition)",
}
