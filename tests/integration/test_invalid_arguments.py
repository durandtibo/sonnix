r"""Test that every public function and module rejects an invalid
argument, with the same exception type and message for a function and
its module (properties P2 and P6).

``INVALID_ARGUMENTS`` maps a function and its module to the invalid
keyword arguments and the expected exception. ``MODULE_ARGUMENTS``
lists the invalid arguments of the objects that have no function.
"""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, Any

import pytest
import torch
from torch import nn

import sonnix.functional as sf
import sonnix.modules as sm

if TYPE_CHECKING:
    from collections.abc import Callable

NAN = float("nan")
INF = float("inf")


def _loss(function: Callable[..., Any]) -> Callable[..., Any]:
    return lambda **kwargs: function(torch.ones(2, 3), torch.ones(2, 3), **kwargs)


def _op(function: Callable[..., Any]) -> Callable[..., Any]:
    return lambda **kwargs: function(torch.ones(2, 3), **kwargs)


def _number(name: str, *, invalid: tuple[Any, ...], match: str) -> list[tuple[dict, type, str]]:
    r"""Return the rows of a number argument: the values of ``invalid``
    raise a ``ValueError`` and a string raises a ``TypeError``."""
    return [
        *[({name: value}, ValueError, match) for value in invalid],
        ({name: "a"}, TypeError, rf"{name} must be a real number"),
    ]


def _flag(name: str) -> list[tuple[dict, type, str]]:
    return [({name: 1}, TypeError, rf"{name} must be an instance of bool")]


NOT_FINITE = (NAN, INF)

# (id, function call, module call, [(invalid keyword arguments, exception, message)])
INVALID_ARGUMENTS = [
    (
        "binary_focal_loss",
        _loss(sf.binary_focal_loss),
        sm.BinaryFocalLoss,
        [
            *_number(
                "alpha", invalid=(*NOT_FINITE, -0.1, 2.0), match=r"alpha must be in the range"
            ),
            *_number("gamma", invalid=NOT_FINITE, match=r"gamma must be finite"),
            ({"gamma": -1.0}, ValueError, r"gamma must be greater than or equal to 0"),
            ({"reduction": "bad"}, ValueError, r"reduction must be one of"),
        ],
    ),
    (
        "binary_focal_loss_with_logits",
        _loss(sf.binary_focal_loss_with_logits),
        sm.BinaryFocalLossWithLogits,
        [
            *_number("alpha", invalid=(*NOT_FINITE, 2.0), match=r"alpha must be in the range"),
            *_number("gamma", invalid=NOT_FINITE, match=r"gamma must be finite"),
            ({"gamma": -1.0}, ValueError, r"gamma must be greater than or equal to 0"),
        ],
    ),
    (
        "binary_poly1_loss",
        _loss(sf.binary_poly1_loss),
        sm.BinaryPoly1Loss,
        _number("alpha", invalid=(*NOT_FINITE, -INF), match=r"alpha must be finite"),
    ),
    (
        "binary_poly1_loss_with_logits",
        _loss(sf.binary_poly1_loss_with_logits),
        sm.BinaryPoly1LossWithLogits,
        _number("alpha", invalid=(*NOT_FINITE, -INF), match=r"alpha must be finite"),
    ),
    (
        "general_robust_regression_loss",
        _loss(sf.general_robust_regression_loss),
        sm.GeneralRobustRegressionLoss,
        [
            *_number("alpha", invalid=(NAN, INF), match=r"alpha cannot be NaN or \+inf"),
            *_number("scale", invalid=(*NOT_FINITE, 0.0, -1.0), match=r"scale must be"),
            *_number("max_val", invalid=(*NOT_FINITE, 0.0, -1.0), match=r"max_val must be"),
        ],
    ),
    (
        "log_cosh_loss",
        _loss(sf.log_cosh_loss),
        sm.LogCoshLoss,
        [
            *_number("scale", invalid=(*NOT_FINITE, 0.0, -1.0), match=r"scale must be"),
            ({"reduction": "bad"}, ValueError, r"reduction must be one of"),
        ],
    ),
    (
        "poisson_regression_loss",
        _loss(sf.poisson_regression_loss),
        sm.PoissonRegressionLoss,
        [
            *_number("eps", invalid=(*NOT_FINITE, 0.0, -1.0), match=r"eps must be"),
            *_flag("log_input"),
        ],
    ),
    (
        "quantile_regression_loss",
        _loss(sf.quantile_regression_loss),
        sm.QuantileRegressionLoss,
        _number("q", invalid=(*NOT_FINITE, -0.1, 2.0), match=r"q must be in the range"),
    ),
    (
        "relative_loss",
        _loss(sf.relative_loss),
        partial(sm.RelativeLoss, nn.MSELoss(reduction="none")),
        [
            *_number("eps", invalid=(*NOT_FINITE, 0.0, -1.0), match=r"eps must be"),
        ],
    ),
    (
        "safe_exp",
        _op(sf.safe_exp),
        sm.SafeExp,
        [
            *_number("max_val", invalid=NOT_FINITE, match=r"max_val must be finite"),
            *_flag("preserve_gradient"),
        ],
    ),
    (
        "safe_log",
        _op(sf.safe_log),
        sm.SafeLog,
        [
            *_number("min_val", invalid=(*NOT_FINITE, 0.0, -1.0), match=r"min_val must be"),
            *_flag("preserve_gradient"),
        ],
    ),
]

FUNCTION_ROWS = [
    pytest.param(call, kwargs, exception, match, id=f"{name}-{next(iter(kwargs))}-{kwargs!r}")
    for name, call, _, rows in INVALID_ARGUMENTS
    for kwargs, exception, match in rows
]
MODULE_ROWS = [
    pytest.param(build, kwargs, exception, match, id=f"{name}-{next(iter(kwargs))}-{kwargs!r}")
    for name, _, build, rows in INVALID_ARGUMENTS
    for kwargs, exception, match in rows
]
PARITY_ROWS = [
    pytest.param(call, build, kwargs, id=f"{name}-{next(iter(kwargs))}-{kwargs!r}")
    for name, call, build, rows in INVALID_ARGUMENTS
    for kwargs, _, _ in rows
]

#########################################################
#     Tests that the functions and modules validate     #
#########################################################


@pytest.mark.parametrize(("call", "kwargs", "exception", "match"), FUNCTION_ROWS)
def test_function_raises(
    call: Callable[..., Any], kwargs: dict, exception: type[Exception], match: str
) -> None:
    with pytest.raises(exception, match=match):
        call(**kwargs)


@pytest.mark.parametrize(("build", "kwargs", "exception", "match"), MODULE_ROWS)
def test_module_raises(
    build: Callable[..., Any], kwargs: dict, exception: type[Exception], match: str
) -> None:
    with pytest.raises(exception, match=match):
        build(**kwargs)


@pytest.mark.parametrize(("call", "build", "kwargs"), PARITY_ROWS)
def test_function_and_module_raise_the_same_error(
    call: Callable[..., Any], build: Callable[..., Any], kwargs: dict
) -> None:
    with pytest.raises((TypeError, ValueError), match=r".+") as function_info:
        call(**kwargs)
    with pytest.raises((TypeError, ValueError), match=r".+") as module_info:
        build(**kwargs)
    assert (function_info.type, str(function_info.value)) == (
        module_info.type,
        str(module_info.value),
    )


####################################
#     Tests for the modules only   #
####################################

_RESOLVED = [
    (INF, ValueError, r"must be finite"),
    (NAN, ValueError, r"must be finite"),
    ("a", TypeError, r"must be a real number"),
]
ALPHA_ACTIVATIONS = [
    sm.ExpSin,
    sm.Gaussian,
    sm.Laplacian,
    sm.MultiQuadratic,
    sm.Quadratic,
    sm.Snake,
]
ENCODER_TENSOR = torch.tensor([[1.0, 2.0]])
CREATE_INPUT = torch.arange(6, dtype=torch.float).view(6, 1)

# (id, build, invalid keyword arguments, exception, message)
MODULE_ARGUMENTS = [
    *[
        (f"{cls.__name__}-learnable", cls, {"learnable": 1}, TypeError, r"learnable must be")
        for cls in ALPHA_ACTIVATIONS
    ],
    *[
        (f"{cls.__name__}-init-{value}", cls, {"init": value}, exception, match)
        for cls in ALPHA_ACTIVATIONS
        for value, exception, match in _RESOLVED
    ],
    *[
        (f"ConcatFusion-dim-{value!r}", sm.ConcatFusion, {"dim": value}, TypeError, r"dim must be")
        for value in (1.5, True)
    ],
    *[
        (
            f"ToCategoricalLabel-dim-{value!r}",
            sm.ToCategoricalLabel,
            {"dim": value},
            TypeError,
            r"dim must be",
        )
        for value in (1.5, True)
    ],
    *[
        (f"ToBinaryLabel-{value}", sm.ToBinaryLabel, {"threshold": value}, exception, match)
        for value, exception, match in _RESOLVED
    ],
    *[
        (f"Clamp-{name}-{value}", sm.Clamp, {name: value}, exception, match)
        for name in ("min_val", "max_val")
        for value, exception, match in _RESOLVED
    ],
    ("Clamp-both-nan", sm.Clamp, {"min_val": NAN, "max_val": NAN}, ValueError, r"min_val must be"),
    *[
        (f"ReLUn-{value}", sm.ReLUn, {"max_val": value}, exception, match)
        for value, exception, match in _RESOLVED
    ],
    *[
        (f"Pow-{value}", sm.Pow, {"exponent": value}, exception, match)
        for value, exception, match in _RESOLVED
    ],
    *[
        (f"{cls.__name__}-{value}", partial(cls, 5), {"alpha_init_value": value}, exception, match)
        for cls in (sm.DynamicAsinh, sm.DynamicTanh)
        for value, exception, match in _RESOLVED
    ],
    *[
        (f"{cls.__name__}-bias", partial(cls, 5), {"bias": 1}, TypeError, r"bias must be")
        for cls in (sm.DynamicAsinh, sm.DynamicTanh, sm.ScaleAndShift)
    ],
    ("NLinear-bias", partial(sm.NLinear, 2, 3, 4), {"bias": 1}, TypeError, r"bias must be"),
    ("ExU-bias", partial(sm.ExU, 3, 4), {"bias": 1}, TypeError, r"bias must be"),
    *[
        (
            f"{cls.__name__}-{value}",
            partial(cls, **{name: ENCODER_TENSOR}),
            {name: torch.tensor([[value, 1.0]])},
            ValueError,
            rf"'{name}' must contain only finite values",
        )
        for cls, name in (
            (sm.AsinhNumericalEncoder, "scale"),
            (sm.PiecewiseLinearNumericalEncoder, "bins"),
        )
        for value in (NAN, INF)
    ],
    *[
        (
            f"{cls.__name__}-{name}-{value}",
            partial(cls, frequency=ENCODER_TENSOR, phase_shift=ENCODER_TENSOR),
            {name: torch.tensor([[value, 1.0]])},
            ValueError,
            rf"'{name}' must contain only finite values",
        )
        for cls in (sm.CosSinNumericalEncoder, sm.AsinhCosSinNumericalEncoder)
        for name in ("frequency", "phase_shift")
        for value in (NAN, INF)
    ],
    *[
        (
            f"{cls.__name__}-learnable",
            partial(cls, **kwargs),
            {"learnable": 1},
            TypeError,
            r"learnable must be",
        )
        for cls, kwargs in (
            (sm.AsinhNumericalEncoder, {"scale": ENCODER_TENSOR}),
            (
                sm.CosSinNumericalEncoder,
                {"frequency": ENCODER_TENSOR, "phase_shift": ENCODER_TENSOR},
            ),
            (
                sm.AsinhCosSinNumericalEncoder,
                {"frequency": ENCODER_TENSOR, "phase_shift": ENCODER_TENSOR},
            ),
        )
    ],
]


@pytest.mark.parametrize(
    ("build", "kwargs", "exception", "match"),
    [pytest.param(*row[1:], id=row[0]) for row in MODULE_ARGUMENTS],
)
def test_module_only_raises(
    build: Callable[..., Any], kwargs: dict, exception: type[Exception], match: str
) -> None:
    with pytest.raises(exception, match=match):
        build(**kwargs)


########################################
#     Tests for the create() methods   #
########################################

CREATE_ARGUMENTS = [
    *[
        (
            f"Asinh-feature_size-{value!r}",
            partial(sm.AsinhNumericalEncoder.create, "linspace", scale_range=(1.0, 2.0)),
            {"feature_size": value},
            TypeError,
            r"feature_size must be an int",
        )
        for value in (2.5, True)
    ],
    (
        "Asinh-feature_size-0",
        partial(sm.AsinhNumericalEncoder.create, "linspace", scale_range=(1.0, 2.0)),
        {"feature_size": 0},
        ValueError,
        r"feature_size must be greater than or equal to 1",
    ),
    *[
        (
            f"{cls.__name__}-num_frequencies-{value!r}",
            partial(cls.create, "linspace", frequency_range=(1.0, 2.0)),
            {"num_frequencies": value},
            TypeError,
            r"num_frequencies must be an int",
        )
        for cls in (sm.CosSinNumericalEncoder, sm.AsinhCosSinNumericalEncoder)
        for value in (2.5, True)
    ],
    (
        "CosSin-num_frequencies-0",
        partial(sm.CosSinNumericalEncoder.create, "linspace", frequency_range=(1.0, 2.0)),
        {"num_frequencies": 0},
        ValueError,
        r"num_frequencies must be greater than or equal to 1",
    ),
    (
        "CosSin-n_features-2.5",
        partial(sm.CosSinNumericalEncoder.create, "linspace", 3, frequency_range=(1.0, 2.0)),
        {"n_features": 2.5},
        TypeError,
        r"n_features must be an int",
    ),
    (
        "CosSin-inf-range",
        partial(sm.CosSinNumericalEncoder.create, "linspace", 3),
        {"frequency_range": (1.0, INF)},
        ValueError,
        r"max_frequency must be finite",
    ),
    *[
        (
            f"Piecewise-n_bins-{value!r}",
            partial(sm.PiecewiseLinearNumericalEncoder.from_quantiles, CREATE_INPUT),
            {"n_bins": value},
            TypeError,
            r"n_bins must be an int",
        )
        for value in (2.5, True)
    ],
]


@pytest.mark.parametrize(
    ("build", "kwargs", "exception", "match"),
    [pytest.param(*row[1:], id=row[0]) for row in CREATE_ARGUMENTS],
)
def test_create_raises(
    build: Callable[..., Any], kwargs: dict, exception: type[Exception], match: str
) -> None:
    with pytest.raises(exception, match=match):
        build(**kwargs)
