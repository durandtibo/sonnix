r"""Test that every public function and module rejects an argument that
is out of its range, with the same exception type and message for a
function and its module (property P2).

The type and the finiteness of the scalar arguments are tested for every
public object by ``test_scalar_arguments.py``, which walks the
signatures. This file lists what the signature cannot tell: the range of
an argument (``gamma >= 0``), the arguments that are not scalars and the
exceptions of the generic rule.

``INVALID_ARGUMENTS`` maps a function and its module to the invalid
keyword arguments and the expected exception. ``MODULE_ARGUMENTS`` lists
the invalid arguments of the objects that have no function.
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


def _range(name: str, *, invalid: tuple[Any, ...], match: str) -> list[tuple[dict, type, str]]:
    r"""Return the rows of an argument that is out of its range: the
    values of ``invalid`` raise a ``ValueError``."""
    return [({name: value}, ValueError, match) for value in invalid]


# (id, function call, module call, [(invalid keyword arguments, exception, message)])
INVALID_ARGUMENTS = [
    (
        "binary_focal_loss",
        _loss(sf.binary_focal_loss),
        sm.BinaryFocalLoss,
        [
            *_range("alpha", invalid=(-0.1, 2.0), match=r"alpha must be in the range"),
            ({"gamma": -1.0}, ValueError, r"gamma must be greater than or equal to 0"),
            ({"reduction": "bad"}, ValueError, r"reduction must be one of"),
        ],
    ),
    (
        "binary_focal_loss_with_logits",
        _loss(sf.binary_focal_loss_with_logits),
        sm.BinaryFocalLossWithLogits,
        [
            *_range("alpha", invalid=(2.0,), match=r"alpha must be in the range"),
            ({"gamma": -1.0}, ValueError, r"gamma must be greater than or equal to 0"),
        ],
    ),
    (
        "binary_poly1_loss",
        _loss(sf.binary_poly1_loss),
        sm.BinaryPoly1Loss,
        _range("alpha", invalid=(-INF,), match=r"alpha must be finite"),
    ),
    (
        "binary_poly1_loss_with_logits",
        _loss(sf.binary_poly1_loss_with_logits),
        sm.BinaryPoly1LossWithLogits,
        _range("alpha", invalid=(-INF,), match=r"alpha must be finite"),
    ),
    (
        "general_robust_regression_loss",
        _loss(sf.general_robust_regression_loss),
        sm.GeneralRobustRegressionLoss,
        [
            # ``alpha`` accepts ``-inf`` (the Welsch loss), so the generic rule skips it.
            *_range("alpha", invalid=(NAN, INF), match=r"alpha cannot be NaN or \+inf"),
            *_range("scale", invalid=(0.0, -1.0), match=r"scale must be"),
            *_range("max_val", invalid=(0.0, -1.0), match=r"max_val must be"),
        ],
    ),
    (
        "log_cosh_loss",
        _loss(sf.log_cosh_loss),
        sm.LogCoshLoss,
        [
            *_range("scale", invalid=(0.0, -1.0), match=r"scale must be"),
            ({"reduction": "bad"}, ValueError, r"reduction must be one of"),
        ],
    ),
    (
        "poisson_regression_loss",
        _loss(sf.poisson_regression_loss),
        sm.PoissonRegressionLoss,
        _range("eps", invalid=(0.0, -1.0), match=r"eps must be"),
    ),
    (
        "quantile_regression_loss",
        _loss(sf.quantile_regression_loss),
        sm.QuantileRegressionLoss,
        _range("q", invalid=(-0.1, 2.0), match=r"q must be in the range"),
    ),
    (
        "relative_loss",
        _loss(sf.relative_loss),
        partial(sm.RelativeLoss, nn.MSELoss(reduction="none")),
        _range("eps", invalid=(0.0, -1.0), match=r"eps must be"),
    ),
    (
        "safe_log",
        _op(sf.safe_log),
        sm.SafeLog,
        _range("min_val", invalid=(0.0, -1.0), match=r"min_val must be"),
    ),
    (
        "safe_pow",
        _op(sf.safe_pow),
        sm.SafePow,
        _range("exponent", invalid=(-1.0,), match=r"exponent must be"),
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

ENCODER_TENSOR = torch.tensor([[1.0, 2.0]])
CREATE_INPUT = torch.arange(6, dtype=torch.float).view(6, 1)

# (id, build, invalid keyword arguments, exception, message)
MODULE_ARGUMENTS = [
    *[
        (f"ToDtype-{value!r}", sm.ToDtype, {"dtype": value}, exception, match)
        for value, exception, match in (
            ("float33", ValueError, r"dtype must be one of"),
            ("Tensor", ValueError, r"dtype must be one of"),
            (1, TypeError, r"dtype must be an instance of"),
            (float, TypeError, r"dtype must be an instance of"),
        )
    ],
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
            (sm.PiecewiseLinearNumericalEncoder, "edges"),
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
    (
        "Asinh-feature_size-0",
        partial(sm.AsinhNumericalEncoder.create, "linspace", scale_range=(1.0, 2.0)),
        {"feature_size": 0},
        ValueError,
        r"feature_size must be greater than or equal to 1",
    ),
    (
        "CosSin-n_frequencies-0",
        partial(sm.CosSinNumericalEncoder.create, "linspace", frequency_range=(1.0, 2.0)),
        {"n_frequencies": 0},
        ValueError,
        r"n_frequencies must be greater than or equal to 1",
    ),
    (
        "CosSin-inf-range",
        partial(sm.CosSinNumericalEncoder.create, "linspace", 3),
        {"frequency_range": (1.0, INF)},
        ValueError,
        r"max_frequency must be finite",
    ),
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
