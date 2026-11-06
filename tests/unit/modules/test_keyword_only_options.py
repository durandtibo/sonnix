from __future__ import annotations

import inspect
from typing import TYPE_CHECKING

import pytest

from sonnix.functional import safe_exp, safe_log
from sonnix.modules import (
    AsinhCosSinNumericalEncoder,
    AsinhNumericalEncoder,
    CosSinNumericalEncoder,
    DynamicAsinh,
    DynamicTanh,
    ExpSin,
    ExU,
    Gaussian,
    NLinear,
    SafeExp,
    SafeLog,
    ScaleAndShift,
)

if TYPE_CHECKING:
    from collections.abc import Callable

##################################################
#     Tests for the signature of the options     #
##################################################

CREATE_METHODS = [
    f"create_{spacing}_{kind}"
    for spacing in ("rand", "linspace", "logspace")
    for kind in ("frequency", "value_range")
]


@pytest.mark.parametrize(
    ("callable_", "name"),
    [
        (safe_exp, "preserve_gradient"),
        (safe_log, "preserve_gradient"),
        (SafeExp, "preserve_gradient"),
        (SafeLog, "preserve_gradient"),
        (NLinear, "bias"),
        (NLinear, "device"),
        (NLinear, "dtype"),
        (ExU, "bias"),
        (ExU, "device"),
        (ExU, "dtype"),
        (ScaleAndShift, "device"),
        (ScaleAndShift, "dtype"),
        (DynamicAsinh, "device"),
        (DynamicTanh, "dtype"),
        (ExpSin, "learnable"),
        (ExpSin, "device"),
        (Gaussian, "dtype"),
        (AsinhNumericalEncoder, "learnable"),
        (CosSinNumericalEncoder, "learnable"),
        (AsinhCosSinNumericalEncoder, "learnable"),
        *[(getattr(CosSinNumericalEncoder, name), "learnable") for name in CREATE_METHODS],
        *[(getattr(AsinhCosSinNumericalEncoder, name), "learnable") for name in CREATE_METHODS],
    ],
)
def test_option_is_keyword_only(callable_: Callable, name: str) -> None:
    assert inspect.signature(callable_).parameters[name].kind is inspect.Parameter.KEYWORD_ONLY
