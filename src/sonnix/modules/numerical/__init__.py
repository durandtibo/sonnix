r"""Contain module to encode or decode numerical values."""

from __future__ import annotations

__all__ = [
    "AsinhCosSinNumericalEncoder",
    "AsinhNumericalEncoder",
    "CosSinNumericalEncoder",
    "PiecewiseLinearNumericalEncoder",
]

from sonnix.modules.numerical.asinh import AsinhNumericalEncoder
from sonnix.modules.numerical.piecewise import PiecewiseLinearNumericalEncoder
from sonnix.modules.numerical.sine import (
    AsinhCosSinNumericalEncoder,
    CosSinNumericalEncoder,
)
