r"""Test that the options of the public functions are keyword-only
(property P3)."""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING

import pytest
import torch

import sonnix.functional as sf

if TYPE_CHECKING:
    from collections.abc import Callable

TENSOR_PARAMETERS = {"prediction", "target", "loss", "indicator", "input"}

# ``exponent`` and ``reduction`` are required operands, not options.
EXCEPTIONS = {"Reduction", "reduce_loss", "safe_pow"}

FUNCTIONS = [
    getattr(sf, name) for name in sf.__all__ if name not in EXCEPTIONS and name != "Reduction"
]


def _positional_parameters(function: Callable) -> set[str]:
    return {
        name
        for name, parameter in inspect.signature(function).parameters.items()
        if parameter.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    }


def test_functions_are_found() -> None:
    assert len(FUNCTIONS) >= 20


@pytest.mark.parametrize("function", FUNCTIONS, ids=lambda f: f.__name__)
def test_only_tensors_are_positional(function: Callable) -> None:
    assert _positional_parameters(function) <= TENSOR_PARAMETERS


def test_log_cosh_loss_scale_is_not_positional() -> None:
    with pytest.raises(TypeError, match=r"positional argument"):
        sf.log_cosh_loss(torch.ones(2), torch.ones(2), 2.0)
