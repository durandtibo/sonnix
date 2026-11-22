r"""Test the positional arguments of the modules (property P3).

``ModuleCase.positional`` lists the constructor arguments that may be
positional: the structural arguments and at most one hyperparameter.
A new module without an entry in the registry fails
``test_registry_covers_every_module``. The rule "at most one hyperparameter" is
reviewed in ``tests.registry.POSITIONAL``.
"""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING

import pytest

from sonnix.modules import (
    AsinhCosSinNumericalEncoder,
    AsinhNumericalEncoder,
    CosSinNumericalEncoder,
    PiecewiseLinearNumericalEncoder,
)
from tests.registry import MODULE_CASES, ModuleCase

if TYPE_CHECKING:
    from collections.abc import Callable


def _positional_parameters(callable_: Callable) -> tuple[str, ...]:
    return tuple(
        name
        for name, parameter in inspect.signature(callable_).parameters.items()
        if parameter.kind
        in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    )


#############################################
#     Tests for the module constructors     #
#############################################


@pytest.mark.parametrize("case", MODULE_CASES, ids=repr)
def test_positional_parameters_of_the_constructor(case: ModuleCase) -> None:
    assert _positional_parameters(type(case.build()).__init__)[1:] == case.positional


########################################
#     Tests for the create methods     #
########################################


@pytest.mark.parametrize(
    ("callable_", "expected"),
    [
        (AsinhNumericalEncoder.create, ("spacing", "feature_size")),
        (CosSinNumericalEncoder.create, ("spacing", "num_frequencies")),
        (AsinhCosSinNumericalEncoder.create, ("spacing", "num_frequencies")),
        (PiecewiseLinearNumericalEncoder.from_quantiles, ("x", "n_bins")),
    ],
)
def test_positional_parameters_of_the_create_methods(
    callable_: Callable, expected: tuple[str, ...]
) -> None:
    assert _positional_parameters(callable_) == expected
