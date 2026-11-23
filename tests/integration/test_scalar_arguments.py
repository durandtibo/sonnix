r"""Test that every scalar argument of the public API is validated (property
P2).

The test walks the signature of each call of ``CALL_CASES``: an ``int``
rejects ``2.5``, ``True`` and ``"a"``, a ``float`` rejects ``"a"``, ``nan``
and ``inf``, and a ``bool`` rejects ``1``. A scalar argument that is not
validated fails at once, so a new option cannot escape the property. The
range of each argument (``gamma >= 0``) is tested in
``test_invalid_arguments.py``.
"""

from __future__ import annotations

import importlib
import inspect
import math

import pytest

from tests.helpers import get_scalar_parameters
from tests.registry import (
    ACCEPTS_NON_FINITE,
    CALL_CASES,
    EXEMPT_FROM_SCALAR_CHECKS,
    CallCase,
)

NAN = float("nan")
INF = float("inf")

# The invalid values of each type: (value, exception).
INVALID_VALUES = {
    "int": [(2.5, TypeError), (True, TypeError), ("a", TypeError)],
    "float": [("a", TypeError), (NAN, ValueError), (INF, ValueError)],
    "bool": [(1, TypeError)],
}
PACKAGES = ["functional", "modules", "validation", "factory", "utils", "testing"]


def _is_accepted(case: CallCase, name: str, value: object) -> bool:
    # A float argument that is documented to accept a non-finite value.
    return (
        isinstance(value, float)
        and not math.isfinite(value)
        and f"{case.name}.{name}" in ACCEPTS_NON_FINITE
    )


ROWS = [
    pytest.param(case, name, value, exception, id=f"{case.name}-{name}-{value!r}")
    for case in CALL_CASES
    for name, kind in get_scalar_parameters(case.target).items()
    for value, exception in INVALID_VALUES[kind]
    if not _is_accepted(case, name, value)
]


@pytest.mark.parametrize(("case", "name", "value", "exception"), ROWS)
def test_invalid_scalar_argument(
    case: CallCase, name: str, value: object, exception: type[Exception]
) -> None:
    with pytest.raises(exception, match=rf"\b{name}\b"):
        case.call(**{name: value})


def _get_pascal_case(name: str) -> str:
    return "".join(word.title() for word in name.split("_"))


CASES = {case.name: case for case in CALL_CASES}
# A function and its module (``safe_exp`` and ``SafeExp``) raise the same error.
PARITY_ROWS = [
    pytest.param(function, CASES[_get_pascal_case(function.name)], name, value, id=row.id)
    for function in CALL_CASES
    for row in ROWS
    if row.values[0] is function
    and _get_pascal_case(function.name) in CASES
    and row.values[1] in get_scalar_parameters(CASES[_get_pascal_case(function.name)].target)
    for name, value in [row.values[1:3]]
]


def _get_error(case: CallCase, name: str, value: object) -> tuple[type, str]:
    with pytest.raises((TypeError, ValueError), match=r".+") as info:
        case.call(**{name: value})
    return info.type, str(info.value)


@pytest.mark.parametrize(("function", "module", "name", "value"), PARITY_ROWS)
def test_function_and_module_raise_the_same_error(
    function: CallCase, module: CallCase, name: str, value: object
) -> None:
    assert _get_error(function, name, value) == _get_error(module, name, value)


def test_parity_rows_are_not_empty() -> None:
    assert len(PARITY_ROWS) > 0


def test_every_scalar_argument_is_reached() -> None:
    # A public name with a scalar argument has a case, or an exemption with a reason.
    covered = {case.name for case in CALL_CASES} | set(EXEMPT_FROM_SCALAR_CHECKS)
    assert sorted(set(_get_public_targets()) - covered) == []


def test_exemptions_are_public_names_with_a_scalar_argument() -> None:
    assert sorted(set(EXEMPT_FROM_SCALAR_CHECKS) - set(_get_public_targets())) == []


def test_non_finite_exceptions_are_scalar_arguments_of_a_case() -> None:
    reached = {
        f"{case.name}.{name}" for case in CALL_CASES for name in get_scalar_parameters(case.target)
    }
    assert sorted(set(ACCEPTS_NON_FINITE) - reached) == []


def _get_public_targets() -> dict[str, object]:
    # The public functions, classes and class methods with a scalar argument of the six packages.
    targets = {}
    for package in PACKAGES:
        module = importlib.import_module(f"sonnix.{package}")
        for name in module.__all__:
            obj = getattr(module, name)
            candidates = {name: obj}
            if inspect.isclass(obj):
                candidates |= {
                    f"{name}.{method}": getattr(obj, method)
                    for method, value in vars(obj).items()
                    if isinstance(value, classmethod)
                }
            targets |= {
                key: value
                for key, value in candidates.items()
                if (inspect.isclass(value) or inspect.isfunction(value) or inspect.ismethod(value))
                and get_scalar_parameters(value)
            }
    return targets
