r"""Test that every argument that accepts a configuration documents the
trust warning (property P14)."""

from __future__ import annotations

import inspect

import pytest

from sonnix import factory, modules


def _accepts_config(obj: object) -> bool:
    # ``is_*_config`` only checks a configuration: it does not call its target.
    if obj.__name__.startswith("is_"):
        return False
    target = obj.__init__ if inspect.isclass(obj) else obj
    return any("dict[" in str(p.annotation) for p in inspect.signature(target).parameters.values())


CONFIG_OBJECTS = [
    obj
    for obj in (
        *(getattr(modules, name) for name in modules.__all__),
        *(getattr(factory, name) for name in factory.__all__),
    )
    if (inspect.isclass(obj) or inspect.isfunction(obj)) and _accepts_config(obj)
]


def test_config_objects_are_found() -> None:
    assert len(CONFIG_OBJECTS) >= 6


@pytest.mark.parametrize("obj", CONFIG_OBJECTS, ids=lambda obj: obj.__name__)
def test_config_argument_has_trust_warning(obj: object) -> None:
    assert "Warning:" in inspect.getdoc(obj)
