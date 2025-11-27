from __future__ import annotations

import pytest

from sonnix.utils.fallback.objectory import AbstractFactory, factory, is_object_config


def test_abstract_factory() -> None:
    class Factory(metaclass=AbstractFactory): ...

    with pytest.raises(RuntimeError, match=r"'objectory' package is required but not installed."):
        Factory.factory()


def test_factory() -> None:
    with pytest.raises(RuntimeError, match=r"'objectory' package is required but not installed."):
        factory()


def test_is_object_config() -> None:
    with pytest.raises(RuntimeError, match=r"'objectory' package is required but not installed."):
        is_object_config()
