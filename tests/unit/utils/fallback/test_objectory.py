from __future__ import annotations

import pytest

from sonnix.utils.fallback.objectory import AbstractFactory, factory


def test_abstract_factory() -> None:
    class Factory(metaclass=AbstractFactory): ...

    with pytest.raises(RuntimeError, match=r"'objectory' package is required but not installed."):
        Factory.factory()


def test_factory() -> None:
    with pytest.raises(RuntimeError, match=r"'objectory' package is required but not installed."):
        factory()
