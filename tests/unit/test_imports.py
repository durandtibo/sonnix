from __future__ import annotations

import importlib
import pkgutil

import pytest

import sonnix

MODULE_NAMES = sorted(m.name for m in pkgutil.walk_packages(sonnix.__path__, prefix="sonnix."))

#################################
#     Tests for all modules     #
#################################


@pytest.mark.parametrize("name", MODULE_NAMES)
def test_import_module(name: str) -> None:
    assert importlib.import_module(name).__name__ == name
