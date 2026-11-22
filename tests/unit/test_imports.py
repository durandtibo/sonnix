from __future__ import annotations

import importlib
import pkgutil
import re
from pathlib import Path

import pytest

import sonnix

MODULE_NAMES = sorted(m.name for m in pkgutil.walk_packages(sonnix.__path__, prefix="sonnix."))

#################################
#     Tests for all modules     #
#################################


@pytest.mark.parametrize("name", MODULE_NAMES)
def test_import_module(name: str) -> None:
    assert importlib.import_module(name).__name__ == name


#############################################
#     Tests for the public import paths     #
#############################################

SOURCE_FILES = sorted(Path(sonnix.__path__[0]).rglob("*.py"))
SUBPACKAGE_IMPORT = re.compile(
    r"^\s*>>> from sonnix\.(modules\.(activation|fusion|loss|numerical)|functional\.loss)"
    r"(\.[a-z]\w*)? import",
    flags=re.MULTILINE,
)


@pytest.mark.parametrize("path", SOURCE_FILES, ids=lambda path: path.name)
def test_docstrings_import_from_the_public_path(path: Path) -> None:
    assert SUBPACKAGE_IMPORT.search(path.read_text()) is None
