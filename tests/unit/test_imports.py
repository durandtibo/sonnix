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

PUBLIC_PACKAGES = ["functional", "modules", "validation", "factory", "utils", "testing"]
SOURCE_FILES = sorted(Path(sonnix.__path__[0]).rglob("*.py"))
# The docstrings import a public name from the flat path (``sonnix.modules``), never from a
# folder (``sonnix.modules.loss``) or from a file (``sonnix.utils.mode``). A private name
# (leading underscore) is only reachable from its file.
FOLDER_IMPORT = re.compile(
    r"^\s*>>> from sonnix\."
    r"((modules\.(activation|fusion|loss|numerical)|functional\.(loss|activation))(\.[a-z]\w*)?"
    r"|(utils|testing)\.[a-z]\w*)"
    r" import (?!_)",
    flags=re.MULTILINE,
)
PUBLIC_NAMES = [
    (package, name)
    for package in PUBLIC_PACKAGES
    for name in importlib.import_module(f"sonnix.{package}").__all__
]
# The ``__init__.py`` of the folders of a public package, for example ``modules/loss``.
FOLDER_INITS = sorted(
    path
    for path in Path(sonnix.__path__[0]).rglob("__init__.py")
    if len(path.relative_to(Path(sonnix.__path__[0])).parts) >= 3
)


@pytest.mark.parametrize("path", SOURCE_FILES, ids=lambda path: path.name)
def test_docstrings_import_from_the_public_path(path: Path) -> None:
    assert FOLDER_IMPORT.search(path.read_text()) is None


@pytest.mark.parametrize("package", PUBLIC_PACKAGES)
def test_public_package_has_an_all(package: str) -> None:
    assert isinstance(importlib.import_module(f"sonnix.{package}").__all__, list)


@pytest.mark.parametrize("package", PUBLIC_PACKAGES)
def test_public_package_all_has_no_duplicate(package: str) -> None:
    names = importlib.import_module(f"sonnix.{package}").__all__
    assert len(names) == len(set(names))


@pytest.mark.parametrize(("package", "name"), PUBLIC_NAMES)
def test_public_name_is_importable_from_the_package(package: str, name: str) -> None:
    assert hasattr(importlib.import_module(f"sonnix.{package}"), name)


@pytest.mark.parametrize("path", FOLDER_INITS, ids=lambda path: str(path.parent.name))
def test_folder_has_no_all(path: Path) -> None:
    assert "__all__" not in path.read_text()
