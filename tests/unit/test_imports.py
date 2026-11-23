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


def test_no_object_is_exported_by_two_packages() -> None:
    # One public path: an object that is exported by two packages (``Reduction`` was exported by
    # ``sonnix.functional`` and by ``sonnix.validation``) has two public paths.
    objects = [
        id(getattr(importlib.import_module(f"sonnix.{package}"), name))
        for package, name in PUBLIC_NAMES
    ]
    assert len(objects) == len(set(objects))


@pytest.mark.parametrize("path", FOLDER_INITS, ids=lambda path: str(path.parent.name))
def test_folder_has_no_all(path: Path) -> None:
    assert "__all__" not in path.read_text()


###############################################
#     Tests for the cross-package imports     #
###############################################

SOURCE_ROOT = Path(sonnix.__path__[0])
# ``from sonnix.<package>.<module> import``: a module path of another package.
MODULE_PATH_IMPORT = re.compile(r"^\s*from sonnix\.(\w+)\.[a-z]\w* import ", flags=re.MULTILINE)
# ``functional`` defines ``Reduction`` in the lowest layer (``validation``) and exports it.
ALLOWED_MODULE_PATH_IMPORTS = {("functional/reduction.py", "validation")}


def _get_package(path: Path) -> str:
    return path.relative_to(SOURCE_ROOT).parts[0]


def _get_cross_package_imports(path: Path) -> set[str]:
    package = _get_package(path)
    imported = set(MODULE_PATH_IMPORT.findall(path.read_text())) - {package}
    relative = path.relative_to(SOURCE_ROOT).as_posix()
    return {name for name in imported if (relative, name) not in ALLOWED_MODULE_PATH_IMPORTS}


@pytest.mark.parametrize(
    "path", SOURCE_FILES, ids=lambda path: path.relative_to(SOURCE_ROOT).as_posix()
)
def test_cross_package_imports_use_the_flat_path(path: Path) -> None:
    assert _get_cross_package_imports(path) == set()


def test_module_path_import_detects_a_module_path() -> None:
    assert MODULE_PATH_IMPORT.findall("    from sonnix.utils.tensor import register_tensor\n") == [
        "utils"
    ]


def test_module_path_import_ignores_the_flat_path() -> None:
    assert MODULE_PATH_IMPORT.findall("from sonnix.utils import register_tensor\n") == []


######################################
#     Tests for the suppressions     #
######################################

SUPPRESSION = re.compile(r"noqa(?!: A002\b)|type: ?ignore|ty: ?ignore|pragma: ?no cover")


@pytest.mark.parametrize(
    "path", SOURCE_FILES, ids=lambda path: path.relative_to(SOURCE_ROOT).as_posix()
)
def test_source_has_no_suppression(path: Path) -> None:
    # ``noqa: A002`` (the argument ``input``, the name of ``torch.nn``) is the only exception.
    assert SUPPRESSION.search(path.read_text()) is None


@pytest.mark.parametrize(
    "line",
    [
        "x = 1  # noqa: E501",
        "x = 1  # {}".format("noqa"),  # built at runtime: the blanket-noqa hook flags a literal
        "x = 1  # type: ignore",
        "x = 1  # ty: ignore[a]",
        "x = 1  # pragma: no cover",
    ],
)
def test_suppression_detects(line: str) -> None:
    assert SUPPRESSION.search(line) is not None


def test_suppression_allows_a002() -> None:
    assert SUPPRESSION.search("def f(input):  # noqa: A002") is None
