from __future__ import annotations

import importlib
import re
from pathlib import Path

import pytest

REFS_DIR = Path(__file__).parents[2] / "docs" / "docs" / "refs"
REFS_TEXT = "\n".join(path.read_text() for path in sorted(REFS_DIR.rglob("*.md")))
PACKAGES = ["functional", "modules", "validation", "factory", "utils", "testing"]
PUBLIC_NAMES = sorted(
    {
        name
        for package in PACKAGES
        for name in getattr(importlib.import_module(f"sonnix.{package}"), "__all__", [])
    }
)

############################################
#     Tests for the reference coverage     #
############################################


@pytest.mark.parametrize("name", PUBLIC_NAMES)
def test_public_name_is_documented_in_the_reference(name: str) -> None:
    # A name is listed explicitly by a ``::: sonnix.<package>.<name>`` directive.
    assert re.search(rf"^#+ ::: sonnix(\.\w+)+\.{name}$", REFS_TEXT, flags=re.MULTILINE)


##############################################
#     Tests for the reference directives     #
##############################################

DIRECTIVES = sorted(set(re.findall(r"^#+ ::: (sonnix(?:\.\w+)+)$", REFS_TEXT, flags=re.MULTILINE)))


def _resolve(path: str) -> object:
    parts = path.split(".")
    for i in range(len(parts), 0, -1):
        try:
            obj = importlib.import_module(".".join(parts[:i]))
        except ModuleNotFoundError:
            continue
        for attribute in parts[i:]:
            obj = getattr(obj, attribute)
        return obj
    raise ModuleNotFoundError(path)


@pytest.mark.parametrize("path", DIRECTIVES)
def test_reference_directive_points_to_an_existing_object(path: str) -> None:
    assert _resolve(path) is not None
