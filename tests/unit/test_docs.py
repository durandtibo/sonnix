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
