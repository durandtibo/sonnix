from __future__ import annotations

import doctest
import importlib
import inspect
import re
import runpy
from pathlib import Path

import pytest

REFS_DIR = Path(__file__).parents[2] / "docs" / "docs" / "refs"
REFS_TEXT = "\n".join(path.read_text() for path in sorted(REFS_DIR.rglob("*.md")))
PACKAGES = ["functional", "modules", "validation", "factory", "utils", "testing"]
PUBLIC_NAMES = [
    (package, name)
    for package in PACKAGES
    for name in getattr(importlib.import_module(f"sonnix.{package}"), "__all__", [])
]

############################################
#     Tests for the reference coverage     #
############################################


@pytest.mark.parametrize(("package", "name"), PUBLIC_NAMES)
def test_public_name_is_documented_under_its_package(package: str, name: str) -> None:
    # A name is listed explicitly by a ``::: sonnix.<package>.<name>`` directive of its own package.
    assert re.search(rf"^#+ ::: sonnix\.{package}\.{name}$", REFS_TEXT, flags=re.MULTILINE)


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


#################################################
#     Tests for the docstrings and the code     #
#################################################

DOCUMENTED_OBJECTS = [
    (package, name)
    for package, name in PUBLIC_NAMES
    if inspect.isclass(getattr(importlib.import_module(f"sonnix.{package}"), name))
    or inspect.isfunction(getattr(importlib.import_module(f"sonnix.{package}"), name))
]
# A class that documents its arguments by referring to its base class: the ``Args:`` block of the
# base is the reference.
ARGS_REFERENCE = {
    "ExpSin": "BaseAlphaActivation",
    "Gaussian": "BaseAlphaActivation",
    "Laplacian": "BaseAlphaActivation",
    "MultiQuadratic": "BaseAlphaActivation",
    "Quadratic": "BaseAlphaActivation",
    "Snake": "BaseAlphaActivation",
}
# Names that a ``Raises:`` block may quote without being a parameter of the constructor or of the
# ``forward``: the type of an argument, and the method that raises.
RAISES_TYPES = {"bool", "int", "float", "forward"}
# The encoders raise when the last dimension of the input is not ``n_features``, an attribute that
# is not a parameter of the constructor of the class that documents it.
RAISES_ATTRIBUTES = {
    "AsinhNumericalEncoder": {"n_features"},
    "BaseNumericalEncoder": {"n_features"},
}


def _get_object(package: str, name: str) -> object:
    return getattr(importlib.import_module(f"sonnix.{package}"), name)


def _get_parameters(obj: object) -> list[str]:
    # The parameters of the function, or of the constructor of the class.
    function = obj.__init__ if inspect.isclass(obj) else obj
    return [
        name
        for name, parameter in inspect.signature(function).parameters.items()
        if name != "self"
        and parameter.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
    ]


def _get_section(obj: object, section: str) -> str:
    # The lines under ``<section>:`` in the docstring, up to the first line that is not indented.
    doc = inspect.getdoc(obj) or ""
    match = re.search(rf"^{section}:\n((?:(?: +.*)?\n)*)", doc + "\n", flags=re.MULTILINE)
    return match.group(1) if match else ""


def _get_documented_arguments(obj: object) -> list[str]:
    return re.findall(
        r"^    \*{0,2}(\w+)(?: \(.*\))?:", _get_section(obj, "Args"), flags=re.MULTILINE
    )


@pytest.mark.parametrize(("package", "name"), DOCUMENTED_OBJECTS)
def test_args_match_the_signature(package: str, name: str) -> None:
    obj = _get_object(package, name)
    reference = _get_object("modules", ARGS_REFERENCE[name]) if name in ARGS_REFERENCE else obj
    assert _get_documented_arguments(reference) == _get_parameters(obj)


@pytest.mark.parametrize(("package", "name"), DOCUMENTED_OBJECTS)
def test_raises_quote_only_parameters(package: str, name: str) -> None:
    obj = _get_object(package, name)
    quoted = set(re.findall(r"``([a-z_]\w*)``", _get_section(obj, "Raises")))
    allowed = set(_get_parameters(obj)) | RAISES_TYPES | RAISES_ATTRIBUTES.get(name, set())
    if inspect.isclass(obj) and hasattr(obj, "forward"):
        # A class documents the errors of its constructor and of its ``forward``.
        allowed |= set(_get_parameters(obj.forward))
    assert quoted - allowed == set()


def test_get_documented_arguments_of_a_function() -> None:
    def function(a: int, *, b: int = 0) -> None:  # noqa: ARG001
        r"""Do nothing.

        Args:
            a: The first argument,
                on two lines.
            b (int): The second argument.

        Returns:
            Nothing.
        """
        return

    assert _get_documented_arguments(function) == ["a", "b"]


def test_get_documented_arguments_without_args() -> None:
    def function() -> None:
        r"""Do nothing."""
        return

    assert _get_documented_arguments(function) == []


def test_get_parameters_of_a_class_skips_self_and_variadic() -> None:
    class Dummy:
        def __init__(self, a: int, *args: int, b: int = 0, **kwargs: int) -> None:
            pass

    assert _get_parameters(Dummy) == ["a", "b"]


########################################
#     Tests for the markdown files     #
########################################

ROOT = Path(__file__).parents[2]
GUIDE_FILES = sorted((ROOT / "docs" / "docs").rglob("*.md"))
PYTHON_BLOCK = re.compile(r"^```python\n(.*?)^```", flags=re.MULTILINE | re.DOTALL)


def test_readme_examples_run() -> None:
    # ``doctest.testfile`` runs the ``>>>`` lines of the README.
    result = doctest.testfile(
        str(ROOT / "README.md"), module_relative=False, optionflags=doctest.ELLIPSIS
    )
    assert result.failed == 0


@pytest.mark.parametrize("path", GUIDE_FILES, ids=lambda path: path.name)
def test_guide_python_blocks_run(path: Path, tmp_path: Path) -> None:
    # The blocks of a file run as one script, like the cells of a notebook.
    script = tmp_path / "script.py"
    script.write_text("\n".join(PYTHON_BLOCK.findall(path.read_text())))
    runpy.run_path(str(script))


def test_python_block_regex_finds_the_blocks() -> None:
    text = "a\n```python\nx = 1\n```\nb\n```python\ny = 2\n```\n"
    assert PYTHON_BLOCK.findall(text) == ["x = 1\n", "y = 2\n"]


def _normalize(text: str) -> str:
    # The reference tables and the docstrings differ by the backticks.
    return " ".join(text.replace("`", "").split())


def _get_first_line(obj: object) -> str:
    return _normalize(inspect.getdoc(obj).split("\n\n")[0])


TABLE_ROW = re.compile(r"^\| \[`(\w+)`\]\(#sonnix\.(\w+)\.\w+\) +\| (.+?) +\|$", flags=re.MULTILINE)
TABLE_ROWS = [
    (package, name, description) for name, package, description in TABLE_ROW.findall(REFS_TEXT)
]


@pytest.mark.parametrize(("package", "name", "description"), TABLE_ROWS)
def test_reference_row_is_the_first_line_of_the_docstring(
    package: str, name: str, description: str
) -> None:
    assert _normalize(description) == _get_first_line(_get_object(package, name))


def test_reference_table_has_rows() -> None:
    assert len(TABLE_ROWS) > 0


def test_normalize_removes_backticks_and_extra_spaces() -> None:
    assert _normalize("Use ``a``  and `b`.") == "Use a and b."


######################################
#     Tests for the design rules     #
######################################

RULE_ROW = re.compile(r"^\| (P\d+) +\| \*\*(.+?)\*\*", flags=re.MULTILINE)


def _get_rules(path: Path) -> dict[str, str]:
    # The number and the bold title of each rule; a title may end with a period.
    return {number: title.rstrip(".") for number, title in RULE_ROW.findall(path.read_text())}


def test_conventions_and_target_state_have_the_same_rules() -> None:
    assert _get_rules(ROOT / "docs" / "docs" / "conventions.md") == _get_rules(
        ROOT / "review" / "target_state.md"
    )


def test_claude_md_names_the_last_property() -> None:
    last = max(_get_rules(ROOT / "docs" / "docs" / "conventions.md"), key=lambda p: int(p[1:]))
    assert f"properties P1-{last}" in (ROOT / "CLAUDE.md").read_text()


def test_get_rules_strips_the_period_of_the_title(tmp_path: Path) -> None:
    path = tmp_path / "rules.md"
    path.write_text("| P1  | **Title.** Text |\n| P2 | **Other** Text |\n")
    assert _get_rules(path) == {"P1": "Title", "P2": "Other"}
