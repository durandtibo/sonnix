from __future__ import annotations

import pytest

from sonnix.testing import objectory_available, objectory_not_available
from sonnix.utils.imports import (
    check_objectory,
    is_objectory_available,
)


def my_function(n: int = 0) -> int:
    return 42 + n


#####################
#     objectory     #
#####################


@objectory_available
def test_check_objectory_with_package() -> None:
    check_objectory()


@objectory_not_available
def test_check_objectory_without_package() -> None:
    with pytest.raises(RuntimeError, match=r"'objectory' package is required but not installed."):
        check_objectory()


@objectory_available
def test_is_objectory_available_true() -> None:
    assert is_objectory_available()


@objectory_not_available
def test_is_objectory_available_false() -> None:
    assert not is_objectory_available()
