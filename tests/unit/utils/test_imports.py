from __future__ import annotations

from unittest.mock import patch

import pytest

from sonnix.utils.imports import (
    check_objectory,
    is_objectory_available,
    objectory_available,
    raise_error_objectory_missing,
)


def my_function(n: int = 0) -> int:
    return 42 + n


#####################
#     objectory     #
#####################


def test_check_objectory_with_package() -> None:
    with patch("sonnix.utils.imports.is_objectory_available", lambda: True):
        check_objectory()


def test_check_objectory_without_package() -> None:
    with (
        patch("sonnix.utils.imports.is_objectory_available", lambda: False),
        pytest.raises(RuntimeError, match=r"'objectory' package is required but not installed."),
    ):
        check_objectory()


def test_is_objectory_available() -> None:
    assert isinstance(is_objectory_available(), bool)


def test_objectory_available_with_package() -> None:
    with patch("sonnix.utils.imports.is_objectory_available", lambda: True):
        fn = objectory_available(my_function)
        assert fn(2) == 44


def test_objectory_available_without_package() -> None:
    with patch("sonnix.utils.imports.is_objectory_available", lambda: False):
        fn = objectory_available(my_function)
        assert fn(2) is None


def test_objectory_available_decorator_with_package() -> None:
    with patch("sonnix.utils.imports.is_objectory_available", lambda: True):

        @objectory_available
        def fn(n: int = 0) -> int:
            return 42 + n

        assert fn(2) == 44


def test_objectory_available_decorator_without_package() -> None:
    with patch("sonnix.utils.imports.is_objectory_available", lambda: False):

        @objectory_available
        def fn(n: int = 0) -> int:
            return 42 + n

        assert fn(2) is None


def test_raise_error_objectory_missing() -> None:
    with pytest.raises(RuntimeError, match=r"'objectory' package is required but not installed."):
        raise_error_objectory_missing()
