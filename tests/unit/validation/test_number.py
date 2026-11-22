from __future__ import annotations

import pytest

from sonnix.validation import validate_positive

#######################################
#     Tests for validate_positive     #
#######################################


@pytest.mark.parametrize("value", [1e-8, 0.5, 1, 1e300])
def test_validate_positive_valid(value: float) -> None:
    assert validate_positive(value) is None


@pytest.mark.parametrize("value", [0, 0.0, -1, -0.5])
def test_validate_positive_not_positive(value: float) -> None:
    with pytest.raises(ValueError, match=r"value must be greater than 0"):
        validate_positive(value)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_validate_positive_not_finite(value: float) -> None:
    with pytest.raises(ValueError, match=r"value must be finite"):
        validate_positive(value)


@pytest.mark.parametrize("value", ["a", None, True])
def test_validate_positive_incorrect_type(value: object) -> None:
    with pytest.raises(TypeError, match=r"value must be a real number"):
        validate_positive(value)


def test_validate_positive_name_in_message() -> None:
    with pytest.raises(ValueError, match=r"eps must be greater than 0"):
        validate_positive(0.0, name="eps")
