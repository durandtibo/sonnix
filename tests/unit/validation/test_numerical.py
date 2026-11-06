from __future__ import annotations

import pytest

from sonnix.validation import validate_abs_range, validate_frequency

#####################################
#     Tests for validate_abs_range     #
#####################################


def test_validate_abs_range_valid() -> None:
    validate_abs_range(min_abs_value=1, max_abs_value=5)


def test_validate_abs_range_invalid_min_abs_value() -> None:
    with pytest.raises(ValueError, match=r"min_abs_value must be greater than 0"):
        validate_abs_range(min_abs_value=-1, max_abs_value=5)


def test_validate_abs_range_invalid_max_abs_value() -> None:
    with pytest.raises(
        ValueError, match=r"'max_abs_value' .* has to be greater than 'min_abs_value'"
    ):
        validate_abs_range(min_abs_value=5, max_abs_value=1)


#####################################
#     Tests for validate_frequency     #
#####################################


def test_validate_frequency_valid() -> None:
    validate_frequency(num_frequencies=3, min_frequency=0.2, max_frequency=1)


def test_validate_frequency_invalid_num_frequencies() -> None:
    with pytest.raises(ValueError, match=r"num_frequencies must be greater than or equal to 1"):
        validate_frequency(num_frequencies=0, min_frequency=0.2, max_frequency=1)


def test_validate_frequency_invalid_min_frequency() -> None:
    with pytest.raises(ValueError, match=r"min_frequency must be greater than 0"):
        validate_frequency(num_frequencies=3, min_frequency=-2, max_frequency=1)


def test_validate_frequency_invalid_max_frequency() -> None:
    with pytest.raises(
        ValueError, match=r"'max_frequency' .* has to be greater than 'min_frequency'"
    ):
        validate_frequency(num_frequencies=3, min_frequency=2, max_frequency=1)
