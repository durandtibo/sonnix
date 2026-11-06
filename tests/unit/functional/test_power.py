from __future__ import annotations

import pytest
import torch
from coola.equality import objects_are_allclose, objects_are_equal

from sonnix.functional.power import safe_pow

##############################
#     Tests for safe_pow     #
##############################


def testsafe_pow_positive_values() -> None:
    assert objects_are_allclose(safe_pow(torch.tensor([4.0, 9.0]), 0.5), torch.tensor([2.0, 3.0]))


def testsafe_pow_zero_value() -> None:
    assert objects_are_equal(safe_pow(torch.zeros(2), 0.5), torch.zeros(2))


def testsafe_pow_gradient_zero() -> None:
    x = torch.tensor([0.0, 4.0], requires_grad=True)
    safe_pow(x, 0.5).sum().backward()
    assert objects_are_equal(x.grad, torch.tensor([0.0, 0.25]))


def testsafe_pow_exponent() -> None:
    assert objects_are_allclose(safe_pow(torch.tensor([8.0]), 1 / 3), torch.tensor([2.0]))


def testsafe_pow_float64() -> None:
    assert safe_pow(torch.tensor([4.0], dtype=torch.float64), 0.5).dtype == torch.float64


def test_safe_pow_gradient_exponent_two() -> None:
    x = torch.tensor([0.0, 3.0], requires_grad=True)
    safe_pow(x, 2.0).sum().backward()
    assert objects_are_equal(x.grad, torch.tensor([0.0, 6.0]))


def test_safe_pow_exponent_zero() -> None:
    assert objects_are_equal(safe_pow(torch.tensor([0.0, 2.0]), 0.0), torch.ones(2))


def test_safe_pow_negative_exponent() -> None:
    with pytest.raises(ValueError, match=r"exponent"):
        safe_pow(torch.ones(2), -1.0)
