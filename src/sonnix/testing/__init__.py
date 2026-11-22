r"""Provide testing utilities for the sonnix package."""

from __future__ import annotations

__all__ = [
    "DummyDataset",
    "is_loss_decreasing",
    "is_loss_decreasing_with_adam",
    "is_loss_decreasing_with_sgd",
]

from sonnix.testing.dummy import DummyDataset
from sonnix.testing.loss import (
    is_loss_decreasing,
    is_loss_decreasing_with_adam,
    is_loss_decreasing_with_sgd,
)
