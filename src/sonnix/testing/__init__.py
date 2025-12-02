r"""Provide testing utilities and fixtures for the sonnix package.

This module exports reusable pytest markers for conditional test
execution based on the availability of optional dependencies or hardware
features.
"""

from __future__ import annotations

__all__ = ["cuda_available", "cuda_not_available", "objectory_available", "objectory_not_available"]

from sonnix.testing.fixtures import (
    cuda_available,
    cuda_not_available,
    objectory_available,
    objectory_not_available,
)
