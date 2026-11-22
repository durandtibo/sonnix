r"""Provide utility functions for the sonnix package.

This subpackage contains the helpers that the modules use to register
their tensors and to restore their mode.
"""

from __future__ import annotations

__all__ = ["module_mode", "register_tensor"]

from sonnix.utils.mode import module_mode
from sonnix.utils.tensor import register_tensor
