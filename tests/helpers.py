r"""Define some utility functions/classes."""

from __future__ import annotations

__all__ = ["DEVICES_WITHOUT_MPS", "ExamplePair"]

from dataclasses import dataclass
from typing import Any

from coola.utils.tensor import get_available_devices

DEVICES_WITHOUT_MPS = (d for d in get_available_devices() if not d.startswith("mps:"))


@dataclass
class ExamplePair:
    actual: Any
    expected: Any
    expected_message: str | None = None
    atol: float = 0.0
    rtol: float = 0.0
