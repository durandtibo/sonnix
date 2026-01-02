r"""The sonnix package - A library of PyTorch modules.

Sonnix provides a collection of custom PyTorch modules, layers, and loss
functions for building deep learning models.
"""

from __future__ import annotations

__all__ = ["__version__"]

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version(__name__)
except PackageNotFoundError:  # pragma: no cover
    # Package is not installed, fallback if needed
    __version__ = "0.0.0"
