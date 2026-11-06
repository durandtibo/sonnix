r"""The sonnix package - A library of PyTorch modules.

Sonnix provides a collection of custom PyTorch modules, layers, and loss
functions for building deep learning models.
"""

from __future__ import annotations

__all__ = ["__version__"]

from importlib.metadata import version

__version__ = version(__name__)
