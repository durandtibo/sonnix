r"""Define some PyTest fixtures."""

from __future__ import annotations

__all__ = ["cuda_available", "cuda_not_available", "objectory_available", "objectory_not_available"]

import pytest
import torch

from sonnix.utils.imports import is_objectory_available

cuda_available = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Requires a device with CUDA"
)
cuda_not_available = pytest.mark.skipif(
    torch.cuda.is_available(), reason="Skip if CUDA is available"
)

objectory_available = pytest.mark.skipif(not is_objectory_available(), reason="Require objectory")
objectory_not_available = pytest.mark.skipif(
    is_objectory_available(), reason="Skip if objectory is available"
)
