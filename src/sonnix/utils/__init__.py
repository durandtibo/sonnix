r"""Provide utility functions for the sonnix package.

This subpackage contains the helpers that the modules use to register
their tensors and to restore their mode, and some functions to inspect,
freeze and load modules.
"""

from __future__ import annotations

__all__ = [
    "freeze_module",
    "get_module_device",
    "get_module_devices",
    "has_learnable_parameters",
    "has_parameters",
    "is_module_on_device",
    "load_state_dict_to_module",
    "module_mode",
    "num_learnable_parameters",
    "num_parameters",
    "register_tensor",
    "unfreeze_module",
]

from sonnix.utils.device import get_module_device, get_module_devices, is_module_on_device
from sonnix.utils.mode import module_mode
from sonnix.utils.params import (
    freeze_module,
    has_learnable_parameters,
    has_parameters,
    num_learnable_parameters,
    num_parameters,
    unfreeze_module,
)
from sonnix.utils.state_dict import load_state_dict_to_module
from sonnix.utils.tensor import register_tensor
