"""Contain functions to instantiate objects from their configuration.

A configuration is a dictionary with a ``_target_`` key that is the
import path of a class or a function, and the other keys are its
arguments. The ``resolve_*`` functions import the ``_target_`` and call
it, and then check the type of the result. The call happens before the
check, so a configuration can run any importable callable
(``{"_target_": "os.system", "command": ...}``). This is the same trust
model as Hydra: **only use configurations from a trusted source**, never
a file or a message provided by an untrusted user. The ``is_*_config``
functions import the target but do not call it, so they can be used to
check a configuration before the instantiation, but a callable without a
return type hint is reported as invalid.
"""

from __future__ import annotations

__all__ = [
    "create_sequential",
    "is_dataset_config",
    "is_module_config",
    "is_optimizer_config",
    "resolve_dataset",
    "resolve_module",
    "resolve_optimizer",
]

from sonnix.factory.dataset import is_dataset_config, resolve_dataset
from sonnix.factory.module import create_sequential, is_module_config, resolve_module
from sonnix.factory.optimizer import is_optimizer_config, resolve_optimizer
