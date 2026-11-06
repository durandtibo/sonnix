"""Contain functions to instantiate objects from their configuration."""

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
