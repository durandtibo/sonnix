r"""Contain utility functions to manipulate ``torch.nn.Module``'s state
dict."""

from __future__ import annotations

__all__ = ["load_state_dict_to_module"]

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from torch import nn

logger: logging.Logger = logging.getLogger(__name__)


def _find_module_state_dict(
    state_dict: dict[Any, Any] | list[Any] | tuple[Any, ...] | set[Any], module_keys: set[Any]
) -> dict[Any, Any]:
    r"""Try to find automatically the part of the state dict related to a
    module.

    The user should specify the set of module's keys:
    ``set(module.state_dict().keys())``. This function assumes that
    the set of keys only exists at one location in the state dict.
    If the set of keys exists at several locations in the state dict,
    only the first one is returned.

    A state dict saved from a wrapped module (``torch.nn.DataParallel``
    adds the prefix ``module.``, ``torch.compile`` adds ``_orig_mod.``)
    has the same keys with a common prefix. This prefix is stripped
    before comparing the keys, and the returned dict has the stripped
    keys, so it can be loaded in the module.

    Args:
        state_dict: The state dict. This function is called recursively
            on this input to find the queried state dict.
        module_keys: The set of module keys.

    Returns:
        The part of the state dict related to a module if it is
            found, otherwise an empty dict.

    Example:
        ```pycon
        >>> import torch
        >>> from sonnix.utils.state_dict import _find_module_state_dict
        >>> state = {
        ...     "model": {
        ...         "weight": 42,
        ...         "network": {
        ...             "weight": torch.ones(5, 4),
        ...             "bias": 2 * torch.ones(5),
        ...         },
        ...     }
        ... }
        >>> module = torch.nn.Linear(4, 5)
        >>> state_dict = _find_module_state_dict(state, module_keys=set(module.state_dict().keys()))
        >>> state_dict
        {'weight': tensor([[1., 1., 1., 1.],
                [1., 1., 1., 1.],
                [1., 1., 1., 1.],
                [1., 1., 1., 1.],
                [1., 1., 1., 1.]]), 'bias': tensor([2., 2., 2., 2., 2.])}

        ```
    """
    if isinstance(state_dict, dict):
        if set(state_dict.keys()) == module_keys:
            return state_dict
        stripped = _strip_prefix(state_dict, module_keys)
        if stripped:
            return stripped
        for value in state_dict.values():
            found = _find_module_state_dict(value, module_keys)
            if found:
                return found
    elif isinstance(state_dict, (list, tuple, set)):
        for value in state_dict:
            found = _find_module_state_dict(value, module_keys)
            if found:
                return found
    return {}


def _strip_prefix(state_dict: dict[Any, Any], module_keys: set[Any]) -> dict[Any, Any]:
    r"""Strip the common prefix of the keys of a state dict.

    Args:
        state_dict: The state dict.
        module_keys: The set of module keys. It must contain only
            strings to find a prefix.

    Returns:
        The state dict without the prefix if all the keys start with
            the same prefix, which ends with a dot, and if the keys
            without the prefix are the module keys, otherwise an empty
            dict.

    Example:
        ```pycon
        >>> from sonnix.utils.state_dict import _strip_prefix
        >>> _strip_prefix({"module.weight": 1, "module.bias": 2}, {"weight", "bias"})
        {'weight': 1, 'bias': 2}

        ```
    """
    if not module_keys or not all(isinstance(key, str) for key in module_keys):
        return {}
    if not all(isinstance(key, str) for key in state_dict):
        return {}
    anchor = min(module_keys)
    for key in state_dict:
        prefix = key[: len(key) - len(anchor)]
        if not (key.endswith(anchor) and prefix.endswith(".")):
            continue
        if all(k.startswith(prefix) for k in state_dict):
            candidate = {k[len(prefix) :]: v for k, v in state_dict.items()}
            if set(candidate) == module_keys:
                return candidate
    return {}


def load_state_dict_to_module(
    state_dict: dict[Any, Any], module: nn.Module, *, strict: bool = True
) -> None:
    r"""Load a state dict into a given module.

    This function will automatically try to find the module state dict
    in the given state dict. With ``strict=False``, torch does not raise
    when keys are missing, so the search is run when none of the module
    keys is found in the given state dict. If the module has no state,
    nothing is searched. A common prefix of the keys (``module.`` from
    ``torch.nn.DataParallel``, ``_orig_mod.`` from ``torch.compile``)
    is ignored when the state dict is searched.

    Args:
        state_dict: The state dict.
        module: The module. This function changes the weights of this
            module.
        strict: whether to strictly enforce that the
            keys in ``state_dict`` match the keys returned by this
            module's ``torch.nn.Module.state_dict`` function.

    Example:
        ```pycon
        >>> import torch
        >>> from sonnix.utils.state_dict import load_state_dict_to_module
        >>> state = {
        ...     "model": {
        ...         "weight": 42,
        ...         "network": {
        ...             "weight": torch.ones(5, 4),
        ...             "bias": 2 * torch.ones(5),
        ...         },
        ...     }
        ... }
        >>> module = torch.nn.Linear(4, 5)
        >>> load_state_dict_to_module(state, module)
        >>> out = module(torch.ones(2, 4))
        >>> out
        tensor([[6., 6., 6., 6., 6.],
                [6., 6., 6., 6., 6.]], grad_fn=<AddmmBackward0>)

        ```
    """
    module_keys = set(module.state_dict().keys())
    try:
        result = module.load_state_dict(state_dict, strict=strict)
    except RuntimeError:
        logger.warning(
            "Could not load the state dict. Try to find automatically the part of the state dict "
            "that matches with the module."
        )
        found = _find_module_state_dict(state_dict, module_keys)
        if not found:
            logger.info("Could not find a part of the state dict that matches with the module.")
            raise
        logger.info(
            "Found a part of the state dict that matches with the module. Try to load it in the "
            "module."
        )
        module.load_state_dict(found, strict=strict)
    else:
        if not strict and module_keys and set(result.missing_keys) == module_keys:
            logger.warning(
                "No key of the module was found in the state dict. Try to find automatically the "
                "part of the state dict that matches with the module."
            )
            found = _find_module_state_dict(state_dict, module_keys)
            if not found:
                logger.warning(
                    "Could not find a part of the state dict that matches with the module. "
                    "No weight was loaded. Missing keys: %s. Unexpected keys: %s.",
                    result.missing_keys,
                    result.unexpected_keys,
                )
                return
            result = module.load_state_dict(found, strict=strict)
        if result.missing_keys or result.unexpected_keys:
            logger.warning(
                "Missing keys: %s. Unexpected keys: %s.",
                result.missing_keys,
                result.unexpected_keys,
            )
    logger.info("The weights are loaded in the module.")
