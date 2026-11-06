from __future__ import annotations

import contextlib

import pytest
from torch import nn

from sonnix.utils.mode import module_mode

#################################
#     Tests for module_mode     #
#################################


def _states(module: nn.ModuleDict) -> tuple[bool, bool, bool]:
    return (module.training, module["module1"].training, module["module2"].training)


def _make_module() -> nn.ModuleDict:
    return nn.ModuleDict({"module1": nn.Linear(4, 6), "module2": nn.Linear(2, 4).eval()})


def test_module_mode_train_inside() -> None:
    module = _make_module()
    with module_mode(module):
        module.train()
        states = _states(module)
    assert states == (True, True, True)


def test_module_mode_train_restored() -> None:
    module = _make_module()
    with module_mode(module):
        module.train()
    assert _states(module) == (True, True, False)


def test_module_mode_eval_inside() -> None:
    module = _make_module()
    with module_mode(module):
        module.eval()
        states = _states(module)
    assert states == (False, False, False)


def test_module_mode_eval_restored() -> None:
    module = _make_module()
    with module_mode(module):
        module.eval()
    assert _states(module) == (True, True, False)


def test_module_mode_with_exception_raises() -> None:
    module = _make_module()
    error = RuntimeError("Exception")
    with pytest.raises(RuntimeError, match=r"Exception"), module_mode(module):
        raise error


def test_module_mode_with_exception_restored() -> None:
    module = _make_module()
    error = RuntimeError("Exception")
    with contextlib.suppress(RuntimeError), module_mode(module):
        module.eval()
        raise error
    assert _states(module) == (True, True, False)
