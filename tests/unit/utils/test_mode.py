from __future__ import annotations

import contextlib

import pytest
from torch import nn

from sonnix.utils.mode import module_mode, top_module_mode

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


#####################################
#     Tests for top_module_mode     #
#####################################


def test_top_module_mode_train_inside() -> None:
    module = nn.Linear(4, 6)
    with top_module_mode(module):
        module.eval()
        training = module.training
    assert not training


def test_top_module_mode_train_restored() -> None:
    module = nn.Linear(4, 6)
    with top_module_mode(module):
        module.eval()
    assert module.training


def test_top_module_mode_eval_inside() -> None:
    module = nn.Linear(4, 6).eval()
    with top_module_mode(module):
        module.train()
        training = module.training
    assert training


def test_top_module_mode_eval_restored() -> None:
    module = nn.Linear(4, 6).eval()
    with top_module_mode(module):
        module.train()
    assert not module.training


def test_top_module_mode_with_exception_raises() -> None:
    module = nn.Linear(4, 6)
    error = RuntimeError("Exception")
    with pytest.raises(RuntimeError, match=r"Exception"), top_module_mode(module):
        raise error


def test_top_module_mode_with_exception_restored() -> None:
    module = nn.Linear(4, 6)
    error = RuntimeError("Exception")
    with contextlib.suppress(RuntimeError), top_module_mode(module):
        module.eval()
        raise error
    assert module.training


def test_module_mode_shared_module() -> None:
    shared = nn.Linear(2, 2).eval()
    module = nn.ModuleDict({"a": shared, "b": shared})
    with module_mode(module):
        module.train()
    assert not shared.training


def test_module_mode_module_added_in_context() -> None:
    module = nn.Sequential(nn.Linear(2, 2))
    with module_mode(module):
        module.append(nn.Dropout(0.1))
    assert module.training


def test_module_mode_module_replaced_in_context() -> None:
    module = nn.Sequential(nn.Linear(2, 2).eval(), nn.Linear(2, 2))
    with module_mode(module):
        module[0] = nn.Linear(2, 2)
        module.train()
    assert module.training
