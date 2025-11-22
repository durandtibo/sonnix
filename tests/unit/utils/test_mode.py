from __future__ import annotations

import pytest
from torch import nn

from sonnix.utils.mode import module_mode, top_module_mode

#################################
#     Tests for module_mode     #
#################################


def test_module_mode_train() -> None:
    module = nn.ModuleDict({"module1": nn.Linear(4, 6), "module2": nn.Linear(2, 4).eval()})
    assert module.training
    assert module["module1"].training
    assert not module["module2"].training
    with module_mode(module):
        module.train()
        assert module.training
        assert module["module1"].training
        assert module["module2"].training
    assert module.training
    assert module["module1"].training
    assert not module["module2"].training


def test_module_mode_eval() -> None:
    module = nn.ModuleDict({"module1": nn.Linear(4, 6), "module2": nn.Linear(2, 4).eval()})
    assert module.training
    assert module["module1"].training
    assert not module["module2"].training
    with module_mode(module):
        module.eval()
        assert not module.training
        assert not module["module1"].training
        assert not module["module2"].training
    assert module.training
    assert module["module1"].training
    assert not module["module2"].training


def test_module_mode_with_exception() -> None:
    module = nn.ModuleDict({"module1": nn.Linear(4, 6), "module2": nn.Linear(2, 4).eval()})
    assert module.training
    with pytest.raises(RuntimeError, match=r"Exception"), module_mode(module):  # noqa: PT012
        module.eval()
        assert not module.training
        assert not module["module1"].training
        assert not module["module2"].training
        msg = "Exception"
        raise RuntimeError(msg)

    assert module.training
    assert module["module1"].training
    assert not module["module2"].training


#####################################
#     Tests for top_module_mode     #
#####################################


def test_top_module_mode_train() -> None:
    module = nn.Linear(4, 6)
    assert module.training
    with top_module_mode(module):
        module.eval()
        assert not module.training
    assert module.training


def test_top_module_mode_eval() -> None:
    module = nn.Linear(4, 6)
    module.eval()
    assert not module.training
    with top_module_mode(module):
        module.train()
        assert module.training
    assert not module.training


def test_top_module_mode_with_exception() -> None:
    module = nn.Linear(4, 6)
    assert module.training
    with pytest.raises(RuntimeError, match=r"Exception"), top_module_mode(module):  # noqa: PT012
        module.eval()
        assert not module.training
        msg = "Exception"
        raise RuntimeError(msg)
    assert module.training
