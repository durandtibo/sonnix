from __future__ import annotations

import inspect

import pytest

from tests.registry import MODULE_CASES, ModuleCase

SINGLE_INPUT_CASES = [case for case in MODULE_CASES if len(case.make_inputs()) == 1]

###################################################
#     Tests for the signature of the forwards     #
###################################################


@pytest.mark.parametrize("case", SINGLE_INPUT_CASES, ids=repr)
def test_forward_single_tensor_is_named_input(case: ModuleCase) -> None:
    parameters = list(inspect.signature(case.build().forward).parameters)
    assert parameters == ["input"]
