from __future__ import annotations

import pytest
import torch
from coola.equality import objects_are_equal
from torch import nn

from sonnix.utils.state_dict import find_module_state_dict, load_state_dict_to_module

LINEAR_STATE_DICT = {"weight": torch.ones(5, 4), "bias": 2 * torch.ones(5)}

STATE_DICTS = [
    {"model": {"network": LINEAR_STATE_DICT}},
    {"list": ["weight", "bias"], "model": {"network": LINEAR_STATE_DICT}},  # should not be detected
    {"set": {"weight", "bias"}, "model": {"network": LINEAR_STATE_DICT}},  # should not be detected
    {
        "tuple": ("weight", "bias"),
        "model": {"network": LINEAR_STATE_DICT},
    },  # should not be detected
    {"list": ["weight", "bias", LINEAR_STATE_DICT], "abc": None},
]


############################################
#     Tests for find_module_state_dict     #
############################################


def test_find_module_state_dict() -> None:
    state_dict = {"weight": torch.ones(5, 4), "bias": 2 * torch.ones(5)}
    assert objects_are_equal(state_dict, find_module_state_dict(state_dict, {"weight", "bias"}))


@pytest.mark.parametrize("state_dict", STATE_DICTS)
def test_find_module_state_dict_nested(state_dict: dict) -> None:
    assert objects_are_equal(
        LINEAR_STATE_DICT, find_module_state_dict(state_dict, {"bias", "weight"})
    )


def test_find_module_state_dict_missing_key() -> None:
    assert find_module_state_dict({"weight": torch.ones(5, 4)}, {"bias", "weight"}) == {}


###############################################
#     Tests for load_state_dict_to_module     #
###############################################


@pytest.mark.parametrize("state_dict", STATE_DICTS)
def test_load_state_dict_to_module_find_module(state_dict: dict) -> None:
    module = nn.Linear(4, 5)
    load_state_dict_to_module(state_dict, module)
    assert objects_are_equal(module(torch.ones(2, 4)), torch.full((2, 5), fill_value=6.0))


def test_load_state_dict_to_module_incompatible_module() -> None:
    with pytest.raises(RuntimeError, match=r"Error\(s\) in loading state_dict for Linear:"):
        load_state_dict_to_module(LINEAR_STATE_DICT, nn.Linear(6, 10))


def test_load_state_dict_to_module_partial_state_dict() -> None:
    with pytest.raises(RuntimeError, match=r"Error\(s\) in loading state_dict for Linear:"):
        load_state_dict_to_module({"weight": torch.ones(5, 4)}, nn.Linear(4, 5))


def test_load_state_dict_to_module_dict_strict_false_partial_state() -> None:
    module = nn.Linear(4, 5)
    load_state_dict_to_module({"weight": torch.ones(5, 4)}, module, strict=False)

    out = module(torch.ones(2, 4))
    assert out.shape == (
        2,
        5,
    )  # The bias is randomly initialized so it is not possible to know the exact value.
    assert module.weight.equal(torch.ones(5, 4))
