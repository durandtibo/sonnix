r"""Test the state of every registered module (property P8): the state
dict is complete, holds nothing derived, and a round trip restores the
behaviour."""

from __future__ import annotations

import pytest
import torch
from coola.equality import objects_are_equal

from tests.registry import MODULE_CASES, ModuleCase

OTHER_CASES = [case for case in MODULE_CASES if case.other is not None]
LEARNABLE_CASES = [case for case in MODULE_CASES if case.has_learnable_option]
DERIVED_CASES = [case for case in MODULE_CASES if case.derived]


def _build_with_seed(case: ModuleCase, seed: int) -> torch.nn.Module:
    torch.manual_seed(seed)
    return case.build()


def _build_other(case: ModuleCase) -> torch.nn.Module:
    # A random initialization already differs from the first build.
    return case.other() if case.other is not None else _build_with_seed(case, seed=1)


####################################
#     Tests for the round trip     #
####################################


@pytest.mark.parametrize("case", MODULE_CASES, ids=repr)
def test_load_state_dict_restores_the_output(case: ModuleCase) -> None:
    source = _build_with_seed(case, seed=0)
    target = _build_other(case)
    target.load_state_dict(source.state_dict())
    inputs = case.make_inputs()
    assert objects_are_equal(target(*inputs), source(*inputs))


@pytest.mark.parametrize("case", OTHER_CASES, ids=repr)
def test_other_build_has_a_different_output(case: ModuleCase) -> None:
    inputs = case.make_inputs()
    assert not objects_are_equal(case.other()(*inputs), case.build()(*inputs))


@pytest.mark.parametrize("case", MODULE_CASES, ids=repr)
def test_load_state_dict_has_no_missing_or_unexpected_key(case: ModuleCase) -> None:
    keys = _build_other(case).load_state_dict(_build_with_seed(case, seed=0).state_dict())
    assert (keys.missing_keys, keys.unexpected_keys) == ([], [])


#########################################
#     Tests for the state dict keys     #
#########################################


@pytest.mark.parametrize("case", LEARNABLE_CASES, ids=repr)
def test_state_dict_keys_do_not_depend_on_learnable(case: ModuleCase) -> None:
    assert list(case.build(learnable=True).state_dict()) == list(
        case.build(learnable=False).state_dict()
    )


@pytest.mark.parametrize("case", DERIVED_CASES, ids=repr)
def test_derived_tensors_are_not_in_the_state_dict(case: ModuleCase) -> None:
    assert set(case.derived).isdisjoint(case.build().state_dict())


@pytest.mark.parametrize("case", DERIVED_CASES, ids=repr)
def test_derived_tensors_are_buffers(case: ModuleCase) -> None:
    module = case.build()
    assert set(case.derived) <= {name for name, _ in module.named_buffers()}
