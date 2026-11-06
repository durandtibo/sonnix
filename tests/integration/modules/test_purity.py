r"""Test that the forward of every registered module does not mutate its
inputs (property P7)."""

from __future__ import annotations

import pytest
import torch

from tests.registry import MODULE_CASES, ModuleCase


@pytest.mark.parametrize("case", MODULE_CASES, ids=repr)
def test_forward_does_not_mutate_inputs(case: ModuleCase) -> None:
    module = case.build()
    inputs = case.make_inputs()
    expected = tuple(x.clone() for x in inputs)
    module(*inputs)
    assert all(torch.equal(x, y) for x, y in zip(inputs, expected, strict=True))
