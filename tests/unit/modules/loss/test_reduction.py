r"""Test the ``reduction`` argument of every loss module that has one.

A loss module is covered by registering it in ``tests/registry.py``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from sonnix.modules import BaseLoss
from tests.registry import MODULE_CASES, ModuleCase

if TYPE_CHECKING:
    import torch

LOSS_CASES = [case for case in MODULE_CASES if isinstance(case.build(), BaseLoss)]


def _loss(case: ModuleCase, reduction: str) -> torch.Tensor:
    return case.build(reduction=reduction)(*case.make_inputs())


@pytest.mark.parametrize("case", LOSS_CASES, ids=repr)
def test_loss_reduction_incorrect(case: ModuleCase) -> None:
    with pytest.raises(ValueError, match=r"reduction must be one of"):
        case.build(reduction="incorrect")


@pytest.mark.parametrize("reduction", ["none", "mean", "sum", "batchmean"])
@pytest.mark.parametrize("case", LOSS_CASES, ids=repr)
def test_loss_reduction_is_stored(case: ModuleCase, reduction: str) -> None:
    assert case.build(reduction=reduction).reduction == reduction


@pytest.mark.parametrize("case", LOSS_CASES, ids=repr)
def test_loss_default_reduction_is_mean(case: ModuleCase) -> None:
    assert case.build().reduction == "mean"


@pytest.mark.parametrize("case", LOSS_CASES, ids=repr)
def test_loss_repr_contains_reduction(case: ModuleCase) -> None:
    assert "reduction=mean" in repr(case.build())


@pytest.mark.parametrize("case", LOSS_CASES, ids=repr)
def test_loss_reduction_none_keeps_the_shape(case: ModuleCase) -> None:
    assert _loss(case, "none").shape == case.make_inputs()[0].shape


@pytest.mark.parametrize("case", LOSS_CASES, ids=repr)
def test_loss_reduction_sum_is_the_sum_of_none(case: ModuleCase) -> None:
    assert _loss(case, "sum").allclose(_loss(case, "none").sum())


@pytest.mark.parametrize("case", LOSS_CASES, ids=repr)
def test_loss_reduction_mean_is_the_mean_of_none(case: ModuleCase) -> None:
    assert _loss(case, "mean").allclose(_loss(case, "none").mean())


@pytest.mark.parametrize("case", LOSS_CASES, ids=repr)
def test_loss_reduction_batchmean_is_the_sum_divided_by_the_batch_size(case: ModuleCase) -> None:
    batch_size = case.make_inputs()[0].shape[0]
    assert _loss(case, "batchmean").allclose(_loss(case, "none").sum() / batch_size)


@pytest.mark.parametrize("case", LOSS_CASES, ids=repr)
def test_loss_is_deterministic(case: ModuleCase) -> None:
    criterion = case.build()
    inputs = case.make_inputs()
    assert criterion(*inputs).equal(criterion(*inputs))
