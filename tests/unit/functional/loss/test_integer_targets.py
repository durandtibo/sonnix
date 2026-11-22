r"""Test that the binary losses accept integer and ``bool`` targets, like
the labels of a ``DataLoader``."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import torch
from coola.equality import objects_are_equal

from sonnix.functional import (
    binary_focal_loss,
    binary_focal_loss_with_logits,
    binary_poly1_loss,
    binary_poly1_loss_with_logits,
)
from tests.helpers import make_generator

if TYPE_CHECKING:
    from collections.abc import Callable

PROBABILITY_LOSSES = [binary_focal_loss, binary_poly1_loss]
LOGIT_LOSSES = [binary_focal_loss_with_logits, binary_poly1_loss_with_logits]
LOSSES = [*PROBABILITY_LOSSES, *LOGIT_LOSSES]

LABELS = torch.tensor([[1, 0, 1], [0, 1, 0]])
PREDICTION = torch.rand(2, 3, generator=make_generator()) * 0.8 + 0.1

#################################################
#     Tests for the integer and bool target     #
#################################################


@pytest.mark.parametrize("loss", LOSSES)
@pytest.mark.parametrize("dtype", [torch.long, torch.int32, torch.uint8, torch.bool])
def test_integer_target_gives_the_loss_of_the_float_target(
    loss: Callable, dtype: torch.dtype
) -> None:
    assert objects_are_equal(
        loss(PREDICTION, LABELS.to(dtype)), loss(PREDICTION, LABELS.to(torch.float))
    )


@pytest.mark.parametrize("loss", LOSSES)
def test_integer_target_reduction_none_has_the_shape_of_the_prediction(loss: Callable) -> None:
    assert loss(PREDICTION, LABELS, reduction="none").shape == (2, 3)


@pytest.mark.parametrize("loss", LOSSES)
def test_integer_target_gradient_is_finite(loss: Callable) -> None:
    prediction = PREDICTION.clone().requires_grad_()
    loss(prediction, LABELS).backward()
    assert prediction.grad.isfinite().all()


@pytest.mark.parametrize("loss", LOSSES)
def test_integer_target_float64_prediction(loss: Callable) -> None:
    assert loss(PREDICTION.double(), LABELS).dtype == torch.float64


@pytest.mark.parametrize("loss", LOSSES)
def test_integer_target_does_not_modify_the_target(loss: Callable) -> None:
    target = LABELS.clone()
    loss(PREDICTION, target)
    assert objects_are_equal(target, LABELS)


@pytest.mark.parametrize("loss", LOSSES)
def test_integer_target_shape_mismatch(loss: Callable) -> None:
    with pytest.raises(ValueError, match=r"prediction and target must have the same shape"):
        loss(PREDICTION, LABELS[:1])
