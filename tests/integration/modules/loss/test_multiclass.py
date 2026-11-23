from __future__ import annotations

import pytest
import torch
from coola.equality import objects_are_allclose
from torch import nn

from sonnix.modules import MulticlassFlatten
from tests.helpers import make_generator

#######################################
#     Tests for MulticlassFlatten     #
#######################################


def test_multiclass_flatten_cross_entropy_non_contiguous_prediction() -> None:
    generator = make_generator(1)
    prediction = torch.randn(4, 6, 3, generator=generator).transpose(0, 1)
    target = torch.randint(0, 3, (6, 4), generator=generator)
    out = MulticlassFlatten(nn.CrossEntropyLoss())(prediction, target)
    expected = nn.functional.cross_entropy(prediction.reshape(24, 3), target.reshape(24))
    assert objects_are_allclose(out, expected)


def test_multiclass_flatten_cross_entropy_backward() -> None:
    prediction = torch.randn(2, 5, 3, generator=make_generator(2), requires_grad=True)
    MulticlassFlatten(nn.CrossEntropyLoss())(
        prediction, torch.zeros(2, 5, dtype=torch.long)
    ).backward()
    assert prediction.grad.isfinite().all()


def test_multiclass_flatten_cross_entropy_rows_of_different_shapes_raise() -> None:
    # The same number of elements: the loss would pair unrelated rows.
    with pytest.raises(ValueError, match=r"got \(2, 6, 3\) and \(4, 3\)"):
        MulticlassFlatten(nn.CrossEntropyLoss())(torch.randn(2, 6, 3), torch.zeros(4, 3).long())


def test_multiclass_flatten_cross_entropy_missing_class_dimension_raises() -> None:
    with pytest.raises(ValueError, match=r"got \(4, 3, 5\) and \(4, 2\)"):
        MulticlassFlatten(nn.CrossEntropyLoss())(torch.randn(4, 3, 5), torch.zeros(4, 2).long())
