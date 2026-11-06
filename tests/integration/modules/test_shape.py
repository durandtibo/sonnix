from __future__ import annotations

import torch
from coola.equality import objects_are_allclose
from torch import nn

from sonnix.modules import MulticlassFlatten

#######################################
#     Tests for MulticlassFlatten     #
#######################################


def test_multiclass_flatten_cross_entropy_non_contiguous_prediction() -> None:
    generator = torch.Generator().manual_seed(1)
    prediction = torch.randn(4, 6, 3, generator=generator).transpose(0, 1)
    target = torch.randint(0, 3, (6, 4), generator=generator)
    out = MulticlassFlatten(nn.CrossEntropyLoss())(prediction, target)
    expected = nn.functional.cross_entropy(prediction.reshape(24, 3), target.reshape(24))
    assert objects_are_allclose(out, expected)


def test_multiclass_flatten_cross_entropy_backward() -> None:
    prediction = torch.randn(
        2, 5, 3, generator=torch.Generator().manual_seed(2), requires_grad=True
    )
    MulticlassFlatten(nn.CrossEntropyLoss())(
        prediction, torch.zeros(2, 5, dtype=torch.long)
    ).backward()
    assert prediction.grad.isfinite().all()
