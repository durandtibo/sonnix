from __future__ import annotations

import pytest
import torch

from sonnix.functional.reduction import VALID_REDUCTIONS
from sonnix.modules.loss import BaseLoss


class MyLoss(BaseLoss):
    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return prediction.sub(target)


##############################
#     Tests for BaseLoss     #
##############################


def test_base_loss_reduction_default() -> None:
    assert MyLoss().reduction == "mean"


@pytest.mark.parametrize("reduction", VALID_REDUCTIONS)
def test_base_loss_reduction(reduction: str) -> None:
    assert MyLoss(reduction=reduction).reduction == reduction


def test_base_loss_reduction_incorrect() -> None:
    with pytest.raises(ValueError, match=r"reduction must be one of"):
        MyLoss(reduction="incorrect")


def test_base_loss_is_module() -> None:
    assert isinstance(MyLoss(), torch.nn.Module)
