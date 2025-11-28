from __future__ import annotations

import pytest
import torch
from coola import objects_are_equal
from coola.utils.tensor import get_available_devices

from sonnix.modules import ToFloat, ToLong

#############################
#     Tests for ToFloat     #
#############################


@pytest.mark.parametrize("device", get_available_devices())
def test_to_float_forward(device: str) -> None:
    device = torch.device(device)
    module = ToFloat().to(device=device)
    output = module(torch.tensor([1, 2, 3, 4], dtype=torch.long, device=device))
    assert objects_are_equal(
        output, torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float, device=device)
    )


############################
#     Tests for ToLong     #
############################


@pytest.mark.parametrize("device", get_available_devices())
def test_to_long_forward(device: str) -> None:
    device = torch.device(device)
    module = ToLong().to(device=device)
    output = module(torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float, device=device))
    assert objects_are_equal(output, torch.tensor([1, 2, 3, 4], dtype=torch.long, device=device))
