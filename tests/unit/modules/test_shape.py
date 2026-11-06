from __future__ import annotations

import pytest
import torch
from coola.equality import objects_are_equal
from coola.utils.tensor import get_available_devices

from sonnix.modules import Squeeze

#############################
#     Tests for Squeeze     #
#############################


def test_squeeze_str() -> None:
    assert str(Squeeze()).startswith("Squeeze(")


@pytest.mark.parametrize("device", get_available_devices())
def test_squeeze_forward_dim_none(device: str) -> None:
    device = torch.device(device)
    module = Squeeze().to(device=device)
    assert objects_are_equal(
        module(torch.ones(2, 1, 3, 1, 4, device=device)),
        torch.ones(2, 3, 4, device=device),
    )


@pytest.mark.parametrize("device", get_available_devices())
def test_squeeze_forward_dim_1(device: str) -> None:
    device = torch.device(device)
    module = Squeeze(dim=1).to(device=device)
    assert objects_are_equal(
        module(torch.ones(2, 1, 3, 1, 4, device=device)),
        torch.ones(2, 3, 1, 4, device=device),
    )


def test_squeeze_forward_dim_tuple() -> None:
    assert objects_are_equal(Squeeze(dim=(1, 3))(torch.ones(2, 1, 3, 1, 4)), torch.ones(2, 3, 4))


def test_squeeze_forward_dim_list() -> None:
    assert objects_are_equal(Squeeze(dim=[1, 3])(torch.ones(2, 1, 3, 1, 4)), torch.ones(2, 3, 4))


def test_squeeze_forward_dim_tuple_skips_non_singleton() -> None:
    assert Squeeze(dim=(1, 2))(torch.ones(2, 1, 3)).shape == (2, 3)


def test_squeeze_dim_list_is_converted_to_tuple() -> None:
    assert Squeeze(dim=[1, 3]).dim == (1, 3)


def test_squeeze_dim_int() -> None:
    assert Squeeze(dim=1).dim == 1


def test_squeeze_extra_repr_dim_tuple() -> None:
    assert Squeeze(dim=(1, 3)).extra_repr() == "dim=(1, 3)"
