from __future__ import annotations

import pytest
import torch
from coola import objects_are_equal
from coola.utils.tensor import get_available_devices

from sonnix.modules import ConcatFusion

SIZES = (1, 2, 3)

##################################
#     Tests for ConcatFusion     #
##################################


def test_concat_fusion_str() -> None:
    assert str(ConcatFusion()).startswith("ConcatFusion(")


def test_concat_fusion_forward_0_input() -> None:
    module = ConcatFusion()
    with pytest.raises(RuntimeError, match=r"ConcatFusion needs at least one tensor as input"):
        module()


@pytest.mark.parametrize("device", get_available_devices())
def test_concat_fusion_forward(device: str) -> None:
    device = torch.device(device)
    net = ConcatFusion().to(device=device)
    out = net(
        torch.tensor([[2, 3, 4], [5, 6, 7]], device=device),
        torch.tensor([[12, 13, 14], [15, 16, 17]], device=device),
    )
    assert objects_are_equal(
        out, torch.tensor([[2, 3, 4, 12, 13, 14], [5, 6, 7, 15, 16, 17]], device=device)
    )


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("batch_size", SIZES)
def test_concat_fusion_forward_1_input(device: str, batch_size: int) -> None:
    device = torch.device(device)
    net = ConcatFusion().to(device=device)
    out = net(torch.ones(batch_size, 3, device=device))
    assert objects_are_equal(out, torch.ones(batch_size, 3, device=device))


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("batch_size", SIZES)
def test_concat_fusion_forward_2_inputs(device: str, batch_size: int) -> None:
    device = torch.device(device)
    net = ConcatFusion().to(device=device)
    out = net(torch.ones(batch_size, 3, device=device), torch.ones(batch_size, 4, device=device))
    assert objects_are_equal(out, torch.ones(batch_size, 7, device=device))


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("batch_size", SIZES)
def test_concat_fusion_forward_3_inputs(device: str, batch_size: int) -> None:
    device = torch.device(device)
    net = ConcatFusion().to(device=device)
    out = net(
        torch.ones(batch_size, 3, device=device),
        torch.ones(batch_size, 4, device=device),
        torch.ones(batch_size, 5, device=device),
    )
    assert objects_are_equal(out, torch.ones(batch_size, 12, device=device))


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("seq_len", SIZES)
@pytest.mark.parametrize("batch_size", SIZES)
def test_concat_fusion_forward_3d_inputs(device: str, seq_len: int, batch_size: int) -> None:
    device = torch.device(device)
    net = ConcatFusion().to(device=device)
    out = net(
        torch.ones(batch_size, seq_len, 3, device=device),
        torch.ones(batch_size, seq_len, 4, device=device),
    )
    assert objects_are_equal(out, torch.ones(batch_size, seq_len, 7, device=device))


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("seq_len", SIZES)
@pytest.mark.parametrize("batch_size", SIZES)
@pytest.mark.parametrize("input_size", SIZES)
def test_concat_fusion_forward_dim_0(
    device: str, seq_len: int, batch_size: int, input_size: int
) -> None:
    device = torch.device(device)
    net = ConcatFusion(dim=0).to(device=device)
    out = net(
        torch.ones(seq_len, batch_size, input_size, device=device),
        torch.ones(seq_len, batch_size, input_size, device=device),
    )
    assert objects_are_equal(out, torch.ones(2 * seq_len, batch_size, input_size, device=device))


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("seq_len", SIZES)
@pytest.mark.parametrize("batch_size", SIZES)
@pytest.mark.parametrize("input_size", SIZES)
def test_concat_fusion_forward_dim_1(
    device: str, seq_len: int, batch_size: int, input_size: int
) -> None:
    device = torch.device(device)
    net = ConcatFusion(dim=1).to(device=device)
    out = net(
        torch.ones(batch_size, seq_len, input_size, device=device),
        torch.ones(batch_size, seq_len, input_size, device=device),
    )
    assert objects_are_equal(out, torch.ones(batch_size, 2 * seq_len, input_size, device=device))


@pytest.mark.parametrize("device", get_available_devices())
def test_concat_fusion_backward(device: str) -> None:
    device = torch.device(device)
    module = ConcatFusion().to(device=device)
    out = module(
        torch.ones(2, 4, device=device, requires_grad=True),
        torch.ones(2, 4, device=device, requires_grad=True),
        torch.ones(2, 4, device=device, requires_grad=True),
    )
    out.mean().backward()
    assert out.equal(torch.ones(2, 12, device=device))
