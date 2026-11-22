from __future__ import annotations

import torch
from coola.testing.marks import torch_cuda_available
from torch import nn

from sonnix.utils.device import (
    _iter_tensors,
    get_module_device,
    get_module_devices,
    is_module_on_device,
)


class BufferModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("value", torch.ones(2))


#######################################
#     Tests for get_module_device     #
#######################################


def test_get_module_device_cpu() -> None:
    assert get_module_device(nn.Linear(4, 5)) == torch.device("cpu")


@torch_cuda_available
def test_get_module_device_cuda() -> None:
    assert get_module_device(nn.Linear(4, 5).to(device=torch.device("cuda:0"))) == torch.device(
        "cuda:0"
    )


def test_get_module_device_no_parameter() -> None:
    assert get_module_device(nn.Identity()) == torch.device("cpu")


########################################
#     Tests for get_module_devices     #
########################################


def test_get_module_devices_cpu() -> None:
    assert get_module_devices(nn.Linear(4, 5)) == [torch.device("cpu")]


@torch_cuda_available
def test_get_module_devices_cuda() -> None:
    assert get_module_devices(nn.Linear(4, 5).to(device=torch.device("cuda:0"))) == [
        torch.device("cuda:0")
    ]


def test_get_module_devices_sorted_by_name() -> None:
    net = nn.Sequential(nn.Linear(4, 5).to(device="meta"), nn.Linear(4, 5))
    assert get_module_devices(net) == [torch.device("cpu"), torch.device("meta")]


def test_get_module_devices_unique() -> None:
    assert get_module_devices(nn.Sequential(nn.Linear(4, 5), nn.Linear(5, 4))) == [
        torch.device("cpu")
    ]


def test_get_module_devices_no_parameter() -> None:
    assert get_module_devices(nn.Identity()) == []


@torch_cuda_available
def test_get_module_devices_cpu_cuda() -> None:
    net = nn.Sequential(nn.Linear(4, 5), nn.Linear(4, 5).to(device=torch.device("cuda:0")))
    assert set(get_module_devices(net)) == {torch.device("cpu"), torch.device("cuda:0")}


#########################################
#     Tests for is_module_on_device     #
#########################################


def test_is_module_on_device_true() -> None:
    assert is_module_on_device(nn.Linear(4, 5), torch.device("cpu"))


def test_is_module_on_device_false() -> None:
    assert not is_module_on_device(nn.Linear(4, 5), torch.device("cuda:0"))


def test_get_module_device_buffer_only() -> None:
    assert get_module_device(BufferModule().to("meta")) == torch.device("meta")


def test_get_module_devices_buffer_only() -> None:
    assert get_module_devices(BufferModule().to("meta")) == [torch.device("meta")]


def test_get_module_devices_parameter_and_buffer() -> None:
    module = nn.Sequential(nn.Linear(4, 5), BufferModule().to("meta"))
    assert get_module_devices(module) == [torch.device("cpu"), torch.device("meta")]


def test_is_module_on_device_buffer_only_true() -> None:
    assert is_module_on_device(BufferModule(), torch.device("cpu"))


def test_is_module_on_device_buffer_only_false() -> None:
    assert not is_module_on_device(BufferModule().to("meta"), torch.device("cpu"))


def test_is_module_on_device_buffer_on_other_device() -> None:
    module = nn.Sequential(nn.Linear(4, 5), BufferModule().to("meta"))
    assert not is_module_on_device(module, torch.device("cpu"))


#################################
#     Tests for _iter_tensors     #
#################################


def test_iter_tensors_parameters_then_buffers() -> None:
    module = nn.Sequential(BufferModule(), nn.Linear(2, 1))
    assert len(list(_iter_tensors(module))) == 3


def test_iter_tensors_empty() -> None:
    assert list(_iter_tensors(nn.ReLU())) == []
