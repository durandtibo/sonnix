from __future__ import annotations

import torch
from torch import nn

from sonnix.testing import cuda_available
from sonnix.utils.device import (
    get_module_device,
    get_module_devices,
    is_module_on_device,
)

#######################################
#     Tests for get_module_device     #
#######################################


def test_get_module_device_cpu() -> None:
    assert get_module_device(nn.Linear(4, 5)) == torch.device("cpu")


@cuda_available
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


@cuda_available
def test_get_module_devices_cuda() -> None:
    assert get_module_devices(nn.Linear(4, 5).to(device=torch.device("cuda:0"))) == [
        torch.device("cuda:0")
    ]


@cuda_available
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
