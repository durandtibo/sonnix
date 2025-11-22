from __future__ import annotations

import pytest
from torch import nn

from sonnix.utils.iterator import get_named_modules


@pytest.fixture
def nested_module() -> nn.Module:
    return nn.Sequential(
        nn.Linear(4, 6),
        nn.ReLU(),
        nn.Sequential(
            nn.Linear(6, 6),
            nn.Dropout(0.1),
            nn.Sequential(nn.Linear(6, 6), nn.PReLU()),
            nn.Linear(6, 3),
        ),
    )


#######################################
#     Tests for get_named_modules     #
#######################################


@pytest.mark.parametrize("depth", [-1, 0, 1, 2, 3])
def test_get_named_modules_depth_linear(depth: int) -> None:
    linear = nn.Linear(4, 6)
    named_modules = list(get_named_modules(linear, depth=depth))
    assert named_modules == [("[root]", linear)]


def test_get_named_modules_depth_0_sequential(nested_module: nn.Module) -> None:
    named_modules = list(get_named_modules(nested_module))
    assert named_modules == [("[root]", nested_module)]


def test_get_named_modules_depth_1_sequential(nested_module: nn.Module) -> None:
    named_modules = list(get_named_modules(nested_module, depth=1))
    assert named_modules == [
        ("[root]", nested_module),
        ("0", nested_module[0]),
        ("1", nested_module[1]),
        ("2", nested_module[2]),
    ]


def test_get_named_modules_depth_2_sequential(nested_module: nn.Module) -> None:
    named_modules = list(get_named_modules(nested_module, depth=2))
    assert named_modules == [
        ("[root]", nested_module),
        ("0", nested_module[0]),
        ("1", nested_module[1]),
        ("2", nested_module[2]),
        ("2.0", nested_module[2][0]),
        ("2.1", nested_module[2][1]),
        ("2.2", nested_module[2][2]),
        ("2.3", nested_module[2][3]),
    ]
