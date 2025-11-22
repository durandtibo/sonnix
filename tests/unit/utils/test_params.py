from __future__ import annotations

import pytest
from torch import nn

from sonnix.utils.params import (
    freeze_module,
    has_learnable_parameters,
    has_parameters,
    num_learnable_parameters,
    num_parameters,
    unfreeze_module,
)

####################################
#     Tests for has_parameters     #
####################################


def test_has_parameters_true() -> None:
    assert has_parameters(nn.Linear(4, 5))


def test_has_parameters_false() -> None:
    assert not has_parameters(nn.Tanh())


##############################################
#     Tests for has_learnable_parameters     #
##############################################


def test_has_learnable_parameters_true() -> None:
    assert has_learnable_parameters(nn.Linear(4, 5))


def test_has_learnable_parameters_false() -> None:
    assert not has_learnable_parameters(nn.Tanh())


####################################
#     Tests for num_parameters     #
####################################


def test_num_parameters_0() -> None:
    assert num_parameters(nn.Tanh()) == 0


def test_num_parameters_15() -> None:
    assert num_parameters(nn.Linear(2, 5)) == 15  # 10 (weight) + 5 (bias)


def test_num_parameters_25() -> None:
    assert num_parameters(nn.Linear(4, 5)) == 25  # 20 (weight) + 5 (bias)


def test_num_parameters_25_frozen() -> None:
    module = nn.Linear(4, 5)
    freeze_module(module)
    assert num_parameters(module) == 25  # 20 (weight) + 5 (bias)


def test_num_parameters_2_layers() -> None:
    fc1 = nn.Linear(4, 5)
    fc2 = nn.Linear(5, 8)
    model = nn.Sequential(fc1, fc2)
    assert num_parameters(model) == 25 + 48
    # Freeze the parameters of FC2.
    freeze_module(fc2)
    assert num_parameters(model) == 25 + 48


##############################################
#     Tests for num_learnable_parameters     #
##############################################


def test_num_learnable_parameters_0() -> None:
    assert num_learnable_parameters(nn.Tanh()) == 0


def test_num_learnable_parameters_15() -> None:
    assert num_learnable_parameters(nn.Linear(2, 5)) == 15  # 10 (weight) + 5 (bias)


def test_num_learnable_parameters_25() -> None:
    assert num_learnable_parameters(nn.Linear(4, 5)) == 25  # 20 (weight) + 5 (bias)


def test_num_learnable_parameters_2_layers() -> None:
    fc1 = nn.Linear(4, 5)
    fc2 = nn.Linear(5, 8)
    model = nn.Sequential(fc1, fc2)
    assert num_learnable_parameters(model) == 25 + 48
    # Freeze the parameters of FC2.
    freeze_module(fc2)
    assert num_learnable_parameters(model) == 25  # 20 (weight) + 5 (bias)


###################################
#     Tests for freeze_module     #
###################################


@pytest.mark.parametrize("module", [nn.Tanh(), nn.Linear(2, 5), nn.Linear(4, 5)])
def test_freeze_module(module: nn.Module) -> None:
    freeze_module(module)
    assert num_learnable_parameters(module) == 0


#####################################
#     Tests for unfreeze_module     #
#####################################


def test_unfreeze_module_tanh() -> None:
    module = nn.Tanh()
    freeze_module(module)
    assert num_learnable_parameters(module) == 0
    unfreeze_module(module)
    assert num_learnable_parameters(module) == 0


def test_unfreeze_module_linear() -> None:
    module = nn.Linear(4, 5)
    freeze_module(module)
    assert num_learnable_parameters(module) == 0
    unfreeze_module(module)
    assert num_learnable_parameters(module) == 25  # 20 (weight) + 5 (bias)
