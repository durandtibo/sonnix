from __future__ import annotations

import math

import pytest
import torch
from coola.equality import objects_are_allclose, objects_are_equal
from coola.utils.tensor import get_available_devices

from sonnix.modules import (
    BaseAlphaActivation,
    ExpSin,
    Gaussian,
    Laplacian,
    MultiQuadratic,
    Quadratic,
    Snake,
)

SIZES = (1, 2, 3)
SHAPES = [(2,), (2, 3), (2, 3, 4), (2, 3, 4, 5)]

############################
#     Tests for ExpSin     #
############################


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("batch_size", SIZES)
@pytest.mark.parametrize("feature_size", SIZES)
def test_exp_sin_forward(device: str, batch_size: int, feature_size: int) -> None:
    device = torch.device(device)
    module = ExpSin(num_parameters=feature_size).to(device=device)
    out = module(torch.randn(batch_size, feature_size, device=device, requires_grad=True))
    out.mean().backward()
    assert out.shape == (batch_size, feature_size)
    assert out.dtype == torch.float
    assert out.device == device


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("shape", SHAPES)
def test_exp_sin_forward_shape(device: str, shape: tuple[int, ...]) -> None:
    device = torch.device(device)
    module = ExpSin().to(device=device)
    assert objects_are_equal(
        module(torch.zeros(*shape, dtype=torch.float, device=device)),
        torch.ones(*shape, dtype=torch.float, device=device),
    )


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("num_parameters", [1, 4])
def test_exp_sin_forward_num_parameters(device: str, num_parameters: int) -> None:
    device = torch.device(device)
    module = ExpSin(num_parameters).to(device=device)
    assert objects_are_equal(
        module(torch.zeros(2, 4, dtype=torch.float, device=device)),
        torch.ones(2, 4, dtype=torch.float, device=device),
    )


@pytest.mark.parametrize("device", get_available_devices())
def test_exp_sin_forward_init_1(device: str) -> None:
    device = torch.device(device)
    module = ExpSin().to(device=device)
    assert objects_are_allclose(
        module(torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=torch.float, device=device)),
        torch.tensor(
            [2.4825777280150008, 2.319776824715853, 1.0, 0.43107595064559234, 0.40280712612352804],
            dtype=torch.float,
            device=device,
        ),
    )


@pytest.mark.parametrize("device", get_available_devices())
def test_exp_sin_forward_init_2(device: str) -> None:
    device = torch.device(device)
    module = ExpSin(init=2).to(device=device)
    assert objects_are_allclose(
        module(torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=torch.float, device=device)),
        torch.tensor(
            [0.46916418587400077, 2.4825777280150008, 1.0, 0.40280712612352804, 2.1314499915144016],
            dtype=torch.float,
            device=device,
        ),
    )


##############################
#     Tests for Gaussian     #
##############################


def test_gaussian_float16_large_input_alpha_gradient_is_finite() -> None:
    # exp(-0.5 * (x / alpha) ** 2) is 0 for a large x, and so is its gradient.
    module = Gaussian(dtype=torch.float16)
    module(torch.tensor([1000.0], dtype=torch.float16)).sum().backward()
    assert module.alpha.grad.equal(torch.zeros(1, dtype=torch.float16))


def test_gaussian_float16_large_input_is_zero() -> None:
    assert Gaussian(dtype=torch.float16)(torch.tensor([1000.0], dtype=torch.float16)).equal(
        torch.zeros(1, dtype=torch.float16)
    )


def test_gaussian_forward_alpha_gradient_value() -> None:
    # d/dalpha exp(-0.5 * (x / alpha)^2) at x=1, alpha=1 is exp(-0.5)
    module = Gaussian()
    module(torch.ones(1)).sum().backward()
    assert module.alpha.grad.allclose(torch.tensor([0.6065306597126334]))


def test_gaussian_str() -> None:
    assert str(Gaussian()).startswith("Gaussian(")


@pytest.mark.parametrize("num_parameters", SIZES)
def test_gaussian_num_parameters(num_parameters: int) -> None:
    assert Gaussian(num_parameters).alpha.shape == (num_parameters,)


def test_gaussian_num_parameters_default() -> None:
    assert Gaussian().alpha.shape == (1,)


@pytest.mark.parametrize("init", [0.5, 1.0])
def test_gaussian_init(init: float) -> None:
    assert Gaussian(init=init).alpha.item() == init


def test_gaussian_learnable_true() -> None:
    assert Gaussian().alpha.requires_grad


def test_gaussian_learnable_false() -> None:
    assert not Gaussian(learnable=False).alpha.requires_grad


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("batch_size", SIZES)
@pytest.mark.parametrize("feature_size", SIZES)
def test_gaussian_forward(device: str, batch_size: int, feature_size: int) -> None:
    device = torch.device(device)
    module = Gaussian(num_parameters=feature_size).to(device=device)
    out = module(torch.randn(batch_size, feature_size, device=device, requires_grad=True))
    out.mean().backward()
    assert out.shape == (batch_size, feature_size)
    assert out.dtype == torch.float
    assert out.device == device


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("shape", SHAPES)
def test_gaussian_forward_shape(device: str, shape: tuple[int, ...]) -> None:
    device = torch.device(device)
    module = Gaussian().to(device=device)
    assert objects_are_equal(
        module(torch.zeros(*shape, dtype=torch.float, device=device)),
        torch.ones(*shape, dtype=torch.float, device=device),
    )


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("num_parameters", [1, 4])
def test_gaussian_forward_num_parameters(device: str, num_parameters: int) -> None:
    device = torch.device(device)
    module = Gaussian(num_parameters).to(device=device)
    assert objects_are_equal(
        module(torch.zeros(2, 4, dtype=torch.float, device=device)),
        torch.ones(2, 4, dtype=torch.float, device=device),
    )


@pytest.mark.parametrize("device", get_available_devices())
def test_gaussian_forward_init_1(device: str) -> None:
    device = torch.device(device)
    module = Gaussian().to(device=device)
    assert objects_are_allclose(
        module(torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=torch.float, device=device)),
        torch.tensor(
            [0.1353352832366127, 0.6065306597126334, 1.0, 0.6065306597126334, 0.1353352832366127],
            dtype=torch.float,
            device=device,
        ),
    )


@pytest.mark.parametrize("device", get_available_devices())
def test_gaussian_forward_init_2(device: str) -> None:
    device = torch.device(device)
    module = Gaussian(init=2).to(device=device)
    assert objects_are_allclose(
        module(torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=torch.float, device=device)),
        torch.tensor(
            [0.6065306597126334, 0.8824969025845955, 1.0, 0.8824969025845955, 0.6065306597126334],
            dtype=torch.float,
            device=device,
        ),
    )


###############################
#     Tests for Laplacian     #
###############################


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("batch_size", SIZES)
@pytest.mark.parametrize("feature_size", SIZES)
def test_laplacian_forward(device: str, batch_size: int, feature_size: int) -> None:
    device = torch.device(device)
    module = Laplacian(num_parameters=feature_size).to(device=device)
    out = module(torch.randn(batch_size, feature_size, device=device, requires_grad=True))
    out.mean().backward()
    assert out.shape == (batch_size, feature_size)
    assert out.dtype == torch.float
    assert out.device == device


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("shape", SHAPES)
def test_laplacian_forward_shape(device: str, shape: tuple[int, ...]) -> None:
    device = torch.device(device)
    module = Laplacian().to(device=device)
    assert objects_are_equal(
        module(torch.zeros(*shape, dtype=torch.float, device=device)),
        torch.ones(*shape, dtype=torch.float, device=device),
    )


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("num_parameters", [1, 4])
def test_laplacian_forward_num_parameters(device: str, num_parameters: int) -> None:
    device = torch.device(device)
    module = Laplacian(num_parameters).to(device=device)
    assert objects_are_equal(
        module(torch.zeros(2, 4, dtype=torch.float, device=device)),
        torch.ones(2, 4, dtype=torch.float, device=device),
    )


@pytest.mark.parametrize("device", get_available_devices())
def test_laplacian_forward_init_1(device: str) -> None:
    device = torch.device(device)
    module = Laplacian().to(device=device)
    assert objects_are_allclose(
        module(torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=torch.float, device=device)),
        torch.tensor(
            [0.1353352832366127, 0.36787944117144233, 1.0, 0.36787944117144233, 0.1353352832366127],
            dtype=torch.float,
            device=device,
        ),
    )


@pytest.mark.parametrize("device", get_available_devices())
def test_laplacian_forward_init_2(device: str) -> None:
    device = torch.device(device)
    module = Laplacian(init=2).to(device=device)
    assert objects_are_allclose(
        module(torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=torch.float, device=device)),
        torch.tensor(
            [0.36787944117144233, 0.6065306597126334, 1.0, 0.6065306597126334, 0.36787944117144233],
            dtype=torch.float,
            device=device,
        ),
    )


####################################
#     Tests for MultiQuadratic     #
####################################


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("batch_size", SIZES)
@pytest.mark.parametrize("feature_size", SIZES)
def test_multi_quadratic_forward(device: str, batch_size: int, feature_size: int) -> None:
    device = torch.device(device)
    module = MultiQuadratic(num_parameters=feature_size).to(device=device)
    out = module(torch.randn(batch_size, feature_size, device=device, requires_grad=True))
    out.mean().backward()
    assert out.shape == (batch_size, feature_size)
    assert out.dtype == torch.float
    assert out.device == device


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("shape", SHAPES)
def test_multi_quadratic_forward_shape(device: str, shape: tuple[int, ...]) -> None:
    device = torch.device(device)
    module = MultiQuadratic().to(device=device)
    assert objects_are_equal(
        module(torch.zeros(*shape, dtype=torch.float, device=device)),
        torch.ones(*shape, dtype=torch.float, device=device),
    )


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("num_parameters", [1, 4])
def test_multi_quadratic_forward_num_parameters(device: str, num_parameters: int) -> None:
    device = torch.device(device)
    module = MultiQuadratic(num_parameters).to(device=device)
    assert objects_are_equal(
        module(torch.zeros(2, 4, dtype=torch.float, device=device)),
        torch.ones(2, 4, dtype=torch.float, device=device),
    )


@pytest.mark.parametrize("device", get_available_devices())
def test_multi_quadratic_forward_init_1(device: str) -> None:
    device = torch.device(device)
    module = MultiQuadratic().to(device=device)
    assert objects_are_allclose(
        module(torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=torch.float, device=device)),
        torch.tensor(
            [0.4472135954999579, 0.7071067811865475, 1.0, 0.7071067811865475, 0.4472135954999579],
            dtype=torch.float,
            device=device,
        ),
    )


@pytest.mark.parametrize("device", get_available_devices())
def test_multi_quadratic_forward_init_2(device: str) -> None:
    device = torch.device(device)
    module = MultiQuadratic(init=2).to(device=device)
    assert objects_are_allclose(
        module(torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=torch.float, device=device)),
        torch.tensor(
            [0.24253562503633297, 0.4472135954999579, 1.0, 0.4472135954999579, 0.24253562503633297],
            dtype=torch.float,
            device=device,
        ),
    )


###############################
#     Tests for Quadratic     #
###############################


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("batch_size", SIZES)
@pytest.mark.parametrize("feature_size", SIZES)
def test_quadratic_forward(device: str, batch_size: int, feature_size: int) -> None:
    device = torch.device(device)
    module = Quadratic(num_parameters=feature_size).to(device=device)
    out = module(torch.randn(batch_size, feature_size, device=device, requires_grad=True))
    out.mean().backward()
    assert out.shape == (batch_size, feature_size)
    assert out.dtype == torch.float
    assert out.device == device


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("shape", SHAPES)
def test_quadratic_forward_shape(device: str, shape: tuple[int, ...]) -> None:
    device = torch.device(device)
    module = Quadratic().to(device=device)
    assert objects_are_equal(
        module(torch.zeros(*shape, dtype=torch.float, device=device)),
        torch.ones(*shape, dtype=torch.float, device=device),
    )


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("num_parameters", [1, 4])
def test_quadratic_forward_num_parameters(device: str, num_parameters: int) -> None:
    device = torch.device(device)
    module = Quadratic(num_parameters).to(device=device)
    assert objects_are_equal(
        module(torch.zeros(2, 4, dtype=torch.float, device=device)),
        torch.ones(2, 4, dtype=torch.float, device=device),
    )


@pytest.mark.parametrize("device", get_available_devices())
def test_quadratic_forward_init_1(device: str) -> None:
    device = torch.device(device)
    module = Quadratic().to(device=device)
    assert objects_are_allclose(
        module(torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=torch.float, device=device)),
        torch.tensor([0.2, 0.5, 1.0, 0.5, 0.2], dtype=torch.float, device=device),
    )


@pytest.mark.parametrize("device", get_available_devices())
def test_quadratic_forward_init_2(device: str) -> None:
    device = torch.device(device)
    module = Quadratic(init=2).to(device=device)
    assert objects_are_allclose(
        module(torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=torch.float, device=device)),
        torch.tensor(
            [0.058823529411764705, 0.2, 1.0, 0.2, 0.058823529411764705],
            dtype=torch.float,
            device=device,
        ),
    )


def test_gaussian_frozen_alpha_is_buffer_learnable() -> None:
    module = Gaussian(learnable=False)
    assert not module.learnable


def test_gaussian_frozen_alpha_is_buffer_check2() -> None:
    module = Gaussian(learnable=False)
    assert len(list(module.parameters())) == 0


def test_gaussian_frozen_alpha_is_buffer_check3() -> None:
    module = Gaussian(learnable=False)
    assert "alpha" in module.state_dict()


def test_gaussian_learnable_alpha_is_parameter_learnable() -> None:
    module = Gaussian()
    assert module.learnable


def test_gaussian_learnable_alpha_is_parameter_check2() -> None:
    module = Gaussian()
    assert [n for n, _ in module.named_parameters()] == ["alpha"]


@pytest.mark.parametrize("dtype", [torch.float64, torch.bfloat16])
def test_gaussian_dtype(dtype: torch.dtype) -> None:
    module = Gaussian(dtype=dtype)
    assert module.alpha.dtype == dtype
    assert module(torch.ones(2, 3, dtype=dtype)).dtype == dtype


@pytest.mark.parametrize("module", [ExpSin, Gaussian, Laplacian, MultiQuadratic, Quadratic])
def test_alpha_activation_invalid_num_parameters(module: type[torch.nn.Module]) -> None:
    with pytest.raises(ValueError, match=r"num_parameters must be greater than or equal to 1"):
        module(num_parameters=0)


ALPHA_ACTIVATIONS = [ExpSin, Gaussian, Laplacian, MultiQuadratic, Quadratic, Snake]


@pytest.mark.parametrize("module", [Gaussian, Snake])
def test_alpha_activation_zero_init_is_rejected(module: type[torch.nn.Module]) -> None:
    with pytest.raises(ValueError, match=r"init must be different from 0, got 0"):
        module(init=0)


def test_laplacian_zero_init_is_rejected() -> None:
    with pytest.raises(ValueError, match=r"init must be greater than 0"):
        Laplacian(init=0)


@pytest.mark.parametrize("module", [ExpSin, MultiQuadratic, Quadratic])
def test_alpha_activation_zero_init_is_accepted(module: type[torch.nn.Module]) -> None:
    assert module(init=0).alpha.item() == 0.0


@pytest.mark.parametrize("module", ALPHA_ACTIVATIONS)
def test_alpha_activation_channel_dim_default(module: type[torch.nn.Module]) -> None:
    assert module().channel_dim == -1


@pytest.mark.parametrize("module", ALPHA_ACTIVATIONS)
def test_alpha_activation_channel_dim_1(module: type[torch.nn.Module]) -> None:
    out = module(num_parameters=3, channel_dim=1)(torch.randn(2, 3, 5))
    assert out.shape == (2, 3, 5)


@pytest.mark.parametrize("module", ALPHA_ACTIVATIONS)
def test_alpha_activation_channel_dim_last_is_the_default_broadcast(
    module: type[torch.nn.Module],
) -> None:
    with pytest.raises(ValueError, match=r"channel_dim=-1.*num_parameters=3.*\(2, 3, 5\)"):
        module(num_parameters=3)(torch.randn(2, 3, 5))


@pytest.mark.parametrize("module", [Gaussian, MultiQuadratic, Quadratic])
def test_alpha_activation_channel_dim_matches_manual_broadcast(
    module: type[torch.nn.Module],
) -> None:
    x = torch.randn(2, 3, 5)
    channel_first = module(num_parameters=3, init=2.0, channel_dim=1)
    channel_last = module(num_parameters=3, init=2.0)
    with torch.no_grad():
        channel_first.alpha.copy_(torch.tensor([1.0, 2.0, 3.0]))
        channel_last.alpha.copy_(torch.tensor([1.0, 2.0, 3.0]))
    assert channel_first(x).allclose(channel_last(x.transpose(1, 2)).transpose(1, 2))


def test_alpha_activation_channel_dim_negative() -> None:
    x = torch.randn(2, 3, 5)
    assert Gaussian(num_parameters=3, channel_dim=-2)(x).allclose(
        Gaussian(num_parameters=3, channel_dim=1)(x)
    )


def test_alpha_activation_channel_dim_invalid_type() -> None:
    with pytest.raises(TypeError, match=r"channel_dim"):
        Gaussian(channel_dim=1.5)


def test_alpha_activation_extra_repr_channel_dim() -> None:
    assert (
        Gaussian(num_parameters=3, channel_dim=1).extra_repr()
        == "num_parameters=3, learnable=True, channel_dim=1"
    )


def test_alpha_activation_extra_repr_single_parameter() -> None:
    assert Gaussian(channel_dim=1).extra_repr() == "num_parameters=1, learnable=True"


def test_alpha_activation_broadcast_alpha_single_parameter() -> None:
    module = Gaussian(init=2.0)
    assert module._broadcast_alpha(torch.ones(2, 3, 4)).equal(module.alpha)


def test_alpha_activation_broadcast_alpha_channel_dim_1() -> None:
    module = Gaussian(num_parameters=3, channel_dim=1)
    assert module._broadcast_alpha(torch.ones(2, 3, 4)).shape == (1, 3, 1)


def test_alpha_activation_broadcast_alpha_channel_dim_last() -> None:
    module = Gaussian(num_parameters=4)
    assert module._broadcast_alpha(torch.ones(2, 3, 4)).shape == (1, 1, 4)


def test_alpha_activation_broadcast_alpha_1d_input() -> None:
    module = Gaussian(num_parameters=4)
    assert module._broadcast_alpha(torch.ones(4)).shape == (4,)


##############################################
#     Tests for the init and size checks     #
##############################################


def test_laplacian_negative_init() -> None:
    with pytest.raises(ValueError, match=r"init must be greater than 0"):
        Laplacian(init=-1.0)


def test_laplacian_positive_init() -> None:
    assert Laplacian(init=0.5).alpha.item() == 0.5


@pytest.mark.parametrize("module", [ExpSin, Gaussian, MultiQuadratic, Quadratic, Snake])
def test_negative_init_is_allowed(module: type) -> None:
    assert module(init=-1.0).alpha.item() == -1.0


@pytest.mark.parametrize("init", [float("nan"), float("inf"), float("-inf")])
def test_gaussian_non_finite_init(init: float) -> None:
    with pytest.raises(ValueError, match=r"init must be finite"):
        Gaussian(init=init)


@pytest.mark.parametrize("num_parameters", [2.5, True])
def test_gaussian_incorrect_num_parameters_type(num_parameters: object) -> None:
    with pytest.raises(TypeError, match=r"num_parameters must be an int"):
        Gaussian(num_parameters=num_parameters)


#############################################
#     Tests for the _validate_init hook     #
#############################################


@pytest.mark.parametrize("init", [float("nan"), float("inf")])
def test_base_alpha_activation_validate_init_not_finite(init: float) -> None:
    with pytest.raises(ValueError, match=r"init must be finite"):
        BaseAlphaActivation()._validate_init(init)


@pytest.mark.parametrize("init", [-2.0, 0.0, 2.0])
def test_base_alpha_activation_validate_init_accepts_any_finite_value(init: float) -> None:
    BaseAlphaActivation()._validate_init(init)


@pytest.mark.parametrize("module", [Gaussian, Snake])
def test_alpha_activation_negative_init_is_accepted(module: type[torch.nn.Module]) -> None:
    assert module(init=-1.0).alpha.item() == -1.0


###########################
#     Tests for Snake     #
###########################


def test_snake_str() -> None:
    assert str(Snake()).startswith("Snake(")


def test_snake_forward_frequency_default() -> None:
    module = Snake()
    assert torch.allclose(
        module(torch.tensor([[1.0, 0.0, -1.0], [-2.0, 0.0, 2.0]])),
        torch.tensor(
            [
                [1.708073418273571, 0.0, -0.2919265817264288],
                [-1.173178189568194, 0.0, 2.826821810431806],
            ]
        ),
    )


def test_snake_forward_frequency_2() -> None:
    module = Snake(init=2.0)
    assert torch.allclose(
        module(torch.tensor([[1.0, 0.0, -1.0], [-2.0, 0.0, 2.0]])),
        torch.tensor(
            [
                [1.413410905215903, 0.0, -0.586589094784097],
                [-1.7136249915478468, 0.0, 2.2863750084521532],
            ]
        ),
    )


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("shape", SHAPES)
def test_snake_forward_shape(shape: tuple[int, ...], device: str) -> None:
    device = torch.device(device)
    module = Snake().to(device=device)
    out = module(torch.randn(*shape, device=device))
    assert out.shape == shape
    assert out.dtype == torch.float
    assert out.device == device


def test_snake_zero_init() -> None:
    with pytest.raises(ValueError, match=r"init must be different from 0, got 0"):
        Snake(init=0)


def test_snake_extra_repr() -> None:
    assert (
        Snake(num_parameters=3, learnable=False).extra_repr()
        == "num_parameters=3, learnable=False, channel_dim=-1"
    )


def test_snake_learnable_by_default() -> None:
    assert Snake().alpha.requires_grad


def test_snake_frozen_alpha_is_buffer() -> None:
    assert [n for n, _ in Snake(learnable=False).named_parameters()] == []


def test_snake_forward_per_channel_alpha() -> None:
    module = Snake(num_parameters=2, init=2.0)
    assert module(torch.tensor([[1.0, 0.0], [-1.0, 2.0]])).allclose(
        torch.tensor([[1.413410905215903, 0.0], [-0.586589094784097, 2.2863750084521532]])
    )


def test_snake_forward_alpha_gradient_is_finite() -> None:
    module = Snake()
    module(torch.randn(4, 3)).sum().backward()
    assert module.alpha.grad.isfinite().all()


@pytest.mark.parametrize("x", [-1e3, -1.0, 1.0, 1e3])
def test_snake_forward_equals_paper_formula(x: float) -> None:
    module = Snake(init=1.5)
    expected = x + math.sin(1.5 * x) ** 2 / 1.5
    assert module(torch.tensor([x], dtype=torch.float64)).item() == pytest.approx(expected)


############################################
#     Tests for the shape of the input     #
############################################


@pytest.mark.parametrize("module", ALPHA_ACTIVATIONS)
def test_alpha_activation_forward_size_one_dimension_is_not_expanded(
    module: type[torch.nn.Module],
) -> None:
    with pytest.raises(ValueError, match=r"num_parameters=3.*\(4, 1\)"):
        module(num_parameters=3)(torch.randn(4, 1))


@pytest.mark.parametrize("module", ALPHA_ACTIVATIONS)
def test_alpha_activation_forward_channel_dim_out_of_range_for_1d_input(
    module: type[torch.nn.Module],
) -> None:
    with pytest.raises(
        ValueError,
        match=r"channel_dim=1 is out of range for an input with 1 dimensions \(num_parameters=3\)",
    ):
        module(num_parameters=3, channel_dim=1)(torch.randn(5))


def test_alpha_activation_forward_channel_dim_equal_to_ndim_is_out_of_range() -> None:
    with pytest.raises(ValueError, match=r"channel_dim=2 is out of range for an input with 2"):
        Gaussian(num_parameters=3, channel_dim=2)(torch.randn(4, 3))


def test_alpha_activation_forward_negative_channel_dim_below_minus_ndim_is_out_of_range() -> None:
    with pytest.raises(ValueError, match=r"channel_dim=-3 is out of range for an input with 2"):
        Gaussian(num_parameters=3, channel_dim=-3)(torch.randn(4, 3))


def test_alpha_activation_forward_channel_dim_minus_ndim_is_valid() -> None:
    assert Gaussian(num_parameters=4, channel_dim=-2)(torch.randn(4, 3)).shape == (4, 3)


def test_alpha_activation_forward_wrong_size_message() -> None:
    with pytest.raises(
        ValueError,
        match=r"The size of the dimension channel_dim=1 of input must be num_parameters=3, "
        r"got shape \(2, 4, 5\)",
    ):
        Gaussian(num_parameters=3, channel_dim=1)(torch.randn(2, 4, 5))


def test_alpha_activation_forward_one_parameter_accepts_any_shape() -> None:
    assert Gaussian()(torch.randn(4, 1)).shape == (4, 1)
