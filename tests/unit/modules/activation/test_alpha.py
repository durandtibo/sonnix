from __future__ import annotations

import pytest
import torch
from coola.equality import objects_are_allclose, objects_are_equal
from coola.utils.tensor import get_available_devices

from sonnix.modules import ExpSin, Gaussian, Laplacian, MultiQuadratic, Quadratic, Snake

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


def test_gaussian_frozen_alpha_is_buffer() -> None:
    module = Gaussian(learnable=False)
    assert not module.learnable
    assert len(list(module.parameters())) == 0
    assert "alpha" in module.state_dict()


def test_gaussian_learnable_alpha_is_parameter() -> None:
    module = Gaussian()
    assert module.learnable
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


@pytest.mark.parametrize("module", [Gaussian, Laplacian, Snake])
def test_alpha_activation_zero_init_is_rejected(module: type[torch.nn.Module]) -> None:
    with pytest.raises(ValueError, match=r"'init' has to be different from 0 \(received 0\)"):
        module(init=0)


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
    with pytest.raises(RuntimeError, match=r"size"):
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
    with pytest.raises(ValueError, match=r"'init' has to be positive"):
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
