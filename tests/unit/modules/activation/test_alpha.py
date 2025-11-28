from __future__ import annotations

import pytest
import torch
from coola import objects_are_allclose, objects_are_equal
from coola.utils.tensor import get_available_devices

from sonnix.modules import ExpSin, Gaussian, Laplacian, MultiQuadratic, Quadratic

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
            [0.40280712612352804, 0.43107595064559234, 1.0, 2.319776824715853, 2.4825777280150008],
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
            [2.1314499915144016, 0.40280712612352804, 1.0, 2.4825777280150008, 0.46916418587400077],
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
