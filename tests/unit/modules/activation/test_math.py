from __future__ import annotations

import pytest
import torch
from coola.equality import objects_are_allclose
from coola.utils.tensor import get_available_devices

from sonnix.modules import (
    Asinh,
    Exp,
    Expm1,
    Log,
    Log1p,
    Pow,
    SafeExp,
    SafeLog,
    Sin,
    Sinh,
    Square,
)

SHAPES = [(2,), (2, 3), (2, 3, 4), (2, 3, 4, 5)]

###########################
#     Tests for Asinh     #
###########################


@pytest.mark.parametrize("device", get_available_devices())
def test_asinh_forward(device: str) -> None:
    device = torch.device(device)
    module = Asinh().to(device=device)
    assert objects_are_allclose(
        module(torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0], device=device)),
        torch.tensor(
            [-1.4436354637145996, -0.8813735842704773, 0.0, 0.8813735842704773, 1.4436354637145996],
            dtype=torch.float,
            device=device,
        ),
    )


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("shape", SHAPES)
def test_asinh_forward_shape(device: str, shape: tuple[int, ...]) -> None:
    device = torch.device(device)
    module = Asinh().to(device=device)
    assert objects_are_allclose(
        module(torch.ones(*shape, device=device)),
        torch.full(shape, fill_value=0.8813735842704773, device=device),
    )


#########################
#     Tests for Exp     #
#########################


@pytest.mark.parametrize("device", get_available_devices())
def test_exp_forward(device: str) -> None:
    device = torch.device(device)
    module = Exp().to(device=device)
    assert objects_are_allclose(
        module(torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0], device=device)),
        torch.tensor(
            [0.1353352814912796, 0.3678794503211975, 1.0, 2.7182817459106445, 7.389056205749512],
            dtype=torch.float,
            device=device,
        ),
    )


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("shape", SHAPES)
def test_exp_forward_shape(device: str, shape: tuple[int, ...]) -> None:
    device = torch.device(device)
    module = Exp().to(device=device)
    assert objects_are_allclose(
        module(torch.zeros(*shape, device=device)), torch.ones(*shape, device=device)
    )


###########################
#     Tests for Expm1     #
###########################


@pytest.mark.parametrize("device", get_available_devices())
def test_expm1_forward(device: str) -> None:
    device = torch.device(device)
    module = Expm1().to(device=device)
    assert objects_are_allclose(
        module(torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0], device=device)),
        torch.tensor(
            [-0.8646647334098816, -0.6321205496788025, 0.0, 1.718281865119934, 6.389056205749512],
            dtype=torch.float,
            device=device,
        ),
    )


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("shape", SHAPES)
def test_expm1_forward_shape(device: str, shape: tuple[int, ...]) -> None:
    device = torch.device(device)
    module = Expm1().to(device=device)
    assert objects_are_allclose(
        module(torch.zeros(*shape, device=device)), torch.zeros(*shape, device=device)
    )


#########################
#     Tests for Log     #
#########################


@pytest.mark.parametrize("device", get_available_devices())
def test_log_forward(device: str) -> None:
    device = torch.device(device)
    module = Log().to(device=device)
    assert objects_are_allclose(
        module(torch.tensor([1.0, 2.0, 3.0], device=device)),
        torch.tensor(
            [0.0, 0.6931471805599453, 1.0986122886681098], dtype=torch.float, device=device
        ),
    )


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("shape", SHAPES)
def test_log_forward_shape(device: str, shape: tuple[int, ...]) -> None:
    device = torch.device(device)
    module = Log().to(device=device)
    assert objects_are_allclose(
        module(torch.ones(*shape, device=device)),
        torch.zeros(*shape, device=device),
    )


###########################
#     Tests for Log1p     #
###########################


@pytest.mark.parametrize("device", get_available_devices())
def test_log1p_forward(device: str) -> None:
    device = torch.device(device)
    module = Log1p().to(device=device)
    assert objects_are_allclose(
        module(torch.tensor([0.0, 1.0, 2.0], device=device)),
        torch.tensor(
            [0.0, 0.6931471805599453, 1.0986122886681098], dtype=torch.float, device=device
        ),
    )


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("shape", SHAPES)
def test_log1p_forward_shape(device: str, shape: tuple[int, ...]) -> None:
    device = torch.device(device)
    module = Log1p().to(device=device)
    assert objects_are_allclose(
        module(torch.zeros(*shape, device=device)), torch.zeros(*shape, device=device)
    )


#########################
#     Tests for Pow     #
#########################


@pytest.mark.parametrize("device", get_available_devices())
def test_pow_2_forward(device: str) -> None:
    device = torch.device(device)
    module = Pow(exponent=2.0).to(device=device)
    assert module(
        torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=torch.float, device=device)
    ).allclose(
        torch.tensor(
            [4.0, 1.0, 0.0, 1.0, 4.0],
            dtype=torch.float,
            device=device,
        )
    )


@pytest.mark.parametrize("device", get_available_devices())
def test_pow_3_forward(device: str) -> None:
    device = torch.device(device)
    module = Pow(exponent=3.0).to(device=device)
    assert module(
        torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=torch.float, device=device)
    ).allclose(
        torch.tensor(
            [-8.0, -1.0, 0.0, 1.0, 8.0],
            dtype=torch.float,
            device=device,
        )
    )


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("exponent", [-2.0, -1.0, 0.0, 1.0, 2.0])
def test_pow_forward_shape_value(device: str, shape: tuple[int, ...], exponent: float) -> None:
    device = torch.device(device)
    module = Pow(exponent).to(device=device)
    out = module(torch.randn(*shape, device=device, requires_grad=True))
    out.mean().backward()
    assert out.shape == shape


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("exponent", [-2.0, -1.0, 0.0, 1.0, 2.0])
def test_pow_forward_shape_dtype(device: str, shape: tuple[int, ...], exponent: float) -> None:
    device = torch.device(device)
    module = Pow(exponent).to(device=device)
    out = module(torch.randn(*shape, device=device, requires_grad=True))
    out.mean().backward()
    assert out.dtype == torch.float


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("exponent", [-2.0, -1.0, 0.0, 1.0, 2.0])
def test_pow_forward_shape_device(device: str, shape: tuple[int, ...], exponent: float) -> None:
    device = torch.device(device)
    module = Pow(exponent).to(device=device)
    out = module(torch.randn(*shape, device=device, requires_grad=True))
    out.mean().backward()
    assert out.device == device


#############################
#     Tests for SafeExp     #
#############################


def test_safe_exp_str() -> None:
    assert str(SafeExp()).startswith("SafeExp(")


@pytest.mark.parametrize("device", get_available_devices())
def test_safe_exp_forward(device: str) -> None:
    device = torch.device(device)
    module = SafeExp().to(device=device)
    assert objects_are_allclose(
        module(torch.tensor([-1, 0, 1, 10, 100], dtype=torch.float, device=device)),
        torch.tensor(
            [0.3678794503211975, 1.0, 2.7182817459106445, 22026.46484375, 485165184.0],
            dtype=torch.float,
            device=device,
        ),
    )


@pytest.mark.parametrize("device", get_available_devices())
def test_safe_exp_forward_max_10(device: str) -> None:
    device = torch.device(device)
    module = SafeExp(max_val=10).to(device=device)
    assert objects_are_allclose(
        module(torch.tensor([-1, 0, 1, 10, 100], dtype=torch.float, device=device)),
        torch.tensor(
            [0.3678794503211975, 1.0, 2.7182817459106445, 22026.46484375, 22026.46484375],
            dtype=torch.float,
            device=device,
        ),
    )


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("shape", SHAPES)
def test_safe_exp_forward_shape(device: str, shape: tuple[int, ...]) -> None:
    device = torch.device(device)
    module = SafeExp().to(device=device)
    assert objects_are_allclose(
        module(torch.zeros(*shape, device=device)), torch.ones(*shape, device=device)
    )


#############################
#     Tests for SafeLog     #
#############################


def test_safe_log_str() -> None:
    assert str(SafeLog()).startswith("SafeLog(")


@pytest.mark.parametrize("device", get_available_devices())
def test_safe_log_forward(device: str) -> None:
    device = torch.device(device)
    module = SafeLog().to(device=device)
    assert objects_are_allclose(
        module(torch.tensor([-1, 0, 1, 2], dtype=torch.float, device=device)),
        torch.tensor(
            [-18.420680743952367, -18.420680743952367, 0.0, 0.6931471805599453],
            dtype=torch.float,
            device=device,
        ),
    )


@pytest.mark.parametrize("device", get_available_devices())
def test_safe_log_forward_min_1(device: str) -> None:
    device = torch.device(device)
    module = SafeLog(min_val=1).to(device=device)
    assert objects_are_allclose(
        module(torch.tensor([-1, 0, 1, 2], dtype=torch.float, device=device)),
        torch.tensor([0.0, 0.0, 0.0, 0.6931471805599453], dtype=torch.float, device=device),
    )


@pytest.mark.parametrize("min_val", [-1.0, 0.0])
def test_safe_log_invalid_min_val(min_val: float) -> None:
    with pytest.raises(ValueError, match=r"min_val must be greater than 0"):
        SafeLog(min_val=min_val)


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("shape", SHAPES)
def test_safe_log_forward_shape(device: str, shape: tuple[int, ...]) -> None:
    device = torch.device(device)
    module = SafeLog().to(device=device)
    assert objects_are_allclose(
        module(torch.ones(*shape, device=device)), torch.zeros(*shape, device=device)
    )


##########################
#     Tests for Sinh     #
##########################


@pytest.mark.parametrize("device", get_available_devices())
def test_sinh_forward(device: str) -> None:
    device = torch.device(device)
    module = Sinh().to(device=device)
    assert objects_are_allclose(
        module(torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0], device=device)),
        torch.tensor(
            [-3.6268603801727295, -1.175201177597046, 0.0, 1.175201177597046, 3.6268603801727295],
            dtype=torch.float,
            device=device,
        ),
    )


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("shape", SHAPES)
def test_sinh_forward_shape(device: str, shape: tuple[int, ...]) -> None:
    device = torch.device(device)
    module = Sinh().to(device=device)
    assert objects_are_allclose(
        module(torch.ones(*shape, device=device)),
        torch.full(shape, fill_value=1.175201177597046, device=device),
    )


#########################
#     Tests for Sin     #
#########################


@pytest.mark.parametrize("device", get_available_devices())
def test_sin_forward(device: str) -> None:
    device = torch.device(device)
    module = Sin().to(device=device)
    assert module(
        torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=torch.float, device=device)
    ).allclose(
        torch.tensor(
            [-0.9092974268256817, -0.8414709848078965, 0.0, 0.8414709848078965, 0.9092974268256817],
            dtype=torch.float,
            device=device,
        )
    )


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("shape", SHAPES)
def test_sin_forward_shape_value(device: str, shape: tuple[int, ...]) -> None:
    device = torch.device(device)
    module = Sin().to(device=device)
    out = module(torch.randn(*shape, device=device, requires_grad=True))
    out.mean().backward()
    assert out.shape == shape


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("shape", SHAPES)
def test_sin_forward_shape_dtype(device: str, shape: tuple[int, ...]) -> None:
    device = torch.device(device)
    module = Sin().to(device=device)
    out = module(torch.randn(*shape, device=device, requires_grad=True))
    out.mean().backward()
    assert out.dtype == torch.float


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("shape", SHAPES)
def test_sin_forward_shape_device(device: str, shape: tuple[int, ...]) -> None:
    device = torch.device(device)
    module = Sin().to(device=device)
    out = module(torch.randn(*shape, device=device, requires_grad=True))
    out.mean().backward()
    assert out.device == device


def test_pow_repr() -> None:
    assert repr(Pow(exponent=3)) == "Pow(exponent=3.0)"


############################
#     Tests for Square     #
############################


@pytest.mark.parametrize("device", get_available_devices())
def test_square_forward(device: str) -> None:
    device = torch.device(device)
    module = Square().to(device=device)
    assert module(
        torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=torch.float, device=device)
    ).allclose(
        torch.tensor(
            [4.0, 1.0, 0.0, 1.0, 4.0],
            dtype=torch.float,
            device=device,
        )
    )


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("shape", SHAPES)
def test_square_forward_shape_value(device: str, shape: tuple[int, ...]) -> None:
    device = torch.device(device)
    module = Square().to(device=device)
    out = module(torch.randn(*shape, device=device, requires_grad=True))
    out.mean().backward()
    assert out.shape == shape


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("shape", SHAPES)
def test_square_forward_shape_dtype(device: str, shape: tuple[int, ...]) -> None:
    device = torch.device(device)
    module = Square().to(device=device)
    out = module(torch.randn(*shape, device=device, requires_grad=True))
    out.mean().backward()
    assert out.dtype == torch.float


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("shape", SHAPES)
def test_square_forward_shape_device(device: str, shape: tuple[int, ...]) -> None:
    device = torch.device(device)
    module = Square().to(device=device)
    out = module(torch.randn(*shape, device=device, requires_grad=True))
    out.mean().backward()
    assert out.device == device


def test_safe_exp_module_preserve_gradient_check1() -> None:
    module = SafeExp(max_val=20.0, preserve_gradient=True)
    assert repr(module) == "SafeExp(max_val=20.0, preserve_gradient=True)"


def test_safe_exp_module_preserve_gradient_grad() -> None:
    module = SafeExp(max_val=20.0, preserve_gradient=True)
    x = torch.tensor([100.0], requires_grad=True)
    module(x).backward()
    assert x.grad.item() > 0


def test_safe_log_module_preserve_gradient_check1() -> None:
    module = SafeLog(min_val=1e-8, preserve_gradient=True)
    assert repr(module) == "SafeLog(min_val=1e-08, preserve_gradient=True)"


def test_safe_log_module_preserve_gradient_grad() -> None:
    module = SafeLog(min_val=1e-8, preserve_gradient=True)
    x = torch.tensor([1e-10], requires_grad=True)
    module(x).backward()
    assert x.grad.item() > 0


def test_square_repr() -> None:
    assert repr(Square()) == "Square()"


def test_square_is_not_a_pow() -> None:
    assert not isinstance(Square(), Pow)


def test_square_forward_equals_pow() -> None:
    x = torch.tensor([-2.0, -0.5, 0.0, 3.0])
    assert Square()(x).equal(Pow(exponent=2)(x))
