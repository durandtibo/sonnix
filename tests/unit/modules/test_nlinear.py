from __future__ import annotations

import pytest
import torch
from coola.equality import objects_are_allclose
from coola.utils.tensor import get_available_devices

from sonnix.modules import NLinear

SIZES = (1, 2, 3)


#############################
#     Tests for NLinear     #
#############################


@pytest.fixture
def nlinear_fixed_weights() -> NLinear:
    module = NLinear(n=3, in_features=4, out_features=6)
    with torch.no_grad():
        module.weight.data = torch.tensor(
            [
                [
                    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                ],
                [
                    [2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
                    [2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
                    [2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
                    [2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
                ],
                [
                    [3.0, 3.0, 3.0, 3.0, 3.0, 3.0],
                    [3.0, 3.0, 3.0, 3.0, 3.0, 3.0],
                    [3.0, 3.0, 3.0, 3.0, 3.0, 3.0],
                    [3.0, 3.0, 3.0, 3.0, 3.0, 3.0],
                ],
            ]
        )
        module.bias.data = torch.tensor(
            [
                [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                [2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
                [3.0, 3.0, 3.0, 3.0, 3.0, 3.0],
            ]
        )
    return module


def test_nlinear_str() -> None:
    assert str(NLinear(n=3, in_features=4, out_features=6)).startswith("NLinear(")


@pytest.mark.parametrize("n", SIZES)
def test_nlinear_init_n_weight_shape(n: int) -> None:
    module = NLinear(n=n, in_features=4, out_features=6)
    assert module.weight.shape[0] == n


@pytest.mark.parametrize("n", SIZES)
def test_nlinear_init_n_bias_shape(n: int) -> None:
    module = NLinear(n=n, in_features=4, out_features=6)
    assert module.bias.shape[0] == n


@pytest.mark.parametrize("in_features", SIZES)
def test_nlinear_init_in_features(in_features: int) -> None:
    module = NLinear(n=3, in_features=in_features, out_features=6)
    assert module.weight.shape[1] == in_features


@pytest.mark.parametrize("out_features", SIZES)
def test_nlinear_init_out_features_weight_shape(out_features: int) -> None:
    module = NLinear(n=3, in_features=4, out_features=out_features)
    assert module.weight.shape[2] == out_features


@pytest.mark.parametrize("out_features", SIZES)
def test_nlinear_init_out_features_bias_shape(out_features: int) -> None:
    module = NLinear(n=3, in_features=4, out_features=out_features)
    assert module.bias.shape[1] == out_features


def test_nlinear_init_no_bias() -> None:
    module = NLinear(n=3, in_features=4, out_features=6, bias=False)
    assert module.bias is None


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("batch_size", SIZES)
@pytest.mark.parametrize("n", SIZES)
@pytest.mark.parametrize("in_features", SIZES)
@pytest.mark.parametrize("out_features", SIZES)
@pytest.mark.parametrize("mode", [True, False])
@pytest.mark.parametrize("bias", [True, False])
def test_nlinear_forward_3d(
    device: str,
    batch_size: int,
    n: int,
    in_features: int,
    out_features: int,
    mode: bool,
    bias: bool,
) -> None:
    device = torch.device(device)
    module = NLinear(n=n, in_features=in_features, out_features=out_features, bias=bias).to(
        device=device
    )
    module.train(mode)
    out = module(torch.rand(batch_size, n, in_features, device=device))
    assert out.shape == (batch_size, n, out_features)
    assert out.device == device
    assert out.dtype == torch.float


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("d1", SIZES)
@pytest.mark.parametrize("d2", SIZES)
@pytest.mark.parametrize("n", SIZES)
@pytest.mark.parametrize("in_features", SIZES)
@pytest.mark.parametrize("out_features", SIZES)
@pytest.mark.parametrize("mode", [True, False])
def test_nlinear_forward_4d(
    device: str, d1: int, d2: int, n: int, in_features: int, out_features: int, mode: bool
) -> None:
    device = torch.device(device)
    module = NLinear(n=n, in_features=in_features, out_features=out_features).to(device=device)
    module.train(mode)
    out = module(torch.rand(d1, d2, n, in_features, device=device))
    assert out.shape == (d1, d2, n, out_features)
    assert out.device == device
    assert out.dtype == torch.float


def test_nlinear_forward_predefined_weights_3d(nlinear_fixed_weights: NLinear) -> None:
    assert objects_are_allclose(
        nlinear_fixed_weights(
            torch.tensor(
                [
                    [[-1.0, 0.0, 1.0, 2.0], [1.0, 2.0, 3.0, 4.0], [-1.0, -2.0, 2.0, 1.0]],
                    [[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]],
                ]
            )
        ),
        torch.tensor(
            [
                [
                    [3.0, 3.0, 3.0, 3.0, 3.0, 3.0],
                    [22.0, 22.0, 22.0, 22.0, 22.0, 22.0],
                    [3.0, 3.0, 3.0, 3.0, 3.0, 3.0],
                ],
                [
                    [5.0, 5.0, 5.0, 5.0, 5.0, 5.0],
                    [10.0, 10.0, 10.0, 10.0, 10.0, 10.0],
                    [15.0, 15.0, 15.0, 15.0, 15.0, 15.0],
                ],
            ]
        ),
    )


def test_nlinear_forward_predefined_weights_4d(nlinear_fixed_weights: NLinear) -> None:
    assert objects_are_allclose(
        nlinear_fixed_weights(
            torch.tensor(
                [
                    [
                        [[-1.0, 0.0, 1.0, 2.0], [1.0, 2.0, 3.0, 4.0], [-1.0, -2.0, 2.0, 1.0]],
                        [
                            [-1.0, -1.0, -1.0, -1.0],
                            [-1.0, -1.0, -1.0, -1.0],
                            [-1.0, -1.0, -1.0, -1.0],
                        ],
                        [
                            [-2.0, -2.0, -2.0, -2.0],
                            [-2.0, -2.0, -2.0, -2.0],
                            [-2.0, -2.0, -2.0, -2.0],
                        ],
                    ],
                    [
                        [[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]],
                        [[2.0, 2.0, 2.0, 2.0], [2.0, 2.0, 2.0, 2.0], [2.0, 2.0, 2.0, 2.0]],
                        [[3.0, 3.0, 3.0, 3.0], [3.0, 3.0, 3.0, 3.0], [3.0, 3.0, 3.0, 3.0]],
                    ],
                ]
            )
        ),
        torch.tensor(
            [
                [
                    [
                        [3.0, 3.0, 3.0, 3.0, 3.0, 3.0],
                        [22.0, 22.0, 22.0, 22.0, 22.0, 22.0],
                        [3.0, 3.0, 3.0, 3.0, 3.0, 3.0],
                    ],
                    [
                        [-3.0, -3.0, -3.0, -3.0, -3.0, -3.0],
                        [-6.0, -6.0, -6.0, -6.0, -6.0, -6.0],
                        [-9.0, -9.0, -9.0, -9.0, -9.0, -9.0],
                    ],
                    [
                        [-7.0, -7.0, -7.0, -7.0, -7.0, -7.0],
                        [-14.0, -14.0, -14.0, -14.0, -14.0, -14.0],
                        [-21.0, -21.0, -21.0, -21.0, -21.0, -21.0],
                    ],
                ],
                [
                    [
                        [5.0, 5.0, 5.0, 5.0, 5.0, 5.0],
                        [10.0, 10.0, 10.0, 10.0, 10.0, 10.0],
                        [15.0, 15.0, 15.0, 15.0, 15.0, 15.0],
                    ],
                    [
                        [9.0, 9.0, 9.0, 9.0, 9.0, 9.0],
                        [18.0, 18.0, 18.0, 18.0, 18.0, 18.0],
                        [27.0, 27.0, 27.0, 27.0, 27.0, 27.0],
                    ],
                    [
                        [13.0, 13.0, 13.0, 13.0, 13.0, 13.0],
                        [26.0, 26.0, 26.0, 26.0, 26.0, 26.0],
                        [39.0, 39.0, 39.0, 39.0, 39.0, 39.0],
                    ],
                ],
            ]
        ),
    )


def test_nlinear_init_matches_linear_scale_weight_std() -> None:
    torch.manual_seed(0)
    module = NLinear(n=3, in_features=16, out_features=64)
    reference = torch.nn.Linear(16, 64)
    assert module.weight.std().item() == pytest.approx(reference.weight.std().item(), rel=0.1)


def test_nlinear_init_matches_linear_scale_weight_abs_max() -> None:
    torch.manual_seed(0)
    module = NLinear(n=3, in_features=16, out_features=64)
    torch.nn.Linear(16, 64)
    bound = 1 / 16**0.5
    assert module.weight.abs().max().item() <= bound


def test_nlinear_init_matches_linear_scale_bias_abs_max() -> None:
    torch.manual_seed(0)
    module = NLinear(n=3, in_features=16, out_features=64)
    torch.nn.Linear(16, 64)
    bound = 1 / 16**0.5
    assert module.bias.abs().max().item() <= bound


def test_nlinear_init_matches_linear_scale_bias_abs_max_2() -> None:
    torch.manual_seed(0)
    module = NLinear(n=3, in_features=16, out_features=64)
    torch.nn.Linear(16, 64)
    bound = 1 / 16**0.5
    assert module.bias.abs().max().item() > bound / 2


def test_nlinear_forward_non_contiguous_check1() -> None:
    NLinear(n=3, in_features=4, out_features=6)
    x = torch.randn(4, 3, 2).transpose(0, 2)  # (2, 3, 4), non-contiguous
    assert not x.is_contiguous()


def test_nlinear_forward_non_contiguous_check2() -> None:
    module = NLinear(n=3, in_features=4, out_features=6)
    x = torch.randn(4, 3, 2).transpose(0, 2)  # (2, 3, 4), non-contiguous
    assert module(x).equal(module(x.contiguous()))


######################################
#     Tests for the input checks     #
######################################


@pytest.mark.parametrize("value", [0, -1])
@pytest.mark.parametrize("argument", ["n", "in_features", "out_features"])
def test_nlinear_incorrect_size(argument: str, value: int) -> None:
    kwargs = {"n": 3, "in_features": 4, "out_features": 5, argument: value}
    with pytest.raises(ValueError, match=rf"{argument} must be greater than or equal to 1"):
        NLinear(**kwargs)


@pytest.mark.parametrize("value", [2.5, True])
def test_nlinear_incorrect_size_type(value: object) -> None:
    with pytest.raises(TypeError, match=r"n must be an int"):
        NLinear(n=value, in_features=4, out_features=5)


def test_nlinear_forward_0d_input() -> None:
    with pytest.raises(ValueError, match=r"Expected an input of shape \(\*, n=3, in_features=4\)"):
        NLinear(3, 4, 5)(torch.ones(4))


def test_nlinear_forward_1d_input() -> None:
    with pytest.raises(ValueError, match=r"received \(4,\)"):
        NLinear(3, 4, 5)(torch.ones(4))


def test_nlinear_forward_incorrect_n() -> None:
    with pytest.raises(ValueError, match=r"n=3.*received \(2, 2, 4\)"):
        NLinear(3, 4, 5)(torch.ones(2, 2, 4))


def test_nlinear_sizes_are_plain_attributes() -> None:
    assert {"n", "in_features", "out_features"} <= NLinear(3, 4, 5).__dict__.keys()
