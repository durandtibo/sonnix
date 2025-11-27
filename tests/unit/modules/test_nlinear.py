from __future__ import annotations

import pytest
import torch
from coola import objects_are_allclose
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
def test_nlinear_init_n(n: int) -> None:
    module = NLinear(n=n, in_features=4, out_features=6)
    assert module.weight.shape[0] == n
    assert module.bias.shape[0] == n


@pytest.mark.parametrize("in_features", SIZES)
def test_nlinear_init_in_features(in_features: int) -> None:
    module = NLinear(n=3, in_features=in_features, out_features=6)
    assert module.weight.shape[1] == in_features


@pytest.mark.parametrize("out_features", SIZES)
def test_nlinear_init_out_features(out_features: int) -> None:
    module = NLinear(n=3, in_features=4, out_features=out_features)
    assert module.weight.shape[2] == out_features
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
