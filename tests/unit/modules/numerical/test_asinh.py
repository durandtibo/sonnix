from __future__ import annotations

import pytest
import torch
from coola.utils.tensor import get_available_devices

from sonnix.modules import AsinhNumericalEncoder

SIZES = (1, 2, 3)

###########################################
#     Tests for AsinhNumericalEncoder     #
###########################################


def test_asinh_numerical_encoder_str() -> None:
    assert str(
        AsinhNumericalEncoder(scale=torch.tensor([[1.0, 2.0, 4.0], [2.0, 3.0, 4.0]]))
    ).startswith("AsinhNumericalEncoder(")


@pytest.mark.parametrize("scale", [torch.tensor([1.0, 2.0, 4.0]), torch.tensor([[1.0, 2.0, 4.0]])])
def test_asinh_numerical_encoder_scale_1_feature(scale: torch.Tensor) -> None:
    assert AsinhNumericalEncoder(scale).scale.equal(torch.tensor([[1.0, 2.0, 4.0]]))


def test_asinh_numerical_encoder_scale_2_features() -> None:
    assert AsinhNumericalEncoder(torch.tensor([[1.0, 2.0, 4.0], [2.0, 3.0, 4.0]])).scale.equal(
        torch.tensor([[1.0, 2.0, 4.0], [2.0, 3.0, 4.0]])
    )


def test_asinh_numerical_encoder_scale_incorrect_shape() -> None:
    with pytest.raises(RuntimeError, match=r"Incorrect shape for 'scale':"):
        AsinhNumericalEncoder(torch.ones(2, 3, 4))


def test_asinh_numerical_encoder_input_size() -> None:
    assert AsinhNumericalEncoder(torch.tensor([[1.0, 2.0, 4.0], [2.0, 3.0, 4.0]])).input_size == 2


def test_asinh_numerical_encoder_output_size() -> None:
    assert AsinhNumericalEncoder(torch.tensor([[1.0, 2.0, 4.0], [2.0, 3.0, 4.0]])).output_size == 3


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("batch_size", SIZES)
@pytest.mark.parametrize("n_features", SIZES)
@pytest.mark.parametrize("feature_size", SIZES)
@pytest.mark.parametrize("mode", [True, False])
def test_asinh_numerical_encoder_forward_2d(
    device: str, batch_size: int, n_features: int, feature_size: int, mode: bool
) -> None:
    device = torch.device(device)
    module = AsinhNumericalEncoder(scale=torch.rand(n_features, feature_size)).to(device=device)
    module.train(mode)
    out = module(torch.rand(batch_size, n_features, device=device))
    assert out.shape == (batch_size, n_features, feature_size)
    assert out.device == device
    assert out.dtype == torch.float


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("d1", SIZES)
@pytest.mark.parametrize("d2", SIZES)
@pytest.mark.parametrize("n_features", SIZES)
@pytest.mark.parametrize("feature_size", SIZES)
@pytest.mark.parametrize("mode", [True, False])
def test_asinh_numerical_encoder_forward_4d(
    device: str, d1: int, d2: int, n_features: int, feature_size: int, mode: bool
) -> None:
    device = torch.device(device)
    module = AsinhNumericalEncoder(scale=torch.rand(n_features, feature_size)).to(device=device)
    module.train(mode)
    out = module(torch.rand(d1, d2, n_features, device=device))
    assert out.shape == (d1, d2, n_features, feature_size)
    assert out.device == device
    assert out.dtype == torch.float


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("batch_size", SIZES)
@pytest.mark.parametrize("learnable", [True, False])
@pytest.mark.parametrize("mode", [True, False])
def test_asinh_numerical_encoder_backward(
    device: str, batch_size: int, learnable: bool, mode: bool
) -> None:
    device = torch.device(device)
    module = AsinhNumericalEncoder(scale=torch.rand(3, 6), learnable=learnable).to(device=device)
    module.train(mode)
    out = module(torch.rand(batch_size, 3, device=device, requires_grad=True))
    out.mean().backward()
    assert out.shape == (batch_size, 3, 6)
    assert out.device == device
    assert out.dtype == torch.float


@pytest.mark.parametrize("learnable", [True, False])
def test_asinh_numerical_encoder_learnable(learnable: bool) -> None:
    assert (
        AsinhNumericalEncoder(scale=torch.rand(3, 6), learnable=learnable).scale.requires_grad
        == learnable
    )


def test_asinh_numerical_encoder_forward_scale_1_feature() -> None:
    module = AsinhNumericalEncoder(scale=torch.tensor([[1.0, 2.0, 3.0]]))
    assert module(torch.tensor([[-1.0], [0.0], [1.0]])).allclose(
        torch.tensor(
            [
                [[-0.881373587019543, -1.4436354751788103, -1.8184464592320668]],
                [[0.0, 0.0, 0.0]],
                [[0.881373587019543, 1.4436354751788103, 1.8184464592320668]],
            ],
        ),
    )


def test_asinh_numerical_encoder_forward_scale_2_features() -> None:
    module = AsinhNumericalEncoder(scale=torch.tensor([[1.0, 2.0, 3.0], [1.0, 2.0, 4.0]]))
    assert module(torch.tensor([[-1.0, 0.0], [0.0, 1.0], [1.0, -1.0]])).allclose(
        torch.tensor(
            [
                [
                    [-0.8813735842704773, -1.4436354637145996, -1.8184465169906616],
                    [0.0, 0.0, 0.0],
                ],
                [
                    [0.0, 0.0, 0.0],
                    [0.8813735842704773, 1.4436354637145996, 2.094712495803833],
                ],
                [
                    [0.8813735842704773, 1.4436354637145996, 1.8184465169906616],
                    [-0.8813735842704773, -1.4436354637145996, -2.094712495803833],
                ],
            ],
        ),
    )


def test_asinh_numerical_encoder_forward_scale_2_features_same_scale() -> None:
    module = AsinhNumericalEncoder(scale=torch.tensor([[1.0, 2.0, 3.0]]))
    assert module(torch.tensor([[-1.0, 0.0], [0.0, 1.0], [1.0, -1.0]])).allclose(
        torch.tensor(
            [
                [
                    [-0.8813735842704773, -1.4436354637145996, -1.8184465169906616],
                    [0.0, 0.0, 0.0],
                ],
                [
                    [0.0, 0.0, 0.0],
                    [0.8813735842704773, 1.4436354637145996, 1.8184465169906616],
                ],
                [
                    [0.8813735842704773, 1.4436354637145996, 1.8184465169906616],
                    [-0.8813735842704773, -1.4436354637145996, -1.8184465169906616],
                ],
            ],
        ),
    )
