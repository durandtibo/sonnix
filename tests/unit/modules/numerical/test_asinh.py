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
    with pytest.raises(ValueError, match=r"Incorrect shape for 'scale':"):
        AsinhNumericalEncoder(torch.ones(2, 3, 4))


def test_asinh_numerical_encoder_input_size() -> None:
    assert AsinhNumericalEncoder(torch.tensor([[1.0, 2.0, 4.0], [2.0, 3.0, 4.0]])).n_features == 2


def test_asinh_numerical_encoder_output_size() -> None:
    assert AsinhNumericalEncoder(torch.tensor([[1.0, 2.0, 4.0], [2.0, 3.0, 4.0]])).feature_size == 3


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("batch_size", SIZES)
@pytest.mark.parametrize("n_features", SIZES)
@pytest.mark.parametrize("feature_size", SIZES)
@pytest.mark.parametrize("mode", [True, False])
def test_asinh_numerical_encoder_forward_2d_shape(
    device: str, batch_size: int, n_features: int, feature_size: int, mode: bool
) -> None:
    device = torch.device(device)
    module = AsinhNumericalEncoder(scale=torch.rand(n_features, feature_size)).to(device=device)
    module.train(mode)
    out = module(torch.rand(batch_size, n_features, device=device))
    assert out.shape == (batch_size, n_features, feature_size)


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("batch_size", SIZES)
@pytest.mark.parametrize("n_features", SIZES)
@pytest.mark.parametrize("feature_size", SIZES)
@pytest.mark.parametrize("mode", [True, False])
def test_asinh_numerical_encoder_forward_2d_device(
    device: str, batch_size: int, n_features: int, feature_size: int, mode: bool
) -> None:
    device = torch.device(device)
    module = AsinhNumericalEncoder(scale=torch.rand(n_features, feature_size)).to(device=device)
    module.train(mode)
    out = module(torch.rand(batch_size, n_features, device=device))
    assert out.device == device


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("batch_size", SIZES)
@pytest.mark.parametrize("n_features", SIZES)
@pytest.mark.parametrize("feature_size", SIZES)
@pytest.mark.parametrize("mode", [True, False])
def test_asinh_numerical_encoder_forward_2d_dtype(
    device: str, batch_size: int, n_features: int, feature_size: int, mode: bool
) -> None:
    device = torch.device(device)
    module = AsinhNumericalEncoder(scale=torch.rand(n_features, feature_size)).to(device=device)
    module.train(mode)
    out = module(torch.rand(batch_size, n_features, device=device))
    assert out.dtype == torch.float


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("d1", SIZES)
@pytest.mark.parametrize("d2", SIZES)
@pytest.mark.parametrize("n_features", SIZES)
@pytest.mark.parametrize("feature_size", SIZES)
@pytest.mark.parametrize("mode", [True, False])
def test_asinh_numerical_encoder_forward_4d_shape(
    device: str, d1: int, d2: int, n_features: int, feature_size: int, mode: bool
) -> None:
    device = torch.device(device)
    module = AsinhNumericalEncoder(scale=torch.rand(n_features, feature_size)).to(device=device)
    module.train(mode)
    out = module(torch.rand(d1, d2, n_features, device=device))
    assert out.shape == (d1, d2, n_features, feature_size)


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("d1", SIZES)
@pytest.mark.parametrize("d2", SIZES)
@pytest.mark.parametrize("n_features", SIZES)
@pytest.mark.parametrize("feature_size", SIZES)
@pytest.mark.parametrize("mode", [True, False])
def test_asinh_numerical_encoder_forward_4d_device(
    device: str, d1: int, d2: int, n_features: int, feature_size: int, mode: bool
) -> None:
    device = torch.device(device)
    module = AsinhNumericalEncoder(scale=torch.rand(n_features, feature_size)).to(device=device)
    module.train(mode)
    out = module(torch.rand(d1, d2, n_features, device=device))
    assert out.device == device


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("d1", SIZES)
@pytest.mark.parametrize("d2", SIZES)
@pytest.mark.parametrize("n_features", SIZES)
@pytest.mark.parametrize("feature_size", SIZES)
@pytest.mark.parametrize("mode", [True, False])
def test_asinh_numerical_encoder_forward_4d_dtype(
    device: str, d1: int, d2: int, n_features: int, feature_size: int, mode: bool
) -> None:
    device = torch.device(device)
    module = AsinhNumericalEncoder(scale=torch.rand(n_features, feature_size)).to(device=device)
    module.train(mode)
    out = module(torch.rand(d1, d2, n_features, device=device))
    assert out.dtype == torch.float


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("batch_size", SIZES)
@pytest.mark.parametrize("learnable", [True, False])
@pytest.mark.parametrize("mode", [True, False])
def test_asinh_numerical_encoder_backward_shape(
    device: str, batch_size: int, learnable: bool, mode: bool
) -> None:
    device = torch.device(device)
    module = AsinhNumericalEncoder(scale=torch.rand(3, 6), learnable=learnable).to(device=device)
    module.train(mode)
    out = module(torch.rand(batch_size, 3, device=device, requires_grad=True))
    out.mean().backward()
    assert out.shape == (batch_size, 3, 6)


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("batch_size", SIZES)
@pytest.mark.parametrize("learnable", [True, False])
@pytest.mark.parametrize("mode", [True, False])
def test_asinh_numerical_encoder_backward_device(
    device: str, batch_size: int, learnable: bool, mode: bool
) -> None:
    device = torch.device(device)
    module = AsinhNumericalEncoder(scale=torch.rand(3, 6), learnable=learnable).to(device=device)
    module.train(mode)
    out = module(torch.rand(batch_size, 3, device=device, requires_grad=True))
    out.mean().backward()
    assert out.device == device


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("batch_size", SIZES)
@pytest.mark.parametrize("learnable", [True, False])
@pytest.mark.parametrize("mode", [True, False])
def test_asinh_numerical_encoder_backward_dtype(
    device: str, batch_size: int, learnable: bool, mode: bool
) -> None:
    device = torch.device(device)
    module = AsinhNumericalEncoder(scale=torch.rand(3, 6), learnable=learnable).to(device=device)
    module.train(mode)
    out = module(torch.rand(batch_size, 3, device=device, requires_grad=True))
    out.mean().backward()
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


def test_asinh_numerical_encoder_frozen_scale_is_buffer_learnable() -> None:
    module = AsinhNumericalEncoder(scale=torch.ones(2, 3))
    assert not module.learnable


def test_asinh_numerical_encoder_frozen_scale_is_buffer_check2() -> None:
    module = AsinhNumericalEncoder(scale=torch.ones(2, 3))
    assert len(list(module.parameters())) == 0


def test_asinh_numerical_encoder_frozen_scale_is_buffer_check3() -> None:
    module = AsinhNumericalEncoder(scale=torch.ones(2, 3))
    assert "scale" in module.state_dict()


def test_asinh_numerical_encoder_learnable_scale_is_parameter_learnable() -> None:
    module = AsinhNumericalEncoder(scale=torch.ones(2, 3), learnable=True)
    assert module.learnable


def test_asinh_numerical_encoder_learnable_scale_is_parameter_check2() -> None:
    module = AsinhNumericalEncoder(scale=torch.ones(2, 3), learnable=True)
    assert [n for n, _ in module.named_parameters()] == ["scale"]


@pytest.mark.parametrize("learnable", [True, False])
def test_asinh_numerical_encoder_scale_is_a_copy(learnable: bool) -> None:
    scale = torch.tensor([[1.0, 2.0, 4.0]])
    module = AsinhNumericalEncoder(scale, learnable=learnable)
    scale.add_(1.0)
    assert module.scale.equal(torch.tensor([[1.0, 2.0, 4.0]]))


################################################
#     Tests for AsinhNumericalEncoder.create     #
################################################


def test_asinh_numerical_encoder_create_linspace_scale_range() -> None:
    module = AsinhNumericalEncoder.create("linspace", 3, scale_range=(0.2, 1))
    assert module.scale.equal(torch.tensor([[0.2, 0.6, 1.0]]))


def test_asinh_numerical_encoder_create_linspace_value_range() -> None:
    module = AsinhNumericalEncoder.create("linspace", 3, value_range=(0.2, 1))
    assert module.scale.equal(torch.tensor([[1.0, 3.0, 5.0]]))


def test_asinh_numerical_encoder_create_logspace_scale_range() -> None:
    module = AsinhNumericalEncoder.create("logspace", 3, scale_range=(0.01, 1))
    assert module.scale.equal(torch.tensor([[0.01, 0.1, 1.0]]))


def test_asinh_numerical_encoder_create_logspace_value_range() -> None:
    module = AsinhNumericalEncoder.create("logspace", 3, value_range=(0.01, 1.0))
    assert module.scale.equal(torch.tensor([[1.0, 10.0, 100.0]]))


def test_asinh_numerical_encoder_create_rand_shape() -> None:
    module = AsinhNumericalEncoder.create("rand", 5, scale_range=(0.2, 1), n_features=3)
    assert module.scale.shape == (3, 5)


def test_asinh_numerical_encoder_create_rand_in_range() -> None:
    module = AsinhNumericalEncoder.create("rand", 50, scale_range=(0.2, 1), generator=3)
    assert ((module.scale >= 0.2) & (module.scale <= 1.0)).all()


def test_asinh_numerical_encoder_create_rand_same_seed() -> None:
    assert AsinhNumericalEncoder.create("rand", 5, scale_range=(0.2, 1), generator=3).scale.equal(
        AsinhNumericalEncoder.create("rand", 5, scale_range=(0.2, 1), generator=3).scale
    )


def test_asinh_numerical_encoder_create_n_features() -> None:
    module = AsinhNumericalEncoder.create("linspace", 3, scale_range=(0.2, 1), n_features=2)
    assert module.n_features == 2


@pytest.mark.parametrize("learnable", [True, False])
def test_asinh_numerical_encoder_create_learnable(learnable: bool) -> None:
    module = AsinhNumericalEncoder.create("linspace", 3, scale_range=(0.2, 1), learnable=learnable)
    assert module.learnable == learnable


@pytest.mark.parametrize("dtype", [torch.float64, torch.bfloat16])
def test_asinh_numerical_encoder_create_dtype(dtype: torch.dtype) -> None:
    module = AsinhNumericalEncoder.create("linspace", 3, scale_range=(0.2, 1), dtype=dtype)
    assert module.scale.dtype == dtype


def test_asinh_numerical_encoder_create_forward() -> None:
    module = AsinhNumericalEncoder.create("linspace", 3, scale_range=(1, 3), n_features=2)
    assert module(torch.ones(4, 2)).shape == (4, 2, 3)


def test_asinh_numerical_encoder_create_invalid_spacing() -> None:
    with pytest.raises(ValueError, match=r"spacing"):
        AsinhNumericalEncoder.create("incorrect", 3, scale_range=(0.2, 1))


def test_asinh_numerical_encoder_create_no_range() -> None:
    with pytest.raises(ValueError, match=r"Exactly one of 'scale_range' and 'value_range'"):
        AsinhNumericalEncoder.create("linspace", 3)


def test_asinh_numerical_encoder_create_two_ranges() -> None:
    with pytest.raises(ValueError, match=r"Exactly one of 'scale_range' and 'value_range'"):
        AsinhNumericalEncoder.create("linspace", 3, scale_range=(0.2, 1), value_range=(0.2, 1))


def test_asinh_numerical_encoder_create_invalid_feature_size() -> None:
    with pytest.raises(ValueError, match=r"feature_size must be greater than or equal to 1"):
        AsinhNumericalEncoder.create("linspace", 0, scale_range=(0.2, 1))


def test_asinh_numerical_encoder_create_non_positive_scale() -> None:
    with pytest.raises(ValueError, match=r"min_scale must be greater than 0"):
        AsinhNumericalEncoder.create("linspace", 3, scale_range=(0, 1))


def test_asinh_numerical_encoder_create_invalid_scale_range() -> None:
    with pytest.raises(ValueError, match=r"'max_scale' .* has to be greater than"):
        AsinhNumericalEncoder.create("linspace", 3, scale_range=(1, 0.2))


def test_asinh_numerical_encoder_create_invalid_value_range() -> None:
    with pytest.raises(ValueError, match=r"'max_abs_value' .* has to be greater than"):
        AsinhNumericalEncoder.create("linspace", 3, value_range=(1, 0.2))


def test_asinh_numerical_encoder_create_invalid_n_features() -> None:
    with pytest.raises(ValueError, match=r"n_features must be greater than or equal to 1"):
        AsinhNumericalEncoder.create("linspace", 3, scale_range=(0.2, 1), n_features=0)
