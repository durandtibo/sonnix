from __future__ import annotations

from unittest.mock import patch

import pytest
import torch
from coola.equality import objects_are_allclose, objects_are_equal
from coola.utils.tensor import get_available_devices

from sonnix.modules import AsinhCosSinNumericalEncoder, CosSinNumericalEncoder
from sonnix.modules.numerical.sine import (
    check_abs_range,
    check_frequency,
    prepare_tensor_param,
)

SIZES = (1, 2, 3)


############################################
#     Tests for CosSinNumericalEncoder     #
############################################


def test_cos_sin_numerical_encoder_str() -> None:
    assert str(
        CosSinNumericalEncoder(
            frequency=torch.tensor([[1.0, 2.0, 4.0], [2.0, 3.0, 4.0]]),
            phase_shift=torch.zeros(2, 3),
        )
    ).startswith("CosSinNumericalEncoder(")


@pytest.mark.parametrize(
    "frequency",
    [torch.tensor([1.0, 2.0, 4.0]), torch.tensor([[1.0, 2.0, 4.0]])],
)
def test_cos_sin_numerical_encoder_frequency_1_feature(frequency: torch.Tensor) -> None:
    assert CosSinNumericalEncoder(
        frequency=frequency, phase_shift=torch.zeros(1, 3)
    ).frequency.equal(torch.tensor([[1.0, 2.0, 4.0, 1.0, 2.0, 4.0]]))


def test_cos_sin_numerical_encoder_frequency_2_features() -> None:
    assert CosSinNumericalEncoder(
        frequency=torch.tensor([[1.0, 2.0, 4.0], [2.0, 3.0, 4.0]]),
        phase_shift=torch.zeros(2, 3),
    ).frequency.equal(
        torch.tensor([[1.0, 2.0, 4.0, 1.0, 2.0, 4.0], [2.0, 3.0, 4.0, 2.0, 3.0, 4.0]])
    )


@pytest.mark.parametrize(
    "phase_shift",
    [torch.tensor([2.0, 1.0, 0.0]), torch.tensor([[2.0, 1.0, 0.0]])],
)
def test_cos_sin_numerical_encoder_phase_shift_1_feature(phase_shift: torch.Tensor) -> None:
    assert CosSinNumericalEncoder(
        frequency=torch.tensor([[1.0, 2.0, 4.0]]), phase_shift=phase_shift
    ).phase_shift.equal(torch.tensor([[2.0, 1.0, 0.0, 2.0, 1.0, 0.0]]))


def test_cos_sin_numerical_encoder_phase_shift_2_features() -> None:
    assert CosSinNumericalEncoder(
        frequency=torch.tensor([[1.0, 2.0, 4.0], [2.0, 3.0, 4.0]]),
        phase_shift=torch.tensor([[2.0, 1.0, 0.0], [1.0, 2.0, 3.0]]),
    ).phase_shift.equal(
        torch.tensor([[2.0, 1.0, 0.0, 2.0, 1.0, 0.0], [1.0, 2.0, 3.0, 1.0, 2.0, 3.0]])
    )


def test_cos_sin_numerical_encoder_different_shape() -> None:
    with pytest.raises(RuntimeError, match=r"'frequency' and 'phase_shift' shapes do not match:"):
        CosSinNumericalEncoder(
            frequency=torch.ones(2, 4),
            phase_shift=torch.zeros(2, 3),
        )


def test_cos_sin_numerical_encoder_input_size() -> None:
    assert (
        CosSinNumericalEncoder(
            frequency=torch.tensor([[1.0, 2.0, 4.0], [2.0, 3.0, 4.0]]),
            phase_shift=torch.zeros(2, 3),
        ).input_size
        == 2
    )


def test_cos_sin_numerical_encoder_output_size() -> None:
    assert (
        CosSinNumericalEncoder(
            frequency=torch.tensor([[1.0, 2.0, 4.0], [2.0, 3.0, 4.0]]),
            phase_shift=torch.zeros(2, 3),
        ).output_size
        == 6
    )


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("batch_size", SIZES)
@pytest.mark.parametrize("n_features", SIZES)
@pytest.mark.parametrize("feature_size", SIZES)
@pytest.mark.parametrize("mode", [True, False])
def test_cos_sin_numerical_encoder_forward_2d(
    device: str, batch_size: int, n_features: int, feature_size: int, mode: bool
) -> None:
    device = torch.device(device)
    module = CosSinNumericalEncoder(
        frequency=torch.rand(n_features, feature_size),
        phase_shift=torch.rand(n_features, feature_size),
    ).to(device=device)
    module.train(mode)
    out = module(torch.rand(batch_size, n_features, device=device))
    assert out.shape == (batch_size, n_features, feature_size * 2)
    assert out.device == device
    assert out.dtype == torch.float


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("d1", SIZES)
@pytest.mark.parametrize("d2", SIZES)
@pytest.mark.parametrize("n_features", SIZES)
@pytest.mark.parametrize("feature_size", SIZES)
@pytest.mark.parametrize("mode", [True, False])
def test_cos_sin_numerical_encoder_forward_3d(
    device: str, d1: int, d2: int, n_features: int, feature_size: int, mode: bool
) -> None:
    device = torch.device(device)
    module = CosSinNumericalEncoder(
        frequency=torch.rand(n_features, feature_size),
        phase_shift=torch.rand(n_features, feature_size),
    ).to(device=device)
    module.train(mode)
    out = module(torch.rand(d1, d2, n_features, device=device))
    assert out.shape == (d1, d2, n_features, feature_size * 2)
    assert out.device == device
    assert out.dtype == torch.float


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("batch_size", SIZES)
@pytest.mark.parametrize("learnable", [True, False])
@pytest.mark.parametrize("mode", [True, False])
def test_cos_sin_numerical_encoder_backward(
    device: str, batch_size: int, learnable: bool, mode: bool
) -> None:
    device = torch.device(device)
    module = CosSinNumericalEncoder(
        frequency=torch.rand(3, 6), phase_shift=torch.rand(3, 6), learnable=learnable
    ).to(device=device)
    module.train(mode)
    out = module(torch.rand(batch_size, 3, device=device, requires_grad=True))
    out.mean().backward()
    assert out.shape == (batch_size, 3, 12)
    assert out.device == device
    assert out.dtype == torch.float


@pytest.mark.parametrize("learnable", [True, False])
def test_cos_sin_numerical_encoder_learnable(learnable: bool) -> None:
    module = CosSinNumericalEncoder(
        frequency=torch.rand(3, 6), phase_shift=torch.rand(3, 6), learnable=learnable
    )
    assert module.frequency.requires_grad == learnable
    assert module.phase_shift.requires_grad == learnable


def test_cos_sin_numerical_encoder_forward_1_feature() -> None:
    module = CosSinNumericalEncoder(
        frequency=torch.tensor([[1.0, 2.0, 3.0]]),
        phase_shift=torch.zeros(1, 3),
    )
    assert objects_are_allclose(
        module(torch.tensor([[-1], [0], [1]])),
        torch.tensor(
            [
                [
                    [
                        -0.8414709848078965,
                        -0.9092974268256817,
                        -0.1411200080598672,
                        0.5403023058681398,
                        -0.4161468365471424,
                        -0.9899924966004454,
                    ]
                ],
                [[0.0, 0.0, 0.0, 1.0, 1.0, 1.0]],
                [
                    [
                        0.8414709848078965,
                        0.9092974268256817,
                        0.1411200080598672,
                        0.5403023058681398,
                        -0.4161468365471424,
                        -0.9899924966004454,
                    ]
                ],
            ],
        ),
    )


def test_cos_sin_numerical_encoder_forward_2_features() -> None:
    module = CosSinNumericalEncoder(
        frequency=torch.tensor([[1.0, 2.0, 3.0], [2.0, 4.0, 6.0]]),
        phase_shift=torch.tensor([[1.0, 0.0, -1.0], [-1.0, 0.0, 1.0]]),
    )
    assert objects_are_allclose(
        module(torch.tensor([[-1, -2], [0, 0], [1, 2]])),
        torch.tensor(
            [
                [
                    [
                        0.0,
                        -0.9092974066734314,
                        0.756802499294281,
                        1.0,
                        -0.416146844625473,
                        -0.6536436080932617,
                    ],
                    [
                        0.9589242935180664,
                        -0.9893582463264465,
                        0.9999902248382568,
                        0.28366219997406006,
                        -0.1455000340938568,
                        0.004425697959959507,
                    ],
                ],
                [
                    [
                        0.8414709568023682,
                        0.0,
                        -0.8414709568023682,
                        0.5403023362159729,
                        1.0,
                        0.5403023362159729,
                    ],
                    [
                        -0.8414709568023682,
                        0.0,
                        0.8414709568023682,
                        0.5403023362159729,
                        1.0,
                        0.5403023362159729,
                    ],
                ],
                [
                    [
                        0.9092974066734314,
                        0.9092974066734314,
                        0.9092974066734314,
                        -0.416146844625473,
                        -0.416146844625473,
                        -0.416146844625473,
                    ],
                    [
                        0.14112000167369843,
                        0.9893582463264465,
                        0.4201670289039612,
                        -0.9899924993515015,
                        -0.1455000340938568,
                        0.9074468016624451,
                    ],
                ],
            ]
        ),
    )


def test_cos_sin_numerical_encoder_forward_2_features_same() -> None:
    module = CosSinNumericalEncoder(
        frequency=torch.tensor([[1.0, 2.0, 3.0]]),
        phase_shift=torch.tensor([[0.0, 0.0, 0.0]]),
    )
    assert objects_are_allclose(
        module(torch.tensor([[-1, -2], [0, 0], [1, 2]])),
        torch.tensor(
            [
                [
                    [
                        -0.8414709568023682,
                        -0.9092974066734314,
                        -0.14112000167369843,
                        0.5403023362159729,
                        -0.416146844625473,
                        -0.9899924993515015,
                    ],
                    [
                        -0.9092974066734314,
                        0.756802499294281,
                        0.279415488243103,
                        -0.416146844625473,
                        -0.6536436080932617,
                        0.9601702690124512,
                    ],
                ],
                [[0.0, 0.0, 0.0, 1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]],
                [
                    [
                        0.8414709568023682,
                        0.9092974066734314,
                        0.14112000167369843,
                        0.5403023362159729,
                        -0.416146844625473,
                        -0.9899924993515015,
                    ],
                    [
                        0.9092974066734314,
                        -0.756802499294281,
                        -0.279415488243103,
                        -0.416146844625473,
                        -0.6536436080932617,
                        0.9601702690124512,
                    ],
                ],
            ]
        ),
    )


@patch(
    "sonnix.modules.numerical.sine.torch.rand",
    lambda *args, **kwargs: torch.tensor([0.0, 0.5, 1.0]),  # noqa: ARG005
)
def test_cos_sin_numerical_encoder_create_rand_frequency() -> None:
    module = CosSinNumericalEncoder.create_rand_frequency(
        num_frequencies=3, min_frequency=0.2, max_frequency=1
    )
    assert module.frequency.data.equal(torch.tensor([[0.2, 0.6, 1.0, 0.2, 0.6, 1.0]]))
    assert module.phase_shift.data.equal(torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]))


@pytest.mark.parametrize("learnable", [True, False])
def test_cos_sin_scalar_encoder_create_rand_frequency_learnable(learnable: bool) -> None:
    module = CosSinNumericalEncoder.create_rand_frequency(
        num_frequencies=3, min_frequency=0.2, max_frequency=1, learnable=learnable
    )
    assert module.frequency.requires_grad == learnable
    assert module.phase_shift.requires_grad == learnable


@patch(
    "sonnix.modules.numerical.sine.torch.rand",
    lambda *args, **kwargs: torch.tensor([0.0, 0.5, 1.0]),  # noqa: ARG005
)
def test_cos_sin_numerical_encoder_create_rand_value_range() -> None:
    module = CosSinNumericalEncoder.create_rand_value_range(
        num_frequencies=3, min_abs_value=0.2, max_abs_value=1
    )
    assert module.frequency.data.equal(torch.tensor([[1.0, 3.0, 5.0, 1.0, 3.0, 5.0]]))
    assert module.phase_shift.data.equal(torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]))


@pytest.mark.parametrize("learnable", [True, False])
def test_cos_sin_scalar_encoder_create_rand_value_range_learnable(learnable: bool) -> None:
    module = CosSinNumericalEncoder.create_rand_value_range(
        num_frequencies=3, min_abs_value=0.2, max_abs_value=1, learnable=learnable
    )
    assert module.frequency.requires_grad == learnable
    assert module.phase_shift.requires_grad == learnable


def test_cos_sin_numerical_encoder_create_linspace_frequency() -> None:
    module = CosSinNumericalEncoder.create_linspace_frequency(
        num_frequencies=3, min_frequency=0.2, max_frequency=1
    )
    assert module.frequency.data.equal(torch.tensor([[0.2, 0.6, 1.0, 0.2, 0.6, 1.0]]))
    assert module.phase_shift.data.equal(torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]))


@pytest.mark.parametrize("learnable", [True, False])
def test_cos_sin_scalar_encoder_create_linspace_frequency_learnable(learnable: bool) -> None:
    module = CosSinNumericalEncoder.create_linspace_frequency(
        num_frequencies=3, min_frequency=0.2, max_frequency=1, learnable=learnable
    )
    assert module.frequency.requires_grad == learnable
    assert module.phase_shift.requires_grad == learnable


def test_cos_sin_numerical_encoder_create_linspace_value_range() -> None:
    module = CosSinNumericalEncoder.create_linspace_value_range(
        num_frequencies=3, min_abs_value=0.2, max_abs_value=1
    )
    assert module.frequency.data.equal(torch.tensor([[1.0, 3.0, 5.0, 1.0, 3.0, 5.0]]))
    assert module.phase_shift.data.equal(torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]))


@pytest.mark.parametrize("learnable", [True, False])
def test_cos_sin_scalar_encoder_create_linspace_value_range_learnable(learnable: bool) -> None:
    module = CosSinNumericalEncoder.create_linspace_value_range(
        num_frequencies=3, min_abs_value=0.2, max_abs_value=1, learnable=learnable
    )
    assert module.frequency.requires_grad == learnable
    assert module.phase_shift.requires_grad == learnable


def test_cos_sin_numerical_encoder_create_logspace_frequency() -> None:
    module = CosSinNumericalEncoder.create_logspace_frequency(
        num_frequencies=3, min_frequency=0.01, max_frequency=1
    )
    assert module.frequency.data.equal(torch.tensor([[0.01, 0.1, 1.0, 0.01, 0.1, 1.0]]))
    assert module.phase_shift.data.equal(torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]))


@pytest.mark.parametrize("learnable", [True, False])
def test_cos_sin_scalar_encoder_create_logspace_frequency_learnable(learnable: bool) -> None:
    module = CosSinNumericalEncoder.create_logspace_frequency(
        num_frequencies=3, min_frequency=0.01, max_frequency=1, learnable=learnable
    )
    assert module.frequency.requires_grad == learnable
    assert module.phase_shift.requires_grad == learnable


def test_cos_sin_numerical_encoder_create_logspace_value_range() -> None:
    module = CosSinNumericalEncoder.create_logspace_value_range(
        num_frequencies=3, min_abs_value=0.01, max_abs_value=1.0
    )
    assert module.frequency.data.equal(torch.tensor([[1.0, 10.0, 100.0, 1.0, 10.0, 100.0]]))
    assert module.phase_shift.data.equal(torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]))


@pytest.mark.parametrize("learnable", [True, False])
def test_cos_sin_scalar_encoder_create_logspace_value_range_learnable(learnable: bool) -> None:
    module = CosSinNumericalEncoder.create_logspace_value_range(
        num_frequencies=3, min_abs_value=0.01, max_abs_value=1.0, learnable=learnable
    )
    assert module.frequency.requires_grad == learnable
    assert module.phase_shift.requires_grad == learnable


#################################################
#     Tests for AsinhCosSinNumericalEncoder     #
#################################################


def test_asinh_cos_sin_numerical_encoder_str() -> None:
    assert str(
        AsinhCosSinNumericalEncoder(
            frequency=torch.tensor([[1.0, 2.0, 4.0], [2.0, 3.0, 4.0]]),
            phase_shift=torch.zeros(2, 3),
        )
    ).startswith("AsinhCosSinNumericalEncoder(")


@pytest.mark.parametrize(
    "frequency",
    [torch.tensor([1.0, 2.0, 4.0]), torch.tensor([[1.0, 2.0, 4.0]])],
)
def test_asinh_cos_sin_numerical_encoder_frequency_1_feature(frequency: torch.Tensor) -> None:
    assert AsinhCosSinNumericalEncoder(
        frequency=frequency, phase_shift=torch.zeros(1, 3)
    ).frequency.equal(torch.tensor([[1.0, 2.0, 4.0, 1.0, 2.0, 4.0]]))


def test_asinh_cos_sin_numerical_encoder_frequency_2_features() -> None:
    assert AsinhCosSinNumericalEncoder(
        frequency=torch.tensor([[1.0, 2.0, 4.0], [2.0, 3.0, 4.0]]),
        phase_shift=torch.zeros(2, 3),
    ).frequency.equal(
        torch.tensor([[1.0, 2.0, 4.0, 1.0, 2.0, 4.0], [2.0, 3.0, 4.0, 2.0, 3.0, 4.0]])
    )


@pytest.mark.parametrize(
    "phase_shift",
    [torch.tensor([2.0, 1.0, 0.0]), torch.tensor([[2.0, 1.0, 0.0]])],
)
def test_asinh_cos_sin_numerical_encoder_phase_shift_1_feature(phase_shift: torch.Tensor) -> None:
    assert AsinhCosSinNumericalEncoder(
        frequency=torch.tensor([[1.0, 2.0, 4.0]]), phase_shift=phase_shift
    ).phase_shift.equal(torch.tensor([[2.0, 1.0, 0.0, 2.0, 1.0, 0.0]]))


def test_asinh_cos_sin_numerical_encoder_phase_shift_2_features() -> None:
    assert AsinhCosSinNumericalEncoder(
        frequency=torch.tensor([[1.0, 2.0, 4.0], [2.0, 3.0, 4.0]]),
        phase_shift=torch.tensor([[2.0, 1.0, 0.0], [1.0, 2.0, 3.0]]),
    ).phase_shift.equal(
        torch.tensor([[2.0, 1.0, 0.0, 2.0, 1.0, 0.0], [1.0, 2.0, 3.0, 1.0, 2.0, 3.0]])
    )


def test_asinh_cos_sin_numerical_encoder_different_shape() -> None:
    with pytest.raises(RuntimeError, match=r"'frequency' and 'phase_shift' shapes do not match:"):
        AsinhCosSinNumericalEncoder(
            frequency=torch.ones(2, 4),
            phase_shift=torch.zeros(2, 3),
        )


def test_asinh_cos_sin_numerical_encoder_input_size() -> None:
    assert (
        AsinhCosSinNumericalEncoder(
            frequency=torch.tensor([[1.0, 2.0, 4.0], [2.0, 3.0, 4.0]]),
            phase_shift=torch.zeros(2, 3),
        ).input_size
        == 2
    )


def test_asinh_cos_sin_numerical_encoder_output_size() -> None:
    assert (
        AsinhCosSinNumericalEncoder(
            frequency=torch.tensor([[1.0, 2.0, 4.0], [2.0, 3.0, 4.0]]),
            phase_shift=torch.zeros(2, 3),
        ).output_size
        == 7
    )


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("batch_size", SIZES)
@pytest.mark.parametrize("n_features", SIZES)
@pytest.mark.parametrize("feature_size", SIZES)
@pytest.mark.parametrize("mode", [True, False])
def test_asinh_cos_sin_numerical_encoder_forward_2d(
    device: str, batch_size: int, n_features: int, feature_size: int, mode: bool
) -> None:
    device = torch.device(device)
    module = AsinhCosSinNumericalEncoder(
        frequency=torch.rand(n_features, feature_size),
        phase_shift=torch.rand(n_features, feature_size),
    ).to(device=device)
    module.train(mode)
    out = module(torch.rand(batch_size, n_features, device=device))
    assert out.shape == (batch_size, n_features, feature_size * 2 + 1)
    assert out.device == device
    assert out.dtype == torch.float


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("d1", SIZES)
@pytest.mark.parametrize("d2", SIZES)
@pytest.mark.parametrize("n_features", SIZES)
@pytest.mark.parametrize("feature_size", SIZES)
@pytest.mark.parametrize("mode", [True, False])
def test_asinh_cos_sin_numerical_encoder_forward_4d(
    device: str, d1: int, d2: int, n_features: int, feature_size: int, mode: bool
) -> None:
    device = torch.device(device)
    module = AsinhCosSinNumericalEncoder(
        frequency=torch.rand(n_features, feature_size),
        phase_shift=torch.rand(n_features, feature_size),
    ).to(device=device)
    module.train(mode)
    out = module(torch.rand(d1, d2, n_features, device=device))
    assert out.shape == (d1, d2, n_features, feature_size * 2 + 1)
    assert out.device == device
    assert out.dtype == torch.float


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("batch_size", SIZES)
@pytest.mark.parametrize("learnable", [True, False])
@pytest.mark.parametrize("mode", [True, False])
def test_asinh_cos_sin_numerical_encoder_backward(
    device: str, batch_size: int, learnable: bool, mode: bool
) -> None:
    device = torch.device(device)
    module = AsinhCosSinNumericalEncoder(
        frequency=torch.rand(3, 6), phase_shift=torch.rand(3, 6), learnable=learnable
    ).to(device=device)
    module.train(mode)
    out = module(torch.rand(batch_size, 3, device=device, requires_grad=True))
    out.mean().backward()
    assert out.shape == (batch_size, 3, 13)
    assert out.device == device
    assert out.dtype == torch.float


@pytest.mark.parametrize("learnable", [True, False])
def test_asinh_cos_sin_numerical_encoder_learnable(learnable: bool) -> None:
    module = AsinhCosSinNumericalEncoder(
        frequency=torch.rand(3, 6), phase_shift=torch.rand(3, 6), learnable=learnable
    )
    assert module.frequency.requires_grad == learnable
    assert module.phase_shift.requires_grad == learnable


def test_asinh_cos_sin_numerical_encoder_forward_1_feature() -> None:
    module = AsinhCosSinNumericalEncoder(
        frequency=torch.tensor([[1.0, 2.0, 3.0]]),
        phase_shift=torch.zeros(1, 3),
    )
    assert objects_are_allclose(
        module(torch.tensor([[-1], [0], [1]])),
        torch.tensor(
            [
                [
                    [
                        -0.8414709848078965,
                        -0.9092974268256817,
                        -0.1411200080598672,
                        0.5403023058681398,
                        -0.4161468365471424,
                        -0.9899924966004454,
                        -0.881373587019543,
                    ]
                ],
                [[0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0]],
                [
                    [
                        0.8414709848078965,
                        0.9092974268256817,
                        0.1411200080598672,
                        0.5403023058681398,
                        -0.4161468365471424,
                        -0.9899924966004454,
                        0.881373587019543,
                    ]
                ],
            ],
        ),
    )


def test_asinh_cos_sin_numerical_encoder_forward_2_features() -> None:
    module = AsinhCosSinNumericalEncoder(
        frequency=torch.tensor([[1.0, 2.0, 3.0], [2.0, 4.0, 6.0]]),
        phase_shift=torch.tensor([[1.0, 0.0, -1.0], [-1.0, 0.0, 1.0]]),
    )
    assert objects_are_allclose(
        module(torch.tensor([[-1, -2], [0, 0], [1, 2]])),
        torch.tensor(
            [
                [
                    [
                        0.0,
                        -0.9092974066734314,
                        0.756802499294281,
                        1.0,
                        -0.416146844625473,
                        -0.6536436080932617,
                        -0.881373587019543,
                    ],
                    [
                        0.9589242935180664,
                        -0.9893582463264465,
                        0.9999902248382568,
                        0.28366219997406006,
                        -0.1455000340938568,
                        0.004425697959959507,
                        -1.4436354751788103,
                    ],
                ],
                [
                    [
                        0.8414709568023682,
                        0.0,
                        -0.8414709568023682,
                        0.5403023362159729,
                        1.0,
                        0.5403023362159729,
                        0.0,
                    ],
                    [
                        -0.8414709568023682,
                        0.0,
                        0.8414709568023682,
                        0.5403023362159729,
                        1.0,
                        0.5403023362159729,
                        0.0,
                    ],
                ],
                [
                    [
                        0.9092974066734314,
                        0.9092974066734314,
                        0.9092974066734314,
                        -0.416146844625473,
                        -0.416146844625473,
                        -0.416146844625473,
                        0.881373587019543,
                    ],
                    [
                        0.14112000167369843,
                        0.9893582463264465,
                        0.4201670289039612,
                        -0.9899924993515015,
                        -0.1455000340938568,
                        0.9074468016624451,
                        1.4436354751788103,
                    ],
                ],
            ]
        ),
    )


def test_asinh_cos_sin_numerical_encoder_forward_2_features_same() -> None:
    module = AsinhCosSinNumericalEncoder(
        frequency=torch.tensor([[1.0, 2.0, 3.0]]),
        phase_shift=torch.tensor([[0.0, 0.0, 0.0]]),
    )
    assert objects_are_allclose(
        module(torch.tensor([[-1, -2], [0, 0], [1, 2]])),
        torch.tensor(
            [
                [
                    [
                        -0.8414709568023682,
                        -0.9092974066734314,
                        -0.14112000167369843,
                        0.5403023362159729,
                        -0.416146844625473,
                        -0.9899924993515015,
                        -0.881373587019543,
                    ],
                    [
                        -0.9092974066734314,
                        0.756802499294281,
                        0.279415488243103,
                        -0.416146844625473,
                        -0.6536436080932617,
                        0.9601702690124512,
                        -1.4436354751788103,
                    ],
                ],
                [
                    [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0],
                ],
                [
                    [
                        0.8414709568023682,
                        0.9092974066734314,
                        0.14112000167369843,
                        0.5403023362159729,
                        -0.416146844625473,
                        -0.9899924993515015,
                        0.881373587019543,
                    ],
                    [
                        0.9092974066734314,
                        -0.756802499294281,
                        -0.279415488243103,
                        -0.416146844625473,
                        -0.6536436080932617,
                        0.9601702690124512,
                        1.4436354751788103,
                    ],
                ],
            ]
        ),
    )


##########################################
#     Tests for prepare_tensor_param     #
##########################################


@pytest.mark.parametrize("tensor", [torch.tensor([1.0, 2.0, 4.0]), torch.tensor([[1.0, 2.0, 4.0]])])
def test_prepare_tensor_param_1d(tensor: torch.Tensor) -> None:
    assert objects_are_equal(
        prepare_tensor_param(tensor, name="scale"), torch.tensor([[1.0, 2.0, 4.0]])
    )


def test_prepare_tensor_param_2d() -> None:
    assert objects_are_equal(
        prepare_tensor_param(torch.tensor([[1.0, 2.0, 4.0], [2.0, 3.0, 4.0]]), name="scale"),
        torch.tensor([[1.0, 2.0, 4.0], [2.0, 3.0, 4.0]]),
    )


def test_prepare_tensor_param_incorrect_shape() -> None:
    with pytest.raises(RuntimeError, match=r"Incorrect shape for 'scale':"):
        prepare_tensor_param(torch.ones(2, 3, 4), name="scale")


#####################################
#     Tests for check_abs_range     #
#####################################


def test_check_abs_range_valid() -> None:
    check_abs_range(min_abs_value=1, max_abs_value=5)


def test_check_abs_range_invalid_min_abs_value() -> None:
    with pytest.raises(RuntimeError, match=r"'min_abs_value' has to be greater than 0"):
        check_abs_range(min_abs_value=-1, max_abs_value=5)


def test_check_abs_range_invalid_max_abs_value() -> None:
    with pytest.raises(
        RuntimeError, match=r"'max_abs_value' .* has to be greater than 'min_abs_value'"
    ):
        check_abs_range(min_abs_value=5, max_abs_value=1)


#####################################
#     Tests for check_frequency     #
#####################################


def test_check_frequency_valid() -> None:
    check_frequency(num_frequencies=3, min_frequency=0.2, max_frequency=1)


def test_check_frequency_invalid_num_frequencies() -> None:
    with pytest.raises(RuntimeError, match=r"'num_frequencies' has to be greater or equal to 1"):
        check_frequency(num_frequencies=0, min_frequency=0.2, max_frequency=1)


def test_check_frequency_invalid_min_frequency() -> None:
    with pytest.raises(RuntimeError, match=r"'min_frequency' has to be greater than 0"):
        check_frequency(num_frequencies=3, min_frequency=-2, max_frequency=1)


def test_check_frequency_invalid_max_frequency() -> None:
    with pytest.raises(
        RuntimeError, match=r"'max_frequency' .* has to be greater than 'min_frequency'"
    ):
        check_frequency(num_frequencies=3, min_frequency=2, max_frequency=1)
