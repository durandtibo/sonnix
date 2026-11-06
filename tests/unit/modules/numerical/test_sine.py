from __future__ import annotations

from unittest.mock import patch

import pytest
import torch
from coola.equality import objects_are_allclose
from coola.random import make_torch_generator
from coola.utils.tensor import get_available_devices

from sonnix.modules import AsinhCosSinNumericalEncoder, CosSinNumericalEncoder

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
    ).frequency.equal(torch.tensor([[1.0, 2.0, 4.0]]))


def test_cos_sin_numerical_encoder_frequency_2_features() -> None:
    assert CosSinNumericalEncoder(
        frequency=torch.tensor([[1.0, 2.0, 4.0], [2.0, 3.0, 4.0]]),
        phase_shift=torch.zeros(2, 3),
    ).frequency.equal(torch.tensor([[1.0, 2.0, 4.0], [2.0, 3.0, 4.0]]))


@pytest.mark.parametrize(
    "phase_shift",
    [torch.tensor([2.0, 1.0, 0.0]), torch.tensor([[2.0, 1.0, 0.0]])],
)
def test_cos_sin_numerical_encoder_phase_shift_1_feature(phase_shift: torch.Tensor) -> None:
    assert CosSinNumericalEncoder(
        frequency=torch.tensor([[1.0, 2.0, 4.0]]), phase_shift=phase_shift
    ).phase_shift.equal(torch.tensor([[2.0, 1.0, 0.0]]))


def test_cos_sin_numerical_encoder_phase_shift_2_features() -> None:
    assert CosSinNumericalEncoder(
        frequency=torch.tensor([[1.0, 2.0, 4.0], [2.0, 3.0, 4.0]]),
        phase_shift=torch.tensor([[2.0, 1.0, 0.0], [1.0, 2.0, 3.0]]),
    ).phase_shift.equal(torch.tensor([[2.0, 1.0, 0.0], [1.0, 2.0, 3.0]]))


def test_cos_sin_numerical_encoder_different_shape() -> None:
    with pytest.raises(ValueError, match=r"'frequency' and 'phase_shift' must have the same shape"):
        CosSinNumericalEncoder(
            frequency=torch.ones(2, 4),
            phase_shift=torch.zeros(2, 3),
        )


def test_cos_sin_numerical_encoder_input_size() -> None:
    assert (
        CosSinNumericalEncoder(
            frequency=torch.tensor([[1.0, 2.0, 4.0], [2.0, 3.0, 4.0]]),
            phase_shift=torch.zeros(2, 3),
        ).n_features
        == 2
    )


def test_cos_sin_numerical_encoder_output_size() -> None:
    assert (
        CosSinNumericalEncoder(
            frequency=torch.tensor([[1.0, 2.0, 4.0], [2.0, 3.0, 4.0]]),
            phase_shift=torch.zeros(2, 3),
        ).feature_size
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
def test_cos_sin_numerical_encoder_learnable_frequency_requires_grad(learnable: bool) -> None:
    module = CosSinNumericalEncoder(
        frequency=torch.rand(3, 6), phase_shift=torch.rand(3, 6), learnable=learnable
    )
    assert module.frequency.requires_grad == learnable


@pytest.mark.parametrize("learnable", [True, False])
def test_cos_sin_numerical_encoder_learnable_phase_shift_requires_grad(learnable: bool) -> None:
    module = CosSinNumericalEncoder(
        frequency=torch.rand(3, 6), phase_shift=torch.rand(3, 6), learnable=learnable
    )
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
    lambda *args, **kwargs: torch.tensor([[0.0, 0.5, 1.0]]),  # noqa: ARG005
)
def test_cos_sin_numerical_encoder_create_rand_frequency_range() -> None:
    module = CosSinNumericalEncoder.create("rand", 3, frequency_range=(0.2, 1))
    assert module.frequency.data.equal(torch.tensor([[0.2, 0.6, 1.0]]))


@patch(
    "sonnix.modules.numerical.sine.torch.rand",
    lambda *args, **kwargs: torch.tensor([[0.0, 0.5, 1.0]]),  # noqa: ARG005
)
def test_cos_sin_numerical_encoder_create_rand_value_range() -> None:
    module = CosSinNumericalEncoder.create("rand", 3, value_range=(0.2, 1))
    assert module.frequency.data.equal(torch.tensor([[1.0, 3.0, 5.0]]))


def test_cos_sin_numerical_encoder_create_rand_shape() -> None:
    module = CosSinNumericalEncoder.create("rand", 4, frequency_range=(0.2, 1))
    assert module.frequency.shape == (1, 4)


def test_cos_sin_numerical_encoder_create_rand_in_range_frequency() -> None:
    torch.manual_seed(1)
    module = CosSinNumericalEncoder.create("rand", 100, frequency_range=(0.2, 1))
    assert module.frequency.min() >= 0.2


def test_cos_sin_numerical_encoder_create_rand_in_range_frequency_2() -> None:
    torch.manual_seed(1)
    module = CosSinNumericalEncoder.create("rand", 100, frequency_range=(0.2, 1))
    assert module.frequency.max() <= 1.0


def test_cos_sin_numerical_encoder_create_linspace_frequency_range() -> None:
    module = CosSinNumericalEncoder.create("linspace", 3, frequency_range=(0.2, 1))
    assert module.frequency.data.equal(torch.tensor([[0.2, 0.6, 1.0]]))


def test_cos_sin_numerical_encoder_create_linspace_value_range() -> None:
    module = CosSinNumericalEncoder.create("linspace", 3, value_range=(0.2, 1))
    assert module.frequency.data.equal(torch.tensor([[1.0, 3.0, 5.0]]))


def test_cos_sin_numerical_encoder_create_logspace_frequency_range() -> None:
    module = CosSinNumericalEncoder.create("logspace", 3, frequency_range=(0.01, 1))
    assert module.frequency.data.equal(torch.tensor([[0.01, 0.1, 1.0]]))


def test_cos_sin_numerical_encoder_create_logspace_value_range() -> None:
    module = CosSinNumericalEncoder.create("logspace", 3, value_range=(0.01, 1.0))
    assert module.frequency.data.equal(torch.tensor([[1.0, 10.0, 100.0]]))


@pytest.mark.parametrize("spacing", ["rand", "linspace", "logspace"])
def test_cos_sin_numerical_encoder_create_phase_shift_is_zero(spacing: str) -> None:
    module = CosSinNumericalEncoder.create(spacing, 3, frequency_range=(0.2, 1))
    assert module.phase_shift.data.equal(torch.zeros(1, 3))


@pytest.mark.parametrize("spacing", ["rand", "linspace", "logspace"])
@pytest.mark.parametrize("learnable", [True, False])
def test_cos_sin_numerical_encoder_create_learnable(spacing: str, learnable: bool) -> None:
    module = CosSinNumericalEncoder.create(
        spacing, 3, frequency_range=(0.2, 1), learnable=learnable
    )
    assert module.learnable == learnable


def test_cos_sin_numerical_encoder_create_single_frequency() -> None:
    module = CosSinNumericalEncoder.create("linspace", 1, frequency_range=(0.5, 0.5))
    assert module.frequency.data.equal(torch.tensor([[0.5]]))


def test_cos_sin_numerical_encoder_create_returns_the_subclass() -> None:
    assert isinstance(
        AsinhCosSinNumericalEncoder.create("linspace", 3, frequency_range=(0.2, 1)),
        AsinhCosSinNumericalEncoder,
    )


def test_cos_sin_numerical_encoder_create_invalid_spacing() -> None:
    with pytest.raises(ValueError, match=r"spacing"):
        CosSinNumericalEncoder.create("incorrect", 3, frequency_range=(0.2, 1))


def test_cos_sin_numerical_encoder_create_no_range() -> None:
    with pytest.raises(ValueError, match=r"Exactly one of 'frequency_range' and 'value_range'"):
        CosSinNumericalEncoder.create("linspace", 3)


def test_cos_sin_numerical_encoder_create_two_ranges() -> None:
    with pytest.raises(ValueError, match=r"Exactly one of 'frequency_range' and 'value_range'"):
        CosSinNumericalEncoder.create("linspace", 3, frequency_range=(0.2, 1), value_range=(0.2, 1))


def test_cos_sin_numerical_encoder_create_invalid_num_frequencies() -> None:
    with pytest.raises(ValueError, match=r"num_frequencies must be greater than or equal to 1"):
        CosSinNumericalEncoder.create("linspace", 0, frequency_range=(0.2, 1))


def test_cos_sin_numerical_encoder_create_invalid_frequency_range() -> None:
    with pytest.raises(ValueError, match=r"'max_frequency' .* has to be greater than"):
        CosSinNumericalEncoder.create("linspace", 3, frequency_range=(1, 0.2))


def test_cos_sin_numerical_encoder_create_invalid_value_range() -> None:
    with pytest.raises(ValueError, match=r"'max_abs_value' .* has to be greater than"):
        CosSinNumericalEncoder.create("linspace", 3, value_range=(1, 0.2))


def test_cos_sin_numerical_encoder_create_non_positive_value_range() -> None:
    with pytest.raises(ValueError, match=r"min_abs_value must be greater than 0"):
        CosSinNumericalEncoder.create("linspace", 3, value_range=(0, 1))


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
    ).frequency.equal(torch.tensor([[1.0, 2.0, 4.0]]))


def test_asinh_cos_sin_numerical_encoder_frequency_2_features() -> None:
    assert AsinhCosSinNumericalEncoder(
        frequency=torch.tensor([[1.0, 2.0, 4.0], [2.0, 3.0, 4.0]]),
        phase_shift=torch.zeros(2, 3),
    ).frequency.equal(torch.tensor([[1.0, 2.0, 4.0], [2.0, 3.0, 4.0]]))


@pytest.mark.parametrize(
    "phase_shift",
    [torch.tensor([2.0, 1.0, 0.0]), torch.tensor([[2.0, 1.0, 0.0]])],
)
def test_asinh_cos_sin_numerical_encoder_phase_shift_1_feature(phase_shift: torch.Tensor) -> None:
    assert AsinhCosSinNumericalEncoder(
        frequency=torch.tensor([[1.0, 2.0, 4.0]]), phase_shift=phase_shift
    ).phase_shift.equal(torch.tensor([[2.0, 1.0, 0.0]]))


def test_asinh_cos_sin_numerical_encoder_phase_shift_2_features() -> None:
    assert AsinhCosSinNumericalEncoder(
        frequency=torch.tensor([[1.0, 2.0, 4.0], [2.0, 3.0, 4.0]]),
        phase_shift=torch.tensor([[2.0, 1.0, 0.0], [1.0, 2.0, 3.0]]),
    ).phase_shift.equal(torch.tensor([[2.0, 1.0, 0.0], [1.0, 2.0, 3.0]]))


def test_asinh_cos_sin_numerical_encoder_different_shape() -> None:
    with pytest.raises(ValueError, match=r"'frequency' and 'phase_shift' must have the same shape"):
        AsinhCosSinNumericalEncoder(
            frequency=torch.ones(2, 4),
            phase_shift=torch.zeros(2, 3),
        )


def test_asinh_cos_sin_numerical_encoder_input_size() -> None:
    assert (
        AsinhCosSinNumericalEncoder(
            frequency=torch.tensor([[1.0, 2.0, 4.0], [2.0, 3.0, 4.0]]),
            phase_shift=torch.zeros(2, 3),
        ).n_features
        == 2
    )


def test_asinh_cos_sin_numerical_encoder_output_size() -> None:
    assert (
        AsinhCosSinNumericalEncoder(
            frequency=torch.tensor([[1.0, 2.0, 4.0], [2.0, 3.0, 4.0]]),
            phase_shift=torch.zeros(2, 3),
        ).feature_size
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
def test_asinh_cos_sin_numerical_encoder_learnable_frequency_requires_grad(learnable: bool) -> None:
    module = AsinhCosSinNumericalEncoder(
        frequency=torch.rand(3, 6), phase_shift=torch.rand(3, 6), learnable=learnable
    )
    assert module.frequency.requires_grad == learnable


@pytest.mark.parametrize("learnable", [True, False])
def test_asinh_cos_sin_numerical_encoder_learnable_phase_shift_requires_grad(
    learnable: bool,
) -> None:
    module = AsinhCosSinNumericalEncoder(
        frequency=torch.rand(3, 6), phase_shift=torch.rand(3, 6), learnable=learnable
    )
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


def test_cos_sin_numerical_encoder_sin_cos_share_parameters_shape() -> None:
    module = CosSinNumericalEncoder(
        frequency=torch.tensor([[1.0, 2.0, 4.0]]), phase_shift=torch.zeros(1, 3), learnable=True
    )
    assert module.frequency.shape == (1, 3)
    assert module.phase_shift.shape == (1, 3)


def test_cos_sin_numerical_encoder_sin_cos_share_parameters_feature_size() -> None:
    module = CosSinNumericalEncoder(
        frequency=torch.tensor([[1.0, 2.0, 4.0]]), phase_shift=torch.zeros(1, 3), learnable=True
    )
    assert module.feature_size == 6


def test_cos_sin_numerical_encoder_sin_cos_share_parameters_check3() -> None:
    module = CosSinNumericalEncoder(
        frequency=torch.tensor([[1.0, 2.0, 4.0]]), phase_shift=torch.zeros(1, 3), learnable=True
    )
    assert sum(p.numel() for p in module.parameters()) == 6


def test_cos_sin_numerical_encoder_sin_cos_stay_consistent_after_update() -> None:
    module = CosSinNumericalEncoder(
        frequency=torch.tensor([[1.0, 2.0]]), phase_shift=torch.zeros(1, 2), learnable=True
    )
    optimizer = torch.optim.SGD(module.parameters(), lr=0.1)
    module(torch.tensor([[1.0]])).sum().backward()
    optimizer.step()
    out = module(torch.tensor([[1.0]]))
    assert out[..., :2].pow(2).add(out[..., 2:].pow(2)).allclose(torch.ones(1, 1, 2))


def test_cos_sin_numerical_encoder_frozen_is_buffer_learnable() -> None:
    module = CosSinNumericalEncoder(frequency=torch.ones(1, 3), phase_shift=torch.zeros(1, 3))
    assert not module.learnable


def test_cos_sin_numerical_encoder_frozen_is_buffer_check2() -> None:
    module = CosSinNumericalEncoder(frequency=torch.ones(1, 3), phase_shift=torch.zeros(1, 3))
    assert len(list(module.parameters())) == 0


def test_cos_sin_numerical_encoder_frozen_is_buffer_state_dict() -> None:
    module = CosSinNumericalEncoder(frequency=torch.ones(1, 3), phase_shift=torch.zeros(1, 3))
    assert set(module.state_dict()) == {"frequency", "phase_shift"}


def test_cos_sin_numerical_encoder_learnable_is_parameter_learnable() -> None:
    module = CosSinNumericalEncoder(
        frequency=torch.ones(1, 3), phase_shift=torch.zeros(1, 3), learnable=True
    )
    assert module.learnable


def test_cos_sin_numerical_encoder_learnable_is_parameter_check2() -> None:
    module = CosSinNumericalEncoder(
        frequency=torch.ones(1, 3), phase_shift=torch.zeros(1, 3), learnable=True
    )
    assert {n for n, _ in module.named_parameters()} == {"frequency", "phase_shift"}


def test_cos_sin_numerical_encoder_create_rand_same_seed() -> None:
    module1 = CosSinNumericalEncoder.create("rand", 5, frequency_range=(0.2, 1), generator=42)
    module2 = CosSinNumericalEncoder.create("rand", 5, frequency_range=(0.2, 1), generator=42)
    assert module1.frequency.equal(module2.frequency)


def test_cos_sin_numerical_encoder_create_rand_different_seeds() -> None:
    module1 = CosSinNumericalEncoder.create("rand", 5, frequency_range=(0.2, 1), generator=1)
    module2 = CosSinNumericalEncoder.create("rand", 5, frequency_range=(0.2, 1), generator=2)
    assert not module1.frequency.equal(module2.frequency)


def test_cos_sin_numerical_encoder_create_rand_generator_object() -> None:
    module1 = CosSinNumericalEncoder.create(
        "rand", 5, frequency_range=(0.2, 1), generator=make_torch_generator(7)
    )
    module2 = CosSinNumericalEncoder.create("rand", 5, frequency_range=(0.2, 1), generator=7)
    assert module1.frequency.equal(module2.frequency)


def test_cos_sin_numerical_encoder_create_rand_generator_state_advances() -> None:
    generator = make_torch_generator(7)
    module1 = CosSinNumericalEncoder.create(
        "rand", 5, frequency_range=(0.2, 1), generator=generator
    )
    module2 = CosSinNumericalEncoder.create(
        "rand", 5, frequency_range=(0.2, 1), generator=generator
    )
    assert not module1.frequency.equal(module2.frequency)


def test_cos_sin_numerical_encoder_create_rand_seed_does_not_change_global_rng() -> None:
    torch.manual_seed(0)
    expected = torch.rand(3)
    torch.manual_seed(0)
    CosSinNumericalEncoder.create("rand", 5, frequency_range=(0.2, 1), generator=42)
    assert torch.rand(3).equal(expected)


def test_cos_sin_numerical_encoder_create_rand_global_rng_without_generator() -> None:
    torch.manual_seed(0)
    module1 = CosSinNumericalEncoder.create("rand", 5, frequency_range=(0.2, 1))
    torch.manual_seed(0)
    module2 = CosSinNumericalEncoder.create("rand", 5, frequency_range=(0.2, 1))
    assert module1.frequency.equal(module2.frequency)


@pytest.mark.parametrize("spacing", ["rand", "linspace", "logspace"])
@pytest.mark.parametrize("dtype", [torch.float64, torch.bfloat16])
def test_cos_sin_numerical_encoder_create_dtype(spacing: str, dtype: torch.dtype) -> None:
    module = CosSinNumericalEncoder.create(spacing, 3, frequency_range=(0.2, 1), dtype=dtype)
    assert module.frequency.dtype == dtype
    assert module.phase_shift.dtype == dtype


@pytest.mark.parametrize("spacing", ["rand", "linspace", "logspace"])
def test_cos_sin_numerical_encoder_create_dtype_default(spacing: str) -> None:
    module = CosSinNumericalEncoder.create(spacing, 3, frequency_range=(0.2, 1))
    assert module.frequency.dtype == torch.get_default_dtype()


@pytest.mark.parametrize("spacing", ["rand", "linspace", "logspace"])
@pytest.mark.parametrize("device", get_available_devices())
def test_cos_sin_numerical_encoder_create_device(spacing: str, device: str) -> None:
    module = CosSinNumericalEncoder.create(
        spacing, 3, frequency_range=(0.2, 1), device=torch.device(device)
    )
    assert module.frequency.device == torch.device(device)
    assert module.phase_shift.device == torch.device(device)


def test_cos_sin_numerical_encoder_create_n_features_shape() -> None:
    module = CosSinNumericalEncoder.create("linspace", 3, frequency_range=(0.2, 1), n_features=4)
    assert module.frequency.shape == (4, 3)


def test_cos_sin_numerical_encoder_create_n_features_phase_shift_shape() -> None:
    module = CosSinNumericalEncoder.create("linspace", 3, frequency_range=(0.2, 1), n_features=4)
    assert module.phase_shift.shape == (4, 3)


def test_cos_sin_numerical_encoder_create_n_features_is_keyword_only() -> None:
    with pytest.raises(TypeError, match=r"positional argument"):
        CosSinNumericalEncoder.create("linspace", 3, (0.2, 1), 2)


def test_cos_sin_numerical_encoder_create_n_features_invalid() -> None:
    with pytest.raises(ValueError, match=r"n_features must be greater than or equal to 1"):
        CosSinNumericalEncoder.create("linspace", 3, frequency_range=(0.2, 1), n_features=0)
