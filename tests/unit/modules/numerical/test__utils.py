from __future__ import annotations

import pytest
import torch
from coola.equality import objects_are_equal

from sonnix.modules import (
    AsinhCosSinNumericalEncoder,
    AsinhNumericalEncoder,
    CosSinNumericalEncoder,
    PiecewiseLinearNumericalEncoder,
)
from sonnix.modules.numerical._utils import (
    NumericalEncoder,
    create_spaced_values,
    prepare_tensor_param,
    resolve_range,
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
    with pytest.raises(ValueError, match=r"Incorrect shape for 'scale':"):
        prepare_tensor_param(torch.ones(2, 3, 4), name="scale")


##########################################
#     Tests for create_spaced_values     #
##########################################


def test_create_spaced_values_linspace() -> None:
    assert create_spaced_values("linspace", 3, low=1.0, high=3.0).equal(
        torch.tensor([[1.0, 2.0, 3.0]])
    )


def test_create_spaced_values_logspace() -> None:
    assert create_spaced_values("logspace", 3, low=0.01, high=1.0).equal(
        torch.tensor([[0.01, 0.1, 1.0]])
    )


@pytest.mark.parametrize("spacing", ["linspace", "logspace"])
def test_create_spaced_values_deterministic_features_are_equal(spacing: str) -> None:
    values = create_spaced_values(spacing, 4, low=1.0, high=3.0, n_features=3)
    assert values[0].equal(values[2])


def test_create_spaced_values_rand_shape() -> None:
    assert create_spaced_values("rand", 4, low=1.0, high=3.0, n_features=3).shape == (3, 4)


def test_create_spaced_values_rand_in_range() -> None:
    values = create_spaced_values("rand", 100, low=1.0, high=3.0, generator=1)
    assert ((values >= 1.0) & (values <= 3.0)).all()


def test_create_spaced_values_rand_features_are_independent() -> None:
    values = create_spaced_values("rand", 4, low=1.0, high=3.0, n_features=2, generator=1)
    assert not values[0].equal(values[1])


def test_create_spaced_values_rand_same_seed() -> None:
    assert create_spaced_values("rand", 4, low=1.0, high=3.0, generator=1).equal(
        create_spaced_values("rand", 4, low=1.0, high=3.0, generator=1)
    )


def test_create_spaced_values_dtype() -> None:
    assert create_spaced_values("linspace", 3, low=1.0, high=3.0, dtype=torch.float64).dtype == (
        torch.float64
    )


def test_create_spaced_values_invalid_spacing() -> None:
    with pytest.raises(ValueError, match=r"spacing"):
        create_spaced_values("incorrect", 3, low=1.0, high=3.0)


@pytest.mark.parametrize("n_features", [0, -1])
def test_create_spaced_values_invalid_n_features(n_features: int) -> None:
    with pytest.raises(ValueError, match=r"n_features must be greater than or equal to 1"):
        create_spaced_values("linspace", 3, low=1.0, high=3.0, n_features=n_features)


def test_create_spaced_values_invalid_n_features_type() -> None:
    with pytest.raises(TypeError, match=r"n_features must be an int"):
        create_spaced_values("linspace", 3, low=1.0, high=3.0, n_features=1.5)


####################################
#     Tests for resolve_range      #
####################################


def test_resolve_range_range() -> None:
    assert resolve_range("scale", (1.0, 3.0), None) == (1.0, 3.0)


def test_resolve_range_value_range() -> None:
    assert resolve_range("scale", None, (0.5, 4.0)) == (0.25, 2.0)


def test_resolve_range_both() -> None:
    with pytest.raises(
        ValueError, match=r"Exactly one of 'scale_range' and 'value_range' has to be provided"
    ):
        resolve_range("scale", (1.0, 3.0), (0.5, 4.0))


def test_resolve_range_none() -> None:
    with pytest.raises(
        ValueError, match=r"Exactly one of 'frequency_range' and 'value_range' has to be provided"
    ):
        resolve_range("frequency", None, None)


def test_resolve_range_invalid_value_range() -> None:
    with pytest.raises(ValueError, match=r"'max_abs_value'"):
        resolve_range("scale", None, (4.0, 0.5))


def test_resolve_range_does_not_validate_range() -> None:
    assert resolve_range("scale", (-1.0, -2.0), None) == (-1.0, -2.0)


#########################################
#     Tests for NumericalEncoder        #
#########################################

ENCODERS = [
    AsinhNumericalEncoder(torch.ones(3, 2)),
    CosSinNumericalEncoder(torch.ones(3, 2), torch.zeros(3, 2)),
    AsinhCosSinNumericalEncoder(torch.ones(3, 2), torch.zeros(3, 2)),
    PiecewiseLinearNumericalEncoder(torch.tensor([[0.0, 1.0, 2.0]] * 3)),
]
SINGLE_FEATURE_ENCODERS = [
    AsinhNumericalEncoder(torch.ones(1, 2)),
    CosSinNumericalEncoder(torch.ones(1, 2), torch.zeros(1, 2)),
    AsinhCosSinNumericalEncoder(torch.ones(1, 2), torch.zeros(1, 2)),
    PiecewiseLinearNumericalEncoder(torch.tensor([[0.0, 1.0, 2.0]])),
]


def test_numerical_encoder_abstract() -> None:
    with pytest.raises(TypeError, match=r"abstract"):
        NumericalEncoder()


@pytest.mark.parametrize("encoder", ENCODERS, ids=lambda e: type(e).__name__)
def test_numerical_encoder_input_incorrect_last_dimension(encoder: NumericalEncoder) -> None:
    with pytest.raises(
        ValueError, match=r"Expected an input of shape \(\*, n_features=3\) but received \(5, 4\)"
    ):
        encoder(torch.ones(5, 4))


@pytest.mark.parametrize("encoder", ENCODERS, ids=lambda e: type(e).__name__)
def test_numerical_encoder_input_scalar(encoder: NumericalEncoder) -> None:
    with pytest.raises(ValueError, match=r"but received \(\)"):
        encoder(torch.tensor(1.0))


@pytest.mark.parametrize("encoder", SINGLE_FEATURE_ENCODERS, ids=lambda e: type(e).__name__)
def test_numerical_encoder_single_feature_does_not_broadcast(encoder: NumericalEncoder) -> None:
    with pytest.raises(
        ValueError, match=r"Expected an input of shape \(\*, n_features=1\) but received \(5, 4\)"
    ):
        encoder(torch.ones(5, 4))


@pytest.mark.parametrize("encoder", SINGLE_FEATURE_ENCODERS, ids=lambda e: type(e).__name__)
def test_numerical_encoder_single_feature_accepts_last_dimension_1(
    encoder: NumericalEncoder,
) -> None:
    assert encoder(torch.ones(5, 1)).shape[:2] == (5, 1)
