from __future__ import annotations

import pytest
import torch
from coola.equality import objects_are_equal

from sonnix.modules import (
    AsinhCosSinNumericalEncoder,
    AsinhNumericalEncoder,
    BaseNumericalEncoder,
    CosSinNumericalEncoder,
    PiecewiseLinearNumericalEncoder,
)

#########################################
#     Tests for BaseNumericalEncoder    #
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
        BaseNumericalEncoder()


@pytest.mark.parametrize("encoder", ENCODERS, ids=lambda e: type(e).__name__)
def test_numerical_encoder_input_incorrect_last_dimension(encoder: BaseNumericalEncoder) -> None:
    with pytest.raises(
        ValueError, match=r"Expected an input of shape \(\*, n_features=3\) but received \(5, 4\)"
    ):
        encoder(torch.ones(5, 4))


@pytest.mark.parametrize("encoder", ENCODERS, ids=lambda e: type(e).__name__)
def test_numerical_encoder_input_scalar(encoder: BaseNumericalEncoder) -> None:
    with pytest.raises(ValueError, match=r"but received \(\)"):
        encoder(torch.tensor(1.0))


@pytest.mark.parametrize("encoder", SINGLE_FEATURE_ENCODERS, ids=lambda e: type(e).__name__)
def test_numerical_encoder_single_feature_does_not_broadcast(encoder: BaseNumericalEncoder) -> None:
    with pytest.raises(
        ValueError, match=r"Expected an input of shape \(\*, n_features=1\) but received \(5, 4\)"
    ):
        encoder(torch.ones(5, 4))


@pytest.mark.parametrize("encoder", SINGLE_FEATURE_ENCODERS, ids=lambda e: type(e).__name__)
def test_numerical_encoder_single_feature_accepts_last_dimension_1(
    encoder: BaseNumericalEncoder,
) -> None:
    assert encoder(torch.ones(5, 1)).shape[:2] == (5, 1)


class IdentityEncoder(BaseNumericalEncoder):
    @property
    def n_features(self) -> int:
        return 2

    def _encode(self, input: torch.Tensor) -> torch.Tensor:  # noqa: A002
        return input.unsqueeze(-1)


def test_numerical_encoder_subclass_forward() -> None:
    assert objects_are_equal(IdentityEncoder()(torch.ones(4, 2)), torch.ones(4, 2, 1))


def test_numerical_encoder_subclass_checks_the_input() -> None:
    with pytest.raises(ValueError, match=r"Expected an input of shape \(\*, n_features=2\)"):
        IdentityEncoder()(torch.ones(4, 3))


@pytest.mark.parametrize("encoder", ENCODERS, ids=lambda e: type(e).__name__)
def test_numerical_encoder_is_a_base_numerical_encoder(encoder: BaseNumericalEncoder) -> None:
    assert isinstance(encoder, BaseNumericalEncoder)
