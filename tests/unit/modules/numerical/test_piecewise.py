from __future__ import annotations

import pytest
import torch
from coola.equality import objects_are_equal
from coola.utils.tensor import get_available_devices

from sonnix.modules import PiecewiseLinearNumericalEncoder

SIZES = (1, 2, 3)

#####################################################
#     Tests for PiecewiseLinearNumericalEncoder     #
#####################################################


def test_piecewise_linear_numerical_encoder_str() -> None:
    assert str(
        PiecewiseLinearNumericalEncoder(
            bins=torch.tensor([[0.0, 1.0, 2.0, 4.0], [2.0, 4.0, 6.0, 8.0]])
        )
    ).startswith("PiecewiseLinearNumericalEncoder(")


@pytest.mark.parametrize(
    "bins", [torch.tensor([1.0, 2.0, 4.0, 6.0]), torch.tensor([[1.0, 2.0, 4.0, 6.0]])]
)
def test_piecewise_linear_numerical_encoder_bins_1_feature(bins: torch.Tensor) -> None:
    module = PiecewiseLinearNumericalEncoder(bins)
    assert module.edges.equal(torch.tensor([[1.0, 2.0, 4.0]]))
    assert module.width.equal(torch.tensor([[1.0, 2.0, 2.0]]))


def test_piecewise_linear_numerical_encoder_bins_2_features() -> None:
    module = PiecewiseLinearNumericalEncoder(
        bins=torch.tensor([[0.0, 1.0, 2.0, 4.0], [2.0, 4.0, 6.0, 8.0]])
    )
    assert module.edges.equal(torch.tensor([[0.0, 1.0, 2.0], [2.0, 4.0, 6.0]]))
    assert module.width.equal(torch.tensor([[1.0, 1.0, 2.0], [2.0, 2.0, 2.0]]))


def test_piecewise_linear_numerical_encoder_bins_1() -> None:
    module = PiecewiseLinearNumericalEncoder(bins=torch.tensor([[0.0]]))
    assert module.edges.equal(torch.tensor([[0.0]]))
    assert module.width.equal(torch.tensor([[1.0]]))


def test_piecewise_linear_numerical_encoder_bins_sort() -> None:
    module = PiecewiseLinearNumericalEncoder(
        bins=torch.tensor([[4.0, 2.0, 1.0, 0.0], [6.0, 4.0, 2.0, 8.0]])
    )
    assert module.edges.equal(torch.tensor([[0.0, 1.0, 2.0], [2.0, 4.0, 6.0]]))
    assert module.width.equal(torch.tensor([[1.0, 1.0, 2.0], [2.0, 2.0, 2.0]]))


def test_piecewise_linear_numerical_encoder_bins_incorrect_shape() -> None:
    with pytest.raises(RuntimeError, match=r"Incorrect shape for 'bins':"):
        PiecewiseLinearNumericalEncoder(torch.ones(2, 3, 4))


def test_piecewise_linear_numerical_encoder_input_size() -> None:
    assert (
        PiecewiseLinearNumericalEncoder(
            bins=torch.tensor([[0.0, 1.0, 2.0, 4.0], [2.0, 4.0, 6.0, 8.0]])
        ).input_size
        == 2
    )


def test_piecewise_linear_numerical_encoder_output_size() -> None:
    assert (
        PiecewiseLinearNumericalEncoder(
            bins=torch.tensor([[0.0, 1.0, 2.0, 4.0], [2.0, 4.0, 6.0, 8.0]])
        ).output_size
        == 3
    )


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("batch_size", SIZES)
@pytest.mark.parametrize("n_features", SIZES)
@pytest.mark.parametrize("feature_size", SIZES)
@pytest.mark.parametrize("mode", [True, False])
def test_piecewise_linear_numerical_encoder_forward_2d(
    device: str, batch_size: int, n_features: int, feature_size: int, mode: bool
) -> None:
    device = torch.device(device)
    module = PiecewiseLinearNumericalEncoder(bins=torch.rand(n_features, feature_size + 1)).to(
        device=device
    )
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
def test_piecewise_linear_numerical_encoder_forward_3d(
    device: str, d1: int, d2: int, n_features: int, feature_size: int, mode: bool
) -> None:
    device = torch.device(device)
    module = PiecewiseLinearNumericalEncoder(bins=torch.rand(n_features, feature_size + 1)).to(
        device=device
    )
    module.train(mode)
    out = module(torch.rand(d1, d2, n_features, device=device))
    assert out.shape == (d1, d2, n_features, feature_size)
    assert out.device == device
    assert out.dtype == torch.float


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("batch_size", SIZES)
@pytest.mark.parametrize("mode", [True, False])
def test_piecewise_linear_numerical_encoder_backward(
    device: str, batch_size: int, mode: bool
) -> None:
    device = torch.device(device)
    module = PiecewiseLinearNumericalEncoder(bins=torch.rand(3, 6)).to(device=device)
    module.train(mode)
    out = module(torch.rand(batch_size, 3, device=device, requires_grad=True))
    out.mean().backward()
    assert out.shape == (batch_size, 3, 5)
    assert out.device == device
    assert out.dtype == torch.float


def test_piecewise_linear_numerical_encoder_forward_bins_1_feature() -> None:
    module = PiecewiseLinearNumericalEncoder(bins=torch.tensor([[0.0, 1.0, 2.0, 4.0]]))
    assert objects_are_equal(
        module(torch.tensor([[-1.0], [0.0], [0.5], [1.0], [1.5], [2.0], [3.0], [4.0], [5.0]])),
        torch.tensor(
            [
                [[-1.0, 0.0, 0.0]],
                [[0.0, 0.0, 0.0]],
                [[0.5, 0.0, 0.0]],
                [[1.0, 0.0, 0.0]],
                [[1.0, 0.5, 0.0]],
                [[1.0, 1.0, 0.0]],
                [[1.0, 1.0, 0.5]],
                [[1.0, 1.0, 1.0]],
                [[1.0, 1.0, 1.5]],
            ],
        ),
    )


def test_piecewise_linear_numerical_encoder_forward_bins_2_features() -> None:
    module = PiecewiseLinearNumericalEncoder(
        bins=torch.tensor([[0.0, 1.0, 2.0, 4.0], [1.0, 2.0, 4.0, 8.0]])
    )
    assert objects_are_equal(
        module(
            torch.tensor(
                [
                    [-1.0, 0.0],
                    [0.0, 1.0],
                    [0.5, 1.5],
                    [1.0, 2.0],
                    [1.5, 3.0],
                    [2.0, 4.0],
                    [3.0, 6.0],
                    [4.0, 8.0],
                    [5.0, 9.0],
                ]
            )
        ),
        torch.tensor(
            [
                [[-1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]],
                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                [[0.5, 0.0, 0.0], [0.5, 0.0, 0.0]],
                [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                [[1.0, 0.5, 0.0], [1.0, 0.5, 0.0]],
                [[1.0, 1.0, 0.0], [1.0, 1.0, 0.0]],
                [[1.0, 1.0, 0.5], [1.0, 1.0, 0.5]],
                [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
                [[1.0, 1.0, 1.5], [1.0, 1.0, 1.25]],
            ],
        ),
    )


def test_piecewise_linear_numerical_encoder_forward_bins_2_features_same_bins() -> None:
    module = PiecewiseLinearNumericalEncoder(bins=torch.tensor([[0.0, 1.0, 2.0, 4.0]]))
    assert objects_are_equal(
        module(
            torch.tensor(
                [
                    [-1.0, 5.0],
                    [0.0, 4.0],
                    [0.5, 3.0],
                    [1.0, 2.0],
                    [1.5, 1.5],
                    [2.0, 1.0],
                    [3.0, 0.5],
                    [4.0, 0.0],
                    [5.0, -1.0],
                ]
            )
        ),
        torch.tensor(
            [
                [[-1.0, 0.0, 0.0], [1.0, 1.0, 1.5]],
                [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
                [[0.5, 0.0, 0.0], [1.0, 1.0, 0.5]],
                [[1.0, 0.0, 0.0], [1.0, 1.0, 0.0]],
                [[1.0, 0.5, 0.0], [1.0, 0.5, 0.0]],
                [[1.0, 1.0, 0.0], [1.0, 0.0, 0.0]],
                [[1.0, 1.0, 0.5], [0.5, 0.0, 0.0]],
                [[1.0, 1.0, 1.0], [0.0, 0.0, 0.0]],
                [[1.0, 1.0, 1.5], [-1.0, 0.0, 0.0]],
            ],
        ),
    )


def test_piecewise_linear_numerical_encoder_forward_1_bin() -> None:
    module = PiecewiseLinearNumericalEncoder(bins=torch.tensor([[0.0]]))
    assert objects_are_equal(
        module(
            torch.tensor(
                [
                    [-1.0, 5.0],
                    [0.0, 4.0],
                    [0.5, 3.0],
                    [1.0, 2.0],
                    [1.5, 1.5],
                    [2.0, 1.0],
                    [3.0, 0.5],
                    [4.0, 0.0],
                    [5.0, -1.0],
                ]
            )
        ),
        torch.tensor(
            [
                [[-1.0], [5.0]],
                [[0.0], [4.0]],
                [[0.5], [3.0]],
                [[1.0], [2.0]],
                [[1.5], [1.5]],
                [[2.0], [1.0]],
                [[3.0], [0.5]],
                [[4.0], [0.0]],
                [[5.0], [-1.0]],
            ]
        ),
    )


def test_piecewise_linear_numerical_encoder_forward_same_bin() -> None:
    module = PiecewiseLinearNumericalEncoder(bins=torch.tensor([[0.0, 0.0]]))
    assert objects_are_equal(
        module(
            torch.tensor(
                [
                    [-1.0, 5.0],
                    [0.0, 4.0],
                    [0.5, 3.0],
                    [1.0, 2.0],
                    [1.5, 1.5],
                    [2.0, 1.0],
                    [3.0, 0.5],
                    [4.0, 0.0],
                    [5.0, -1.0],
                ]
            )
        ),
        torch.tensor(
            [
                [[-1.0], [5.0]],
                [[0.0], [4.0]],
                [[0.5], [3.0]],
                [[1.0], [2.0]],
                [[1.5], [1.5]],
                [[2.0], [1.0]],
                [[3.0], [0.5]],
                [[4.0], [0.0]],
                [[5.0], [-1.0]],
            ]
        ),
    )


def test_piecewise_linear_numerical_encoder_forward_same_bins() -> None:
    module = PiecewiseLinearNumericalEncoder(bins=torch.tensor([[0.0, 0.0, 1.0, 1.0]]))
    assert objects_are_equal(
        module(
            torch.tensor(
                [
                    [-1.0, 2.0],
                    [0.0, 1.0],
                    [0.5, 0.5],
                    [1.0, 0.0],
                    [2.0, -1.0],
                ]
            )
        ),
        torch.tensor(
            [
                [[-1.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
                [[0.0, 0.0, 0.0], [1.0, 1.0, 0.0]],
                [[0.5, 0.5, 0.0], [0.5, 0.5, 0.0]],
                [[1.0, 1.0, 0.0], [0.0, 0.0, 0.0]],
                [[1.0, 1.0, 1.0], [-1.0, 0.0, 0.0]],
            ]
        ),
    )
