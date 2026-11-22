from __future__ import annotations

import copy
import pickle
import warnings

import pytest
import torch
from coola.equality import objects_are_equal
from coola.utils.tensor import get_available_devices
from torch import nn

from sonnix.modules import PiecewiseLinearNumericalEncoder

SIZES = (1, 2, 3)

#####################################################
#     Tests for PiecewiseLinearNumericalEncoder     #
#####################################################


def test_piecewise_linear_numerical_encoder_str() -> None:
    assert str(
        PiecewiseLinearNumericalEncoder(
            edges=torch.tensor([[0.0, 1.0, 2.0, 4.0], [2.0, 4.0, 6.0, 8.0]])
        )
    ).startswith("PiecewiseLinearNumericalEncoder(")


@pytest.mark.parametrize(
    "edges", [torch.tensor([1.0, 2.0, 4.0, 6.0]), torch.tensor([[1.0, 2.0, 4.0, 6.0]])]
)
def test_piecewise_linear_numerical_encoder_edges_1_feature_edges(edges: torch.Tensor) -> None:
    module = PiecewiseLinearNumericalEncoder(edges)
    assert module.edges.equal(torch.tensor([[1.0, 2.0, 4.0, 6.0]]))


@pytest.mark.parametrize(
    "edges", [torch.tensor([1.0, 2.0, 4.0, 6.0]), torch.tensor([[1.0, 2.0, 4.0, 6.0]])]
)
def test_piecewise_linear_numerical_encoder_edges_1_feature_width(edges: torch.Tensor) -> None:
    module = PiecewiseLinearNumericalEncoder(edges)
    assert module.width.equal(torch.tensor([[1.0, 2.0, 2.0]]))


def test_piecewise_linear_numerical_encoder_edges_2_features_edges() -> None:
    module = PiecewiseLinearNumericalEncoder(
        edges=torch.tensor([[0.0, 1.0, 2.0, 4.0], [2.0, 4.0, 6.0, 8.0]])
    )
    assert module.edges.equal(torch.tensor([[0.0, 1.0, 2.0, 4.0], [2.0, 4.0, 6.0, 8.0]]))


def test_piecewise_linear_numerical_encoder_edges_2_features_width() -> None:
    module = PiecewiseLinearNumericalEncoder(
        edges=torch.tensor([[0.0, 1.0, 2.0, 4.0], [2.0, 4.0, 6.0, 8.0]])
    )
    assert module.width.equal(torch.tensor([[1.0, 1.0, 2.0], [2.0, 2.0, 2.0]]))


def test_piecewise_linear_numerical_encoder_edges_sort_edges() -> None:
    module = PiecewiseLinearNumericalEncoder(
        edges=torch.tensor([[4.0, 2.0, 1.0, 0.0], [6.0, 4.0, 2.0, 8.0]])
    )
    assert module.edges.equal(torch.tensor([[0.0, 1.0, 2.0, 4.0], [2.0, 4.0, 6.0, 8.0]]))


def test_piecewise_linear_numerical_encoder_edges_sort_width() -> None:
    module = PiecewiseLinearNumericalEncoder(
        edges=torch.tensor([[4.0, 2.0, 1.0, 0.0], [6.0, 4.0, 2.0, 8.0]])
    )
    assert module.width.equal(torch.tensor([[1.0, 1.0, 2.0], [2.0, 2.0, 2.0]]))


def test_piecewise_linear_numerical_encoder_edges_incorrect_shape() -> None:
    with pytest.raises(ValueError, match=r"Incorrect shape for 'edges':"):
        PiecewiseLinearNumericalEncoder(torch.ones(2, 3, 4))


def test_piecewise_linear_numerical_encoder_input_size() -> None:
    assert (
        PiecewiseLinearNumericalEncoder(
            edges=torch.tensor([[0.0, 1.0, 2.0, 4.0], [2.0, 4.0, 6.0, 8.0]])
        ).n_features
        == 2
    )


def test_piecewise_linear_numerical_encoder_output_size() -> None:
    assert (
        PiecewiseLinearNumericalEncoder(
            edges=torch.tensor([[0.0, 1.0, 2.0, 4.0], [2.0, 4.0, 6.0, 8.0]])
        ).feature_size
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
    module = PiecewiseLinearNumericalEncoder(edges=torch.rand(n_features, feature_size + 1)).to(
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
    module = PiecewiseLinearNumericalEncoder(edges=torch.rand(n_features, feature_size + 1)).to(
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
    module = PiecewiseLinearNumericalEncoder(edges=torch.rand(3, 6)).to(device=device)
    module.train(mode)
    out = module(torch.rand(batch_size, 3, device=device, requires_grad=True))
    out.mean().backward()
    assert out.shape == (batch_size, 3, 5)
    assert out.device == device
    assert out.dtype == torch.float


def test_piecewise_linear_numerical_encoder_forward_edges_1_feature() -> None:
    module = PiecewiseLinearNumericalEncoder(edges=torch.tensor([[0.0, 1.0, 2.0, 4.0]]))
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


def test_piecewise_linear_numerical_encoder_forward_edges_2_features() -> None:
    module = PiecewiseLinearNumericalEncoder(
        edges=torch.tensor([[0.0, 1.0, 2.0, 4.0], [1.0, 2.0, 4.0, 8.0]])
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


def test_piecewise_linear_numerical_encoder_forward_edges_2_features_same_edges() -> None:
    module = PiecewiseLinearNumericalEncoder(
        edges=torch.tensor([[0.0, 1.0, 2.0, 4.0], [0.0, 1.0, 2.0, 4.0]])
    )
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


def test_piecewise_linear_numerical_encoder_single_edge() -> None:
    with pytest.raises(ValueError, match=r"n_edges must be greater than or equal to 2"):
        PiecewiseLinearNumericalEncoder(edges=torch.tensor([[0.0], [0.0]]))


def test_piecewise_linear_numerical_encoder_forward_same_bin() -> None:
    module = PiecewiseLinearNumericalEncoder(edges=torch.tensor([[0.0, 0.0], [0.0, 0.0]]))
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


def test_piecewise_linear_numerical_encoder_forward_same_edges() -> None:
    with pytest.warns(UserWarning, match=r"duplicate values"):
        module = PiecewiseLinearNumericalEncoder(
            edges=torch.tensor([[0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 1.0, 1.0]])
        )
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
                [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
                [[0.0, 0.0, 0.0], [1.0, 1.0, 0.0]],
                [[1.0, 0.5, 0.0], [1.0, 0.5, 0.0]],
                [[1.0, 1.0, 0.0], [0.0, 0.0, 0.0]],
                [[1.0, 1.0, 1.0], [0.0, 0.0, 0.0]],
            ]
        ),
    )


def test_piecewise_linear_numerical_encoder_duplicate_edges_width() -> None:
    with pytest.warns(UserWarning, match=r"duplicate values"):
        module = PiecewiseLinearNumericalEncoder(edges=torch.tensor([[0.0, 0.0, 1.0]]))
    assert module.width.equal(torch.tensor([[1.0, 1.0]]))


def test_piecewise_linear_numerical_encoder_duplicate_edges_saturate() -> None:
    with pytest.warns(UserWarning, match=r"duplicate values"):
        module = PiecewiseLinearNumericalEncoder(edges=torch.tensor([[0.0, 0.0, 1.0]]))
    assert module(torch.tensor([[0.5]])).allclose(torch.tensor([[[1.0, 0.5]]]))


def test_piecewise_linear_numerical_encoder_single_segment_duplicate_width() -> None:
    module = PiecewiseLinearNumericalEncoder(edges=torch.tensor([[0.0, 0.0]]))
    assert module.width.equal(torch.tensor([[1.0]]))


def test_piecewise_linear_numerical_encoder_no_warning_without_duplicate() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        PiecewiseLinearNumericalEncoder(edges=torch.tensor([[0.0, 1.0, 2.0]]))


def test_piecewise_linear_numerical_encoder_duplicate_edges_is_step() -> None:
    with pytest.warns(UserWarning, match=r"duplicate values"):
        module = PiecewiseLinearNumericalEncoder(edges=torch.tensor([[0.0, 0.0, 1.0]]))
    assert module.is_step.equal(torch.tensor([[True, False]]))


def test_piecewise_linear_numerical_encoder_single_segment_is_not_step() -> None:
    module = PiecewiseLinearNumericalEncoder(edges=torch.tensor([[0.0, 0.0]]))
    assert module.is_step.equal(torch.tensor([[False]]))


def test_piecewise_linear_numerical_encoder_duplicate_edges_values_are_bounded() -> None:
    with pytest.warns(UserWarning, match=r"duplicate values"):
        module = PiecewiseLinearNumericalEncoder(edges=torch.tensor([[0.0, 0.0, 1.0, 1.0]]))
    assert module(torch.tensor([[-100.0], [100.0]])).abs().max() <= 1.0


def test_piecewise_linear_numerical_encoder_is_step_not_in_state_dict() -> None:
    module = PiecewiseLinearNumericalEncoder(edges=torch.tensor([[0.0, 1.0, 2.0]]))
    assert sorted(module.state_dict().keys()) == ["edges"]


def test_piecewise_linear_numerical_encoder_from_quantiles_edges() -> None:
    module = PiecewiseLinearNumericalEncoder.from_quantiles(
        torch.arange(11, dtype=torch.float).view(11, 1), n_bins=2
    )
    assert objects_are_equal(module.edges, torch.tensor([[0.0, 5.0, 10.0]]))


def test_piecewise_linear_numerical_encoder_from_quantiles_width() -> None:
    module = PiecewiseLinearNumericalEncoder.from_quantiles(
        torch.arange(11, dtype=torch.float).view(11, 1), n_bins=2
    )
    assert objects_are_equal(module.width, torch.tensor([[5.0, 5.0]]))


def test_piecewise_linear_numerical_encoder_from_quantiles_1d() -> None:
    module = PiecewiseLinearNumericalEncoder.from_quantiles(
        torch.arange(11, dtype=torch.float), n_bins=2
    )
    assert objects_are_equal(module.edges, torch.tensor([[0.0, 5.0, 10.0]]))


def test_piecewise_linear_numerical_encoder_from_quantiles_two_features() -> None:
    x = torch.stack(
        [torch.arange(11, dtype=torch.float), 2 * torch.arange(11, dtype=torch.float)], 1
    )
    module = PiecewiseLinearNumericalEncoder.from_quantiles(x, n_bins=4)
    assert objects_are_equal(
        module.edges,
        torch.tensor([[0.0, 2.5, 5.0, 7.5, 10.0], [0.0, 5.0, 10.0, 15.0, 20.0]]),
    )


def test_piecewise_linear_numerical_encoder_from_quantiles_shape() -> None:
    module = PiecewiseLinearNumericalEncoder.from_quantiles(torch.randn(50, 3), n_bins=4)
    assert (module.n_features, module.feature_size) == (3, 4)


def test_piecewise_linear_numerical_encoder_from_quantiles_equals_constructor() -> None:
    torch.manual_seed(0)
    x = torch.randn(100, 2)
    module = PiecewiseLinearNumericalEncoder.from_quantiles(x, n_bins=4)
    expected = PiecewiseLinearNumericalEncoder(
        torch.quantile(x, torch.linspace(0.0, 1.0, 5), dim=0).t()
    )
    assert objects_are_equal(module(x), expected(x))


def test_piecewise_linear_numerical_encoder_from_quantiles_encodes_the_range() -> None:
    torch.manual_seed(0)
    x = torch.randn(100, 2)
    module = PiecewiseLinearNumericalEncoder.from_quantiles(x, n_bins=4)
    assert module(x.amax(dim=0, keepdim=True)).eq(1.0).all()


def test_piecewise_linear_numerical_encoder_from_quantiles_float64() -> None:
    module = PiecewiseLinearNumericalEncoder.from_quantiles(
        torch.arange(11, dtype=torch.float64), n_bins=2
    )
    assert module.edges.dtype == torch.float64


def test_piecewise_linear_numerical_encoder_from_quantiles_duplicates_warn() -> None:
    with pytest.warns(UserWarning, match=r"duplicate values"):
        PiecewiseLinearNumericalEncoder.from_quantiles(torch.zeros(10, 1), n_bins=4)


@pytest.mark.parametrize("n_bins", [0, -2])
def test_piecewise_linear_numerical_encoder_from_quantiles_invalid_n_bins(n_bins: int) -> None:
    with pytest.raises(ValueError, match=r"n_bins must be greater than or equal to 1"):
        PiecewiseLinearNumericalEncoder.from_quantiles(torch.randn(10, 1), n_bins=n_bins)


def test_piecewise_linear_numerical_encoder_from_quantiles_invalid_shape() -> None:
    with pytest.raises(ValueError, match=r"Incorrect shape for 'x':"):
        PiecewiseLinearNumericalEncoder.from_quantiles(torch.randn(10, 2, 3), n_bins=3)


def test_piecewise_linear_numerical_encoder_from_quantiles_does_not_modify_the_input() -> None:
    x = torch.tensor([[3.0], [1.0], [2.0]])
    PiecewiseLinearNumericalEncoder.from_quantiles(x, n_bins=3)
    assert objects_are_equal(x, torch.tensor([[3.0], [1.0], [2.0]]))


@pytest.mark.parametrize(
    ("edges", "lower", "upper"),
    [
        ([[1.0, 2.0]], [-torch.inf], [torch.inf]),
        ([[1.0, 2.0, 4.0]], [-torch.inf, 0.0], [1.0, torch.inf]),
        ([[1.0, 2.0, 4.0, 8.0]], [-torch.inf, 0.0, 0.0], [1.0, 1.0, torch.inf]),
    ],
)
def test_piecewise_linear_numerical_encoder_clamp_bounds(
    edges: list[list[float]], lower: list[float], upper: list[float]
) -> None:
    module = PiecewiseLinearNumericalEncoder(torch.tensor(edges))
    assert objects_are_equal(
        (module.lower, module.upper), (torch.tensor(lower), torch.tensor(upper))
    )


def test_piecewise_linear_numerical_encoder_clamp_bounds_not_in_state_dict() -> None:
    module = PiecewiseLinearNumericalEncoder(torch.tensor([[1.0, 2.0, 4.0]]))
    assert sorted(module.state_dict()) == ["edges"]


def test_piecewise_linear_numerical_encoder_clamp_bounds_follow_dtype() -> None:
    module = PiecewiseLinearNumericalEncoder(torch.tensor([[1.0, 2.0, 4.0]])).to(torch.float64)
    assert module.lower.dtype == torch.float64


def test_piecewise_linear_numerical_encoder_forward_matches_reference() -> None:
    bins = torch.tensor([[1.0, 2.0, 4.0, 8.0], [0.0, 2.0, 4.0, 6.0]])
    x = torch.tensor([[-5.0, 100.0], [1.5, 3.0], [2.0, 4.0], [9.0, -1.0]])
    expected = (x[..., None] - bins[:, :-1]) / bins.diff()
    expected = torch.cat(
        [
            expected[..., :1].clamp_max(1.0),
            expected[..., 1:-1].clamp(0.0, 1.0),
            expected[..., -1:].clamp_min(0.0),
        ],
        dim=-1,
    )
    assert PiecewiseLinearNumericalEncoder(bins)(x).allclose(expected)


####################################
#     Tests for the edges input    #
####################################


def test_piecewise_linear_numerical_encoder_integer_edges_dtype() -> None:
    assert PiecewiseLinearNumericalEncoder(
        edges=torch.tensor([[1, 2, 4, 8]])
    ).edges.is_floating_point()


def test_piecewise_linear_numerical_encoder_integer_edges_forward() -> None:
    module = PiecewiseLinearNumericalEncoder(edges=torch.tensor([[1, 2, 4, 8]]))
    assert module(torch.tensor([[3.0]])).allclose(torch.tensor([[[1.0, 0.5, 0.0]]]))


@pytest.mark.parametrize("n_edges", [0, 1])
def test_piecewise_linear_numerical_encoder_not_enough_edges(n_edges: int) -> None:
    with pytest.raises(ValueError, match=r"n_edges must be greater than or equal to 2"):
        PiecewiseLinearNumericalEncoder(edges=torch.zeros(1, n_edges))


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_piecewise_linear_numerical_encoder_non_finite_edges(value: float) -> None:
    with pytest.raises(ValueError, match=r"'edges' must contain only finite values"):
        PiecewiseLinearNumericalEncoder(edges=torch.tensor([[1.0, value, 4.0]]))


def test_piecewise_linear_numerical_encoder_on_duplicate_ignore_is_silent() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        PiecewiseLinearNumericalEncoder(
            edges=torch.tensor([[1.0, 1.0, 2.0]]), on_duplicate="ignore"
        )


def test_piecewise_linear_numerical_encoder_on_duplicate_ignore_keeps_step() -> None:
    module = PiecewiseLinearNumericalEncoder(
        edges=torch.tensor([[1.0, 1.0, 2.0]]), on_duplicate="ignore"
    )
    assert module.is_step.equal(torch.tensor([[True, False]]))


def test_piecewise_linear_numerical_encoder_on_duplicate_incorrect() -> None:
    with pytest.raises(ValueError, match=r"on_duplicate"):
        PiecewiseLinearNumericalEncoder(edges=torch.tensor([[1.0, 2.0]]), on_duplicate="incorrect")


def test_piecewise_linear_numerical_encoder_from_quantiles_on_duplicate_ignore_is_silent() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        PiecewiseLinearNumericalEncoder.from_quantiles(
            torch.zeros(10, 1), n_bins=4, on_duplicate="ignore"
        )


#################################
#     Tests for the state     #
#################################

DUPLICATE_EDGES = torch.tensor([[0.0, 1.0, 1.0, 3.0]])
DISTINCT_EDGES = torch.tensor([[0.0, 1.0, 2.0, 3.0]])


def _load_duplicate_edges() -> PiecewiseLinearNumericalEncoder:
    module = PiecewiseLinearNumericalEncoder(DISTINCT_EDGES)
    module.load_state_dict(
        PiecewiseLinearNumericalEncoder(DUPLICATE_EDGES, on_duplicate="ignore").state_dict()
    )
    return module


def test_piecewise_linear_numerical_encoder_load_state_dict_forward() -> None:
    assert objects_are_equal(
        _load_duplicate_edges()(torch.tensor([[1.5]])), torch.tensor([[[1.0, 1.0, 0.25]]])
    )


def test_piecewise_linear_numerical_encoder_load_state_dict_is_step() -> None:
    assert _load_duplicate_edges().is_step.equal(torch.tensor([[False, True, False]]))


def test_piecewise_linear_numerical_encoder_load_state_dict_width() -> None:
    assert _load_duplicate_edges().width.equal(torch.tensor([[1.0, 1.0, 2.0]]))


def test_piecewise_linear_numerical_encoder_load_state_dict_in_a_parent() -> None:
    target = nn.Sequential(PiecewiseLinearNumericalEncoder(DISTINCT_EDGES))
    source = nn.Sequential(PiecewiseLinearNumericalEncoder(DUPLICATE_EDGES, on_duplicate="ignore"))
    target.load_state_dict(source.state_dict())
    assert objects_are_equal(target(torch.tensor([[1.5]])), torch.tensor([[[1.0, 1.0, 0.25]]]))


def test_piecewise_linear_numerical_encoder_load_state_dict_old_width_key_is_unexpected() -> None:
    module = PiecewiseLinearNumericalEncoder(DISTINCT_EDGES)
    state_dict = {**module.state_dict(), "width": torch.ones(1, 3)}
    assert module.load_state_dict(state_dict, strict=False).unexpected_keys == ["width"]


def test_piecewise_linear_numerical_encoder_load_state_dict_old_width_key_fails_if_strict() -> None:
    module = PiecewiseLinearNumericalEncoder(DISTINCT_EDGES)
    state_dict = {**module.state_dict(), "width": torch.ones(1, 3)}
    with pytest.raises(RuntimeError, match=r"Unexpected key\(s\) in state_dict: \"width\""):
        module.load_state_dict(state_dict)


def test_piecewise_linear_numerical_encoder_load_state_dict_different_n_edges() -> None:
    module = PiecewiseLinearNumericalEncoder(DISTINCT_EDGES)
    with pytest.raises(RuntimeError, match=r"size mismatch for edges"):
        module.load_state_dict(
            PiecewiseLinearNumericalEncoder(torch.arange(5.0).view(1, 5)).state_dict()
        )


def test_piecewise_linear_numerical_encoder_deepcopy_keeps_the_hook() -> None:
    module = copy.deepcopy(PiecewiseLinearNumericalEncoder(DISTINCT_EDGES))
    module.load_state_dict(
        PiecewiseLinearNumericalEncoder(DUPLICATE_EDGES, on_duplicate="ignore").state_dict()
    )
    assert module.is_step.equal(torch.tensor([[False, True, False]]))


def test_piecewise_linear_numerical_encoder_pickle() -> None:
    module = pickle.loads(pickle.dumps(PiecewiseLinearNumericalEncoder(DISTINCT_EDGES)))  # noqa: S301
    assert objects_are_equal(module(torch.tensor([[1.5]])), torch.tensor([[[1.0, 0.5, 0.0]]]))


def test_piecewise_linear_numerical_encoder_derive_after_edges_change() -> None:
    module = PiecewiseLinearNumericalEncoder(DISTINCT_EDGES)
    module.edges.copy_(torch.tensor([[0.0, 2.0, 4.0, 8.0]]))
    module._derive()
    assert module.width.equal(torch.tensor([[2.0, 2.0, 4.0]]))


def test_piecewise_linear_numerical_encoder_rederive_hook() -> None:
    module = PiecewiseLinearNumericalEncoder(DISTINCT_EDGES)
    module.edges.copy_(torch.tensor([[0.0, 2.0, 4.0, 8.0]]))
    PiecewiseLinearNumericalEncoder._rederive(module, None)
    assert module.width.equal(torch.tensor([[2.0, 2.0, 4.0]]))


def test_piecewise_linear_numerical_encoder_derived_tensors_follow_dtype() -> None:
    module = PiecewiseLinearNumericalEncoder(DISTINCT_EDGES).to(torch.float64)
    assert module.width.dtype == torch.float64


def test_piecewise_linear_numerical_encoder_from_quantiles_feature_size_is_n_bins() -> None:
    module = PiecewiseLinearNumericalEncoder.from_quantiles(torch.randn(50, 2), n_bins=3)
    assert module.feature_size == 3


@pytest.mark.parametrize("n_edges", [2, 3, 5])
def test_piecewise_linear_numerical_encoder_feature_size_is_n_edges_minus_one(
    n_edges: int,
) -> None:
    assert PiecewiseLinearNumericalEncoder(torch.rand(2, n_edges)).feature_size == n_edges - 1


def test_piecewise_linear_numerical_encoder_extra_repr() -> None:
    assert (
        repr(PiecewiseLinearNumericalEncoder(torch.rand(2, 4)))
        == "PiecewiseLinearNumericalEncoder(n_features=2, feature_size=3)"
    )


def test_piecewise_linear_numerical_encoder_start() -> None:
    assert PiecewiseLinearNumericalEncoder(DISTINCT_EDGES).start.equal(
        torch.tensor([[0.0, 1.0, 2.0]])
    )
