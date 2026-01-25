from __future__ import annotations

from unittest.mock import patch

import pytest
import torch
from coola.equality import objects_are_allclose, objects_are_equal
from coola.utils.tensor import get_available_devices

from sonnix.functional.reduction import VALID_REDUCTIONS
from sonnix.modules import GeneralRobustRegressionLoss

SIZES = (1, 2, 3)


#################################################
#     Tests for GeneralRobustRegressionLoss     #
#################################################


def test_general_robust_regression_loss_str() -> None:
    assert str(GeneralRobustRegressionLoss()).startswith("GeneralRobustRegressionLoss(")


@pytest.mark.parametrize("alpha", [-1, 0, 1, 2])
def test_general_robust_regression_loss_alpha(alpha: float) -> None:
    assert GeneralRobustRegressionLoss(alpha=alpha)._alpha == alpha


def test_general_robust_regression_loss_alpha_default() -> None:
    assert GeneralRobustRegressionLoss()._alpha == 2.0


@pytest.mark.parametrize("scale", [1, 2])
def test_general_robust_regression_loss_scale(scale: float) -> None:
    assert GeneralRobustRegressionLoss(scale=scale)._scale == scale


def test_general_robust_regression_loss_scale_default() -> None:
    assert GeneralRobustRegressionLoss()._scale == 1.0


def test_general_robust_regression_loss_incorrect_scale() -> None:
    with pytest.raises(ValueError, match=r"scale has to be greater than 0 but received"):
        GeneralRobustRegressionLoss(scale=0)


@pytest.mark.parametrize("reduction", VALID_REDUCTIONS)
def test_general_robust_regression_loss_reduction(reduction: str) -> None:
    assert GeneralRobustRegressionLoss(reduction=reduction).reduction == reduction


def test_general_robust_regression_loss_reduction_default() -> None:
    assert GeneralRobustRegressionLoss().reduction == "mean"


def test_general_robust_regression_loss_incorrect_reduction() -> None:
    with pytest.raises(ValueError, match=r"Incorrect reduction: incorrect"):
        GeneralRobustRegressionLoss(reduction="incorrect")


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("batch_size", SIZES)
@pytest.mark.parametrize("alpha", [0, 1, 2])
def test_general_robust_regression_loss_forward_1d(
    device: str, batch_size: int, alpha: float
) -> None:
    device = torch.device(device)
    criterion = GeneralRobustRegressionLoss(alpha=alpha)
    out = criterion(
        prediction=torch.randn(batch_size, dtype=torch.float, device=device, requires_grad=True),
        target=torch.randn(batch_size, dtype=torch.float, device=device),
    )
    out.backward()
    assert out.numel() == 1
    assert out.dtype == torch.float
    assert out.device == device


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("batch_size", SIZES)
@pytest.mark.parametrize("feature_size", SIZES)
@pytest.mark.parametrize("alpha", [0, 1, 2])
def test_general_robust_regression_loss_forward_2d(
    device: str, batch_size: int, feature_size: int, alpha: float
) -> None:
    device = torch.device(device)
    criterion = GeneralRobustRegressionLoss(alpha=alpha)
    out = criterion(
        prediction=torch.randn(
            batch_size, feature_size, dtype=torch.float, device=device, requires_grad=True
        ),
        target=torch.randn(batch_size, feature_size, dtype=torch.float, device=device),
    )
    out.backward()
    assert out.numel() == 1
    assert out.dtype == torch.float
    assert out.device == device


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("batch_size", SIZES)
@pytest.mark.parametrize("seq_len", SIZES)
@pytest.mark.parametrize("feature_size", SIZES)
@pytest.mark.parametrize("alpha", [0, 1, 2])
def test_general_robust_regression_loss_forward_3d(
    device: str, batch_size: int, seq_len: int, feature_size: int, alpha: float
) -> None:
    device = torch.device(device)
    criterion = GeneralRobustRegressionLoss(alpha=alpha)
    out = criterion(
        prediction=torch.randn(
            batch_size, seq_len, feature_size, dtype=torch.float, device=device, requires_grad=True
        ),
        target=torch.randn(batch_size, seq_len, feature_size, dtype=torch.float, device=device),
    )
    out.backward()
    assert out.numel() == 1
    assert out.dtype == torch.float
    assert out.device == device


def test_general_robust_regression_loss_forward_alpha_2_scale_1() -> None:
    criterion = GeneralRobustRegressionLoss(alpha=2.0, scale=1.0)
    assert objects_are_equal(
        criterion(
            prediction=torch.ones(2, 3, dtype=torch.float),
            target=-torch.ones(2, 3, dtype=torch.float),
        ),
        torch.tensor(4.0),
    )


def test_general_robust_regression_loss_forward_alpha_2_scale_2() -> None:
    criterion = GeneralRobustRegressionLoss(alpha=2.0, scale=2.0)
    assert objects_are_equal(
        criterion(
            prediction=torch.ones(2, 3, dtype=torch.float),
            target=-torch.ones(2, 3, dtype=torch.float),
        ),
        torch.tensor(1.0),
    )


def test_general_robust_regression_loss_forward_alpha_1_scale_1() -> None:
    criterion = GeneralRobustRegressionLoss(alpha=1.0, scale=1.0)
    assert objects_are_allclose(
        criterion(
            prediction=torch.ones(2, 3, dtype=torch.float),
            target=-torch.ones(2, 3, dtype=torch.float),
        ),
        torch.tensor(1.2360679774997898),
    )


def test_general_robust_regression_loss_forward_alpha_1_scale_2() -> None:
    criterion = GeneralRobustRegressionLoss(alpha=1.0, scale=2.0)
    assert objects_are_allclose(
        criterion(
            prediction=torch.ones(2, 3, dtype=torch.float),
            target=-torch.ones(2, 3, dtype=torch.float),
        ),
        torch.tensor(0.41421356237309515),
    )


def test_general_robust_regression_loss_forward_alpha_0_scale_1() -> None:
    criterion = GeneralRobustRegressionLoss(alpha=0.0, scale=1.0)
    assert objects_are_allclose(
        criterion(
            prediction=torch.ones(2, 3, dtype=torch.float),
            target=-torch.ones(2, 3, dtype=torch.float),
        ),
        torch.tensor(1.0986122886681098),
    )


def test_general_robust_regression_loss_forward_alpha_0_scale_2() -> None:
    criterion = GeneralRobustRegressionLoss(alpha=0.0, scale=2.0)
    assert objects_are_allclose(
        criterion(
            prediction=torch.ones(2, 3, dtype=torch.float),
            target=-torch.ones(2, 3, dtype=torch.float),
        ),
        torch.tensor(0.4054651081081644),
    )


def test_general_robust_regression_loss_forward_alpha_minus_2_scale_1() -> None:
    criterion = GeneralRobustRegressionLoss(alpha=-2.0, scale=1.0)
    assert objects_are_allclose(
        criterion(
            prediction=torch.ones(2, 3, dtype=torch.float),
            target=-torch.ones(2, 3, dtype=torch.float),
        ),
        torch.tensor(1.0),
    )


def test_general_robust_regression_loss_forward_alpha_minus_2_scale_2() -> None:
    criterion = GeneralRobustRegressionLoss(alpha=-2.0, scale=2.0)
    assert objects_are_allclose(
        criterion(
            prediction=torch.ones(2, 3, dtype=torch.float),
            target=-torch.ones(2, 3, dtype=torch.float),
        ),
        torch.tensor(0.4),
    )


def test_general_robust_regression_loss_forward_reduction_mean() -> None:
    criterion = GeneralRobustRegressionLoss(alpha=2.0, scale=1.0, reduction="mean")
    assert objects_are_allclose(
        criterion(
            prediction=torch.tensor([[0, 1, 2], [2, 1, 0]], dtype=torch.float),
            target=torch.tensor([[0, 1, 2], [1, 1, 1]], dtype=torch.float),
        ),
        torch.tensor(1 / 3),
    )


def test_general_robust_regression_loss_forward_reduction_sum() -> None:
    criterion = GeneralRobustRegressionLoss(alpha=2.0, scale=1.0, reduction="sum")
    assert objects_are_equal(
        criterion(
            prediction=torch.tensor([[0, 1, 2], [2, 1, 0]], dtype=torch.float),
            target=torch.tensor([[0, 1, 2], [1, 1, 1]], dtype=torch.float),
        ),
        torch.tensor(2.0),
    )


def test_general_robust_regression_loss_forward_reduction_none() -> None:
    criterion = GeneralRobustRegressionLoss(alpha=2.0, scale=1.0, reduction="none")
    assert objects_are_equal(
        criterion(
            prediction=torch.tensor([[0, 1, 2], [2, 1, 0]], dtype=torch.float),
            target=torch.tensor([[0, 1, 2], [1, 1, 1]], dtype=torch.float),
        ),
        torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 1.0]]),
    )


def test_general_robust_regression_loss_forward_max() -> None:
    criterion = GeneralRobustRegressionLoss(alpha=2.0, scale=1.0, max=0.5, reduction="none")
    assert objects_are_equal(
        criterion(
            prediction=torch.tensor([[0, 1, 2], [2, 1, 0]], dtype=torch.float),
            target=torch.tensor([[0, 1, 2], [1, 1, 1]], dtype=torch.float),
        ),
        torch.tensor([[0.0, 0.0, 0.0], [0.5, 0.0, 0.5]]),
    )


@pytest.mark.parametrize("alpha", [1.0, 2.0])
@pytest.mark.parametrize("scale", [1.0, 2.0])
@pytest.mark.parametrize("max_value", [None, 1.0])
@pytest.mark.parametrize("reduction", ["mean", "sum"])
def test_general_robust_regression_loss_forward_mock(
    alpha: float, scale: float, max_value: float | None, reduction: str
) -> None:
    criterion = GeneralRobustRegressionLoss(
        alpha=alpha, scale=scale, max=max_value, reduction=reduction
    )
    with patch("sonnix.modules.loss.general_robust.general_robust_regression_loss") as loss_mock:
        criterion(prediction=torch.tensor([1.0]), target=torch.tensor([1.0]))
        loss_mock.assert_called_once_with(
            prediction=torch.tensor([1.0]),
            target=torch.tensor([1.0]),
            alpha=alpha,
            scale=scale,
            max=max_value,
            reduction=reduction,
        )
