from __future__ import annotations

import pytest
import torch
from coola.utils.tensor import get_available_devices

from sonnix.functional import general_robust_regression_loss

SIZES = (1, 2, 3)


####################################################
#     Tests for general_robust_regression_loss     #
####################################################


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("batch_size", SIZES)
@pytest.mark.parametrize("alpha", [0.0, 1.0, 2.0])
def test_general_robust_regression_loss_1d(device: str, batch_size: int, alpha: float) -> None:
    device = torch.device(device)
    out = general_robust_regression_loss(
        prediction=torch.randn(batch_size, device=device, requires_grad=True),
        target=torch.randn(batch_size, device=device),
        alpha=alpha,
    )
    out.backward()
    assert out.numel() == 1
    assert out.dtype == torch.float
    assert out.device == device


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("batch_size", SIZES)
@pytest.mark.parametrize("feature_size", SIZES)
@pytest.mark.parametrize("alpha", [0.0, 1.0, 2.0])
def test_general_robust_regression_loss_2d(
    device: str, batch_size: int, feature_size: int, alpha: float
) -> None:
    device = torch.device(device)
    out = general_robust_regression_loss(
        prediction=torch.randn(batch_size, feature_size, device=device, requires_grad=True),
        target=torch.randn(batch_size, feature_size, device=device),
        alpha=alpha,
    )
    out.backward()
    assert out.numel() == 1
    assert out.dtype == torch.float
    assert out.device == device


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("batch_size", SIZES)
@pytest.mark.parametrize("seq_len", SIZES)
@pytest.mark.parametrize("feature_size", SIZES)
@pytest.mark.parametrize("alpha", [0.0, 1.0, 2.0])
def test_general_robust_regression_loss_3d(
    device: str, batch_size: int, seq_len: int, feature_size: int, alpha: float
) -> None:
    device = torch.device(device)
    out = general_robust_regression_loss(
        prediction=torch.randn(
            batch_size, seq_len, feature_size, device=device, requires_grad=True
        ),
        target=torch.randn(batch_size, seq_len, feature_size, device=device),
        alpha=alpha,
    )
    out.backward()
    assert out.numel() == 1
    assert out.dtype == torch.float
    assert out.device == device


def test_general_robust_regression_loss_alpha_2_scale_1() -> None:
    assert general_robust_regression_loss(
        prediction=torch.ones(2, 3),
        target=-torch.ones(2, 3),
        alpha=2.0,
        scale=1.0,
    ).equal(torch.tensor(4.0))


def test_general_robust_regression_loss_alpha_2_scale_2() -> None:
    assert general_robust_regression_loss(
        prediction=torch.ones(2, 3),
        target=-torch.ones(2, 3),
        alpha=2.0,
        scale=2.0,
    ).equal(torch.tensor(1.0))


def test_general_robust_regression_loss_alpha_1_scale_1() -> None:
    assert general_robust_regression_loss(
        prediction=torch.ones(2, 3),
        target=-torch.ones(2, 3),
        alpha=1.0,
        scale=1.0,
    ).allclose(
        torch.tensor(1.2360679774997898),
    )


def test_general_robust_regression_loss_alpha_1_scale_2() -> None:
    assert general_robust_regression_loss(
        prediction=torch.ones(2, 3),
        target=-torch.ones(2, 3),
        alpha=1.0,
        scale=2.0,
    ).allclose(
        torch.tensor(0.41421356237309515),
    )


def test_general_robust_regression_loss_alpha_0_scale_1() -> None:
    assert general_robust_regression_loss(
        prediction=torch.ones(2, 3),
        target=-torch.ones(2, 3),
        alpha=0.0,
        scale=1.0,
    ).allclose(
        torch.tensor(1.0986122886681098),
    )


def test_general_robust_regression_loss_alpha_0_scale_2() -> None:
    assert general_robust_regression_loss(
        prediction=torch.ones(2, 3),
        target=-torch.ones(2, 3),
        alpha=0.0,
        scale=2.0,
    ).allclose(
        torch.tensor(0.4054651081081644),
    )


def test_general_robust_regression_loss_alpha_minus_2_scale_1() -> None:
    assert general_robust_regression_loss(
        prediction=torch.ones(2, 3),
        target=-torch.ones(2, 3),
        alpha=-2.0,
        scale=1.0,
    ).allclose(
        torch.tensor(1.0),
    )


def test_general_robust_regression_loss_alpha_minus_2_scale_2() -> None:
    assert general_robust_regression_loss(
        prediction=torch.ones(2, 3),
        target=-torch.ones(2, 3),
        alpha=-2.0,
        scale=2.0,
    ).allclose(
        torch.tensor(0.4),
    )


def test_general_robust_regression_loss_reduction_mean() -> None:
    assert general_robust_regression_loss(
        prediction=torch.tensor([[0.0, 1.0, 2.0], [2.0, 1.0, 0.0]]),
        target=torch.tensor([[0.0, 1.0, 2.0], [1.0, 1.0, 1.0]]),
        reduction="mean",
    ).allclose(
        torch.tensor(1 / 3),
    )


def test_general_robust_regression_loss_reduction_sum() -> None:
    assert general_robust_regression_loss(
        prediction=torch.tensor([[0.0, 1.0, 2.0], [2.0, 1.0, 0.0]]),
        target=torch.tensor([[0.0, 1.0, 2.0], [1.0, 1.0, 1.0]]),
        reduction="sum",
    ).equal(torch.tensor(2.0))


def test_general_robust_regression_loss_reduction_none() -> None:
    assert general_robust_regression_loss(
        prediction=torch.tensor([[0.0, 1.0, 2.0], [2.0, 1.0, 0.0]]),
        target=torch.tensor([[0.0, 1.0, 2.0], [1.0, 1.0, 1.0]]),
        reduction="none",
    ).equal(torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 1.0]]))


def test_general_robust_regression_loss_max() -> None:
    assert general_robust_regression_loss(
        prediction=torch.tensor([[0.0, 1.0, 2.0], [2.0, 1.0, 0.0]]),
        target=torch.tensor([[0.0, 1.0, 2.0], [1.0, 1.0, 1.0]]),
        max=0.5,
        reduction="none",
    ).equal(torch.tensor([[0.0, 0.0, 0.0], [0.5, 0.0, 0.5]]))
