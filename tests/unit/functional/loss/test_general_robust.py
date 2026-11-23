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
def test_general_robust_regression_loss_1d_check1(
    device: str, batch_size: int, alpha: float
) -> None:
    device = torch.device(device)
    out = general_robust_regression_loss(
        prediction=torch.randn(batch_size, device=device, requires_grad=True),
        target=torch.randn(batch_size, device=device),
        alpha=alpha,
    )
    out.backward()
    assert out.numel() == 1


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("batch_size", SIZES)
@pytest.mark.parametrize("alpha", [0.0, 1.0, 2.0])
def test_general_robust_regression_loss_1d_dtype_device(
    device: str, batch_size: int, alpha: float
) -> None:
    device = torch.device(device)
    out = general_robust_regression_loss(
        prediction=torch.randn(batch_size, device=device, requires_grad=True),
        target=torch.randn(batch_size, device=device),
        alpha=alpha,
    )
    out.backward()
    assert out.dtype == torch.float
    assert out.device == device


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("batch_size", SIZES)
@pytest.mark.parametrize("feature_size", SIZES)
@pytest.mark.parametrize("alpha", [0.0, 1.0, 2.0])
def test_general_robust_regression_loss_2d_check1(
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


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("batch_size", SIZES)
@pytest.mark.parametrize("feature_size", SIZES)
@pytest.mark.parametrize("alpha", [0.0, 1.0, 2.0])
def test_general_robust_regression_loss_2d_dtype_device(
    device: str, batch_size: int, feature_size: int, alpha: float
) -> None:
    device = torch.device(device)
    out = general_robust_regression_loss(
        prediction=torch.randn(batch_size, feature_size, device=device, requires_grad=True),
        target=torch.randn(batch_size, feature_size, device=device),
        alpha=alpha,
    )
    out.backward()
    assert out.dtype == torch.float
    assert out.device == device


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("batch_size", SIZES)
@pytest.mark.parametrize("seq_len", SIZES)
@pytest.mark.parametrize("feature_size", SIZES)
@pytest.mark.parametrize("alpha", [0.0, 1.0, 2.0])
def test_general_robust_regression_loss_3d_check1(
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


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("batch_size", SIZES)
@pytest.mark.parametrize("seq_len", SIZES)
@pytest.mark.parametrize("feature_size", SIZES)
@pytest.mark.parametrize("alpha", [0.0, 1.0, 2.0])
def test_general_robust_regression_loss_3d_dtype_device(
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
        max_val=0.5,
        reduction="none",
    ).equal(torch.tensor([[0.0, 0.0, 0.0], [0.5, 0.0, 0.5]]))


@pytest.mark.parametrize("alpha", [1e-3, -1e-3, 1e-7])
def test_general_robust_regression_loss_alpha_near_0_is_stable(alpha: float) -> None:
    torch.manual_seed(0)
    prediction, target = torch.randn(8, 4, dtype=torch.float64), torch.randn(8, 4).double()
    expected = general_robust_regression_loss(prediction, target, alpha=0.0)
    assert general_robust_regression_loss(prediction, target, alpha=alpha).allclose(
        expected, atol=1e-2
    )


def test_general_robust_regression_loss_alpha_near_0_float32_is_stable() -> None:
    torch.manual_seed(0)
    prediction, target = torch.randn(8, 4, dtype=torch.float64), torch.randn(8, 4)
    expected = general_robust_regression_loss(prediction, target.double(), alpha=0.0)
    assert general_robust_regression_loss(prediction.float(), target, alpha=1e-3).allclose(
        expected.float(), atol=1e-2
    )


def test_general_robust_regression_loss_alpha_neg_inf() -> None:
    prediction, target = torch.randn(8, 4), torch.randn(8, 4)
    loss = general_robust_regression_loss(prediction, target, alpha=float("-inf"))
    expected = 1 - torch.exp(-0.5 * (prediction - target) ** 2)
    assert loss.allclose(expected.mean())


@pytest.mark.parametrize("scale", [0.0, -1.0])
def test_general_robust_regression_loss_invalid_scale(scale: float) -> None:
    with pytest.raises(ValueError, match=r"scale must be greater than 0"):
        general_robust_regression_loss(torch.ones(2, 3), torch.ones(2, 3), scale=scale)


@pytest.mark.parametrize("alpha", [float("nan"), float("inf")])
def test_general_robust_regression_loss_invalid_alpha(alpha: float) -> None:
    with pytest.raises(ValueError, match=r"alpha cannot be NaN or \+inf"):
        general_robust_regression_loss(torch.ones(2, 3), torch.ones(2, 3), alpha=alpha)


@pytest.mark.parametrize("max_val", [0.0, -1.0])
def test_general_robust_regression_loss_invalid_max_val(max_val: float) -> None:
    with pytest.raises(ValueError, match=r"max_val must be greater than 0"):
        general_robust_regression_loss(torch.ones(2, 3), torch.ones(2, 3), max_val=max_val)


@pytest.mark.parametrize(
    ("alpha", "expected"), [(1.0, 299.0), (0.0, 10.7109375), (-float("inf"), 1.0)]
)
def test_general_robust_regression_loss_float16_large_difference_is_finite(
    alpha: float, expected: float
) -> None:
    # The difference of 300 is squared in float32: its square (9e4) is not finite in float16.
    loss = general_robust_regression_loss(
        torch.tensor([300.0], dtype=torch.float16),
        torch.zeros(1, dtype=torch.float16),
        alpha=alpha,
    )
    assert loss.equal(torch.tensor(expected, dtype=torch.float16))


def test_general_robust_regression_loss_float16_large_difference_gradient_is_finite() -> None:
    prediction = torch.tensor([300.0], dtype=torch.float16, requires_grad=True)
    general_robust_regression_loss(
        prediction, torch.zeros(1, dtype=torch.float16), alpha=1.0
    ).backward()
    assert prediction.grad.equal(torch.tensor([1.0], dtype=torch.float16))


def test_general_robust_regression_loss_float16_loss_above_the_maximum_is_inf() -> None:
    # 300 ** 2 = 90000 is larger than the maximum of float16 (65504).
    loss = general_robust_regression_loss(
        torch.tensor([300.0], dtype=torch.float16), torch.zeros(1, dtype=torch.float16)
    )
    assert loss.isinf().all()


def test_general_robust_regression_loss_float16_loss_above_the_maximum_gradient_is_finite() -> None:
    prediction = torch.tensor([300.0], dtype=torch.float16, requires_grad=True)
    general_robust_regression_loss(prediction, torch.zeros(1, dtype=torch.float16)).backward()
    assert prediction.grad.equal(torch.tensor([600.0], dtype=torch.float16))


def test_general_robust_regression_loss_float16_max_val_clips_in_float32() -> None:
    loss = general_robust_regression_loss(
        torch.tensor([300.0], dtype=torch.float16),
        torch.zeros(1, dtype=torch.float16),
        max_val=100.0,
    )
    assert loss.equal(torch.tensor(100.0, dtype=torch.float16))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_general_robust_regression_loss_keeps_the_dtype(dtype: torch.dtype) -> None:
    loss = general_robust_regression_loss(
        torch.ones(2, 3, dtype=dtype), torch.zeros(2, 3, dtype=dtype), reduction="none"
    )
    assert loss.dtype == dtype


def test_general_robust_regression_loss_integer_prediction_is_float32() -> None:
    loss = general_robust_regression_loss(
        torch.ones(2, 3, dtype=torch.long), torch.zeros(2, 3), reduction="none"
    )
    assert loss.dtype == torch.float32
