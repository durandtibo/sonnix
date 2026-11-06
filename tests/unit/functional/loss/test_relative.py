from __future__ import annotations

import pytest
import torch
from coola.equality import objects_are_allclose, objects_are_equal
from coola.utils.tensor import get_available_devices

from sonnix.functional import relative_loss
from sonnix.functional.loss.relative import (
    arithmetical_mean_indicator,
    classical_relative_indicator,
    geometric_mean_indicator,
    maximum_mean_indicator,
    minimum_mean_indicator,
    moment_mean_indicator,
    reversed_relative_indicator,
)

###################################
#     Tests for relative_loss     #
###################################


@pytest.mark.parametrize("device", get_available_devices())
def test_relative_loss_reduction_mean(device: str) -> None:
    prediction = torch.tensor(
        [[-2.0, 1.0, 0.0], [-3.0, 5.0, -1.0]], device=device, requires_grad=True
    )
    target = torch.tensor([[0.0, 1.0, -1.0], [3.0, 1.0, -1.0]], device=device)
    loss = relative_loss(
        loss=torch.nn.functional.mse_loss(prediction, target, reduction="none"),
        indicator=classical_relative_indicator(prediction, target),
        eps=1e-5,
    )
    loss.backward()
    assert objects_are_equal(loss, torch.tensor(66671.5, device=device))


@pytest.mark.parametrize("device", get_available_devices())
def test_relative_loss_reduction_sum(device: str) -> None:
    prediction = torch.tensor(
        [[-2.0, 1.0, 0.0], [-3.0, 5.0, -1.0]], device=device, requires_grad=True
    )
    target = torch.tensor([[0.0, 1.0, -1.0], [3.0, 1.0, -1.0]], device=device)
    loss = relative_loss(
        loss=torch.nn.functional.mse_loss(prediction, target, reduction="none"),
        indicator=classical_relative_indicator(prediction, target),
        reduction="sum",
        eps=1e-5,
    )
    loss.backward()
    assert objects_are_equal(loss, torch.tensor(400029.0, device=device))


@pytest.mark.parametrize("device", get_available_devices())
def test_relative_loss_reduction_none(device: str) -> None:
    prediction = torch.tensor(
        [[-2.0, 1.0, 0.0], [-3.0, 5.0, -1.0]], device=device, requires_grad=True
    )
    target = torch.tensor([[0.0, 1.0, -1.0], [3.0, 1.0, -1.0]], device=device)
    loss = relative_loss(
        loss=torch.nn.functional.mse_loss(prediction, target, reduction="none"),
        indicator=classical_relative_indicator(prediction, target),
        reduction="none",
        eps=0.01,
    )
    assert objects_are_equal(
        loss, torch.tensor([[400.0, 0.0, 1.0], [12.0, 16.0, 0.0]], device=device)
    )


def test_relative_loss_reduction_incorrect() -> None:
    prediction = torch.tensor([[-2.0, 1.0, 0.0], [-3.0, 5.0, -1.0]], requires_grad=True)
    target = torch.tensor([[0.0, 1.0, -1.0], [3.0, 1.0, -1.0]])
    with pytest.raises(ValueError, match=r"reduction must be one of"):
        relative_loss(
            loss=torch.nn.functional.mse_loss(prediction, target, reduction="none"),
            indicator=classical_relative_indicator(prediction, target),
            reduction="incorrect",
        )


@pytest.mark.parametrize("device", get_available_devices())
def test_relative_loss_indicator_arithmetical_mean(device: str) -> None:
    prediction = torch.tensor(
        [[-2.0, 1.0, 0.0], [-3.0, 5.0, -1.0]], device=device, requires_grad=True
    )
    target = torch.tensor([[0.0, 1.0, -1.0], [3.0, 1.0, -1.0]], device=device)
    loss = relative_loss(
        loss=torch.nn.functional.mse_loss(prediction, target, reduction="none"),
        indicator=arithmetical_mean_indicator(prediction, target),
        reduction="none",
    )
    assert objects_are_equal(
        loss, torch.tensor([[4.0, 0.0, 2.0], [12.0, 5.333333333333333, 0.0]], device=device)
    )


@pytest.mark.parametrize("device", get_available_devices())
def test_relative_loss_indicator_classical_relative(device: str) -> None:
    prediction = torch.tensor(
        [[-2.0, 1.0, 0.0], [-3.0, 5.0, -1.0]], device=device, requires_grad=True
    )
    target = torch.tensor([[0.0, 1.0, -1.0], [3.0, 1.0, -1.0]], device=device)
    loss = relative_loss(
        loss=torch.nn.functional.mse_loss(prediction, target, reduction="none"),
        indicator=classical_relative_indicator(prediction, target),
        reduction="none",
        eps=0.01,
    )
    assert objects_are_equal(
        loss, torch.tensor([[400.0, 0.0, 1.0], [12.0, 16.0, 0.0]], device=device)
    )


@pytest.mark.parametrize("device", get_available_devices())
def test_relative_loss_indicator_reversed_relative(device: str) -> None:
    prediction = torch.tensor(
        [[-2.0, 1.0, 0.0], [-3.0, 5.0, -1.0]], device=device, requires_grad=True
    )
    target = torch.tensor([[0.0, 1.0, -1.0], [3.0, 1.0, -1.0]], device=device)
    loss = relative_loss(
        loss=torch.nn.functional.mse_loss(prediction, target, reduction="none"),
        indicator=reversed_relative_indicator(prediction, target),
        reduction="none",
        eps=0.01,
    )
    assert objects_are_equal(
        loss, torch.tensor([[2.0, 0.0, 100.0], [12.0, 3.2, 0.0]], device=device)
    )


def test_relative_loss_incorrect_shapes() -> None:
    prediction = torch.tensor([[-2.0, 1.0, 0.0], [-3.0, 5.0, -1.0]], requires_grad=True)
    target = torch.tensor([[0.0, 1.0, -1.0], [3.0, 1.0, -1.0]])
    with pytest.raises(ValueError, match=r"loss and indicator must have the same shape"):
        relative_loss(
            loss=torch.nn.functional.mse_loss(prediction, target),
            indicator=classical_relative_indicator(prediction, target),
        )


#################################################
#     Tests for arithmetical_mean_indicator     #
#################################################


@pytest.mark.parametrize("device", get_available_devices())
def test_arithmetical_mean_indicator(device: str) -> None:
    device = torch.device(device)
    assert objects_are_equal(
        arithmetical_mean_indicator(
            torch.tensor([[0.0, 1.0, -1.0], [3.0, 1.0, -1.0]], device=device),
            torch.tensor([[-2.0, 1.0, 0.0], [-3.0, 5.0, -1.0]], device=device),
        ),
        torch.tensor([[1.0, 1.0, 0.5], [3.0, 3.0, 1.0]], device=device),
    )


##################################################
#     Tests for classical_relative_indicator     #
##################################################


@pytest.mark.parametrize("device", get_available_devices())
def test_classical_relative_indicator(device: str) -> None:
    device = torch.device(device)
    assert objects_are_equal(
        classical_relative_indicator(
            torch.ones(2, 3, device=device),
            torch.tensor([[-2.0, 1.0, 0.0], [-3.0, 5.0, -1.0]], device=device),
        ),
        torch.tensor([[2.0, 1.0, 0.0], [3.0, 5.0, 1.0]], device=device),
    )


##############################################
#     Tests for geometric_mean_indicator     #
##############################################


@pytest.mark.parametrize("device", get_available_devices())
def test_geometric_mean_indicator(device: str) -> None:
    device = torch.device(device)
    assert objects_are_allclose(
        geometric_mean_indicator(
            torch.tensor([[0.0, 1.0, -1.0], [3.0, 1.0, -1.0]], device=device),
            torch.tensor([[-2.0, 1.0, 0.0], [-3.0, 5.0, -1.0]], device=device),
        ),
        torch.tensor([[1e-8, 1.0, 1e-8], [3.0, 2.23606797749979, 1.0]], device=device),
        atol=1e-5,
    )


############################################
#     Tests for maximum_mean_indicator     #
############################################


@pytest.mark.parametrize("device", get_available_devices())
def test_maximum_mean_indicator(device: str) -> None:
    device = torch.device(device)
    assert objects_are_equal(
        maximum_mean_indicator(
            torch.tensor([[0.0, 1.0, -1.0], [3.0, 1.0, -1.0]], device=device),
            torch.tensor([[-2.0, 1.0, 0.0], [-3.0, 5.0, -1.0]], device=device),
        ),
        torch.tensor([[2.0, 1.0, 1.0], [3.0, 5.0, 1.0]], device=device),
    )


############################################
#     Tests for minimum_mean_indicator     #
############################################


@pytest.mark.parametrize("device", get_available_devices())
def test_minimum_mean_indicator(device: str) -> None:
    device = torch.device(device)
    assert objects_are_equal(
        minimum_mean_indicator(
            torch.tensor([[0.0, 1.0, -1.0], [3.0, 1.0, -1.0]], device=device),
            torch.tensor([[-2.0, 1.0, 0.0], [-3.0, 5.0, -1.0]], device=device),
        ),
        torch.tensor([[0.0, 1.0, 0.0], [3.0, 1.0, 1.0]], device=device),
    )


###########################################
#     Tests for moment_mean_indicator     #
###########################################


@pytest.mark.parametrize("device", get_available_devices())
def test_moment_mean_indicator_order_1(device: str) -> None:
    device = torch.device(device)
    assert objects_are_equal(
        moment_mean_indicator(
            torch.tensor([[0.0, 1.0, -1.0], [3.0, 1.0, -1.0]], device=device),
            torch.tensor([[-2.0, 1.0, 0.0], [-3.0, 5.0, -1.0]], device=device),
        ),
        torch.tensor([[1.0, 1.0, 0.5], [3.0, 3.0, 1.0]], device=device),
    )


@pytest.mark.parametrize("device", get_available_devices())
def test_moment_mean_indicator_order_2(device: str) -> None:
    device = torch.device(device)
    assert objects_are_allclose(
        moment_mean_indicator(
            torch.tensor([[0.0, 1.0, -1.0], [3.0, 1.0, -1.0]], device=device),
            torch.tensor([[-2.0, 1.0, 0.0], [-3.0, 5.0, -1.0]], device=device),
            k=2,
        ),
        torch.tensor(
            [[1.4142135623730951, 1.0, 0.7071067811865476], [3.0, 3.605551275463989, 1.0]],
            device=device,
        ),
    )


@pytest.mark.parametrize("device", get_available_devices())
def test_moment_mean_indicator_order_3(device: str) -> None:
    device = torch.device(device)
    assert objects_are_allclose(
        moment_mean_indicator(
            torch.tensor([[0.0, 1.0, -1.0], [3.0, 1.0, -1.0]], device=device),
            torch.tensor([[-2.0, 1.0, 0.0], [-3.0, 5.0, -1.0]], device=device),
            k=3,
        ),
        torch.tensor(
            [[1.5874010519681994, 1.0, 0.7937005259840998], [3.0, 3.9790572078963917, 1.0]],
            device=device,
        ),
    )


#################################################
#     Tests for reversed_relative_indicator     #
#################################################


@pytest.mark.parametrize("device", get_available_devices())
def test_reversed_relative_indicator(device: str) -> None:
    device = torch.device(device)
    assert objects_are_equal(
        reversed_relative_indicator(
            torch.tensor([[-2.0, 1.0, 0.0], [-3.0, 5.0, -1.0]], device=device),
            torch.ones(2, 3, device=device),
        ),
        torch.tensor([[2.0, 1.0, 0.0], [3.0, 5.0, 1.0]], device=device),
    )


@pytest.mark.parametrize("k", [0, -1])
def test_moment_mean_indicator_incorrect_k(k: int) -> None:
    with pytest.raises(ValueError, match="k must be greater than or equal to 1"):
        moment_mean_indicator(torch.randn(2, 3), torch.randn(2, 3), k=k)


#####################################################
#     Tests for finite gradients of the indicators  #
#####################################################


def test_geometric_mean_indicator_gradient_zero_prediction() -> None:
    prediction = torch.tensor([0.0, 2.0], requires_grad=True)
    geometric_mean_indicator(prediction, torch.tensor([3.0, 2.0])).sum().backward()
    assert objects_are_allclose(prediction.grad, torch.tensor([0.0, 0.5]))


def test_geometric_mean_indicator_gradient_zero_target() -> None:
    prediction = torch.tensor([2.0, 1.0], requires_grad=True)
    geometric_mean_indicator(prediction, torch.tensor([0.0, 1.0])).sum().backward()
    assert objects_are_equal(prediction.grad, torch.tensor([0.0, 0.5]))


def test_geometric_mean_indicator_gradient_zero_prediction_and_target() -> None:
    prediction = torch.zeros(2, requires_grad=True)
    geometric_mean_indicator(prediction, torch.zeros(2)).sum().backward()
    assert objects_are_equal(prediction.grad, torch.zeros(2))


def test_geometric_mean_indicator_value_zero() -> None:
    assert objects_are_equal(
        geometric_mean_indicator(torch.zeros(2), torch.tensor([1.0, 0.0])), torch.zeros(2)
    )


@pytest.mark.parametrize("k", [2, 3, 4])
def test_moment_mean_indicator_gradient_zero_prediction_and_target(k: int) -> None:
    prediction = torch.zeros(2, requires_grad=True)
    moment_mean_indicator(prediction, torch.zeros(2), k=k).sum().backward()
    assert objects_are_equal(prediction.grad, torch.zeros(2))


@pytest.mark.parametrize("k", [2, 3, 4])
def test_moment_mean_indicator_gradient_zero_prediction(k: int) -> None:
    prediction = torch.zeros(2, requires_grad=True)
    moment_mean_indicator(prediction, torch.ones(2), k=k).sum().backward()
    assert prediction.grad.isfinite().all()


@pytest.mark.parametrize("k", [2, 3])
def test_moment_mean_indicator_value_zero(k: int) -> None:
    assert objects_are_equal(
        moment_mean_indicator(torch.zeros(2), torch.zeros(2), k=k), torch.zeros(2)
    )


def test_relative_loss_geometric_mean_indicator_gradient_zero_prediction() -> None:
    prediction = torch.tensor([0.0, 1.0, 2.0], requires_grad=True)
    target = torch.tensor([1.0, 2.0, 3.0])
    relative_loss(
        (prediction - target).abs(), geometric_mean_indicator(prediction, target)
    ).backward()
    assert prediction.grad.isfinite().all()


@pytest.mark.parametrize("eps", [0.0, -1.0])
def test_relative_loss_invalid_eps(eps: float) -> None:
    with pytest.raises(ValueError, match=r"eps must be greater than 0"):
        relative_loss(torch.ones(2, 3), torch.ones(2, 3), eps=eps)


##################################################
#     Tests for the overflow of the indicators   #
##################################################


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_geometric_mean_indicator_half_precision_large_values(dtype: torch.dtype) -> None:
    out = geometric_mean_indicator(
        torch.tensor([300.0], dtype=dtype), torch.tensor([400.0], dtype=dtype)
    )
    assert out.float().allclose(torch.tensor([346.41]), rtol=2e-2)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_moment_mean_indicator_half_precision_large_values(dtype: torch.dtype) -> None:
    out = moment_mean_indicator(
        torch.tensor([300.0], dtype=dtype), torch.tensor([400.0], dtype=dtype), k=2
    )
    assert out.float().allclose(torch.tensor([353.55]), rtol=2e-2)


def test_geometric_mean_indicator_float32_large_values() -> None:
    out = geometric_mean_indicator(torch.tensor([3e19]), torch.tensor([3e19]))
    assert out.allclose(torch.tensor([3e19]))


def test_moment_mean_indicator_float32_large_values() -> None:
    out = moment_mean_indicator(torch.tensor([3e19]), torch.tensor([3e19]), k=2)
    assert out.allclose(torch.tensor([3e19]))


def test_moment_mean_indicator_large_order() -> None:
    out = moment_mean_indicator(torch.tensor([50.0]), torch.tensor([60.0]), k=100)
    assert out.allclose(torch.tensor([59.5855]), rtol=1e-4)


def test_moment_mean_indicator_zeros() -> None:
    out = moment_mean_indicator(torch.zeros(2), torch.zeros(2), k=2)
    assert out.equal(torch.zeros(2))


def test_moment_mean_indicator_gradient_zeros_is_finite() -> None:
    prediction = torch.zeros(2, requires_grad=True)
    moment_mean_indicator(prediction, torch.zeros(2), k=2).sum().backward()
    assert prediction.grad.isfinite().all()


def test_moment_mean_indicator_gradient_matches_unscaled_formula() -> None:
    prediction = torch.tensor([1.0, -2.0], requires_grad=True)
    moment_mean_indicator(prediction, torch.tensor([3.0, 1.0]), k=2).sum().backward()
    # d/dp sqrt((t^2 + p^2) / 2) = p / (2 * sqrt((t^2 + p^2) / 2))
    assert prediction.grad.allclose(torch.tensor([0.223606797749979, -0.6324555320336759]))
