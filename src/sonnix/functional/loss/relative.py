r"""Contain relative loss functions."""

from __future__ import annotations

__all__ = [
    "arithmetical_mean_indicator",
    "classical_relative_indicator",
    "geometric_mean_indicator",
    "maximum_mean_indicator",
    "minimum_mean_indicator",
    "moment_mean_indicator",
    "relative_loss",
    "reversed_relative_indicator",
]

from typing import TYPE_CHECKING

import torch
from coola.validation import validate_gt

from sonnix.functional.power import safe_pow
from sonnix.functional.reduction import reduce_loss
from sonnix.validation import validate_moment_order, validate_same_shape

if TYPE_CHECKING:
    from sonnix.functional.reduction import Reduction


def relative_loss(
    loss: torch.Tensor,
    indicator: torch.Tensor,
    eps: float = 1e-8,
    *,
    reduction: Reduction = "mean",
) -> torch.Tensor:
    r"""Compute the relative loss.

    The indicators are designed based on
    https://en.wikipedia.org/wiki/Relative_change#Indicators_of_relative_change.

    Note:
        The indicator is clamped to ``eps``, so the gradient with
        respect to the indicator is zero where it is lower than ``eps``.

    Args:
        loss: The loss values. The tensor must have the same shape as
            the target.
        indicator: The indicator values.
        eps: An arbitrary small strictly positive number to avoid
            undefined results when the indicator is zero.
        reduction: The reduction strategy: ``'none'``, ``'mean'``, ``'sum'`` or
            ``'batchmean'``. See `sonnix.functional.reduce_loss` for the details.

    Returns:
        The computed relative loss.

    Raises:
        ValueError: if the loss and indicator shapes do not match.
        ValueError: if the reduction is not valid.
        ValueError: if ``eps`` is not strictly positive.

    Example:
        ```pycon
        >>> import torch
        >>> from sonnix.functional import relative_loss
        >>> from sonnix.functional.loss import classical_relative_indicator
        >>> prediction = torch.randn(3, 5, requires_grad=True)
        >>> target = torch.randn(3, 5)
        >>> loss = relative_loss(
        ...     loss=torch.nn.functional.mse_loss(prediction, target, reduction="none"),
        ...     indicator=classical_relative_indicator(prediction, target),
        ... )
        >>> loss
        tensor(..., grad_fn=<MeanBackward0>)
        >>> loss.backward()

        ```
    """
    validate_gt(eps, 0, name="eps")
    validate_same_shape(loss, indicator, name1="loss", name2="indicator")
    return reduce_loss(loss.div(indicator.clamp(min=eps)), reduction)


def arithmetical_mean_indicator(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    r"""Return the arithmetical mean change.

    Args:
        prediction: The predictions.
        target: The target values.

    Returns:
        The indicator values.

    Raises:
        ValueError: If ``prediction`` and ``target`` do not have the same shape.

    Example:
        ```pycon
        >>> import torch
        >>> from sonnix.functional.loss import arithmetical_mean_indicator
        >>> prediction = torch.tensor([[0.0, 1.0, -1.0], [3.0, 1.0, -1.0]], requires_grad=True)
        >>> target = torch.tensor([[-2.0, 1.0, 0.0], [-3.0, 5.0, -1.0]])
        >>> indicator = arithmetical_mean_indicator(prediction, target)
        >>> indicator
        tensor([[1.0000, 1.0000, 0.5000],
                [3.0000, 3.0000, 1.0000]], grad_fn=<MulBackward0>)

        ```
    """
    validate_same_shape(prediction, target, name1="prediction", name2="target")
    return target.abs().add(prediction.abs()).mul(0.5)


def classical_relative_indicator(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    r"""Return the classical relative change.

    Args:
        prediction: The predictions.
        target: The target values.

    Returns:
        The indicator values.

    Raises:
        ValueError: If ``prediction`` and ``target`` do not have the same shape.

    Example:
        ```pycon
        >>> import torch
        >>> from sonnix.functional.loss import classical_relative_indicator
        >>> prediction = torch.tensor([[0.0, 1.0, -1.0], [3.0, 1.0, -1.0]], requires_grad=True)
        >>> target = torch.tensor([[-2.0, 1.0, 0.0], [-3.0, 5.0, -1.0]])
        >>> indicator = classical_relative_indicator(prediction, target)
        >>> indicator
        tensor([[2., 1., 0.],
                [3., 5., 1.]])

        ```
    """
    validate_same_shape(prediction, target, name1="prediction", name2="target")
    return target.abs()


def geometric_mean_indicator(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    r"""Return the geometric mean change.

    Args:
        prediction: The predictions.
        target: The target values.

    Returns:
        The indicator values. The gradient is finite (zero) where the
            prediction or the target is exactly zero.

    Raises:
        ValueError: If ``prediction`` and ``target`` do not have the same shape.

    Example:
        ```pycon
        >>> import torch
        >>> from sonnix.functional.loss import geometric_mean_indicator
        >>> prediction = torch.tensor([[0.0, 1.0, -1.0], [3.0, 1.0, -1.0]], requires_grad=True)
        >>> target = torch.tensor([[-2.0, 1.0, 0.0], [-3.0, 5.0, -1.0]])
        >>> indicator = geometric_mean_indicator(prediction, target)
        >>> indicator
        tensor([[0.0000, 1.0000, 0.0000],
                [3.0000, 2.2361, 1.0000]], grad_fn=<MulBackward0>)

        ```
    """
    validate_same_shape(prediction, target, name1="prediction", name2="target")
    # The product of the two square roots does not overflow when
    # ``|target| * |prediction|`` does.
    return safe_pow(target.abs(), 0.5).mul(safe_pow(prediction.abs(), 0.5))


def maximum_mean_indicator(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    r"""Return the maximum mean change.

    Args:
        prediction: The predictions.
        target: The target values.

    Returns:
        The indicator values.

    Raises:
        ValueError: If ``prediction`` and ``target`` do not have the same shape.

    Example:
        ```pycon
        >>> import torch
        >>> from sonnix.functional.loss import maximum_mean_indicator
        >>> prediction = torch.tensor([[0.0, 1.0, -1.0], [3.0, 1.0, -1.0]], requires_grad=True)
        >>> target = torch.tensor([[-2.0, 1.0, 0.0], [-3.0, 5.0, -1.0]])
        >>> indicator = maximum_mean_indicator(prediction, target)
        >>> indicator
        tensor([[2., 1., 1.],
                [3., 5., 1.]], grad_fn=<MaximumBackward0>)

        ```
    """
    validate_same_shape(prediction, target, name1="prediction", name2="target")
    return torch.maximum(target.abs(), prediction.abs())


def minimum_mean_indicator(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    r"""Return the minimum mean change.

    Args:
        prediction: The predictions.
        target: The target values.

    Returns:
        The indicator values.

    Raises:
        ValueError: If ``prediction`` and ``target`` do not have the same shape.

    Example:
        ```pycon
        >>> import torch
        >>> from sonnix.functional.loss import minimum_mean_indicator
        >>> prediction = torch.tensor([[0.0, 1.0, -1.0], [3.0, 1.0, -1.0]], requires_grad=True)
        >>> target = torch.tensor([[-2.0, 1.0, 0.0], [-3.0, 5.0, -1.0]])
        >>> indicator = minimum_mean_indicator(prediction, target)
        >>> indicator
        tensor([[0., 1., 0.],
                [3., 1., 1.]], grad_fn=<MinimumBackward0>)

        ```
    """
    validate_same_shape(prediction, target, name1="prediction", name2="target")
    return torch.minimum(target.abs(), prediction.abs())


def moment_mean_indicator(
    prediction: torch.Tensor, target: torch.Tensor, k: int = 1
) -> torch.Tensor:
    r"""Return the moment mean change of order k.

    Args:
        prediction: The predictions.
        target: The target values.
        k: The order, which has to be an integer greater than or
            equal to 1.

    Returns:
        The indicator values.

    Raises:
        TypeError: if ``k`` is not an integer.
        ValueError: if ``k`` is lower than 1.
        ValueError: If ``prediction`` and ``target`` do not have the same shape.

    Example:
        ```pycon
        >>> import torch
        >>> from sonnix.functional.loss import moment_mean_indicator
        >>> prediction = torch.tensor([[0.0, 1.0, -1.0], [3.0, 1.0, -1.0]], requires_grad=True)
        >>> target = torch.tensor([[-2.0, 1.0, 0.0], [-3.0, 5.0, -1.0]])
        >>> indicator = moment_mean_indicator(prediction, target)
        >>> indicator
        tensor([[1.0000, 1.0000, 0.5000],
                [3.0000, 3.0000, 1.0000]], grad_fn=<MulBackward0>)

        ```
    """
    validate_moment_order(k)
    validate_same_shape(prediction, target, name1="prediction", name2="target")
    # Scale by the largest magnitude so ``|x| ** k`` cannot overflow.
    target, prediction = target.abs(), prediction.abs()
    scale = torch.maximum(target, prediction)
    safe_scale = torch.where(scale > 0, scale, 1.0)
    mean = (target / safe_scale).pow(k).add((prediction / safe_scale).pow(k)).mul(0.5)
    return scale.mul(safe_pow(mean, 1 / k))


def reversed_relative_indicator(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    r"""Return the reversed relative change.

    Args:
        prediction: The predictions.
        target: The target values.

    Returns:
        The indicator values.

    Raises:
        ValueError: If ``prediction`` and ``target`` do not have the same shape.

    Example:
        ```pycon
        >>> import torch
        >>> from sonnix.functional.loss import reversed_relative_indicator
        >>> prediction = torch.tensor([[0.0, 1.0, -1.0], [3.0, 1.0, -1.0]], requires_grad=True)
        >>> target = torch.tensor([[-2.0, 1.0, 0.0], [-3.0, 5.0, -1.0]])
        >>> indicator = reversed_relative_indicator(prediction, target)
        >>> indicator
        tensor([[0., 1., 1.],
                [3., 1., 1.]], grad_fn=<AbsBackward0>)

        ```
    """
    validate_same_shape(prediction, target, name1="prediction", name2="target")
    return prediction.abs()
