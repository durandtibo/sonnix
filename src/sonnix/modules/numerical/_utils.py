r"""Contain helper functions shared by the numerical encoders."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Literal

import torch
from coola.random import make_torch_generator
from coola.validation import validate_ge, validate_in, validate_int

from sonnix.validation import validate_positive

if TYPE_CHECKING:
    from torch import Tensor

SPACINGS = ("rand", "linspace", "logspace")


def _resolve_torch_generator(generator_or_seed: int | torch.Generator) -> torch.Generator:
    r"""Resolve a ``torch.Generator`` object.

    Args:
        generator_or_seed: A ``torch.Generator`` object or a random
            seed.

    Returns:
        A ``torch.Generator`` object.

    Example:
        ```pycon
        >>> from sonnix.modules.numerical._utils import _resolve_torch_generator
        >>> generator = _resolve_torch_generator(42)
        >>> generator
        <torch._C.Generator object at 0x...>

        ```
    """
    if isinstance(generator_or_seed, torch.Generator):
        return generator_or_seed
    return make_torch_generator(generator_or_seed)


def prepare_tensor_param(tensor: Tensor, name: str) -> Tensor:
    r"""Prepare a tensor parameter to be a finite 2d tensor.

    Args:
        tensor: The tensor to prepare.
        name: The name associated to the tensor.

    Returns:
        The prepared tensor.

    Raises:
        ValueError: if the input tensor is not a 1d or 2d tensor or
            if it contains a ``NaN`` or an infinite value.

    Example:
        ```pycon
        >>> import torch
        >>> from sonnix.modules.numerical._utils import prepare_tensor_param
        >>> prepare_tensor_param(torch.tensor([1.0, 2.0, 4.0]), name="scale")
        tensor([[1., 2., 4.]])

        ```
    """
    if tensor.ndim == 1:
        tensor = tensor.view(1, -1)
    if tensor.ndim != 2:
        msg = f"Incorrect shape for '{name}': {tensor.shape}"
        raise ValueError(msg)
    if not tensor.isfinite().all():
        msg = f"'{name}' must contain only finite values"
        raise ValueError(msg)
    return tensor


def create_spaced_values(
    spacing: Literal["rand", "linspace", "logspace"],
    num_values: int,
    low: float,
    high: float,
    *,
    n_features: int = 1,
    generator: torch.Generator | int | None = None,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> Tensor:
    r"""Create the values of the parameters of a numerical encoder.

    Args:
        spacing: The way the values are spread in the range.
            ``"rand"``: uniformly sampled (independently for each
            feature), ``"linspace"``: evenly spaced, ``"logspace"``:
            evenly spaced in the log space (``low`` has to be strictly
            positive). The deterministic spacings give the same values
            to every feature.
        num_values: The number of values per feature.
        low: The lower bound of the range.
        high: The upper bound of the range.
        n_features: The number of features.
        generator: The random number generator or the random seed
            used by ``spacing="rand"``. ``None`` uses the global
            random number generator. It is ignored by the other
            spacings.
        device: The device of the tensor.
        dtype: The data type of the tensor. ``None`` uses the
            default floating point type.

    Returns:
        A tensor of shape ``(n_features, num_values)``.

    Raises:
        TypeError: if ``n_features`` is not an integer.
        ValueError: if ``spacing`` is not valid or if ``n_features``
            is lower than 1.

    Example:
        ```pycon
        >>> from sonnix.modules.numerical._utils import create_spaced_values
        >>> create_spaced_values("linspace", num_values=3, low=1.0, high=3.0, n_features=2)
        tensor([[1., 2., 3.],
                [1., 2., 3.]])

        ```
    """
    validate_in(spacing, SPACINGS, name="spacing")
    validate_int(n_features, name="n_features")
    validate_ge(n_features, 1, name="n_features")
    if spacing == "rand":
        return (
            torch.rand(
                n_features,
                num_values,
                generator=None if generator is None else _resolve_torch_generator(generator),
                device=device,
                dtype=dtype,
            )
            .mul(high - low)
            .add(low)
        )
    if spacing == "linspace":
        values = torch.linspace(start=low, end=high, steps=num_values, device=device, dtype=dtype)
    else:
        values = torch.logspace(
            start=math.log10(low),
            end=math.log10(high),
            steps=num_values,
            device=device,
            dtype=dtype,
        )
    return values.view(1, -1).repeat(n_features, 1)


def resolve_range(
    name: str,
    range_: tuple[float, float] | None,
    value_range: tuple[float, float] | None,
) -> tuple[float, float]:
    r"""Return the ``(min, max)`` range of a parameter of an encoder.

    The range is given either by the range of the parameter itself or
    by the range of the absolute values to encode, which is converted
    to ``(1 / max_abs_value, 1 / min_abs_value)``.

    Args:
        name: The name of the parameter (used in the error message),
            e.g. ``"scale"`` for the argument ``scale_range``.
        range_: The ``(min, max)`` range of the parameter.
        value_range: The ``(min_abs_value, max_abs_value)`` range of
            the absolute values to encode.

    Returns:
        The ``(min, max)`` range of the parameter, with
            ``0 < min <= max``.

    Raises:
        TypeError: if a bound of the given range is not a number.
        ValueError: if both or none of ``range_`` and ``value_range``
            are given, or if the given range is not
            ``0 < min <= max`` with finite bounds.

    Example:
        ```pycon
        >>> from sonnix.modules.numerical._utils import resolve_range
        >>> resolve_range("scale", None, (0.5, 4.0))
        (0.25, 2.0)
        >>> resolve_range("scale", (1.0, 3.0), None)
        (1.0, 3.0)

        ```
    """
    if range_ is not None and value_range is None:
        (low, high), low_name, high_name = range_, f"min_{name}", f"max_{name}"
    elif range_ is None and value_range is not None:
        (low, high), low_name, high_name = value_range, "min_abs_value", "max_abs_value"
    else:
        msg = f"Exactly one of '{name}_range' and 'value_range' has to be provided"
        raise ValueError(msg)
    validate_positive(low, name=low_name)
    validate_positive(high, name=high_name)
    if high < low:
        msg = (
            f"'{high_name}' (received: {high}) has to be greater than "
            f"'{low_name}' (received: {low})"
        )
        raise ValueError(msg)
    return (low, high) if range_ is not None else (1 / high, 1 / low)
