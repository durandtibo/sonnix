r"""Contain helper functions shared by the numerical encoders."""

from __future__ import annotations

__all__ = ["create_spaced_values", "prepare_tensor_param"]

import math
from typing import TYPE_CHECKING, Literal

import torch
from coola.validation import validate_ge, validate_in, validate_int

from sonnix.utils.random import resolve_torch_generator

if TYPE_CHECKING:
    from torch import Tensor

SPACINGS = ("rand", "linspace", "logspace")


def prepare_tensor_param(tensor: Tensor, name: str) -> Tensor:
    r"""Prepare a tensor parameter to be a 2d tensor.

    Args:
        tensor: The tensor to prepare.
        name: The name associated to the tensor.

    Returns:
        The prepared tensor.

    Raises:
        ValueError: if the input tensor is not a 1d or 2d tensor.

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
                generator=None if generator is None else resolve_torch_generator(generator),
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
