r"""Contain modules to encode numerical values using cosine and sine
functions."""

from __future__ import annotations

__all__ = [
    "AsinhCosSinNumericalEncoder",
    "CosSinNumericalEncoder",
]

import math
from typing import TYPE_CHECKING, Literal

import torch
from coola.validation import validate_in
from torch import nn

from sonnix.modules.numerical._utils import (
    check_abs_range,
    check_frequency,
    prepare_tensor_param,
)
from sonnix.utils.random import resolve_torch_generator
from sonnix.validation import validate_same_shape

if TYPE_CHECKING:
    from typing import Self

    from torch import Tensor

SPACINGS = ("rand", "linspace", "logspace")


class CosSinNumericalEncoder(nn.Module):
    r"""Implement a frequency/phase-shift numerical encoder where the
    periodic functions are cosine and sine.

    Args:
        frequency: The initial frequency values. This input should be
            a tensor of shape ``(n_features, feature_size // 2)`` or
            ``(feature_size // 2,)``.
        phase_shift: The initial phase-shift values. This input should
            be a tensor of shape ``(n_features, feature_size // 2)`` or
            ``(feature_size // 2,)``.
        learnable: If ``True`` the frequencies and phase-shift
            parameters are learnable, otherwise they are frozen.

    Shape:
        - Input: ``(*, n_features)``, where ``*`` means any number of
            dimensions.
        - Output: ``(*, n_features, feature_size)``,  where ``*`` has
            the same shape as the input.

    Example:
        ```pycon
        >>> import torch
        >>> from sonnix.modules import CosSinNumericalEncoder
        >>> # Example with 1 feature
        >>> m = CosSinNumericalEncoder(
        ...     frequency=torch.tensor([[1.0, 2.0, 4.0]]),
        ...     phase_shift=torch.zeros(1, 3),
        ... )
        >>> m
        CosSinNumericalEncoder(frequency=(1, 3), phase_shift=(1, 3), learnable=False)
        >>> out = m(torch.tensor([[0.0], [1.0], [2.0], [3.0]]))
        >>> out
        tensor([[[ 0.0000,  0.0000,  0.0000,  1.0000,  1.0000,  1.0000]],
                [[ 0.8415,  0.9093, -0.7568,  0.5403, -0.4161, -0.6536]],
                [[ 0.9093, -0.7568,  0.9894, -0.4161, -0.6536, -0.1455]],
                [[ 0.1411, -0.2794, -0.5366, -0.9900,  0.9602,  0.8439]]])
        >>> # Example with 2 features
        >>> m = CosSinNumericalEncoder(
        ...     frequency=torch.tensor([[1.0, 2.0, 4.0], [2.0, 4.0, 6.0]]),
        ...     phase_shift=torch.zeros(2, 3),
        ... )
        >>> m
        CosSinNumericalEncoder(frequency=(2, 3), phase_shift=(2, 3), learnable=False)
        >>> out = m(torch.tensor([[0.0, 3.0], [1.0, 2.0], [2.0, 1.0], [3.0, 0.0]]))
        >>> out
        tensor([[[ 0.0000,  0.0000,  0.0000,  1.0000,  1.0000,  1.0000],
                 [-0.2794, -0.5366, -0.7510,  0.9602,  0.8439,  0.6603]],
                [[ 0.8415,  0.9093, -0.7568,  0.5403, -0.4161, -0.6536],
                 [-0.7568,  0.9894, -0.5366, -0.6536, -0.1455,  0.8439]],
                [[ 0.9093, -0.7568,  0.9894, -0.4161, -0.6536, -0.1455],
                 [ 0.9093, -0.7568, -0.2794, -0.4161, -0.6536,  0.9602]],
                [[ 0.1411, -0.2794, -0.5366, -0.9900,  0.9602,  0.8439],
                 [ 0.0000,  0.0000,  0.0000,  1.0000,  1.0000,  1.0000]]])

        ```
    """

    def __init__(self, frequency: Tensor, phase_shift: Tensor, *, learnable: bool = False) -> None:
        super().__init__()
        frequency = prepare_tensor_param(frequency, name="frequency")
        self.frequency: Tensor
        self.phase_shift: Tensor
        self._register_tensor("frequency", frequency, learnable)

        phase_shift = prepare_tensor_param(phase_shift, name="phase_shift")
        self._register_tensor("phase_shift", phase_shift, learnable)
        validate_same_shape(
            self.frequency, self.phase_shift, name1="'frequency'", name2="'phase_shift'"
        )

    @property
    def learnable(self) -> bool:
        r"""Indicate if the frequencies and phase-shifts are
        learnable."""
        return isinstance(self.frequency, nn.Parameter)

    def _register_tensor(self, name: str, value: Tensor, learnable: bool) -> None:
        # Frozen values are buffers: they are saved in the state dict and moved with
        # the module, but they are not returned by ``parameters()``.
        if learnable:
            self.register_parameter(name, nn.Parameter(value.detach().clone()))
        else:
            self.register_buffer(name, value.detach().clone())

    @property
    def n_features(self) -> int:
        r"""Return the input feature size."""
        return self.frequency.shape[0]

    @property
    def feature_size(self) -> int:
        r"""Return the output feature size."""
        return 2 * self.frequency.shape[1]

    def extra_repr(self) -> str:
        return (
            f"frequency={tuple(self.frequency.shape)}, "
            f"phase_shift={tuple(self.phase_shift.shape)}, "
            f"learnable={self.learnable}"
        )

    def forward(self, scalar: Tensor) -> Tensor:
        features = scalar.unsqueeze(dim=-1).mul(self.frequency).add(self.phase_shift)
        return torch.cat((features.sin(), features.cos()), dim=-1)

    @classmethod
    def create(
        cls,
        spacing: Literal["rand", "linspace", "logspace"],
        num_frequencies: int,
        *,
        frequency_range: tuple[float, float] | None = None,
        value_range: tuple[float, float] | None = None,
        learnable: bool = False,
        generator: torch.Generator | int | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> Self:
        r"""Create an encoder for one feature where the frequencies are
        initialized in a range and the phase-shifts are zero.

        The range is given either as a frequency range
        (``frequency_range``) or as the range of the absolute values
        to encode (``value_range``). The latter is converted to the
        frequency range ``(1 / max_abs_value, 1 / min_abs_value)``.

        Args:
            spacing: The way the frequencies are spread in the range.
                ``"rand"``: uniformly sampled, ``"linspace"``: evenly
                spaced, ``"logspace"``: evenly spaced in the log space.
            num_frequencies: The number of frequencies.
            frequency_range: The ``(min_frequency, max_frequency)``
                range. The minimum has to be strictly positive and the
                maximum has to be greater than or equal to the minimum.
            value_range: The ``(min_abs_value, max_abs_value)`` range
                of the absolute values to encode. The minimum has to be
                strictly positive and the maximum has to be greater
                than or equal to the minimum.
            learnable: If ``True`` the parameters are learnable,
                otherwise they are frozen.
            generator: The random number generator or the random seed
                used by ``spacing="rand"``. ``None`` uses the global
                random number generator. A generator has to be on the
                same device as ``device``. It is ignored by the other
                spacings.
            device: The device of the parameters.
            dtype: The data type of the parameters. ``None`` uses the
                default floating point type.

        Returns:
            An instantiated encoder with ``n_features=1``.

        Raises:
            ValueError: if ``spacing`` is not valid, if both or none of
                ``frequency_range`` and ``value_range`` are given, or if
                the range or ``num_frequencies`` are invalid.

        Example:
            ```pycon
            >>> import torch
            >>> from sonnix.modules import CosSinNumericalEncoder
            >>> m = CosSinNumericalEncoder.create(
            ...     "linspace", num_frequencies=5, frequency_range=(0.1, 1.0)
            ... )
            >>> m
            CosSinNumericalEncoder(frequency=(1, 5), phase_shift=(1, 5), learnable=False)
            >>> m = CosSinNumericalEncoder.create(
            ...     "logspace", num_frequencies=5, value_range=(0.1, 10.0)
            ... )
            >>> m
            CosSinNumericalEncoder(frequency=(1, 5), phase_shift=(1, 5), learnable=False)
            >>> m = CosSinNumericalEncoder.create(
            ...     "rand", num_frequencies=5, frequency_range=(0.1, 1.0), generator=42
            ... )
            >>> m.frequency.shape
            torch.Size([1, 5])

            ```
        """
        validate_in(spacing, SPACINGS, name="spacing")
        msg = "Exactly one of 'frequency_range' and 'value_range' has to be provided"
        if frequency_range is not None and value_range is not None:
            raise ValueError(msg)
        if frequency_range is not None:
            min_frequency, max_frequency = frequency_range
        elif value_range is not None:
            check_abs_range(min_abs_value=value_range[0], max_abs_value=value_range[1])
            min_frequency, max_frequency = 1 / value_range[1], 1 / value_range[0]
        else:
            raise ValueError(msg)
        check_frequency(
            num_frequencies=num_frequencies,
            min_frequency=min_frequency,
            max_frequency=max_frequency,
        )
        if spacing == "rand":
            frequency = (
                torch.rand(
                    1,
                    num_frequencies,
                    generator=None if generator is None else resolve_torch_generator(generator),
                    device=device,
                    dtype=dtype,
                )
                .mul(max_frequency - min_frequency)
                .add(min_frequency)
            )
        elif spacing == "linspace":
            frequency = torch.linspace(
                start=min_frequency,
                end=max_frequency,
                steps=num_frequencies,
                device=device,
                dtype=dtype,
            )
        else:
            frequency = torch.logspace(
                start=math.log10(min_frequency),
                end=math.log10(max_frequency),
                steps=num_frequencies,
                device=device,
                dtype=dtype,
            )
        return cls(
            frequency=frequency,
            phase_shift=torch.zeros_like(frequency),
            learnable=learnable,
        )


class AsinhCosSinNumericalEncoder(CosSinNumericalEncoder):
    r"""Extension of ``CosSinNumericalEncoder`` with an additional
    feature built using the inverse hyperbolic sine (arcsinh).

    Args:
        frequency: The initial frequency values. This input should be
            a tensor of shape ``(n_features, feature_size // 2)`` or
            ``(feature_size // 2,)``.
        phase_shift: The initial phase-shift values. This input should
            be a tensor of shape ``(n_features, feature_size // 2)`` or
            ``(feature_size // 2,)``.
        learnable: If ``True`` the frequencies and phase-shift
            parameters are learnable, otherwise they are frozen.

    Shape:
        - Input: ``(*, n_features)``, where ``*`` means any number of
            dimensions.
        - Output: ``(*, n_features, feature_size + 1)``,  where ``*``
            has the same shape as the input.

    Example:
        ```pycon
        >>> import torch
        >>> from sonnix.modules import AsinhCosSinNumericalEncoder
        >>> # Example with 1 feature
        >>> m = AsinhCosSinNumericalEncoder(
        ...     frequency=torch.tensor([[1.0, 2.0, 4.0]]),
        ...     phase_shift=torch.zeros(1, 3),
        ... )
        >>> m
        AsinhCosSinNumericalEncoder(frequency=(1, 3), phase_shift=(1, 3), learnable=False)
        >>> out = m(torch.tensor([[0.0], [1.0], [2.0], [3.0]]))
        >>> out
        tensor([[[ 0.0000,  0.0000,  0.0000,  1.0000,  1.0000,  1.0000,  0.0000]],
                [[ 0.8415,  0.9093, -0.7568,  0.5403, -0.4161, -0.6536,  0.8814]],
                [[ 0.9093, -0.7568,  0.9894, -0.4161, -0.6536, -0.1455,  1.4436]],
                [[ 0.1411, -0.2794, -0.5366, -0.9900,  0.9602,  0.8439,  1.8184]]])
        >>> # Example with 2 features
        >>> m = AsinhCosSinNumericalEncoder(
        ...     frequency=torch.tensor([[1.0, 2.0, 4.0], [2.0, 4.0, 6.0]]),
        ...     phase_shift=torch.zeros(2, 3),
        ... )
        >>> m
        AsinhCosSinNumericalEncoder(frequency=(2, 3), phase_shift=(2, 3), learnable=False)
        >>> out = m(torch.tensor([[0.0, 3.0], [1.0, 2.0], [2.0, 1.0], [3.0, 0.0]]))
        >>> out
        tensor([[[ 0.0000,  0.0000,  0.0000,  1.0000,  1.0000,  1.0000,  0.0000],
                 [-0.2794, -0.5366, -0.7510,  0.9602,  0.8439,  0.6603,  1.8184]],
                [[ 0.8415,  0.9093, -0.7568,  0.5403, -0.4161, -0.6536,  0.8814],
                 [-0.7568,  0.9894, -0.5366, -0.6536, -0.1455,  0.8439,  1.4436]],
                [[ 0.9093, -0.7568,  0.9894, -0.4161, -0.6536, -0.1455,  1.4436],
                 [ 0.9093, -0.7568, -0.2794, -0.4161, -0.6536,  0.9602,  0.8814]],
                [[ 0.1411, -0.2794, -0.5366, -0.9900,  0.9602,  0.8439,  1.8184],
                 [ 0.0000,  0.0000,  0.0000,  1.0000,  1.0000,  1.0000,  0.0000]]])

        ```
    """

    @property
    def feature_size(self) -> int:
        r"""Return the output feature size."""
        return super().feature_size + 1

    def forward(self, scalar: Tensor) -> Tensor:
        features = super().forward(scalar)
        return torch.cat(
            (features, scalar.unsqueeze(dim=-1).asinh()),
            dim=-1,
        )
