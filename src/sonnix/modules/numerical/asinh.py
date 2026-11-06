r"""Contain modules to encode numerical values using the inverse
hyperbolic sine (asinh)."""

from __future__ import annotations

__all__ = ["AsinhNumericalEncoder"]

from typing import TYPE_CHECKING, Literal

from coola.validation import validate_ge, validate_gt
from torch import nn

from sonnix.modules.numerical._utils import create_spaced_values, prepare_tensor_param
from sonnix.validation import validate_abs_range

if TYPE_CHECKING:
    from typing import Self

    import torch
    from torch import Tensor


class AsinhNumericalEncoder(nn.Module):
    r"""Implement a numerical encoder using the inverse hyperbolic sine
    (asinh).

    Args:
        scale: The initial scale values. This input should be a tensor
            of shape ``(n_features, feature_size)`` or
            ``(feature_size,)``.
        learnable: If ``True`` the scales are learnable,
            otherwise they are frozen.

    Shape:
        - Input: ``(*, n_features)``, where ``*`` means any number of
            dimensions.
        - Output: ``(*, n_features, feature_size)``,  where ``*`` has
            the same shape as the input.

    Example:
        ```pycon
        >>> import torch
        >>> from sonnix.modules import AsinhNumericalEncoder
        >>> # Example with 1 feature
        >>> m = AsinhNumericalEncoder(scale=torch.tensor([[1.0, 2.0, 4.0]]))
        >>> m
        AsinhNumericalEncoder(scale=(1, 3), learnable=False)
        >>> out = m(torch.tensor([[0.0], [1.0], [2.0], [3.0]]))
        >>> out
        tensor([[[0.0000, 0.0000, 0.0000]],
                [[0.8814, 1.4436, 2.0947]],
                [[1.4436, 2.0947, 2.7765]],
                [[1.8184, 2.4918, 3.1798]]])
        >>> # Example with 2 features
        >>> m = AsinhNumericalEncoder(scale=torch.tensor([[1.0, 2.0, 4.0], [1.0, 3.0, 6.0]]))
        >>> m
        AsinhNumericalEncoder(scale=(2, 3), learnable=False)
        >>> out = m(torch.tensor([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]))
        >>> out
        tensor([[[0.0000, 0.0000, 0.0000], [0.0000, 0.0000, 0.0000]],
                [[0.8814, 1.4436, 2.0947], [0.8814, 1.8184, 2.4918]],
                [[1.4436, 2.0947, 2.7765], [1.4436, 2.4918, 3.1798]],
                [[1.8184, 2.4918, 3.1798], [1.8184, 2.8934, 3.5843]]])

        ```
    """

    def __init__(self, scale: Tensor, *, learnable: bool = False) -> None:
        super().__init__()
        scale = prepare_tensor_param(scale, name="scale")
        self.scale: Tensor
        if learnable:
            self.scale = nn.Parameter(scale.detach().clone())
        else:
            # Frozen values are buffers: they are saved in the state dict and moved with
            # the module, but they are not returned by ``parameters()``.
            self.register_buffer("scale", scale.detach().clone())

    @classmethod
    def create(
        cls,
        spacing: Literal["rand", "linspace", "logspace"],
        feature_size: int,
        *,
        scale_range: tuple[float, float] | None = None,
        value_range: tuple[float, float] | None = None,
        n_features: int = 1,
        learnable: bool = False,
        generator: torch.Generator | int | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> Self:
        r"""Create an encoder where the scales are initialized in a
        range.

        The range is given either as a scale range (``scale_range``)
        or as the range of the absolute values to encode
        (``value_range``). The latter is converted to the scale range
        ``(1 / max_abs_value, 1 / min_abs_value)``, like
        ``CosSinNumericalEncoder.create``.

        Args:
            spacing: The way the scales are spread in the range.
                ``"rand"``: uniformly sampled, ``"linspace"``: evenly
                spaced, ``"logspace"``: evenly spaced in the log space.
            feature_size: The number of scales per feature i.e. the
                output feature size.
            scale_range: The ``(min_scale, max_scale)`` range. The
                minimum has to be strictly positive and the maximum
                has to be greater than or equal to the minimum.
            value_range: The ``(min_abs_value, max_abs_value)`` range
                of the absolute values to encode. The minimum has to be
                strictly positive and the maximum has to be greater
                than or equal to the minimum.
            n_features: The number of features. The scales are
                sampled independently for each feature with
                ``spacing="rand"``, otherwise every feature has the
                same scales.
            learnable: If ``True`` the scales are learnable,
                otherwise they are frozen.
            generator: The random number generator or the random seed
                used by ``spacing="rand"``. ``None`` uses the global
                random number generator. It is ignored by the other
                spacings.
            device: The device of the parameters.
            dtype: The data type of the parameters. ``None`` uses the
                default floating point type.

        Returns:
            An instantiated encoder with ``n_features`` features.

        Raises:
            ValueError: if ``spacing`` is not valid, if both or none of
                ``scale_range`` and ``value_range`` are given, or if
                the range, ``feature_size`` or ``n_features`` are
                invalid.

        Example:
            ```pycon
            >>> import torch
            >>> from sonnix.modules import AsinhNumericalEncoder
            >>> m = AsinhNumericalEncoder.create(
            ...     "logspace", feature_size=4, value_range=(0.1, 10.0), n_features=2
            ... )
            >>> m
            AsinhNumericalEncoder(scale=(2, 4), learnable=False)
            >>> m.scale
            tensor([[ 0.1000,  0.4642,  2.1544, 10.0000],
                    [ 0.1000,  0.4642,  2.1544, 10.0000]])

            ```
        """
        msg = "Exactly one of 'scale_range' and 'value_range' has to be provided"
        if scale_range is not None and value_range is not None:
            raise ValueError(msg)
        if scale_range is not None:
            min_scale, max_scale = scale_range
        elif value_range is not None:
            validate_abs_range(min_abs_value=value_range[0], max_abs_value=value_range[1])
            min_scale, max_scale = 1 / value_range[1], 1 / value_range[0]
        else:
            raise ValueError(msg)
        validate_ge(feature_size, 1, name="feature_size")
        validate_gt(min_scale, 0, name="min_scale")
        if max_scale < min_scale:
            msg = (
                f"'max_scale' (received: {max_scale}) has to be greater than "
                f"'min_scale' (received: {min_scale})"
            )
            raise ValueError(msg)
        scale = create_spaced_values(
            spacing,
            feature_size,
            min_scale,
            max_scale,
            n_features=n_features,
            generator=generator,
            device=device,
            dtype=dtype,
        )
        return cls(scale=scale, learnable=learnable)

    @property
    def learnable(self) -> bool:
        r"""Indicate if the scales are learnable."""
        return isinstance(self.scale, nn.Parameter)

    @property
    def n_features(self) -> int:
        r"""Return the input feature size."""
        return self.scale.shape[0]

    @property
    def feature_size(self) -> int:
        r"""Return the output feature size."""
        return self.scale.shape[1]

    def extra_repr(self) -> str:
        return f"scale={tuple(self.scale.shape)}, learnable={self.learnable}"

    def forward(self, input: Tensor) -> Tensor:  # noqa: A002
        return input.unsqueeze(dim=-1).mul(self.scale).asinh()
