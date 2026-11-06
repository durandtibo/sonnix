r"""Contain activation modules with a learnable parameter ``alpha``."""

from __future__ import annotations

from coola.validation import validate_finite, validate_ge, validate_int

__all__ = [
    "BaseAlphaActivation",
    "ExpSin",
    "Gaussian",
    "Laplacian",
    "MultiQuadratic",
    "Quadratic",
]

from typing import TYPE_CHECKING

import torch
from torch import nn

if TYPE_CHECKING:
    from torch import Tensor


class BaseAlphaActivation(nn.Module):
    r"""Define a base class to implement an activation layer with a
    learnable parameter ``alpha``.

    When called without arguments, the activation layer uses a single
    parameter ``alpha`` across all input channels. If called with a
    first argument, a separate ``alpha`` is used for each input
    channel.

    Args:
        num_parameters: The number of learnable parameters. Although
            it takes an integer as input, only two values are
            legitimate: ``1``, or the number of channels at input.
        init: The initial value of the learnable parameter(s).
        learnable: If ``True``, the parameters are learned during the
            training, otherwise they are fixed.
        channel_dim: The dimension of the input that ``alpha`` is
            broadcast over when ``num_parameters`` is greater than 1.
            The default ``-1`` is the last dimension. Use ``1`` for
            an input of shape ``(N, C, L)`` like ``torch.nn.PReLU``.
        device: The device of the parameters.
        dtype: The data type of the parameters. ``None`` uses the
            default floating point type.

    Shape:
        - Input: ``(*)``, where ``*`` means any number of dimensions.
        - Output: ``(*)``, same shape as the input.

    Raises:
        TypeError: if ``num_parameters`` or ``channel_dim`` is not an
            integer.
        ValueError: if ``num_parameters`` is lower than 1, if ``init``
            is not finite, if ``init`` is 0 for a module that does not
            allow it, or if ``init`` is negative for a module that
            does not allow it.

    Example:
        ```pycon
        >>> import torch
        >>> from sonnix.modules import MultiQuadratic
        >>> m = MultiQuadratic()
        >>> m
        MultiQuadratic(num_parameters=1, learnable=True)
        >>> out = m(torch.arange(8, dtype=torch.float).view(2, 4))
        >>> out
        tensor([[1.0000, 0.7071, 0.4472, 0.3162],
                [0.2425, 0.1961, 0.1644, 0.1414]], grad_fn=<MulBackward0>)

        ```
    """

    allow_zero_init: bool = True
    allow_negative_init: bool = True

    def __init__(
        self,
        num_parameters: int = 1,
        init: float = 1.0,
        *,
        learnable: bool = True,
        channel_dim: int = -1,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        validate_int(num_parameters, name="num_parameters")
        validate_ge(num_parameters, 1, name="num_parameters")
        validate_finite(init, name="init")
        validate_int(channel_dim, name="channel_dim")
        if init == 0 and not self.allow_zero_init:
            msg = f"'init' has to be different from 0 (received {init})"
            raise ValueError(msg)
        if init < 0 and not self.allow_negative_init:
            msg = f"'init' has to be positive (received {init})"
            raise ValueError(msg)
        self.channel_dim = channel_dim
        self.alpha: Tensor
        alpha = torch.full(
            (num_parameters,),
            init,
            device=device,
            dtype=dtype if dtype is not None else torch.get_default_dtype(),
        )
        if learnable:
            self.alpha = nn.Parameter(alpha)
        else:
            # Frozen values are buffers: they are saved in the state dict and moved with
            # the module, but they are not returned by ``parameters()``.
            self.register_buffer("alpha", alpha)

    @property
    def learnable(self) -> bool:
        r"""Indicate if ``alpha`` is learnable."""
        return isinstance(self.alpha, nn.Parameter)

    def _broadcast_alpha(self, input: torch.Tensor) -> Tensor:  # noqa: A002
        r"""Return ``alpha`` reshaped to broadcast with the input.

        Args:
            input: The input tensor. If ``alpha`` has several values,
                the dimension ``channel_dim`` of the input must have
                the same size.

        Returns:
            ``alpha`` unchanged if it has one value, otherwise a view
                of ``alpha`` with size 1 in all the dimensions of the
                input except ``channel_dim``.
        """
        if self.alpha.numel() == 1:
            return self.alpha
        shape = [1] * input.ndim
        shape[self.channel_dim] = -1
        return self.alpha.view(shape)

    def extra_repr(self) -> str:
        extra = f", channel_dim={self.channel_dim}" if self.alpha.numel() > 1 else ""
        return f"num_parameters={self.alpha.numel()}, learnable={self.learnable}{extra}"


class ExpSin(BaseAlphaActivation):
    r"""Implement the ExpSin activation layer.

    Formula: ``exp(-sin(alpha * x))``

    This activation layer was proposed in the following paper:

        Beyond Periodicity: Towards a Unifying Framework for Activations in Coordinate-MLPs.
        Ramasinghe S., Lucey S.
        ECCV 2022. (http://arxiv.org/pdf/2111.15135)

    Args:
        num_parameters: The number of learnable parameters. Although
            it takes an integer as input, only two values are
            legitimate: ``1``, or the number of channels at input.
        init: The initial value of the learnable parameter(s).
        learnable: If ``True``, the parameters are learned during the
            training, otherwise they are fixed.
        channel_dim: The dimension of the input that ``alpha`` is
            broadcast over when ``num_parameters`` is greater than 1.
            The default ``-1`` is the last dimension. Use ``1`` for
            an input of shape ``(N, C, L)`` like ``torch.nn.PReLU``.
        device: The device of the parameters.
        dtype: The data type of the parameters. ``None`` uses the
            default floating point type.

    Shape:
        - Input: ``(*)``, where ``*`` means any number of dimensions.
        - Output: ``(*)``, same shape as the input.

    Example:
        ```pycon
        >>> import torch
        >>> from sonnix.modules import ExpSin
        >>> m = ExpSin()
        >>> m
        ExpSin(num_parameters=1, learnable=True)
        >>> out = m(torch.arange(8, dtype=torch.float).view(2, 4))
        >>> out
        tensor([[1.0000, 0.4311, 0.4028, 0.8684],
                [2.1314, 2.6089, 1.3224, 0.5184]], grad_fn=<ExpBackward0>)

        ```
    """

    def forward(self, input: torch.Tensor) -> torch.Tensor:  # noqa: A002
        return input.mul(self._broadcast_alpha(input)).sin().neg().exp()


class Gaussian(BaseAlphaActivation):
    r"""Implement the Gaussian activation layer.

    Formula: ``exp(-0.5 * x^2 / alpha^2)``

    Note: the output is ``nan`` or ``inf`` if ``alpha`` reaches ``0``,
    which can happen during training when ``alpha`` is learnable.

    This activation layer was proposed in the following paper:

        Beyond Periodicity: Towards a Unifying Framework for Activations in Coordinate-MLPs.
        Ramasinghe S., Lucey S.
        ECCV 2022. (http://arxiv.org/pdf/2111.15135)

    Args:
        num_parameters: The number of learnable parameters. Although
            it takes an integer as input, only two values are
            legitimate: ``1``, or the number of channels at input.
        init: The initial value of the learnable parameter(s).
        learnable: If ``True``, the parameters are learned during the
            training, otherwise they are fixed.
        channel_dim: The dimension of the input that ``alpha`` is
            broadcast over when ``num_parameters`` is greater than 1.
            The default ``-1`` is the last dimension. Use ``1`` for
            an input of shape ``(N, C, L)`` like ``torch.nn.PReLU``.
        device: The device of the parameters.
        dtype: The data type of the parameters. ``None`` uses the
            default floating point type.

    Shape:
        - Input: ``(*)``, where ``*`` means any number of dimensions.
        - Output: ``(*)``, same shape as the input.

    Raises:
        ValueError: if ``num_parameters`` is lower than 1 or ``init``
            is 0.

    Example:
        ```pycon
        >>> import torch
        >>> from sonnix.modules import Gaussian
        >>> m = Gaussian()
        >>> m
        Gaussian(num_parameters=1, learnable=True)
        >>> out = m(torch.arange(8, dtype=torch.float).view(2, 4))
        >>> out
        tensor([[1.0000e+00, 6.0653e-01, 1.3534e-01, 1.1109e-02],
                [3.3546e-04, 3.7267e-06, 1.5230e-08, 2.2897e-11]], grad_fn=<ExpBackward0>)

        ```
    """

    allow_zero_init = False

    def forward(self, input: torch.Tensor) -> torch.Tensor:  # noqa: A002
        return input.pow(2).mul(-0.5).div(self._broadcast_alpha(input).pow(2)).exp()


class Laplacian(BaseAlphaActivation):
    r"""Implement the Laplacian activation layer.

    Formula: ``exp(-|x| / alpha)``

    Note: the output is ``nan`` or ``inf`` if ``alpha`` reaches ``0``,
    which can happen during training when ``alpha`` is learnable.

    This activation layer was proposed in the following paper:

        Beyond Periodicity: Towards a Unifying Framework for Activations in Coordinate-MLPs.
        Ramasinghe S., Lucey S.
        ECCV 2022. (http://arxiv.org/pdf/2111.15135)

    Args:
        num_parameters: The number of learnable parameters. Although
            it takes an integer as input, only two values are
            legitimate: ``1``, or the number of channels at input.
        init: The initial value of the learnable parameter(s).
        learnable: If ``True``, the parameters are learned during the
            training, otherwise they are fixed.
        channel_dim: The dimension of the input that ``alpha`` is
            broadcast over when ``num_parameters`` is greater than 1.
            The default ``-1`` is the last dimension. Use ``1`` for
            an input of shape ``(N, C, L)`` like ``torch.nn.PReLU``.
        device: The device of the parameters.
        dtype: The data type of the parameters. ``None`` uses the
            default floating point type.

    Shape:
        - Input: ``(*)``, where ``*`` means any number of dimensions.
        - Output: ``(*)``, same shape as the input.

    Raises:
        ValueError: if ``num_parameters`` is lower than 1 or ``init``
            is 0.

    Example:
        ```pycon
        >>> import torch
        >>> from sonnix.modules import Laplacian
        >>> m = Laplacian()
        >>> m
        Laplacian(num_parameters=1, learnable=True)
        >>> out = m(torch.arange(8, dtype=torch.float).view(2, 4))
        >>> out
        tensor([[1.0000e+00, 3.6788e-01, 1.3534e-01, 4.9787e-02],
                [1.8316e-02, 6.7379e-03, 2.4788e-03, 9.1188e-04]], grad_fn=<ExpBackward0>)

        ```
    """

    allow_zero_init = False
    allow_negative_init = False

    def forward(self, input: torch.Tensor) -> torch.Tensor:  # noqa: A002
        return input.abs().mul(-1).div(self._broadcast_alpha(input)).exp()


class MultiQuadratic(BaseAlphaActivation):
    r"""Implement the Multi Quadratic activation layer.

    Formula: ``1 / sqrt(1 + (alpha * x)^2)``

    This activation layer was proposed in the following paper:

        Beyond Periodicity: Towards a Unifying Framework for Activations in Coordinate-MLPs.
        Ramasinghe S., Lucey S.
        ECCV 2022. (http://arxiv.org/pdf/2111.15135)

    Args:
        num_parameters: The number of learnable parameters. Although
            it takes an integer as input, only two values are
            legitimate: ``1``, or the number of channels at input.
        init: The initial value of the learnable parameter(s).
        learnable: If ``True``, the parameters are learned during the
            training, otherwise they are fixed.
        channel_dim: The dimension of the input that ``alpha`` is
            broadcast over when ``num_parameters`` is greater than 1.
            The default ``-1`` is the last dimension. Use ``1`` for
            an input of shape ``(N, C, L)`` like ``torch.nn.PReLU``.
        device: The device of the parameters.
        dtype: The data type of the parameters. ``None`` uses the
            default floating point type.

    Shape:
        - Input: ``(*)``, where ``*`` means any number of dimensions.
        - Output: ``(*)``, same shape as the input.

    Example:
        ```pycon
        >>> import torch
        >>> from sonnix.modules import MultiQuadratic
        >>> m = MultiQuadratic()
        >>> m
        MultiQuadratic(num_parameters=1, learnable=True)
        >>> out = m(torch.arange(8, dtype=torch.float).view(2, 4))
        >>> out
        tensor([[1.0000, 0.7071, 0.4472, 0.3162],
                [0.2425, 0.1961, 0.1644, 0.1414]], grad_fn=<MulBackward0>)

        ```
    """

    def forward(self, input: torch.Tensor) -> torch.Tensor:  # noqa: A002
        return 1.0 / input.mul(self._broadcast_alpha(input)).pow(2).add(1).sqrt()


class Quadratic(BaseAlphaActivation):
    r"""Implement the Quadratic activation layer.

    Formula: ``1 / (1 + (alpha * x)^2)``

    This activation layer was proposed in the following paper:

        Beyond Periodicity: Towards a Unifying Framework for Activations in Coordinate-MLPs.
        Ramasinghe S., Lucey S.
        ECCV 2022. (http://arxiv.org/pdf/2111.15135)

    Args:
        num_parameters: The number of learnable parameters. Although
            it takes an integer as input, only two values are
            legitimate: ``1``, or the number of channels at input.
        init: The initial value of the learnable parameter(s).
        learnable: If ``True``, the parameters are learned during the
            training, otherwise they are fixed.
        channel_dim: The dimension of the input that ``alpha`` is
            broadcast over when ``num_parameters`` is greater than 1.
            The default ``-1`` is the last dimension. Use ``1`` for
            an input of shape ``(N, C, L)`` like ``torch.nn.PReLU``.
        device: The device of the parameters.
        dtype: The data type of the parameters. ``None`` uses the
            default floating point type.

    Shape:
        - Input: ``(*)``, where ``*`` means any number of dimensions.
        - Output: ``(*)``, same shape as the input.

    Example:
        ```pycon
        >>> import torch
        >>> from sonnix.modules import Quadratic
        >>> m = Quadratic()
        >>> m
        Quadratic(num_parameters=1, learnable=True)
        >>> out = m(torch.arange(8, dtype=torch.float).view(2, 4))
        >>> out
        tensor([[1.0000, 0.5000, 0.2000, 0.1000],
                [0.0588, 0.0385, 0.0270, 0.0200]], grad_fn=<MulBackward0>)

        ```
    """

    def forward(self, input: torch.Tensor) -> torch.Tensor:  # noqa: A002
        return 1.0 / input.mul(self._broadcast_alpha(input)).pow(2).add(1)
