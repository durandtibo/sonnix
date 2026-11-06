r"""Contain modules to encode numerical values using piecewise linear
functions."""

from __future__ import annotations

__all__ = ["PiecewiseLinearNumericalEncoder"]

import warnings
from typing import TYPE_CHECKING, Literal

import torch
from coola.validation import validate_ge, validate_in

from sonnix.modules.numerical._utils import NumericalEncoder, prepare_tensor_param

if TYPE_CHECKING:
    from typing import Self

    from torch import Tensor


class PiecewiseLinearNumericalEncoder(NumericalEncoder):
    r"""Implement a numerical encoder using piecewise linear functions.

    This layer was proposed in the following paper:

    On Embeddings for Numerical Features in Tabular Deep Learning
    Yury Gorishniy, Ivan Rubachev, Artem Babenko
    NeurIPS 2022, https://arxiv.org/pdf/2203.05556
    https://github.com/yandex-research/rtdl-num-embeddings

    Args:
        bins: The bins used to compute the piecewise linear
            representations. This input should be a tensor of shape
            ``(n_features, n_bins)`` or ``(n_bins,)``. The bin values
            are sorted by ascending order for each feature. Duplicate
            edges are accepted with a warning when there are more
            than two bins: a segment of zero width is a step function
            (``0`` when the input is not above the edge, ``1``
            otherwise). With only two bins (a single segment), the
            width is set to ``1``. Integer bins are converted to the
            default floating point dtype.
        on_duplicate: The behaviour when ``bins`` contain duplicate
            edges: ``'warn'`` emits a ``UserWarning`` and ``'ignore'``
            is silent. The duplicate edges are step functions in both
            cases.

    Raises:
        ValueError: if ``bins`` is not a 1d or 2d tensor, if it has no
            bin, if it contains a ``NaN`` or an infinite value, or if
            ``on_duplicate`` is not valid.

    Shape:
        - Input: ``(*, n_features)``, where ``*`` means any number of
            dimensions.
        - Output: ``(*, n_features, n_bins - 1)``,  where ``*`` has
            the same shape as the input.

    Example:
        ```pycon
        >>> import torch
        >>> from sonnix.modules import PiecewiseLinearNumericalEncoder
        >>> # Example with 1 feature
        >>> m = PiecewiseLinearNumericalEncoder(bins=torch.tensor([[1.0, 2.0, 4.0, 8.0]]))
        >>> m
        PiecewiseLinearNumericalEncoder(n_features=1, feature_size=3)
        >>> out = m(torch.tensor([[0.0], [1.0], [2.0], [3.0]]))
        >>> out
        tensor([[[-1.0000,  0.0000,  0.0000]],
                [[ 0.0000,  0.0000,  0.0000]],
                [[ 1.0000,  0.0000,  0.0000]],
                [[ 1.0000,  0.5000,  0.0000]]])
        >>> # Example with 2 features
        >>> m = PiecewiseLinearNumericalEncoder(
        ...     bins=torch.tensor([[1.0, 2.0, 4.0, 8.0], [0.0, 2.0, 4.0, 6.0]])
        ... )
        >>> m
        PiecewiseLinearNumericalEncoder(n_features=2, feature_size=3)
        >>> out = m(torch.tensor([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]))
        >>> out
        tensor([[[-1.0000,  0.0000,  0.0000],
                 [ 0.0000,  0.0000,  0.0000]],
                [[ 0.0000,  0.0000,  0.0000],
                 [ 0.5000,  0.0000,  0.0000]],
                [[ 1.0000,  0.0000,  0.0000],
                 [ 1.0000,  0.0000,  0.0000]],
                [[ 1.0000,  0.5000,  0.0000],
                 [ 1.0000,  0.5000,  0.0000]]])

        ```
    """

    edges: Tensor
    width: Tensor
    is_step: Tensor
    lower: Tensor
    upper: Tensor

    def __init__(self, bins: Tensor, *, on_duplicate: Literal["warn", "ignore"] = "warn") -> None:
        super().__init__()
        validate_in(on_duplicate, ("warn", "ignore"), name="on_duplicate")
        bins = prepare_tensor_param(bins, name="bins")
        if not bins.is_floating_point():
            bins = bins.to(torch.get_default_dtype())
        bins = bins.sort(dim=1)[0]
        n_bins = bins.shape[1]
        validate_ge(n_bins, 1, name="n_bins")
        if not bins.isfinite().all():
            msg = "bins must contain only finite values"
            raise ValueError(msg)
        self.register_buffer("edges", bins[:, :-1] if n_bins > 1 else bins)
        width = bins.diff() if n_bins > 1 else torch.ones_like(bins)
        # With a single segment, a zero width is replaced by 1 instead of a step.
        is_step = (width == 0) & (n_bins > 2)
        if on_duplicate == "warn" and is_step.any():
            warnings.warn(
                "The bins contain duplicate edges. The corresponding segments are step functions.",
                UserWarning,
                stacklevel=2,
            )
        width = torch.where(width == 0, 1.0, width)
        self.register_buffer("is_step", is_step, persistent=False)
        self.register_buffer("width", width)
        # The first segment is only clamped above and the last segment only
        # below, the other segments are clamped to ``[0, 1]``. A single
        # segment is not clamped. The bounds are precomputed so ``forward``
        # has a single ``clamp`` and no data-dependent Python branch.
        n_segments = width.shape[1]
        lower = torch.zeros(n_segments, dtype=width.dtype, device=width.device)
        upper = torch.ones(n_segments, dtype=width.dtype, device=width.device)
        if n_segments == 1:
            lower.fill_(-torch.inf)
            upper.fill_(torch.inf)
        else:
            lower[0] = -torch.inf
            upper[-1] = torch.inf
        self.register_buffer("lower", lower, persistent=False)
        self.register_buffer("upper", upper, persistent=False)

    @classmethod
    def from_quantiles(
        cls, x: Tensor, n_bins: int, *, on_duplicate: Literal["warn", "ignore"] = "warn"
    ) -> Self:
        r"""Create an encoder where the bins are the quantiles of the
        data.

        This is the main use case of the paper: the edges of the bins
        of each feature are the empirical quantiles of the feature
        computed on the training data.

        Args:
            x: The data used to compute the quantiles. This input
                should be a floating point tensor of shape
                ``(n_samples, n_features)`` or ``(n_samples,)``.
            n_bins: The number of bins, which has to be greater than
                or equal to 2. The edges are the quantiles
                ``0, 1 / (n_bins - 1), ..., 1`` of each feature.
            on_duplicate: The behaviour when some quantiles are equal,
                which is common for discrete or skewed features:
                ``'warn'`` emits a ``UserWarning`` and ``'ignore'`` is
                silent. The equal quantiles give step segments in both
                cases.

        Returns:
            An instantiated encoder with ``n_bins - 1`` segments per
                feature.

        Raises:
            ValueError: if ``n_bins`` is lower than 2, if ``x`` is
                not a 1d or 2d tensor, or if ``on_duplicate`` is not
                valid.

        Example:
            ```pycon
            >>> import torch
            >>> from sonnix.modules import PiecewiseLinearNumericalEncoder
            >>> m = PiecewiseLinearNumericalEncoder.from_quantiles(
            ...     torch.arange(11, dtype=torch.float).view(11, 1), n_bins=3
            ... )
            >>> m
            PiecewiseLinearNumericalEncoder(n_features=1, feature_size=2)
            >>> m.edges
            tensor([[0., 5.]])

            ```
        """
        validate_ge(n_bins, 2, name="n_bins")
        x = prepare_tensor_param(x.view(-1, 1) if x.ndim == 1 else x, name="x")
        probabilities = torch.linspace(0.0, 1.0, n_bins, dtype=x.dtype, device=x.device)
        return cls(torch.quantile(x, probabilities, dim=0).t(), on_duplicate=on_duplicate)

    @property
    def n_features(self) -> int:
        return self.edges.shape[0]

    @property
    def feature_size(self) -> int:
        r"""Return the output feature size i.e. the number of bins minus
        one."""
        return self.edges.shape[1]

    def extra_repr(self) -> str:
        return f"n_features={self.n_features}, feature_size={self.feature_size}"

    def _encode(self, input: Tensor) -> Tensor:  # noqa: A002
        delta = input[..., None] - self.edges
        x = torch.where(self.is_step, (delta > 0).to(delta.dtype), delta / self.width)
        return x.clamp(min=self.lower, max=self.upper)
