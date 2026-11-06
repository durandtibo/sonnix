r"""Contain modules to encode numerical values using piecewise linear
functions."""

from __future__ import annotations

__all__ = ["PiecewiseLinearNumericalEncoder"]

import warnings
from typing import TYPE_CHECKING

import torch
from torch import nn

from sonnix.modules.numerical._utils import prepare_tensor_param

if TYPE_CHECKING:
    from torch import Tensor


class PiecewiseLinearNumericalEncoder(nn.Module):
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
            width is set to ``1``.

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

    def __init__(self, bins: Tensor) -> None:
        super().__init__()
        bins = prepare_tensor_param(bins, name="bins").sort(dim=1)[0]
        n_bins = bins.shape[1]
        self.register_buffer("edges", bins[:, :-1] if n_bins > 1 else bins)
        width = bins.diff() if n_bins > 1 else torch.ones_like(bins)
        is_duplicate = width == 0
        if n_bins > 2 and is_duplicate.any():
            warnings.warn(
                "The bins contain duplicate edges. The corresponding segments are step functions.",
                UserWarning,
                stacklevel=2,
            )
        else:
            is_duplicate = torch.zeros_like(is_duplicate)
        width[width == 0] = 1.0
        self.register_buffer("is_step", is_duplicate, persistent=False)
        self.register_buffer("width", width)

    @property
    def n_features(self) -> int:
        r"""Return the input feature size i.e. the number of scalar
        values."""
        return self.edges.shape[0]

    @property
    def feature_size(self) -> int:
        r"""Return the output feature size i.e. the number of bins minus
        one."""
        return self.edges.shape[1]

    def extra_repr(self) -> str:
        return f"n_features={self.n_features}, feature_size={self.feature_size}"

    def forward(self, scalar: Tensor) -> Tensor:
        delta = scalar[..., None] - self.edges
        x = torch.where(self.is_step, (delta > 0).to(delta.dtype), delta / self.width)
        n_bins = x.shape[-1]
        if n_bins == 1:
            return x
        return torch.cat(
            [
                x[..., :1].clamp_max(1.0),
                *([] if n_bins == 2 else [x[..., 1:-1].clamp(0.0, 1.0)]),
                x[..., -1:].clamp_min(0.0),
            ],
            dim=-1,
        )
