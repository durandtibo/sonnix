from __future__ import annotations

import pytest
import torch

from sonnix.modules import NLinear

pytest.importorskip("pytest_benchmark")

# (batch_size, n, in_features, out_features)
SHAPES = [(32, 8, 16, 16), (256, 16, 32, 32)]


##################################
#     Benchmarks for NLinear     #
##################################


@pytest.mark.parametrize(("batch_size", "n", "in_features", "out_features"), SHAPES)
def test_nlinear_forward(benchmark, batch_size, n, in_features, out_features) -> None:  # noqa: ANN001
    module = NLinear(n=n, in_features=in_features, out_features=out_features)
    x = torch.randn(batch_size, n, in_features, generator=torch.Generator().manual_seed(1))
    out = benchmark.pedantic(module, args=(x,), rounds=5, warmup_rounds=1)
    assert out.shape == (batch_size, n, out_features)
