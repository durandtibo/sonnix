# Numerical encoders and embeddings

A numerical encoder maps each scalar feature to a vector of `feature_size` values. The input has
shape `(*, n_features)` and the output has shape `(*, n_features, feature_size)`.

| Encoder                           | Encoding of a value `x`                                                |
| --------------------------------- | ---------------------------------------------------------------------- |
| `AsinhNumericalEncoder`           | `asinh(x / scale)`, one value per scale                                |
| `CosSinNumericalEncoder`          | cosine and sine of `x * frequency + phase`                             |
| `AsinhCosSinNumericalEncoder`     | the cosine/sine encoding plus an `asinh` feature                       |
| `PiecewiseLinearNumericalEncoder` | the position of `x` in each bin, as in the "piecewise linear" encoding |

The classes have a `create` (or `from_quantiles`) class method that builds the encoder from the
data or from a range, with one set of parameters per feature (`n_features`).

## Embedding of numerical features

An encoder is usually followed by `NLinear`, which applies a different linear layer to each
feature. The composition below turns a table of `n_features` numbers into
`n_features` embeddings of size `d_embedding`:

```python
import torch
from torch import nn
from sonnix.modules import NLinear, PiecewiseLinearNumericalEncoder

x = torch.randn(128, 3)  # (batch_size, n_features)
encoder = PiecewiseLinearNumericalEncoder.from_quantiles(x, n_bins=8)
embedding = nn.Sequential(
    encoder,
    NLinear(n=3, in_features=encoder.feature_size, out_features=16),
    nn.ReLU(),
)
out = embedding(x)  # shape: (128, 3, 16)
```

`NLinear(n, in_features, out_features)` expects an input of shape `(*, n, in_features)` and
raises a `ValueError` that names `n` when the shape does not match.

See the [reference](../refs/modules/numerical.md) for the list of encoders.
