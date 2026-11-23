# Losses

Every loss exists twice: as a function in `sonnix.functional` and as a module in
`sonnix.modules` (for example `log_cosh_loss` and `LogCoshLoss`). The module
stores the hyperparameters and calls the function.

## Arguments

A loss takes `(prediction, target)`, in this order, and returns a tensor. The
options are keyword-only:

```python
import torch
from sonnix.modules import LogCoshLoss

criterion = LogCoshLoss(reduction="sum")
loss = criterion(torch.randn(4, 3), torch.randn(4, 3))
```

## Reduction

All the losses accept the same `reduction` values. The elementwise loss `l` of a
prediction of shape `(N, ...)` is reduced as follows:

| Value         | Result                                       | Shape       |
| ------------- | -------------------------------------------- | ----------- |
| `"none"`      | no reduction                                 | same as `l` |
| `"mean"`      | the sum divided by the number of elements    | scalar      |
| `"sum"`       | the sum of all the elements                  | scalar      |
| `"batchmean"` | the sum divided by the size of dimension `0` | scalar      |

`"mean"` is the default. `sonnix.functional.reduce_loss` implements these
strategies and validates the value, so a custom loss only has to call it:

```python
import torch
from sonnix.functional import reduce_loss

reduce_loss(torch.ones(4, 3), reduction="batchmean")  # tensor(3.)
```

## General robust loss

`general_robust_regression_loss` (`GeneralRobustRegressionLoss`) is Barron's
loss. The limit cases of `alpha` use their own formula, and the loss is
continuous in `alpha`: `alpha=2` is `0.5 * (x / scale) ** 2`, half of the
squared error, like the value of `alpha=1.99` and `alpha=2.0001`. The default
`alpha=2.0` is therefore half of `torch.nn.functional.mse_loss`; use
`scale=s / sqrt(2)` to get the squared error of scale `s`.

```python
import torch
from sonnix.functional import general_robust_regression_loss

general_robust_regression_loss(torch.tensor([3.0]), torch.zeros(1), alpha=2.0)  # tensor(4.5000)
```

## Guards and the dtype

The losses that divide by a quantity or take a logarithm have an `eps`
(`poisson_regression_loss`, `relative_loss`, `absolute_relative_error`,
`symmetric_absolute_relative_error`). The default `1e-8` is `0` in `float16`, so
`eps` is raised to the smallest normal number of the dtype of the tensor
(`6.1e-05` in `float16`) when it is lower. A value that is already large enough
is kept, and `float32`, `float64` and `bfloat16` are not changed. See
[Numerical stability](numerical_stability.md).

## Composing losses

A loss that is a composition of a `torch.nn` criterion and a transform is a
recipe, not a class.

- `TransformedLoss` applies a module to the prediction and/or to the target
  before the criterion (for example `Asinh` to compare values that span several
  orders of magnitude).
- `RelativeLoss` divides the error of a criterion by an indicator of the
  magnitude of the prediction and the target, so the loss is not dominated by
  the large values. The indicator is a function of `sonnix.functional`
  (`classical_relative_indicator`, `moment_mean_indicator`, ...); use
  `functools.partial` to set its options.

```python
from functools import partial

from torch import nn

from sonnix.functional import geometric_mean_indicator, moment_mean_indicator
from sonnix.modules import Asinh, Log1p, RelativeLoss, TransformedLoss

asinh_mse = TransformedLoss(nn.MSELoss(), prediction_transform=Asinh(), target_transform=Asinh())
asinh_smooth_l1 = TransformedLoss(
    nn.SmoothL1Loss(beta=0.5), prediction_transform=Asinh(), target_transform=Asinh()
)
msle = TransformedLoss(nn.MSELoss(), prediction_transform=Log1p(), target_transform=Log1p())
relative_mse = RelativeLoss(nn.MSELoss(reduction="none"), indicator=geometric_mean_indicator)
relative_smooth_l1 = RelativeLoss(
    nn.SmoothL1Loss(reduction="none"), indicator=partial(moment_mean_indicator, k=2)
)
```

The criterion and the transforms can be modules or
[configurations](configurations.md). The criterion of `RelativeLoss` must
not reduce (`reduction="none"`).

See the [reference](../refs/modules/loss.md) for the list of losses.
