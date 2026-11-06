# Losses

Every loss exists twice: as a function in `sonnix.functional` and as a module in `sonnix.modules`
(for example `log_cosh_loss` and `LogCoshLoss`). The module stores the hyperparameters and calls
the function.

## Arguments

A loss takes `(prediction, target)`, in this order, and returns a tensor. The options are
keyword-only:

```python
import torch
from sonnix.modules import LogCoshLoss

criterion = LogCoshLoss(reduction="sum")
loss = criterion(torch.randn(4, 3), torch.randn(4, 3))
```

## Reduction

All the losses accept the same `reduction` values. The elementwise loss `l` of a prediction of
shape `(N, ...)` is reduced as follows:

| Value         | Result                                       | Shape        |
| ------------- | -------------------------------------------- | ------------ |
| `"none"`      | no reduction                                 | same as `l`  |
| `"mean"`      | the sum divided by the number of elements    | scalar       |
| `"sum"`       | the sum of all the elements                  | scalar       |
| `"batchmean"` | the sum divided by the size of dimension `0` | scalar       |

`"mean"` is the default. `sonnix.functional.reduce_loss` implements these strategies and validates
the value, so a custom loss only has to call it:

```python
import torch
from sonnix.functional import reduce_loss

reduce_loss(torch.ones(4, 3), reduction="batchmean")  # tensor(3.)
```

## Composing losses

- `TransformedLoss` applies a module to the prediction and/or to the target before the criterion
  (for example `Asinh` to compare values that span several orders of magnitude).
- `RelativeLoss` divides the error of a criterion by an indicator of the magnitude of the
  prediction and the target (`ClassicalRelativeIndicator`, `MomentMeanIndicator`, ...), so the
  loss is not dominated by the large values.

The criterion and the transforms can be modules or [configurations](configurations.md).

See the [reference](../refs/modules/loss.md) for the list of losses.
