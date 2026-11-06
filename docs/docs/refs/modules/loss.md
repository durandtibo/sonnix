# Losses

Loss modules, the relative loss indicators and the shared base classes. See also the [`sonnix.modules`](index.md) overview.

| Name                                                                         | Description                                                                                                                                    |
| ---------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| [`ArithmeticalMeanIndicator`](#sonnix.modules.ArithmeticalMeanIndicator)     | Implement the arithmetical mean change indicator function.                                                                                     |
| [`AsinhMSELoss`](#sonnix.modules.AsinhMSELoss)                               | Implement a loss module that computes the mean squared error (MSE) on the inverse hyperbolic sine (asinh) transformed predictions and targets. |
| [`AsinhSmoothL1Loss`](#sonnix.modules.AsinhSmoothL1Loss)                     | Implement a loss module that computes the smooth L1 loss on the inverse hyperbolic sine (asinh) transformed predictions and targets.           |
| [`BaseBinaryFocalLoss`](#sonnix.modules.BaseBinaryFocalLoss)                 | Define the base class of the binary focal losses.                                                                                              |
| [`BaseBinaryPoly1Loss`](#sonnix.modules.BaseBinaryPoly1Loss)                 | Define the base class of the binary Poly-1 losses.                                                                                             |
| [`BaseLoss`](#sonnix.modules.BaseLoss)                                       | Define the base class of the loss modules that have a reduction.                                                                               |
| [`BaseRelativeIndicator`](#sonnix.modules.BaseRelativeIndicator)             | Define the base class to implement a relative indicator function.                                                                              |
| [`BinaryFocalLoss`](#sonnix.modules.BinaryFocalLoss)                         | Implementation of the binary focal loss.                                                                                                       |
| [`BinaryFocalLossWithLogits`](#sonnix.modules.BinaryFocalLossWithLogits)     | Implementation of the binary focal loss.                                                                                                       |
| [`BinaryPoly1Loss`](#sonnix.modules.BinaryPoly1Loss)                         | Implementation of the binary Poly-1 loss for binary targets.                                                                                   |
| [`BinaryPoly1LossWithLogits`](#sonnix.modules.BinaryPoly1LossWithLogits)     | Implementation of the binary Poly-1 loss for binary targets.                                                                                   |
| [`ClassicalRelativeIndicator`](#sonnix.modules.ClassicalRelativeIndicator)   | Implement the classical relative indicator function.                                                                                           |
| [`GeneralRobustRegressionLoss`](#sonnix.modules.GeneralRobustRegressionLoss) | Implement the general robust regression loss a.k.a. Barron robust loss.                                                                        |
| [`GeometricMeanIndicator`](#sonnix.modules.GeometricMeanIndicator)           | Implement the geometric mean indicator function.                                                                                               |
| [`LogCoshLoss`](#sonnix.modules.LogCoshLoss)                                 | Implement a loss module that computes the logarithm of the hyperbolic cosine of the prediction error.                                          |
| [`MSLELoss`](#sonnix.modules.MSLELoss)                                       | Implement a loss module that computes the mean squared error (MSE) on the logarithmic transformed predictions and targets.                     |
| [`MaximumMeanIndicator`](#sonnix.modules.MaximumMeanIndicator)               | Implement the maximum mean change indicator function.                                                                                          |
| [`MinimumMeanIndicator`](#sonnix.modules.MinimumMeanIndicator)               | Implement the minimum mean change indicator function.                                                                                          |
| [`MomentMeanIndicator`](#sonnix.modules.MomentMeanIndicator)                 | Implement the moment mean change of order k indicator function.                                                                                |
| [`PoissonRegressionLoss`](#sonnix.modules.PoissonRegressionLoss)             | Implement a loss module that computes the Poisson regression loss.                                                                             |
| [`QuantileRegressionLoss`](#sonnix.modules.QuantileRegressionLoss)           | Implement a loss module that computes the quantile regression loss.                                                                            |
| [`RelativeLoss`](#sonnix.modules.RelativeLoss)                               | Implement a "generic" relative loss that takes as input a criterion.                                                                           |
| [`RelativeMSELoss`](#sonnix.modules.RelativeMSELoss)                         | Implement the relative MSE loss.                                                                                                               |
| [`RelativeSmoothL1Loss`](#sonnix.modules.RelativeSmoothL1Loss)               | Implement the relative smooth L1 loss.                                                                                                         |
| [`ReversedRelativeIndicator`](#sonnix.modules.ReversedRelativeIndicator)     | Implement the reversed relative indicator function.                                                                                            |
| [`TransformedLoss`](#sonnix.modules.TransformedLoss)                         | Implement a loss function where the predictions and targets are transformed before to be fed to the loss function.                             |

## ::: sonnix.modules.ArithmeticalMeanIndicator

## ::: sonnix.modules.AsinhMSELoss

## ::: sonnix.modules.AsinhSmoothL1Loss

## ::: sonnix.modules.BaseBinaryFocalLoss

## ::: sonnix.modules.BaseBinaryPoly1Loss

## ::: sonnix.modules.BaseLoss

## ::: sonnix.modules.BaseRelativeIndicator

## ::: sonnix.modules.BinaryFocalLoss

## ::: sonnix.modules.BinaryFocalLossWithLogits

## ::: sonnix.modules.BinaryPoly1Loss

## ::: sonnix.modules.BinaryPoly1LossWithLogits

## ::: sonnix.modules.ClassicalRelativeIndicator

## ::: sonnix.modules.GeneralRobustRegressionLoss

## ::: sonnix.modules.GeometricMeanIndicator

## ::: sonnix.modules.LogCoshLoss

## ::: sonnix.modules.MSLELoss

## ::: sonnix.modules.MaximumMeanIndicator

## ::: sonnix.modules.MinimumMeanIndicator

## ::: sonnix.modules.MomentMeanIndicator

## ::: sonnix.modules.PoissonRegressionLoss

## ::: sonnix.modules.QuantileRegressionLoss

## ::: sonnix.modules.RelativeLoss

## ::: sonnix.modules.RelativeMSELoss

## ::: sonnix.modules.RelativeSmoothL1Loss

## ::: sonnix.modules.ReversedRelativeIndicator

## ::: sonnix.modules.TransformedLoss
