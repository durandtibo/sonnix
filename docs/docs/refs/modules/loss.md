# Losses

Loss modules and the shared base classes. See also the [`sonnix.modules`](index.md) overview.

| Name                                                                         | Description                                                                                                                                    |
| ---------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| [`BaseBinaryFocalLoss`](#sonnix.modules.BaseBinaryFocalLoss)                 | Define the base class of the binary focal losses.                                                                                              |
| [`BaseBinaryPoly1Loss`](#sonnix.modules.BaseBinaryPoly1Loss)                 | Define the base class of the binary Poly-1 losses.                                                                                             |
| [`BaseLoss`](#sonnix.modules.BaseLoss)                                       | Define the base class of the loss modules that have a reduction.                                                                               |
| [`BinaryFocalLoss`](#sonnix.modules.BinaryFocalLoss)                         | Implementation of the binary focal loss.                                                                                                       |
| [`BinaryFocalLossWithLogits`](#sonnix.modules.BinaryFocalLossWithLogits)     | Implementation of the binary focal loss.                                                                                                       |
| [`BinaryPoly1Loss`](#sonnix.modules.BinaryPoly1Loss)                         | Implementation of the binary Poly-1 loss for binary targets.                                                                                   |
| [`BinaryPoly1LossWithLogits`](#sonnix.modules.BinaryPoly1LossWithLogits)     | Implementation of the binary Poly-1 loss for binary targets.                                                                                   |
| [`GeneralRobustRegressionLoss`](#sonnix.modules.GeneralRobustRegressionLoss) | Implement the general robust regression loss a.k.a. Barron robust loss.                                                                        |
| [`LogCoshLoss`](#sonnix.modules.LogCoshLoss)                                 | Implement a loss module that computes the logarithm of the hyperbolic cosine of the prediction error.                                          |
| [`MulticlassFlatten`](#sonnix.modules.MulticlassFlatten) | Implement a wrapper to flat the multiclass inputs of a `torch.nn.Module`. |
| [`PoissonRegressionLoss`](#sonnix.modules.PoissonRegressionLoss)             | Implement a loss module that computes the Poisson regression loss.                                                                             |
| [`QuantileRegressionLoss`](#sonnix.modules.QuantileRegressionLoss)           | Implement a loss module that computes the quantile regression loss.                                                                            |
| [`RelativeLoss`](#sonnix.modules.RelativeLoss)                               | Implement a "generic" relative loss that takes as input a criterion.                                                                           |
| [`TransformedLoss`](#sonnix.modules.TransformedLoss)                         | Implement a loss function where the predictions and targets are transformed before to be fed to the loss function.                             |

## ::: sonnix.modules.BaseBinaryFocalLoss

## ::: sonnix.modules.BaseBinaryPoly1Loss

## ::: sonnix.modules.BaseLoss

## ::: sonnix.modules.BinaryFocalLoss

## ::: sonnix.modules.BinaryFocalLossWithLogits

## ::: sonnix.modules.BinaryPoly1Loss

## ::: sonnix.modules.BinaryPoly1LossWithLogits

## ::: sonnix.modules.GeneralRobustRegressionLoss

## ::: sonnix.modules.LogCoshLoss

## ::: sonnix.modules.MulticlassFlatten

## ::: sonnix.modules.PoissonRegressionLoss

## ::: sonnix.modules.QuantileRegressionLoss

## ::: sonnix.modules.RelativeLoss

## ::: sonnix.modules.TransformedLoss
