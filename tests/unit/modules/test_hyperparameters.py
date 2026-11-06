from __future__ import annotations

import pytest
from torch import nn

from sonnix.modules import (
    BinaryFocalLoss,
    BinaryFocalLossWithLogits,
    BinaryPoly1Loss,
    BinaryPoly1LossWithLogits,
    Clamp,
    ConcatFusion,
    DynamicTanh,
    ExU,
    GeneralRobustRegressionLoss,
    NLinear,
    PoissonRegressionLoss,
    Pow,
    QuantileRegressionLoss,
    RelativeLoss,
    ReLUn,
    SafeExp,
    SafeLog,
    ScaleAndShift,
    Squeeze,
    ToBinaryLabel,
)

################################################
#     Tests for the public hyperparameters     #
################################################


@pytest.mark.parametrize(
    ("module", "name", "value"),
    [
        (ReLUn(max_val=3.0), "max_val", 3.0),
        (Clamp(min_val=-2.0, max_val=2.0), "min_val", -2.0),
        (Clamp(min_val=-2.0, max_val=2.0), "max_val", 2.0),
        (Clamp(), "min_val", -1.0),
        (Pow(exponent=3), "exponent", 3.0),
        (SafeExp(max_val=5.0, preserve_gradient=True), "max_val", 5.0),
        (SafeExp(max_val=5.0, preserve_gradient=True), "preserve_gradient", True),
        (SafeLog(min_val=0.1, preserve_gradient=True), "min_val", 0.1),
        (SafeLog(min_val=0.1, preserve_gradient=True), "preserve_gradient", True),
        (Squeeze(dim=1), "dim", 1),
        (ConcatFusion(dim=2), "dim", 2),
        (ScaleAndShift(normalized_shape=4), "normalized_shape", (4,)),
        (DynamicTanh(normalized_shape=4, alpha_init_value=0.3), "alpha_init_value", 0.3),
        (ToBinaryLabel(threshold=0.5), "threshold", 0.5),
        (NLinear(n=3, in_features=4, out_features=5), "n", 3),
        (NLinear(n=3, in_features=4, out_features=5), "in_features", 4),
        (NLinear(n=3, in_features=4, out_features=5), "out_features", 5),
        (ExU(in_features=4, out_features=5), "in_features", 4),
        (QuantileRegressionLoss(q=0.3), "q", 0.3),
        (PoissonRegressionLoss(eps=1e-3), "eps", 1e-3),
        (RelativeLoss(nn.MSELoss(), eps=1e-3, detach_indicator=True), "eps", 1e-3),
        (RelativeLoss(nn.MSELoss(), eps=1e-3, detach_indicator=True), "detach_indicator", True),
        (BinaryFocalLoss(alpha=0.25, gamma=3.0), "alpha", 0.25),
        (BinaryFocalLoss(alpha=0.25, gamma=3.0), "gamma", 3.0),
        (BinaryFocalLoss(alpha=None), "alpha", None),
        (BinaryFocalLossWithLogits(alpha=0.25, gamma=3.0), "gamma", 3.0),
        (BinaryPoly1Loss(alpha=0.5), "alpha", 0.5),
        (BinaryPoly1LossWithLogits(alpha=0.5), "alpha", 0.5),
        (GeneralRobustRegressionLoss(alpha=1.0, scale=2.0, max_val=5.0), "alpha", 1.0),
        (GeneralRobustRegressionLoss(alpha=1.0, scale=2.0, max_val=5.0), "scale", 2.0),
        (GeneralRobustRegressionLoss(alpha=1.0, scale=2.0, max_val=5.0), "max_val", 5.0),
    ],
)
def test_hyperparameter_is_public(module: nn.Module, name: str, value: object) -> None:
    assert getattr(module, name) == value
