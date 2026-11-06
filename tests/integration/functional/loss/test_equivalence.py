from __future__ import annotations

import pytest
import torch
from torch.nn.functional import l1_loss, mse_loss, poisson_nll_loss

from sonnix.functional import (
    binary_focal_loss,
    binary_focal_loss_with_logits,
    binary_poly1_loss,
    binary_poly1_loss_with_logits,
    general_robust_regression_loss,
    poisson_regression_loss,
    quantile_regression_loss,
)

REDUCTIONS = ("none", "mean", "sum", "batchmean")


def _logits_and_soft_targets() -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(42)
    logits = 3 * torch.randn(4, 5, generator=generator, dtype=torch.float64)
    return logits, torch.rand(4, 5, generator=generator, dtype=torch.float64)


def _normal_pair() -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(42)
    return torch.randn(4, 5, generator=generator), torch.randn(4, 5, generator=generator)


#################################################################
#     Tests for with_logits vs probabilities (soft targets)     #
#################################################################


@pytest.mark.parametrize("reduction", REDUCTIONS)
@pytest.mark.parametrize("alpha", [None, 0.25])
@pytest.mark.parametrize("gamma", [0.0, 2.0])
def test_binary_focal_loss_with_logits_equals_probabilities_soft_targets(
    reduction: str, alpha: float | None, gamma: float
) -> None:
    logits, target = _logits_and_soft_targets()
    assert binary_focal_loss_with_logits(
        logits, target, alpha=alpha, gamma=gamma, reduction=reduction
    ).allclose(
        binary_focal_loss(logits.sigmoid(), target, alpha=alpha, gamma=gamma, reduction=reduction),
        atol=1e-10,
    )


@pytest.mark.parametrize("reduction", REDUCTIONS)
@pytest.mark.parametrize("alpha", [0.0, 1.0, 2.0])
def test_binary_poly1_loss_with_logits_equals_probabilities_soft_targets(
    reduction: str, alpha: float
) -> None:
    logits, target = _logits_and_soft_targets()
    assert binary_poly1_loss_with_logits(logits, target, alpha=alpha, reduction=reduction).allclose(
        binary_poly1_loss(logits.sigmoid(), target, alpha=alpha, reduction=reduction),
        atol=1e-10,
    )


########################################
#     Tests for the PyTorch losses     #
########################################


@pytest.mark.parametrize("reduction", REDUCTIONS[:3])
def test_quantile_regression_loss_median_is_half_of_l1_loss(reduction: str) -> None:
    prediction, target = _normal_pair()
    assert quantile_regression_loss(prediction, target, q=0.5, reduction=reduction).allclose(
        0.5 * l1_loss(prediction, target, reduction=reduction)
    )


@pytest.mark.parametrize("reduction", REDUCTIONS[:3])
def test_general_robust_regression_loss_alpha_2_equals_mse_loss(reduction: str) -> None:
    prediction, target = _normal_pair()
    assert general_robust_regression_loss(
        prediction, target, alpha=2.0, reduction=reduction
    ).allclose(mse_loss(prediction, target, reduction=reduction))


@pytest.mark.parametrize("reduction", REDUCTIONS[:3])
def test_poisson_regression_loss_equals_poisson_nll_loss(reduction: str) -> None:
    generator = torch.Generator().manual_seed(42)
    prediction = torch.rand(4, 5, generator=generator) + 0.1
    target = torch.randint(0, 5, (4, 5), generator=generator).float()
    # ``poisson_nll_loss`` adds ``eps`` and the loss clamps to ``eps``: the two are
    # equal up to ``eps`` when the predictions are much larger than ``eps``.
    assert poisson_regression_loss(prediction, target, reduction=reduction).allclose(
        poisson_nll_loss(prediction, target, log_input=False, reduction=reduction),
        atol=1e-6,
    )
