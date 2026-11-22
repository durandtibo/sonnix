r"""Contain helper functions shared by the binary focal and Poly-1
losses."""

from __future__ import annotations

__all__ = ["class_weight", "probability_of_target"]

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


def probability_of_target(probability: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    r"""Compute the probability assigned to the target class (``p_t``).

    ``exp(-cross_entropy) = p_t`` only holds for binary targets, so
    ``p_t`` is computed from the probabilities to support soft targets
    (label smoothing, mixup).

    Args:
        probability: The predicted probabilities of the positive class.
        target: The targets, with the same shape as ``probability``.

    Returns:
        ``probability * target + (1 - probability) * (1 - target)``.

    Example:
        ```pycon
        >>> import torch
        >>> from sonnix.functional.loss._binary import probability_of_target
        >>> probability_of_target(torch.tensor([0.8, 0.8]), torch.tensor([1.0, 0.0]))
        tensor([0.8000, 0.2000])

        ```
    """
    return probability * target + (1 - probability) * (1 - target)


def class_weight(alpha: float, target: torch.Tensor) -> torch.Tensor:
    r"""Compute the weight of each element given its target (``alpha_t``).

    Args:
        alpha: The weight of the positive class. The negative class has
            the weight ``1 - alpha``.
        target: The targets, which can be soft targets.

    Returns:
        ``alpha * target + (1 - alpha) * (1 - target)``.

    Example:
        ```pycon
        >>> import torch
        >>> from sonnix.functional.loss._binary import class_weight
        >>> class_weight(0.25, torch.tensor([1.0, 0.0]))
        tensor([0.2500, 0.7500])

        ```
    """
    return alpha * target + (1 - alpha) * (1 - target)
