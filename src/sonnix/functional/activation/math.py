r"""Contain functional implementation of some activation layers."""

from __future__ import annotations

__all__ = ["safe_exp", "safe_log"]

import torch
from coola.validation import validate_finite, validate_isinstance

from sonnix.utils import resolve_eps, resolve_max_exponent
from sonnix.validation import validate_positive


def _clamp(
    input: torch.Tensor,  # noqa: A002
    *,
    min_val: float | None = None,
    max_val: float | None = None,
    preserve_gradient: bool,
) -> torch.Tensor:
    r"""Clamp the elements, and optionally keep the gradient of the
    input.

    Args:
        input: The input tensor.
        min_val: The minimum value, or ``None`` for no lower bound.
        max_val: The maximum value, or ``None`` for no upper bound.
        preserve_gradient: If ``True``, the value is the clamped value
            and the gradient is the one of the identity
            (straight-through estimator).

    Returns:
        The clamped tensor. Its value does not depend on
            ``preserve_gradient``.

    Example:
        ```pycon
        >>> import torch
        >>> from sonnix.functional.activation.math import _clamp
        >>> _clamp(torch.tensor([-1e30, 0.5, torch.inf]), min_val=0.0, max_val=1.0,
        ...        preserve_gradient=True)
        tensor([0.0000, 0.5000, 1.0000])

        ```
    """
    clamped = input.clamp(min=min_val, max=max_val)
    if not preserve_gradient:
        return clamped
    # The value of ``clamped`` and the gradient of ``input`` (identity). ``input - input.detach()``
    # is 0 for a finite input and ``nan`` for an infinite one, which is replaced by 0. It avoids
    # ``input + (clamped - input).detach()``, which loses ``clamped`` by cancellation when
    # ``|input|`` is large.
    return clamped.detach() + torch.where(input.isfinite(), input - input.detach(), 0.0)


def safe_exp(
    input: torch.Tensor,  # noqa: A002
    *,
    max_val: float = 20.0,
    preserve_gradient: bool = False,
) -> torch.Tensor:
    r"""Compute safely the exponential of the elements.

    The values that are higher than the specified maximum value are
    set to this maximum value. The maximum value is lowered to
    ``log(finfo.max) - 1`` when ``exp`` of it is not finite in the
    dtype of ``input`` (``10.09`` in ``float16``, ``87.7`` in
    ``float32``), so the output has no Inf. A ``NaN`` input stays
    ``NaN``.

    Note:
        ``clamp`` has a zero gradient for the values outside of the
        range. Set ``preserve_gradient=True`` to keep the gradient of
        the unclamped input (straight-through estimator).

    Args:
        input: The input tensor.
        max_val: The maximum value. It is lowered if ``exp`` of it is
            not finite in the dtype of ``input``
            (see ``sonnix.utils.resolve_max_exponent``).
        preserve_gradient: If ``True``, the gradient is computed as
            if the input was not clamped (straight-through). The
            value does not depend on this option.

    Returns:
        A tensor with the exponential of the elements.

    Raises:
        TypeError: If ``max_val`` is not a number or ``preserve_gradient`` is not a
            ``bool``.
        ValueError: If ``max_val`` is not finite.

    Example:
        ```pycon
        >>> import torch
        >>> from sonnix.functional import safe_exp
        >>> output = safe_exp(torch.tensor([1.0, 10.0, 100.0, 1000.0]))
        >>> output
        tensor([2.7183e+00, 2.2026e+04, 4.8517e+08, 4.8517e+08])

        ```
    """
    validate_finite(max_val, name="max_val")
    validate_isinstance(preserve_gradient, bool, name="preserve_gradient")
    clamped = _clamp(
        input, max_val=resolve_max_exponent(max_val, input), preserve_gradient=preserve_gradient
    )
    return clamped.exp()


def safe_log(
    input: torch.Tensor,  # noqa: A002
    *,
    min_val: float = 1e-8,
    preserve_gradient: bool = False,
) -> torch.Tensor:
    r"""Compute safely the natural logarithm of the elements.

    The values that are lower than the specified minimum value are set
    to this minimum value. The minimum value is raised to the smallest
    positive normal number of the dtype of ``input`` when it is lower
    (``6.1e-05`` in ``float16``, as ``1e-8`` is ``0`` there), so the
    output has no NaN or Inf. A ``NaN`` input stays ``NaN``.

    Note:
        ``clamp`` has a zero gradient for the values outside of the
        range. Set ``preserve_gradient=True`` to keep the gradient of
        the unclamped input (straight-through estimator).

    Note:
        With ``preserve_gradient=True``, the gradient is the one of the
        logarithm at the clamped value, ``1 / min_val`` for an input
        lower than ``min_val`` (``1e8`` at ``0`` with the default
        ``min_val``), not a finite or zero gradient.

    Args:
        input: The input tensor.
        min_val: The minimum value. It is raised to the smallest normal
            number of the dtype of ``input``
            (see ``sonnix.utils.resolve_eps``).
        preserve_gradient: If ``True``, the gradient is computed as
            if the input was not clamped (straight-through). The
            value does not depend on this option.

    Returns:
        A tensor with the natural logarithm of the elements.

    Raises:
        TypeError: if ``min_val`` is not a number or ``preserve_gradient`` is not a
            ``bool``.
        ValueError: if ``min_val`` is not finite and strictly positive.

    Example:
        ```pycon
        >>> import torch
        >>> from sonnix.functional import safe_log
        >>> safe_log(torch.tensor([1e-4, 1e-5, 1e-6, 1e-8, 1e-9, 1e-10]))
        tensor([ -9.2103, -11.5129, -13.8155, -18.4207, -18.4207, -18.4207])

        ```
    """
    validate_positive(min_val, name="min_val")
    validate_isinstance(preserve_gradient, bool, name="preserve_gradient")
    clamped = _clamp(
        input, min_val=resolve_eps(min_val, input), preserve_gradient=preserve_gradient
    )
    return clamped.log()
