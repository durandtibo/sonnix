r"""Contain activation modules using mathematical functions."""

from __future__ import annotations

__all__ = [
    "Asinh",
    "Exp",
    "Expm1",
    "Log",
    "Log1p",
    "Pow",
    "SafeExp",
    "SafeLog",
    "SafePow",
    "Sin",
    "Sinh",
    "Square",
]

import torch
from coola.validation import validate_finite, validate_ge, validate_isinstance
from torch import nn

from sonnix.functional import safe_exp, safe_log, safe_pow
from sonnix.validation import validate_positive


class Asinh(nn.Module):
    r"""Implement a ``torch.nn.Module`` to compute the inverse hyperbolic
    sine (arcsinh) of the elements.

    Shape:
        - Input: ``(*)``, where ``*`` means any number of dimensions.
        - Output: ``(*)``, same shape as the input.

    Example:
        ```pycon
        >>> import torch
        >>> from sonnix.modules import Asinh
        >>> m = Asinh()
        >>> m
        Asinh()
        >>> out = m(torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 2.0, 4.0]]))
        >>> out
        tensor([[-0.8814,  0.0000,  0.8814],
                [-1.4436,  1.4436,  2.0947]])

        ```
    """

    def forward(self, input: torch.Tensor) -> torch.Tensor:  # noqa: A002
        return input.asinh()


class Exp(nn.Module):
    r"""Implement a ``torch.nn.Module`` to compute the exponential of the
    input.

    This module is equivalent to  ``exp(input)``

    Shape:
        - Input: ``(*)``, where ``*`` means any number of dimensions.
        - Output: ``(*)``, same shape as the input.

    Example:
        ```pycon
        >>> import torch
        >>> from sonnix.modules import Exp
        >>> m = Exp()
        >>> m
        Exp()
        >>> out = m(torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 2.0, 3.0]]))
        >>> out
        tensor([[ 0.3679,  1.0000,  2.7183],
                [ 0.1353,  7.3891, 20.0855]])

        ```
    """

    def forward(self, input: torch.Tensor) -> torch.Tensor:  # noqa: A002
        return input.exp()


class Expm1(nn.Module):
    r"""Implement a ``torch.nn.Module`` to compute the exponential of the
    elements minus 1 of input.

    This module is equivalent to  ``exp(input) - 1``

    Shape:
        - Input: ``(*)``, where ``*`` means any number of dimensions.
        - Output: ``(*)``, same shape as the input.

    Example:
        ```pycon
        >>> import torch
        >>> from sonnix.modules import Expm1
        >>> m = Expm1()
        >>> m
        Expm1()
        >>> out = m(torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 2.0, 4.0]]))
        >>> out
        tensor([[-0.6321,  0.0000,  1.7183],
                [-0.8647,  6.3891, 53.5981]])

        ```
    """

    def forward(self, input: torch.Tensor) -> torch.Tensor:  # noqa: A002
        return input.expm1()


class Log(nn.Module):
    r"""Implement a ``torch.nn.Module`` to compute the natural logarithm
    of the input.

    This module is equivalent to  ``log(input)``

    Shape:
        - Input: ``(*)``, where ``*`` means any number of dimensions.
        - Output: ``(*)``, same shape as the input.

    Example:
        ```pycon
        >>> import torch
        >>> from sonnix.modules import Log
        >>> m = Log()
        >>> m
        Log()
        >>> out = m(torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
        >>> out
        tensor([[0.0000, 0.6931, 1.0986],
                [1.3863, 1.6094, 1.7918]])

        ```
    """

    def forward(self, input: torch.Tensor) -> torch.Tensor:  # noqa: A002
        return input.log()


class Log1p(nn.Module):
    r"""Implement a ``torch.nn.Module`` to compute the natural logarithm
    of ``(1 + input)``.

    This module is equivalent to  ``log(1 + input)``

    Shape:
        - Input: ``(*)``, where ``*`` means any number of dimensions.
        - Output: ``(*)``, same shape as the input.

    Example:
        ```pycon
        >>> import torch
        >>> from sonnix.modules import Log1p
        >>> m = Log1p()
        >>> m
        Log1p()
        >>> out = m(torch.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]))
        >>> out
        tensor([[0.0000, 0.6931, 1.0986],
                [1.3863, 1.6094, 1.7918]])

        ```
    """

    def forward(self, input: torch.Tensor) -> torch.Tensor:  # noqa: A002
        return input.log1p()


class Pow(nn.Module):
    r"""Implement a ``torch.nn.Module`` to raise the input to a power.

    This module is equivalent to  ``pow(input, exponent)``

    Note:
        The result is ``nan`` for a negative input and a non-integer
        exponent (for example ``Pow(exponent=0.5)`` on negative inputs),
        and no error is raised. Clamp or shift the inputs if they can be
        negative. The gradient of a power with an exponent in ``(0, 1)``
        is infinite at ``0``: use ``SafePow`` to get a gradient of ``0``
        there. ``Pow`` is the plain ``torch.pow``, and it is the only one
        that accepts a negative exponent.

    Args:
        exponent: The exponent. It must be a finite number.

    Shape:
        - Input: ``(*)``, where ``*`` means any number of dimensions.
        - Output: ``(*)``, same shape as the input.

    Raises:
        TypeError: If ``exponent`` is not a number.
        ValueError: If ``exponent`` is not finite.

    Example:
        ```pycon
        >>> import torch
        >>> from sonnix.modules import Pow
        >>> m = Pow(exponent=2)
        >>> m
        Pow(exponent=2.0)
        >>> out = m(torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
        >>> out
        tensor([[ 1.,  4.,  9.],
                [16., 25., 36.]])

        ```
    """

    def __init__(self, exponent: float) -> None:
        super().__init__()
        validate_finite(exponent, name="exponent")
        self.exponent = float(exponent)

    def forward(self, input: torch.Tensor) -> torch.Tensor:  # noqa: A002
        return input.pow(self.exponent)

    def extra_repr(self) -> str:
        return f"exponent={self.exponent}"


class SafeExp(nn.Module):
    r"""Implement a ``torch.nn.Module`` to compute the exponential of the
    elements.

    The values that are higher than the specified maximum value are
    set to this maximum value. The maximum value is lowered to
    ``log(finfo.max) - 1`` when ``exp`` of it is not finite in the
    dtype of the input, so the output has no Inf. A ``NaN`` input
    stays ``NaN``. See ``sonnix.functional.safe_exp``.

    Args:
        max_val: The maximum value before to compute the exponential.
            It is lowered if ``exp`` of it is not finite in the dtype
            of the input.
        preserve_gradient: If ``True``, the gradient is computed as
            if the input was not clamped (straight-through). By
            default, ``clamp`` gives a zero gradient outside the range.
            The value of the output does not depend on this option.

    Shape:
        - Input: ``(*)``, where ``*`` means any number of dimensions.
        - Output: ``(*)``, same shape as the input.

    Raises:
        TypeError: If ``max_val`` is not a number or ``preserve_gradient`` is not a
            ``bool``.
        ValueError: If ``max_val`` is not finite.

    Example:
        ```pycon
        >>> import torch
        >>> from sonnix.modules import SafeExp
        >>> m = SafeExp()
        >>> m
        SafeExp(max_val=20.0)
        >>> out = m(torch.tensor([[0.01, 0.1, 1.0], [10.0, 100.0, 1000.0]]))
        >>> out
        tensor([[1.0101e+00, 1.1052e+00, 2.7183e+00],
                [2.2026e+04, 4.8517e+08, 4.8517e+08]])

        ```
    """

    def __init__(self, max_val: float = 20.0, *, preserve_gradient: bool = False) -> None:
        super().__init__()
        validate_finite(max_val, name="max_val")
        validate_isinstance(preserve_gradient, bool, name="preserve_gradient")
        self.max_val = float(max_val)
        self.preserve_gradient = preserve_gradient

    def extra_repr(self) -> str:
        extra = ", preserve_gradient=True" if self.preserve_gradient else ""
        return f"max_val={self.max_val}{extra}"

    def forward(self, input: torch.Tensor) -> torch.Tensor:  # noqa: A002
        return safe_exp(input, max_val=self.max_val, preserve_gradient=self.preserve_gradient)


class SafeLog(nn.Module):
    r"""Implement a ``torch.nn.Module`` to compute the logarithm natural
    of the elements.

    The values that are lower than the specified minimum value are set
    to this minimum value. The minimum value is raised to the smallest
    positive normal number of the dtype of the input when it is lower,
    so the output has no NaN or Inf. A ``NaN`` input stays ``NaN``.
    See ``sonnix.functional.safe_log``.

    Note:
        With ``preserve_gradient=True``, the gradient is the one of the
        logarithm at the clamped value, ``1 / min_val`` for an input
        lower than ``min_val`` (``1e8`` at ``0`` with the default
        ``min_val``), not a finite or zero gradient.

    Args:
        min_val: The minimum value before to compute the logarithm natural.
            It is raised to the smallest normal number of the dtype of
            the input.
        preserve_gradient: If ``True``, the gradient is computed as
            if the input was not clamped (straight-through). By
            default, ``clamp`` gives a zero gradient outside the range.
            The value of the output does not depend on this option.

    Shape:
        - Input: ``(*)``, where ``*`` means any number of dimensions.
        - Output: ``(*)``, same shape as the input.

    Raises:
        TypeError: if ``min_val`` is not a number or ``preserve_gradient`` is not a
            ``bool``.
        ValueError: if ``min_val`` is not finite and strictly positive.

    Example:
        ```pycon
        >>> import torch
        >>> from sonnix.modules import SafeLog
        >>> m = SafeLog()
        >>> m
        SafeLog(min_val=1e-08)
        >>> out = m(torch.tensor([[1e-4, 1e-5, 1e-6], [1e-8, 1e-9, 1e-10]]))
        >>> out
        tensor([[ -9.2103, -11.5129, -13.8155],
                [-18.4207, -18.4207, -18.4207]])

        ```
    """

    def __init__(self, min_val: float = 1e-8, *, preserve_gradient: bool = False) -> None:
        super().__init__()
        validate_positive(min_val, name="min_val")
        validate_isinstance(preserve_gradient, bool, name="preserve_gradient")
        self.min_val = float(min_val)
        self.preserve_gradient = preserve_gradient

    def extra_repr(self) -> str:
        extra = ", preserve_gradient=True" if self.preserve_gradient else ""
        return f"min_val={self.min_val}{extra}"

    def forward(self, input: torch.Tensor) -> torch.Tensor:  # noqa: A002
        return safe_log(input, min_val=self.min_val, preserve_gradient=self.preserve_gradient)


class SafePow(nn.Module):
    r"""Implement a ``torch.nn.Module`` to raise the input to a power with
    a finite gradient at zero.

    This module is ``torch.pow`` with a gradient of ``0`` at ``0`` and
    nothing else: the result is ``nan`` for a negative input and a
    non-integer exponent, like ``torch.pow``. For an exponent in
    ``(0, 1)``, the derivative of the power is infinite at zero, and
    ``Pow`` gives a ``NaN`` gradient there. See
    ``sonnix.functional.safe_pow``.

    Args:
        exponent: The exponent, which must be finite and non-negative.
            ``Pow`` accepts a negative exponent.

    Shape:
        - Input: ``(*)``, where ``*`` means any number of dimensions.
        - Output: ``(*)``, same shape as the input.

    Raises:
        TypeError: If ``exponent`` is not a number.
        ValueError: If ``exponent`` is negative or not finite.

    Example:
        ```pycon
        >>> import torch
        >>> from sonnix.modules import SafePow
        >>> m = SafePow(exponent=0.5)
        >>> m
        SafePow(exponent=0.5)
        >>> x = torch.tensor([0.0, 4.0, 9.0], requires_grad=True)
        >>> out = m(x)
        >>> out
        tensor([0., 2., 3.], grad_fn=<WhereBackward0>)
        >>> out.sum().backward()
        >>> x.grad
        tensor([0.0000, 0.2500, 0.1667])

        ```
    """

    def __init__(self, exponent: float) -> None:
        super().__init__()
        validate_finite(exponent, name="exponent")
        validate_ge(exponent, 0, name="exponent")
        self.exponent = float(exponent)

    def extra_repr(self) -> str:
        return f"exponent={self.exponent}"

    def forward(self, input: torch.Tensor) -> torch.Tensor:  # noqa: A002
        return safe_pow(input, exponent=self.exponent)


class Sin(nn.Module):
    r"""Implement the sine activation layer.

    Shape:
        - Input: ``(*)``, where ``*`` means any number of dimensions.
        - Output: ``(*)``, same shape as the input.

    Example:
        ```pycon
        >>> import torch
        >>> from sonnix.modules import Sin
        >>> m = Sin()
        >>> m
        Sin()
        >>> out = m(torch.arange(8, dtype=torch.float).view(2, 4))
        >>> out
        tensor([[ 0.0000,  0.8415,  0.9093,  0.1411],
                [-0.7568, -0.9589, -0.2794,  0.6570]])

        ```
    """

    def forward(self, input: torch.Tensor) -> torch.Tensor:  # noqa: A002
        return input.sin()


class Sinh(nn.Module):
    r"""Implement a ``torch.nn.Module`` to compute the hyperbolic sine
    (sinh) of the elements.

    Shape:
        - Input: ``(*)``, where ``*`` means any number of dimensions.
        - Output: ``(*)``, same shape as the input.

    Example:
        ```pycon
        >>> import torch
        >>> from sonnix.modules import Sinh
        >>> m = Sinh()
        >>> m
        Sinh()
        >>> out = m(torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 2.0, 4.0]]))
        >>> out
        tensor([[-1.1752,  0.0000,  1.1752],
                [-3.6269,  3.6269, 27.2899]])

        ```
    """

    def forward(self, input: torch.Tensor) -> torch.Tensor:  # noqa: A002
        return input.sinh()


class Square(nn.Module):
    r"""Implement the square activation.

    It is equivalent to ``Pow(exponent=2)`` with a name that is easier
    to use in a configuration.

    Shape:
        - Input: ``(*)``, where ``*`` means any number of dimensions.
        - Output: ``(*)``, same shape as the input.

    Example:
        ```pycon
        >>> import torch
        >>> from sonnix.modules import Square
        >>> m = Square()
        >>> m
        Square()
        >>> out = m(torch.arange(8, dtype=torch.float).view(2, 4))
        >>> out
        tensor([[ 0.,  1.,  4.,  9.],
                [16., 25., 36., 49.]])

        ```
    """

    def forward(self, input: torch.Tensor) -> torch.Tensor:  # noqa: A002
        return input.pow(2)
