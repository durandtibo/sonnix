r"""Contain activation modules using mathematical functions."""

from __future__ import annotations

from coola.validation import validate_gt

__all__ = ["Asinh", "Exp", "Expm1", "Log", "Log1p", "SafeExp", "SafeLog", "Sin", "Sinh"]

import torch
from torch import nn

from sonnix.functional import safe_exp, safe_log


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

    Shape:
        - Input: ``(*)``, where ``*`` means any number of dimensions.
        - Output: ``(*)``, same shape as the input.

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
        self._exponent = float(exponent)

    def forward(self, input: torch.Tensor) -> torch.Tensor:  # noqa: A002
        return input.pow(self._exponent)

    def extra_repr(self) -> str:
        return f"exponent={self._exponent}"


class SafeExp(nn.Module):
    r"""Implement a ``torch.nn.Module`` to compute the exponential of the
    elements.

    The values that are higher than the specified maximum value are
    set to this maximum value. Using a not too large positive value
    leads to an output tensor without Inf.

    Args:
        max_val: The maximum value before to compute the exponential.
        preserve_gradient: If ``True``, the gradient is computed as
            if the input was not clamped (straight-through). By
            default, ``clamp`` gives a zero gradient outside the range.

    Shape:
        - Input: ``(*)``, where ``*`` means any number of dimensions.
        - Output: ``(*)``, same shape as the input.

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

    def __init__(self, max_val: float = 20.0, preserve_gradient: bool = False) -> None:
        super().__init__()
        self._max = float(max_val)
        self._preserve_gradient = preserve_gradient

    def extra_repr(self) -> str:
        extra = ", preserve_gradient=True" if self._preserve_gradient else ""
        return f"max_val={self._max}{extra}"

    def forward(self, input: torch.Tensor) -> torch.Tensor:  # noqa: A002
        return safe_exp(input, self._max, preserve_gradient=self._preserve_gradient)


class SafeLog(nn.Module):
    r"""Implement a ``torch.nn.Module`` to compute the logarithm natural
    of the elements.

    The values that are lower than the specified minimum value are set
    to this minimum value. Using a small positive value leads to an
    output tensor without NaN or Inf.

    Args:
        min_val: The minimum value before to compute the logarithm natural.
        preserve_gradient: If ``True``, the gradient is computed as
            if the input was not clamped (straight-through). By
            default, ``clamp`` gives a zero gradient outside the range.

    Shape:
        - Input: ``(*)``, where ``*`` means any number of dimensions.
        - Output: ``(*)``, same shape as the input.

    Raises:
        ValueError: if ``min_val`` is not strictly positive.

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

    def __init__(self, min_val: float = 1e-8, preserve_gradient: bool = False) -> None:
        super().__init__()
        validate_gt(min_val, 0, name="min_val")
        self._min = float(min_val)
        self._preserve_gradient = preserve_gradient

    def extra_repr(self) -> str:
        extra = ", preserve_gradient=True" if self._preserve_gradient else ""
        return f"min_val={self._min}{extra}"

    def forward(self, input: torch.Tensor) -> torch.Tensor:  # noqa: A002
        return safe_log(input, self._min, preserve_gradient=self._preserve_gradient)


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
        return input * input
