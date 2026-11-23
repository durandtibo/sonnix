# Numerical stability

The guarantees below hold for `float16`, `bfloat16`, `float32` and `float64`
(property P6 of the [conventions](../conventions.md)): wherever the `float32`
result and its gradient can be represented in the dtype, the output and the
gradient are finite. They are tested on zeros, tiny values and large values for
every module.

- **Guards follow the dtype.** `1e-8` is `0` in `float16` (the smallest normal
  number is `6.1e-05`), so a guard of that size does not protect a division or a
  logarithm. `safe_log`, `SafeLog`, `poisson_regression_loss`, `relative_loss`,
  `absolute_relative_error` and `symmetric_absolute_relative_error` raise their
  `min_val` or `eps` to the smallest normal number of the dtype
  (`sonnix.utils.resolve_eps`). `safe_log(0)` is `-9.7` in `float16`, not `-inf`.
  The attributes and the `repr` of a module keep the value that was passed.
- **Exponential.** `safe_exp` clamps the input before `exp`, and lowers
  `max_val` to `log(finfo.max) - 1` when `exp` of it is not finite in the dtype
  (`sonnix.utils.resolve_max_exponent`: `10.09` in `float16`, `87.7` in
  `float32`), so the output is finite in every dtype for a finite input. It does
  not remove `nan`: a `nan` input stays `nan`.
- **Logarithm.** `safe_log` clamps the input to `min_val`; with
  `preserve_gradient=True` the gradient of the unclamped function is used, so
  at `x = 0` the gradient is `1 / min_val`, which is large. The option changes
  the gradient only: the value is the one of the clamped input for every input,
  including `inf` (`safe_exp(inf, preserve_gradient=True)` is finite). The
  gradient is `0` at an infinite input.
- **Powers.** `safe_pow` is `torch.pow` with a gradient of `0` at `0`, instead
  of `nan`, and nothing else: a negative input with a fractional exponent is
  `nan`, like `torch.pow`. Use it instead of `x ** p` or `sqrt` when the input
  can be `0` and a gradient is needed, and give it the absolute value if the
  input can be negative. `SafePow` is the module.
- **Squares.** `general_robust_regression_loss` squares the difference in at
  least `float32` and casts the loss back, because `(p - t) / scale` squared
  overflows in `float16` for a difference above `255 * scale`. `Gaussian` computes
  `exp(-0.5 * (x / alpha) ** 2)`, so a large `x` gives a gradient of `0` for
  `alpha` instead of `nan`.
- **Poisson loss.** `poisson_regression_loss` with `log_input=False` expects
  positive predictions (the others are clamped to `eps`). If the output of the
  model has no positivity constraint, use `log_input=True`: the prediction is the
  logarithm of the rate, the loss is `exp(prediction) - target * prediction`
  and `eps` is not used.
- **Learnable `alpha`.** `Gaussian` and `Laplacian` divide by `alpha`. A
  learnable `alpha` that reaches `0` makes the training fail and is hard to
  debug: use a small learning rate for it or keep it fixed.
- **Reductions.** `mean` divides by the number of elements, `batchmean` by the
  batch size. A loss summed over a large tensor can overflow in `float16`.

## Not guaranteed

A result that the dtype cannot represent is not finite, and no guard changes
that:

- the square of `1000` in `float16` (`Square`, `SquaredReLU`, an `MSELoss`
  inside a `RelativeLoss`), or a large exponent in `Pow` and `ExU`;
- `exp(100)` in `float32` outside `safe_exp`;
- the probability variants of the binary losses (`binary_focal_loss`,
  `binary_poly1_loss`) with a prediction of exactly `0` or `1` in `float16`:
  the `binary_cross_entropy` of `torch` divides by `0` and its gradient is
  `nan`. Use the `*_with_logits` variants;
- a learnable `alpha` that crosses `0`;
- a module fed a dtype other than its own: it follows the promotion rules of
  `torch`, and `NLinear` raises like `torch.nn.Linear`.

For a loss on `float16`, compare with a `float32` run if a value looks wrong.
