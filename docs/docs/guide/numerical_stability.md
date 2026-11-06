# Numerical stability

- **Large values.** The relative indicators and `exp`/`pow` based functions can overflow in half
  precision. Check the docstring of the function you use, and compare with a `float32` run.
- **Exponential and logarithm.** `safe_exp` clamps the input before `exp`, so the output has no
  `inf`; it does not remove `nan`. `safe_log` clamps the input to `min_val`; with
  `preserve_gradient=True` the gradient of the unclamped function is used, so at `x = 0` the
  gradient is `1 / min_val`, which is large.
- **Powers.** `safe_pow` has a zero gradient at `0` instead of `nan`. Use it instead of
  `x ** p` or `sqrt` when the input can be `0` and a gradient is needed.
- **Poisson loss.** `poisson_regression_loss` expects positive predictions. Apply a positive
  activation (for example `SafeExp`, `Square` or a softplus) to the output of the model.
- **Learnable `alpha`.** `Gaussian` and `Laplacian` divide by `alpha`. A learnable `alpha` that
  reaches `0` makes the training fail and is hard to debug: use a small learning rate for it or
  keep it fixed.
- **Reductions.** `mean` divides by the number of elements, `batchmean` by the batch size. A
  loss summed over a large tensor can overflow in `float16`.
