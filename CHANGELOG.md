# Changelog

All notable changes to this project are documented in this file. The package is still in alpha
(`0.0.1a0`), so breaking changes can happen in any release.

## Unreleased

### Breaking changes

- The hyperparameters of the modules are public attributes named like the constructor arguments
  (`ReLUn.max_val`, `Clamp.min_val`/`max_val`, `Snake.frequency`, `Pow.exponent`,
  `SafeExp.max_val`/`preserve_gradient`, `SafeLog.min_val`/`preserve_gradient`, `Squeeze.dim`,
  `View.shape`, `ConcatFusion.dim`, `ScaleAndShift.normalized_shape`,
  `DynamicTanh.alpha_init_value`, `MomentMeanIndicator.k`, and the `alpha`, `gamma`, `beta`, `q`,
  `eps`, `scale`, `max_val`, `detach_indicator` attributes of the losses). The previous private
  attributes (`_max`, `_alpha`, ...) are removed.

### Fixed

- `RelativeSmoothL1Loss` stores `beta` and shows it in its representation.

### Changed

- `BaseLoss.extra_repr` returns `reduction=...` and the losses extend it, so the representation of
  every loss starts with `reduction=...` followed by its hyperparameters.

### Renamed

- The `check_*` validation functions move to `sonnix.validation` and are renamed `validate_*`:
  `sonnix.functional.check_loss_reduction_strategy` is `validate_loss_reduction_strategy`,
  `check_moment_order` (`sonnix.functional.loss.relative`) is `validate_moment_order`, and
  `check_abs_range`/`check_frequency` (`sonnix.modules.numerical._utils`) are `validate_abs_range`
  and `validate_frequency`. `VALID_REDUCTIONS` moves to `sonnix.validation.loss`.

- The `input_size` and `output_size` properties of the numerical encoders are now `n_features` and
  `feature_size`, like their representation.
- The `prediction` and `target` arguments (and attributes) of `TransformedLoss` are now
  `prediction_transform` and `target_transform`.

### Changed (keyword-only options)

- The options are keyword-only: `preserve_gradient` (`safe_exp`, `safe_log`, `SafeExp`, `SafeLog`),
  `bias` (`NLinear`, `ExU`), `device` and `dtype` (`NLinear`, `ExU`, `ScaleAndShift`, the dynamic
  norms, the alpha activations), `learnable` (the alpha activations, the numerical encoders and their
  `create_*` class methods). `SafeExp(20.0, True)` is now `SafeExp(20.0, preserve_gradient=True)`.

### Snake

- `Snake` is a `BaseAlphaActivation` with a learnable `alpha` (per channel with `num_parameters`),
  as in the paper, and supports `device` and `dtype`. The `frequency` argument is replaced by
  `init`, and `Snake(frequency=2.0)` is now `Snake(init=2.0, learnable=False)`. The forward is
  `x + sin(alpha * x) ** 2 / alpha`, which has the same values with fewer operations.

### Wrapper modules

- Add `ToDtype(dtype)`. `ToFloat` and `ToLong` are now `ToDtype` subclasses (`float32` and `int64`).
- `ToCategoricalLabel` has a `dim` argument (default `-1`).
- `Square` is a `Pow(exponent=2)` subclass (it is no longer a separate implementation) and is listed
  in `sonnix.modules.activation.math.__all__` with `Pow`. Its output on integer tensors is now
  floating point, like `Pow`.

### Normalization layers

- `BaseDynamicNorm` extends `ScaleAndShift` and is abstract (`forward` is an abstract method).
  The subclasses call `ScaleAndShift.forward` for the affine part. `alpha` is registered after
  `weight` and `bias`, so the order of the keys of the state dict changes (the names do not).
- `ScaleAndShift`, `DynamicTanh` and `DynamicAsinh` accept `bias=False` (no shift, `bias` is
  `None`). `bias` is part of their representation.

### Fusion modules

- Add `BaseFusion`: it checks that there is at least one input (the check was repeated in three
  modules) and calls the abstract method `fuse`. `MultiplicationFusion`, `SumFusion`,
  `AverageFusion` and `ConcatFusion` extend it.
- `SumFusion` has no `normalized` argument and `AverageFusion` no longer extends `SumFusion`:
  `SumFusion(normalized=True)` is now `AverageFusion()`.

### Error functions

- `absolute_error`, `absolute_relative_error` and `symmetric_absolute_relative_error` raise a
  `ValueError` when `prediction` and `target` do not have the same shape, as their documentation
  says, instead of broadcasting. This is the same policy as `relative_loss`.

### Shape modules

- `View` uses `reshape`, so it works with non-contiguous inputs (it was `Tensor.view`, which raised).
- `Squeeze` accepts a tuple of dimensions as `torch.Tensor.squeeze` does.

### Utils

- `sonnix.utils` stays a documented toolbox (every module has a page in the API reference).
- `get_named_modules` uses a single traversal for every `depth`.
- `module_mode` restores the mode by module and not by name: a module added, replaced or shared
  under several names in the context no longer breaks the restoration (it raised a `KeyError` for an
  added module).

### Numerical encoders

- `AsinhNumericalEncoder(scale, learnable=True)` copies `scale` (the parameter aliased the tensor of
  the caller, so an in-place change leaked into the module). `CosSinNumericalEncoder` already did.
- `prepare_tensor_param`, `check_frequency` and `check_abs_range` move from
  `sonnix.modules.numerical.sine` to the private module `sonnix.modules.numerical._utils`. They are
  no longer in the `__all__` of `sine`. `check_frequency` and `check_abs_range` use
  `coola.validation`, so the message of the errors for `num_frequencies`, `min_frequency` and
  `min_abs_value` changes (for example `min_frequency must be greater than 0`).

- The six class methods `CosSinNumericalEncoder.create_{rand,linspace,logspace}_{frequency,value_range}`
  are replaced by `CosSinNumericalEncoder.create(spacing, num_frequencies, *, frequency_range=None,
value_range=None, learnable=False)`. For example
  `create_logspace_value_range(5, 0.1, 10.0)` is
  `create("logspace", 5, value_range=(0.1, 10.0))`. It also works for
  `AsinhCosSinNumericalEncoder` and returns the subclass.
- `CosSinNumericalEncoder.create` accepts `generator` (a `torch.Generator` or a seed, used by
  `spacing="rand"`), `device` and `dtype`, so the initialization can be reproduced without touching
  the global random number generator.
- Add `PiecewiseLinearNumericalEncoder.from_quantiles(x, n_bins)`: the bins are the empirical
  quantiles of the data of each feature.

### Alpha activations

- The alpha activations (`ExpSin`, `Gaussian`, `Laplacian`, `MultiQuadratic`, `Quadratic`,
  `Snake`) have a `channel_dim` argument (default `-1`, the previous behaviour): the dimension of the
  input that a per-channel `alpha` is broadcast over. `channel_dim=1` matches `torch.nn.PReLU` for an
  `(N, C, L)` input. It is shown in the representation when `num_parameters > 1`.
- `Gaussian`, `Laplacian` and `Snake` reject `init=0` (they divide by `alpha`).
- The docstrings document `device`, `dtype` and `channel_dim`.

### Structure

- `Clamp` moves from `sonnix.modules.fusion.clamp` to `sonnix.modules.activation.clamp` (next to
  `ReLUn`; it has a single input, so it is not a fusion module). `sonnix.modules.Clamp` and
  `sonnix.modules.activation.Clamp` still work.
- Every base class is part of the API and is exported from `sonnix.modules`: `BaseLoss`,
  `BaseBinaryFocalLoss`, `BaseBinaryPoly1Loss`, `BaseFusion` and `BaseDynamicNorm` (which is also in
  `sonnix.modules.activation.__all__`), next to `BaseAlphaActivation` and `BaseRelativeIndicator`.
- The loss modules import their functions from `sonnix.functional` (`asinh`, `indicators`,
  `poisson` and `quantile` used the file path or `sonnix.functional.loss`). No change for callers.
- `validate_focal_params` moves from `sonnix.functional.loss.focal` to `sonnix.validation` (and
  `sonnix.validation.loss`), with the other validators, and is exported there. It now has tests.
- Add the `LogCoshLoss(scale=1.0, *, reduction="mean")` and `MSLELoss(*, reduction="mean")` modules,
  the counterparts of `log_cosh_loss` and `msle_loss`.
- `PiecewiseLinearNumericalEncoder.forward` clamps with two precomputed non-persistent buffers
  (`lower`, `upper`) instead of building a list and calling `cat` on every call. The output and the
  state dict are unchanged.
