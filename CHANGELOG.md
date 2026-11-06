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
