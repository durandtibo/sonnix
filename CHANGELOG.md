# Changelog

All notable changes to this project are documented in this file. The package is still in alpha
(`0.0.1a0`), so breaking changes can happen in any release.

## Unreleased

### Breaking changes

- Remove `View` (it reshaped the batch dimension too, so it only worked for one batch size): use `nn.Flatten`,
  `nn.Unflatten` or `x.reshape` in the model. The alpha activations validate `init` in an overridable
  `_validate_init` method instead of the class flags `allow_zero_init` / `allow_negative_init` (removed). The focal and
  Poly-1 functions share `probability_of_target` and `class_weight` (`sonnix.functional.loss._binary`), with the same
  values (target state step 9, #423).
- Reorganize the packages (target state step 8, #422):
  `sonnix.utils.loss` (`is_loss_decreasing*`) moves to `sonnix.testing.loss`; `sonnix.factory.dataset` and
  `sonnix.factory.optimizer` (`resolve_dataset`, `is_dataset_config`, `resolve_optimizer`, `is_optimizer_config`)
  are removed (use `coola.factory.resolve_object`); `sonnix.utils.iterator` (`get_named_modules`),
  `top_module_mode` and `sonnix.utils.random` (`resolve_torch_generator`, use `coola.random.make_torch_generator`)
  are removed; `find_module_state_dict` is private. `Reduction` and `VALID_REDUCTIONS` are defined once in
  `sonnix.validation` (`sonnix.functional.Reduction` is the same object). The sub-packages
  `sonnix.modules.{activation,fusion,loss,numerical}` no longer export the classes: import them from
  `sonnix.modules`.
- The numerical encoders (`AsinhNumericalEncoder`, `CosSinNumericalEncoder`, `AsinhCosSinNumericalEncoder`,
  `PiecewiseLinearNumericalEncoder`) check that the last dimension of the input is `n_features` and raise a
  `ValueError` naming it, instead of a torch broadcast `RuntimeError`. An encoder with `n_features == 1` does not
  broadcast over the last dimension anymore: build `n_features=k` explicitly with `create(..., n_features=k)`.
  The range logic of the two `create` methods is shared (`resolve_range`), the encoders share a private base class,
  and `on_duplicate` is typed `Literal["warn", "ignore"]` (target state step 7, #420).
- `BaseDynamicNorm` (and so `DynamicTanh` and `DynamicAsinh`) does not inherit from `ScaleAndShift` anymore: it
  owns one as `affine`. The parameters are `affine.weight` and `affine.bias` (were `weight` and `bias`), so the
  keys of the `state_dict` change, and a checkpoint of an older version does not load. Subclasses implement
  `_activation(input)` instead of `forward`, `ScaleAndShift._affine` is removed, and the representation shows the
  `affine` submodule (target state step 6, #419).
- The tensor argument of `forward` is named `input` in `NLinear`, `ScaleAndShift`, `DynamicTanh`,
  `DynamicAsinh`, `View`, the four numerical encoders (was `x` or `scalar`), `ToBinaryLabel` and
  `ToCategoricalLabel` (was `scores`). Positional calls are unchanged; `forward(x=...)`,
  `forward(scalar=...)` and `forward(scores=...)` are a `TypeError`. The `# noqa: A002` are kept (decision
  D4 is not applied) (target state step 5, #418).
- Remove the losses that are a composition of a `torch.nn` criterion and a transform: `AsinhMSELoss`,
  `AsinhSmoothL1Loss`, `MSLELoss`, `RelativeMSELoss`, `RelativeSmoothL1Loss`, `asinh_mse_loss`,
  `asinh_smooth_l1_loss` and `msle_loss`. Use `TransformedLoss` or `RelativeLoss` (recipes in the losses
  guide). Remove the seven relative indicator modules and `BaseRelativeIndicator`: `RelativeLoss(indicator=...)`
  takes a callable (the `*_indicator` functions of `sonnix.functional`, `functools.partial` for `k`), is not
  a submodule anymore, and a non-callable indicator raises a `TypeError`. `MulticlassFlatten` moves to
  `sonnix.modules.loss` (still exported by `sonnix.modules`) (target state step 4, #417).
- The options of the functions of `sonnix.functional` are keyword-only (`log_cosh_loss(p, t, scale=2.0)`):
  `scale`, `alpha`, `gamma`, `beta`, `q`, `eps`, `k`, `max_val` and `min_val`. Positional calls are a `TypeError`
  (target state step 3, #415).
- Functions of two tensors of the same role (`log_cosh_loss`, `quantile_regression_loss`, `poisson_regression_loss`,
  the focal, Poly-1, general robust, asinh and MSLE losses, and the relative indicators) raise a `ValueError`
  when the shapes differ, instead of a torch error, a warning or a silent broadcast (`quantile_regression_loss`
  returned a scalar for `(4, 3)` vs `(1, 3)`). `BinaryPoly1Loss*(alpha)`, `SafeExp(max_val)`, `Pow(exponent)` and
  `ToBinaryLabel(threshold)` reject non-finite values, and `Squeeze(dim)` rejects non-integer dimensions
  (target state step 2, #414).
- `load_state_dict_to_module(strict=...)`, and `num_iterations`, `random_seed` and `lr` of `is_loss_decreasing*` are
  keyword-only. `random_seed` of `is_loss_decreasing*` also seeds the optimization steps, so the functions are
  deterministic for a module with randomness (review 3.5, #368).
- `ToBinaryLabel.threshold` and `NLinear.n/in_features/out_features` are plain attributes like the other
  hyperparameters (they were a read-only property); `NLinear.forward` is documented (review 3.4, #367).
- `safe_pow` is exported from `sonnix.functional`. The implementation details `ALPHA_EPS`, `SPACINGS` and
  `BaseAlphaActivation.broadcast_alpha` are private (`_ALPHA_EPS`, `_SPACINGS`, `_broadcast_alpha`)
  (review 3.3, #366).
- `Square` is an independent module instead of a subclass of `Pow`: it has no `exponent` attribute anymore.
  `DynamicTanh`, `DynamicAsinh` and the user subclasses of `BaseDynamicNorm` apply the affine transformation
  with the new `ScaleAndShift._affine` method instead of `ScaleAndShift.forward(self, ...)` (review 3.2, #365).
- `BaseLoss(reduction)` is keyword-only like all the losses: `BaseLoss("sum")` is a `TypeError`. The
  redundant `__init__` of `MSLELoss` and `AsinhMSELoss`, and the copy of `extra_repr` of `AsinhMSELoss`, are
  removed (review 3.1, #364).
- The hyperparameters of the modules are public attributes named like the constructor arguments
  (`ReLUn.max_val`, `Clamp.min_val`/`max_val`, `Snake.frequency`, `Pow.exponent`,
  `SafeExp.max_val`/`preserve_gradient`, `SafeLog.min_val`/`preserve_gradient`, `Squeeze.dim`,
  `View.shape`, `ConcatFusion.dim`, `ScaleAndShift.normalized_shape`,
  `DynamicTanh.alpha_init_value`, `MomentMeanIndicator.k`, and the `alpha`, `gamma`, `beta`, `q`,
  `eps`, `scale`, `max_val`, `detach_indicator` attributes of the losses). The previous private
  attributes (`_max`, `_alpha`, ...) are removed.

### Added

- Design rules (P1-P15) in `docs/docs/conventions.md`; import-linter contracts for `factory` (independent
  packages, `functional` does not import `modules`/`factory`, `testing` on top of the layers); tests that
  every argument accepting a configuration documents the trust warning and that no registered module
  mutates its inputs (target state step 1, #413).

### Fixed

- `load_state_dict_to_module` finds the state dict of checkpoints saved from `DataParallel` (`module.` prefix)
  or `torch.compile` (`_orig_mod.` prefix), and `import sonnix` no longer fails when the package is not
  installed (`__version__` is `0.0.0+unknown`) (review 3.5, #368).
- `NLinear`, `ExU`, `ScaleAndShift` (and its subclasses), the alpha activations and `MomentMeanIndicator` validate
  their arguments with `coola.validation` (sizes >= 1, integer types, finite `init`, `Laplacian` needs
  `init > 0`), and `NLinear.forward` names `n` when the input shape does not match (review 2.6, #361).
- `PiecewiseLinearNumericalEncoder` accepts integer bins (converted to the default float dtype) and raises
  a `ValueError` for empty bins and for non-finite bins (review 2.5, #360).
- `poisson_regression_loss` clamps the prediction in the linear term too, so the loss is bounded below for
  negative predictions (review 2.4, #359).
- `log_cosh_loss` validates `scale` like `LogCoshLoss` does: `scale <= 0` raises a `ValueError` instead
  of returning `nan` or a negative loss (review 2.3, #358).
- `general_robust_regression_loss` and `GeneralRobustRegressionLoss` reject `alpha` equal to `NaN` or
  `+inf` and a non-positive `max_val` (review 2.2, #357).
- `geometric_mean_indicator` and `moment_mean_indicator` no longer overflow in half precision, for
  large values or for a large order `k` (review 2.1, #356).
- `sonnix.modules` can be imported again with `coola` 2.0.0a2: `get_torch_generator` is replaced by
  `make_torch_generator` (review 1.1, #354).
- `RelativeSmoothL1Loss` stores `beta` and shows it in its representation.

### Changed

- `AsinhNumericalEncoder.create(...)` builds an encoder with spaced scales like
  `CosSinNumericalEncoder.create`, and both `create` methods have a keyword-only `n_features` option
  (review 3.7, #370).
- The README and the documentation home page state that `sonnix` also ships training and testing helpers
  (`sonnix.factory`, `sonnix.utils`, `sonnix.testing`) (review 3.6, #369).
- The documentation of `sonnix.factory` and of every `Module | dict` argument states that a configuration
  calls its `_target_` and must come from a trusted source (review 2.8, #363).
- `PiecewiseLinearNumericalEncoder` and `from_quantiles` have a keyword-only
  `on_duplicate="warn" | "ignore"` option for the duplicate-edge warning (review 2.5, #360).
- `PoissonRegressionLoss` and `poisson_regression_loss` have a keyword-only `log_input` option, and the
  representation of the module shows `log_input` (review 2.4, #359).
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
