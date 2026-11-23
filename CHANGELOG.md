# Changelog

All notable changes to this project are documented in this file. The package is
still in alpha (`0.0.1a0`), so breaking changes can happen in any release.

## Unreleased

### Migration

| Before                                                                                             | After                                                                                                       |
| -------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------- |
| `log_cosh_loss(p, t, 2.0)`                                                                         | `log_cosh_loss(p, t, scale=2.0)` (every function option is keyword-only)                                    |
| `AsinhMSELoss()`                                                                                   | `TransformedLoss(nn.MSELoss(), prediction_transform=Asinh(), target_transform=Asinh())`                     |
| `AsinhSmoothL1Loss(beta=0.5)`                                                                      | the same with `nn.SmoothL1Loss(beta=0.5)`                                                                   |
| `MSLELoss()`                                                                                       | the same with `Log1p()`                                                                                     |
| `RelativeMSELoss(indicator=GeometricMeanIndicator())`                                              | `RelativeLoss(nn.MSELoss(reduction="none"), indicator=geometric_mean_indicator)`                            |
| `MomentMeanIndicator(k=2)`                                                                         | `partial(moment_mean_indicator, k=2)`                                                                       |
| `sonnix.modules.loss.X` (and `.activation`, `.fusion`, ...)                                        | `sonnix.modules.X`                                                                                          |
| `DynamicTanh(...).weight` / state dict key `weight`                                                | `.affine.weight` / key `affine.weight`                                                                      |
| `forward(x=...)` / `forward(scalar=...)` / `forward(scores=...)`                                   | `forward(input=...)` (positional calls are unchanged)                                                       |
| `View(shape)`                                                                                      | `nn.Flatten`, `nn.Unflatten`, or `x.reshape` in the model's `forward`                                       |
| `sonnix.utils.loss.is_loss_decreasing*`                                                            | `sonnix.testing.is_loss_decreasing*` (flat path since #452)                                                 |
| `sonnix.factory.resolve_dataset` / `resolve_optimizer`                                             | `coola.factory.resolve_object(config, Dataset)` / `(config, Optimizer)`                                     |
| `utils.iterator.get_named_modules`, `top_module_mode`                                              | removed (`module.named_modules()`, `module_mode`)                                                           |
| `utils.random.resolve_torch_generator`                                                             | `coola.random.make_torch_generator`                                                                         |
| `BinaryPoly1Loss(alpha=nan)`, `SafeExp(max_val=-1)`, ...                                           | `ValueError`                                                                                                |
| `quantile_regression_loss(p[4,3], t[1,3])`                                                         | `ValueError` (shapes differ); broadcast explicitly with `expand`                                            |
| `AsinhNumericalEncoder(ones(1, 3))(x[5, 4])` (broadcast)                                           | `AsinhNumericalEncoder.create(..., n_features=4)(x)`                                                        |
| `log_cosh_loss(p, t, scale=inf)`, `SafeLog(min_val=inf)`, `Clamp(nan, nan)`, ...                   | `ValueError` (every float hyperparameter is finite and in its range) (#447)                                 |
| `ConcatFusion(dim=1.5)`, `create(num_frequencies=2.5)`, `from_quantiles(x, 2.5)`                   | `TypeError` naming the argument (#447)                                                                      |
| `AsinhNumericalEncoder(tensor_with_nan)`, `ScaleAndShift(4, bias=1)`                               | `ValueError` / `TypeError` (#447)                                                                           |
| `validate_abs_range`, `validate_frequency`, `validate_moment_order`                                | removed: `validate_int` + `validate_ge`, or `resolve_range` (#447)                                          |
| `PiecewiseLinearNumericalEncoder(bins=...)`, `from_quantiles(x, n_bins=k)`                         | `PiecewiseLinearNumericalEncoder(edges=...)`, `from_quantiles(x, n_bins=k - 1)` for the same encoder (#449) |
| state dict key `width` of the piecewise encoder                                                    | removed (derived); a checkpoint with `width` loads with `strict=False`; an old `edges` does not load (#449) |
| `BinaryFocalLoss(0.25, 2.0)`                                                                       | `BinaryFocalLoss(0.25, gamma=2.0)` (#450)                                                                   |
| `GeneralRobustRegressionLoss(2.0, 1.0, 5.0)`                                                       | `GeneralRobustRegressionLoss(2.0, scale=1.0, max_val=5.0)` (#450)                                           |
| `RelativeLoss(criterion, indicator, 1e-6)`                                                         | `RelativeLoss(criterion, indicator=indicator, eps=1e-6)` (#450)                                             |
| `reduce_loss(x, "sum")`; `safe_pow(x=t, exponent=2)`                                               | `reduce_loss(x, reduction="sum")`; `safe_pow(input=t, exponent=2)` (#450)                                   |
| `BaseBinaryFocalLoss`, `BaseBinaryPoly1Loss`                                                       | subclass `BaseLoss` (#451)                                                                                  |
| `sonnix.modules.numerical._utils.NumericalEncoder`                                                 | `sonnix.modules.BaseNumericalEncoder` (#452)                                                                |
| `general_robust_regression_loss(p, t)` / `GeneralRobustRegressionLoss()` (`alpha=2`): `(x/c)²`      | `0.5 * (x/c)²`, continuous with `alpha != 2`; `scale=s / sqrt(2)` gives the old value (fourth target state step 2, #469) |
| `safe_exp(1e8, preserve_gradient=True)`: `8.9e6`; `safe_log(-1e30, ...)`: `-inf`; `inf` input: `nan` | the value of `safe_exp(x)` / `safe_log(x)`; gradient `1` (`0` at an infinite input) (fourth target state step 2, #469) |
| `is_loss_decreasing*(num_iterations=0)`, `lr=nan`, `max_norm=-1`, `random_seed=1.5`, `DummyDataset(feature_size=2.5)` | `ValueError` / `TypeError` naming the argument (fourth target state step 3, #471) |
| `load_state_dict_to_module(..., strict="yes")`, `register_tensor(..., learnable="yes")`              | `TypeError` (fourth target state step 3, #471) |
| `Clamp(0, 1).min_val` (`int`)                                                                        | `0.0` (`float`) (fourth target state step 3, #471) |
| `MulticlassFlatten(c)(p[2, 6, 3], t[4, 3])` (silent); `(p[4, 3, 5], t[4, 2])` (`RuntimeError`) | `ValueError` naming both shapes (fourth target state step 4, #472) |
| `sonnix.functional.loss.X`, `sonnix.utils.mode.X`, `sonnix.testing.loss.X`                         | `sonnix.functional.X`, `sonnix.utils.X`, `sonnix.testing.X` (#452)                                          |
| `binary_focal_loss(p, t_long)` and the three other binary losses (`RuntimeError`)                  | work: an integer or `bool` target is converted to the dtype of the prediction (#454)                        |
| `safe_log(0)`, `SafeLog()(0)` in `float16`: `-inf`                                                 | `-9.7` (`eps` raised to the smallest normal number of the dtype) (#460)                                     |
| `poisson_regression_loss`, `relative_loss`, `absolute_relative_error` on `0` in `float16`: `nan`   | finite (same rule) (#460)                                                                                   |
| `safe_exp(x)` in `float16` for `x > 11.09`: `inf`; `safe_exp(x, max_val=100)` in `float32`: `inf`  | finite: `max_val` is lowered to `log(finfo.max) - 1` (#460)                                                 |
| `general_robust_regression_loss` in `float16`, `\|p - t\| / scale > 255`: `inf` and `nan` gradient | computed in `float32`, cast back (#460)                                                                     |
| `Gaussian` gradient of `alpha` in `float16` for large `\|x\|`: `nan`                               | `0` (#460)                                                                                                  |
| `safe_pow(negative, 0.5)`: `0`; `safe_pow(-2, 2)`: `0`                                             | `nan` (like `torch.pow`); `4` (#460)                                                                        |
| `ScaleAndShift(5)(x[3, 1])`, `DynamicTanh`, `DynamicAsinh`                                         | `ValueError` naming `normalized_shape`; use `x.expand(3, 5)` to broadcast on purpose (#461)                 |
| `SumFusion()(a[4, 3], b[4, 1])`, `AverageFusion`, `MultiplicationFusion`                           | `ValueError` naming both shapes; `expand` first (#461)                                                      |
| `ResidualBlock` with branches of different shapes                                                  | `ValueError` naming `skip` and `residual` (#461)                                                            |
| `Gaussian(3)(x[4, 1])`, `Gaussian(3, channel_dim=1)(x[5])` (broadcast, `IndexError`)               | `ValueError` naming `channel_dim` and `num_parameters` (#461)                                               |
| `from sonnix.validation import Reduction`                                                          | `from sonnix.functional import Reduction` (#462)                                                            |
| `CosSinNumericalEncoder.create("linspace", num_frequencies=8, ...)`                                | `create("linspace", n_frequencies=8, ...)` (#464)                                                           |

### Breaking changes

- `MulticlassFlatten` checks what it reshapes (fourth target state step 4,
  #472): `forward` raises a `ValueError` naming both shapes when the shape of
  the prediction is not the shape of the target plus a class dimension.
  Before, `prediction (2, 6, 3)` with `target (4, 3)` had the same number of
  elements once flattened, so the loss paired unrelated rows, and
  `prediction (4, 3, 5)` with `target (4, 2)` raised a bare `RuntimeError`.
  The module is in the test registry (export, compile, dtype, purity, state
  and shape tests), and `conventions.md` says that a wrapper that reshapes
  checks its shapes while a wrapper that only forwards (`TransformedLoss`)
  leaves the policy to the wrapped module.
- Validate the whole surface (fourth target state step 3, #471). `sonnix.testing`
  and `sonnix.utils` check their scalar arguments like the modules do:
  `is_loss_decreasing*` require `num_iterations >= 1` and an integer
  `random_seed`, `max_norm` is `None` or finite and strictly positive,
  `is_loss_decreasing_with_adam` and `_with_sgd` require a finite and strictly
  positive `lr`; `DummyDataset` requires integers (a `bool` and a float are
  rejected, there is no `int()` conversion any more), `feature_size >= 1`,
  `num_examples >= 1` and an integer `rng_seed`; `load_state_dict_to_module`
  and `register_tensor` require a `bool` for `strict` and `learnable`. The
  hyperparameters `Clamp.min_val`, `Clamp.max_val`,
  `GeneralRobustRegressionLoss.max_val` and `BaseDynamicNorm.alpha_init_value`
  are stored as `float`. The hand-written table of invalid arguments keeps the
  range rows; the type and finiteness rows are replaced by one test that walks
  the signature of every public function, class and `create` class method
  (`tests/integration/test_scalar_arguments.py`, with `CALL_CASES` in the
  registry): an `int` rejects `2.5`, `True` and `"a"`, a `float` rejects `"a"`,
  `nan` and `inf`, a `bool` rejects `1`, and a scalar argument without a case
  fails a completeness test.
- Two numerical bugs (fourth target state step 2, #469). The general robust
  regression loss with `alpha=2` is `0.5 * (x / scale) ** 2` (Barron's
  definition and the limit of the general formula) instead of `(x / scale) ** 2`,
  so the loss is continuous in `alpha` and the default loss, and its gradient,
  are halved; `scale=s / sqrt(2)` reproduces the old value. `safe_exp`,
  `safe_log`, `SafeExp` and `SafeLog` with `preserve_gradient=True` return the
  value of the clamped input for every input (before, `safe_exp(1e8)` was
  `exp(16)`, `safe_log(-1e30)` was `-inf` and an infinite input gave `nan`);
  the gradient is the identity, and `0` at an infinite input. New tests compare
  the robust loss with the formula of the paper in `float64` and check its
  continuity at `alpha=2`, `0` and `-inf`, and check that the option does not
  change the value.
- `num_frequencies` becomes `n_frequencies` in the `create` class method of
  `CosSinNumericalEncoder` and `AsinhCosSinNumericalEncoder`, so that every
  count is named `n_<things>` (`n_features`, `n_bins`, `n_frequencies`, `n`);
  `num_parameters` is the name of `torch.nn.PReLU` and stays. The migration
  table lists the changes of the third revision with their pull requests (third
  target state step 6, #464).
- One public path for `Reduction` (third target state step 4, #462):
  `sonnix.validation` no longer exports `Reduction`; import it from
  `sonnix.functional` (`from sonnix.functional import Reduction`).
  `VALID_REDUCTIONS` and the validators stay in `sonnix.validation`. The
  reference lists `Reduction` under `sonnix.functional`, `test_docs` requires
  the directive of the package that exports the name, and a new test checks
  that no object is exported by two packages.
- Strict shapes (third target state step 3, #461): nothing broadcasts silently
  any more. `ScaleAndShift`, `DynamicTanh` and `DynamicAsinh` raise a
  `ValueError` naming `normalized_shape` when the last dimensions of the input
  are not `normalized_shape` (`ScaleAndShift(5)(x[3, 1])` no longer returns
  `(3, 5)`; use `x.expand(3, 5)`). `SumFusion`, `AverageFusion` and
  `MultiplicationFusion` raise a `ValueError` naming both shapes when an input
  does not have the shape of the first one (`ConcatFusion` is unchanged).
  `ResidualBlock` raises a `ValueError` naming `skip` and `residual` when the
  two branches have different shapes. The alpha activations with
  `num_parameters > 1` raise a `ValueError` naming `channel_dim` and
  `num_parameters` when `channel_dim` is out of range or the size of this
  dimension is not `num_parameters` (it was a broadcast, or a bare
  `IndexError`). `RelativeLoss` checks that the prediction and the target have
  the same shape before it calls the criterion.
- Numerics in half precision (third target state step 2, #460): the default
  guards are raised to the smallest normal number of the dtype
  (`resolve_eps`: `safe_log`, `SafeLog`, `poisson_regression_loss`,
  `relative_loss`, `absolute_relative_error`,
  `symmetric_absolute_relative_error`), `safe_exp` lowers `max_val` so that the
  output is finite (`resolve_max_exponent`), `general_robust_regression_loss`
  computes in at least `float32`, and `Gaussian` computes
  `exp(-0.5 * (x / alpha) ** 2)`. The results in `float16` that were `inf` or
  `nan` are finite (`safe_log(0)` is `-9.7`, not `-inf`); `float32`, `float64`
  and `bfloat16` are unchanged for the default arguments, except that
  `safe_exp(x, max_val=100)` in `float32` is now finite. `safe_pow` masks the
  zeros only (`input != 0`): `safe_pow(-4, 0.5)` is `nan` like `torch.pow`, and
  `safe_pow(-2, 2)` is `4` (it was `0`).
- One public path for every package (revised target state step 6, #452):
  `sonnix.utils` and `sonnix.testing` export their API (`from sonnix.utils
import module_mode, register_tensor, ...`, `from sonnix.testing import
is_loss_decreasing, ...`); the `__init__.py` of the folders
  `sonnix.functional.loss` and `sonnix.functional.activation` no longer export
  anything (use `sonnix.functional`); the module paths still work as they were
  (`sonnix.utils.mode`, ...) but they are not public. `NumericalEncoder`
  (`sonnix.modules.numerical._utils`) becomes `BaseNumericalEncoder` in
  `sonnix/modules/numerical/base.py` and is exported by `sonnix.modules`.
  The reference lists every exported name of `utils` and `testing`; the
  docstrings import from the flat path. New tests check the six `__all__`
  lists, that no folder has an `__all__`, and that no docstring imports from a
  folder.
- Clean up the public surface (revised target state step 5, #451):
  `BaseBinaryFocalLoss` and `BaseBinaryPoly1Loss` become private
  (`_BinaryFocalLoss`, `_BinaryPoly1Loss`; the public classes document their
  arguments), `ToDtype` accepts a `torch.dtype` or its name
  (`ToDtype("float32")`), `Snake` moves to
  `sonnix/modules/activation/alpha.py`, `AverageFusion` and `SumFusion` share
  `BaseFusion._sum`, and `Gaussian` and `Snake` check `init` with `coola.validation.validate_non_zero`.
  `Laplacian` rejects `init <= 0` with one message (`init must be greater than
0`). `ToFloat`, `ToLong`, `ReLUn`, `Square`, the three error functions and
  `rectifier_asinh_unit` are kept: a named wrapper that users look for is
  allowed (P16).
- Uniform call signatures (revised target state step 4, #450).
  `BinaryFocalLoss`, `BinaryFocalLossWithLogits` take only `alpha`
  positionally (`gamma` is keyword-only); `GeneralRobustRegressionLoss` takes
  only `alpha` (`scale`, `max_val` are keyword-only); `RelativeLoss` takes only
  `criterion` (`indicator`, `eps` are keyword-only). `reduce_loss(loss, *,
reduction="mean")`: the operand is named `loss` and `reduction` is
  keyword-only with a default. `safe_pow(input, exponent)`: the first argument
  is named `input`. `Clamp(min_val, max_val)` keeps two positional arguments,
  like `torch.nn.Hardtanh`. The positional arguments of every registered module
  are listed in `tests/registry.py` and compared with the signature, so a new
  class is checked (the hand-picked `test_option_is_keyword_only` is removed).
- Make the state of the encoders and the activations complete by
  construction (revised target state step 3, #449). New
  `sonnix.utils.register_tensor`, used by the alpha activations and the
  numerical encoders (it replaces three copies of "parameter or buffer").
  `PiecewiseLinearNumericalEncoder` keeps a single persistent buffer,
  `edges`; `width`, `is_step`, `lower`, `upper` and the new `start` are
  non-persistent and computed again after `load_state_dict` (before, `width`
  was in the state dict and `is_step` was not recomputed, so loading the state
  of an encoder with a duplicated edge changed its output). The constructor
  argument `bins` is renamed `edges`, and `edges` now holds all the edges
  (`(n_features, n_edges)`, so `feature_size == n_edges - 1`); a single edge is
  rejected. `from_quantiles(x, n_bins)` now makes `n_bins` segments from
  `n_bins + 1` quantiles, so `feature_size == n_bins`
  (`from_quantiles(x, n_bins=k)` becomes `from_quantiles(x, n_bins=k - 1)` for
  the same encoder). A checkpoint with the key `width` loads with
  `strict=False`; an old `edges` has one value less per feature and does not
  load.
- Validate every hyperparameter completely (revised target state step 2,
  #447): a float must be finite (`log_cosh_loss(scale=inf)`,
  `poisson_regression_loss(eps=inf)`, `binary_focal_loss(gamma=inf)`,
  `SafeLog(min_val=inf)`, `ReLUn(max_val=inf)`, `Clamp(nan, nan)`,
  `DynamicTanh(alpha_init_value=nan)`, an infinite range of an encoder, ...
  now raise `ValueError`); an integer argument is checked before it is used
  (`ConcatFusion(dim=1.5)`, `create(num_frequencies=2.5)`,
  `from_quantiles(n_bins=2.5)` raise `TypeError`); a flag must be a `bool`
  (`learnable`, `bias`, `log_input`, `detach_indicator`,
  `preserve_gradient`); the modules validate before they convert, so
  `GeneralRobustRegressionLoss(alpha="a")` raises `TypeError` like the
  function. The encoders reject a `NaN` or an infinite tensor
  (`AsinhNumericalEncoder`, `CosSinNumericalEncoder`; the piecewise message is
  now `'bins' must contain only finite values`). New
  `sonnix.validation.validate_positive`; removed `validate_abs_range`,
  `validate_frequency` and `validate_moment_order` (`resolve_range` checks the
  range once and names `min_abs_value` / `max_abs_value` for a `value_range`).
- Documentation only: the alpha activations, the binary focal losses and the
  binary Poly-1 losses no longer copy the `Args:` block of their base class.
  The class docstring points to it (target state step 12, #440).
- `resolve_range` (used by `AsinhNumericalEncoder.create`,
  `CosSinNumericalEncoder.create` and `AsinhCosSinNumericalEncoder.create`)
  checks `0 < min <= max` itself, so the check is written once. The error
  messages are unchanged; `scale_range` is now checked before `feature_size`
  (target state step 11, #439).
- Remove `View` (it reshaped the batch dimension too, so it only worked for one
  batch size): use `nn.Flatten`, `nn.Unflatten` or `x.reshape` in the model. The
  alpha activations validate `init` in an overridable `_validate_init` method
  instead of the class flags `allow_zero_init` / `allow_negative_init`
  (removed). The focal and Poly-1 functions share `probability_of_target` and
  `class_weight` (`sonnix.functional.loss._binary`), with the same values
  (target state step 9, #423).
- Reorganize the packages (target state step 8, #422): `sonnix.utils.loss`
  (`is_loss_decreasing*`) moves to `sonnix.testing.loss`;
  `sonnix.factory.dataset` and `sonnix.factory.optimizer` (`resolve_dataset`,
  `is_dataset_config`, `resolve_optimizer`, `is_optimizer_config`) are removed
  (use `coola.factory.resolve_object`); `sonnix.utils.iterator`
  (`get_named_modules`), `top_module_mode` and `sonnix.utils.random`
  (`resolve_torch_generator`, use `coola.random.make_torch_generator`) are
  removed; `find_module_state_dict` is private. `Reduction` and
  `VALID_REDUCTIONS` are defined once in `sonnix.validation`
  (`sonnix.functional.Reduction` is the same object). The sub-packages
  `sonnix.modules.{activation,fusion,loss,numerical}` no longer export the
  classes: import them from `sonnix.modules`.
- The numerical encoders (`AsinhNumericalEncoder`, `CosSinNumericalEncoder`,
  `AsinhCosSinNumericalEncoder`, `PiecewiseLinearNumericalEncoder`) check that
  the last dimension of the input is `n_features` and raise a `ValueError`
  naming it, instead of a torch broadcast `RuntimeError`. An encoder with
  `n_features == 1` does not broadcast over the last dimension anymore: build
  `n_features=k` explicitly with `create(..., n_features=k)`. The range logic of
  the two `create` methods is shared (`resolve_range`), the encoders share a
  private base class, and `on_duplicate` is typed `Literal["warn", "ignore"]`
  (target state step 7, #420).
- `BaseDynamicNorm` (and so `DynamicTanh` and `DynamicAsinh`) does not inherit
  from `ScaleAndShift` anymore: it owns one as `affine`. The parameters are
  `affine.weight` and `affine.bias` (were `weight` and `bias`), so the keys of
  the `state_dict` change, and a checkpoint of an older version does not load.
  Subclasses implement `_activation(input)` instead of `forward`,
  `ScaleAndShift._affine` is removed, and the representation shows the `affine`
  submodule (target state step 6, #419).
- The tensor argument of `forward` is named `input` in `NLinear`,
  `ScaleAndShift`, `DynamicTanh`, `DynamicAsinh`, `View`, the four numerical
  encoders (was `x` or `scalar`), `ToBinaryLabel` and `ToCategoricalLabel` (was
  `scores`). Positional calls are unchanged; `forward(x=...)`,
  `forward(scalar=...)` and `forward(scores=...)` are a `TypeError`. The
  `# noqa: A002` are kept (decision D4 is not applied) (target state
  step 5, #418).
- Remove the losses that are a composition of a `torch.nn` criterion and a
  transform: `AsinhMSELoss`, `AsinhSmoothL1Loss`, `MSLELoss`, `RelativeMSELoss`,
  `RelativeSmoothL1Loss`, `asinh_mse_loss`, `asinh_smooth_l1_loss` and
  `msle_loss`. Use `TransformedLoss` or `RelativeLoss` (recipes in the losses
  guide). Remove the seven relative indicator modules and
  `BaseRelativeIndicator`: `RelativeLoss(indicator=...)` takes a callable (the
  `*_indicator` functions of `sonnix.functional`, `functools.partial` for `k`),
  is not a submodule anymore, and a non-callable indicator raises a `TypeError`.
  `MulticlassFlatten` moves to `sonnix.modules.loss` (still exported by
  `sonnix.modules`) (target state step 4, #417).
- The options of the functions of `sonnix.functional` are keyword-only
  (`log_cosh_loss(p, t, scale=2.0)`): `scale`, `alpha`, `gamma`, `beta`, `q`,
  `eps`, `k`, `max_val` and `min_val`. Positional calls are a `TypeError`
  (target state step 3, #415).
- Functions of two tensors of the same role (`log_cosh_loss`,
  `quantile_regression_loss`, `poisson_regression_loss`, the focal, Poly-1,
  general robust, asinh and MSLE losses, and the relative indicators) raise a
  `ValueError` when the shapes differ, instead of a torch error, a warning or a
  silent broadcast (`quantile_regression_loss` returned a scalar for `(4, 3)` vs
  `(1, 3)`). `BinaryPoly1Loss*(alpha)`, `SafeExp(max_val)`, `Pow(exponent)` and
  `ToBinaryLabel(threshold)` reject non-finite values, and `Squeeze(dim)`
  rejects non-integer dimensions (target state step 2, #414).
- `load_state_dict_to_module(strict=...)`, and `num_iterations`, `random_seed`
  and `lr` of `is_loss_decreasing*` are keyword-only. `random_seed` of
  `is_loss_decreasing*` also seeds the optimization steps, so the functions are
  deterministic for a module with randomness (review 3.5, #368).
- `ToBinaryLabel.threshold` and `NLinear.n/in_features/out_features` are plain
  attributes like the other hyperparameters (they were a read-only property);
  `NLinear.forward` is documented (review 3.4, #367).
- `safe_pow` is exported from `sonnix.functional`. The implementation details
  `ALPHA_EPS`, `SPACINGS` and `BaseAlphaActivation.broadcast_alpha` are private
  (`_ALPHA_EPS`, `_SPACINGS`, `_broadcast_alpha`) (review 3.3, #366).
- `Square` is an independent module instead of a subclass of `Pow`: it has no
  `exponent` attribute anymore. `DynamicTanh`, `DynamicAsinh` and the user
  subclasses of `BaseDynamicNorm` apply the affine transformation with the new
  `ScaleAndShift._affine` method instead of `ScaleAndShift.forward(self, ...)`
  (review 3.2, #365).
- `BaseLoss(reduction)` is keyword-only like all the losses: `BaseLoss("sum")`
  is a `TypeError`. The redundant `__init__` of `MSLELoss` and `AsinhMSELoss`,
  and the copy of `extra_repr` of `AsinhMSELoss`, are removed
  (review 3.1, #364).
- The hyperparameters of the modules are public attributes named like the
  constructor arguments (`ReLUn.max_val`, `Clamp.min_val`/`max_val`,
  `Snake.frequency`, `Pow.exponent`, `SafeExp.max_val`/`preserve_gradient`,
  `SafeLog.min_val`/`preserve_gradient`, `Squeeze.dim`, `View.shape`,
  `ConcatFusion.dim`, `ScaleAndShift.normalized_shape`,
  `DynamicTanh.alpha_init_value`, `MomentMeanIndicator.k`, and the `alpha`,
  `gamma`, `beta`, `q`, `eps`, `scale`, `max_val`, `detach_indicator` attributes
  of the losses). The previous private attributes (`_max`, `_alpha`,
  ...) are removed.

### Added

- `sonnix.modules.SafePow(exponent)`, the module of `safe_pow`, which completes
  `SafeExp`, `SafeLog` and `SafePow` (property P1). A registry test checks that
  the output of every module with a function (the losses, `RectifierAsinhUnit`
  and the three safe operations) equals the output of the function (third
  target state step 4, #462).
- `ModuleCase.wrong_inputs` in the test registry and a test that every module
  raises a `ValueError` on inputs with a wrong shape (the modules that accept
  any shape are listed with the reason), and a `torch.export` test with a
  dynamic batch dimension for every registered module (third target state
  step 3, #461).
- `sonnix.utils.resolve_eps` and `sonnix.utils.resolve_max_exponent`, the two
  helpers that adapt a guard to the dtype of a tensor. New registry tests: every
  module and every function without a module is finite on zeros, tiny and large
  inputs wherever the `float32` result is representable (`float16` and
  `bfloat16`), and `float16` is added to the dtype tests (third target state
  step 2, #460).
- Design rules (P1-P15) in `docs/docs/conventions.md`; import-linter contracts
  for `factory` (independent packages, `functional` does not import
  `modules`/`factory`, `testing` on top of the layers); tests that every
  argument accepting a configuration documents the trust warning and that no
  registered module mutates its inputs (target state step 1, #413).
- Second revision of the design rules (P1-P16) in `docs/docs/conventions.md`,
  and a test that every name exported by a public package is listed in
  `docs/docs/refs`. The reference pages list the `validation` and `factory`
  names one by one, document `rectifier_asinh_unit` and the relative loss
  indicators as `sonnix.functional.<name>`, and `utils.md` no longer renders
  `sonnix.testing.loss` (revised target state step 1, #446).
- Third revision of the design rules (P1-P17) in `docs/docs/conventions.md`,
  and two tests over the six `__all__`: the `Args:` block of every public
  function and class lists exactly the arguments of its signature, and a
  `Raises:` block quotes only parameters. Every exported name is documented
  under its own package. `review/suggestions.md` (obsolete) is deleted, the
  `pre-commit` configuration runs `pyright`, and `docs/docs/index.md` includes
  `README.md` (third target state step 1, #459).

### Fixed

- The guide `numerical_stability.md` describes what the code does (guards that
  follow the dtype, finite `safe_exp`, the cases that are not guaranteed) and
  `losses.md` documents that `eps` is raised to the dtype (third target state
  step 2, #460).
- Docstrings: `Pow` documents `exponent`, `ToBinaryLabel` no longer says it
  raises for a `dim` that does not exist, and the `Args:` of `RelativeLoss` and
  `is_loss_decreasing_with_sgd` follow the order of the signature (third target
  state step 1, #459).
- The four binary losses (`binary_focal_loss`, `binary_focal_loss_with_logits`,
  `binary_poly1_loss`, `binary_poly1_loss_with_logits`, and their modules)
  convert an integer or `bool` target to the dtype of the prediction. Before,
  the labels of a `DataLoader` (`long`) raised a `RuntimeError` of `torch`. A
  floating point target is not converted (revised target state step 8, #454).
- `RelativeLoss.__repr__` shows the `indicator` (`indicator=classical_relative_indicator`
  or `indicator=functools.partial(moment_mean_indicator, k=2)`), and
  `DynamicTanh`, `DynamicAsinh` and `BaseDynamicNorm` show `alpha_init_value`,
  so every hyperparameter is shown by `repr` (revised target state step 8,
  #454). A new test checks it for the hyperparameters of the registry.
- `load_state_dict_to_module` finds the state dict of checkpoints saved from
  `DataParallel` (`module.` prefix) or `torch.compile` (`_orig_mod.` prefix),
  and `import sonnix` no longer fails when the package is not installed
  (`__version__` is `0.0.0+unknown`) (review 3.5, #368).
- `NLinear`, `ExU`, `ScaleAndShift` (and its subclasses), the alpha activations
  and `MomentMeanIndicator` validate their arguments with `coola.validation`
  (sizes >= 1, integer types, finite `init`, `Laplacian` needs `init > 0`), and
  `NLinear.forward` names `n` when the input shape does not match
  (review 2.6, #361).
- `PiecewiseLinearNumericalEncoder` accepts integer bins (converted to the
  default float dtype) and raises a `ValueError` for empty bins and for
  non-finite bins (review 2.5, #360).
- `poisson_regression_loss` clamps the prediction in the linear term too, so the
  loss is bounded below for negative predictions (review 2.4, #359).
- `log_cosh_loss` validates `scale` like `LogCoshLoss` does: `scale <= 0` raises
  a `ValueError` instead of returning `nan` or a negative loss
  (review 2.3, #358).
- `general_robust_regression_loss` and `GeneralRobustRegressionLoss` reject
  `alpha` equal to `NaN` or `+inf` and a non-positive `max_val`
  (review 2.2, #357).
- `geometric_mean_indicator` and `moment_mean_indicator` no longer overflow in
  half precision, for large values or for a large order `k` (review 2.1, #356).
- `sonnix.modules` can be imported again with `coola` 2.0.0a2:
  `get_torch_generator` is replaced by `make_torch_generator`
  (review 1.1, #354).
- `RelativeSmoothL1Loss` stores `beta` and shows it in its representation.

### Changed

- Gates that keep the properties true (fourth target state step 1, #468): the 12
  imports of another package use the flat path (`from sonnix.functional import
  Reduction`, `from sonnix.utils import register_tensor`, `module_mode`); new
  tests check the flat cross-package imports, that `src` has no suppression
  (`noqa: A002` for the argument `input` is the only exception), that the
  `README.md` and the `python` blocks of the guides run, that the descriptions
  of the reference tables are the first line of the docstrings, and that the
  rules of `docs/docs/conventions.md` and of `review/target_state.md` agree.
  `conventions.md` gets the properties P18 (easy to maintain) and P19 (matches
  its definition). No value changes.

- Simplifications without a change of behaviour (third target state step 5,
  #463): the probability and the logits variants of `binary_focal_loss` and of
  `binary_poly1_loss` share one private core each (`_focal`, `_poly1`), so they
  cannot drift; the `target` paragraph of the four binary loss modules refers to
  the function instead of copying it; the dead branch of
  `NLinear.reset_parameters` is removed. `NLinear.forward` keeps `bmm`: an
  `einsum` was measured slower (`22` against `15` us at `32 x 8 x 16 x 16`, `33`
  against `28` us at `256 x 16 x 32 x 32`), and the plan says to skip an item
  whose benchmark regresses.
- `AsinhNumericalEncoder.create(...)` builds an encoder with spaced scales like
  `CosSinNumericalEncoder.create`, and both `create` methods have a keyword-only
  `n_features` option (review 3.7, #370).
- The README and the documentation home page state that `sonnix` also ships
  training and testing helpers (`sonnix.factory`, `sonnix.utils`,
  `sonnix.testing`) (review 3.6, #369).
- The documentation of `sonnix.factory` and of every `Module | dict` argument
  states that a configuration calls its `_target_` and must come from a trusted
  source (review 2.8, #363).
- `PiecewiseLinearNumericalEncoder` and `from_quantiles` have a keyword-only
  `on_duplicate="warn" | "ignore"` option for the duplicate-edge warning
  (review 2.5, #360).
- `PoissonRegressionLoss` and `poisson_regression_loss` have a keyword-only
  `log_input` option, and the representation of the module shows `log_input`
  (review 2.4, #359).
- `BaseLoss.extra_repr` returns `reduction=...` and the losses extend it, so the
  representation of every loss starts with `reduction=...` followed
  by its hyperparameters.

### Renamed

- The `check_*` validation functions move to `sonnix.validation` and are renamed
  `validate_*`: `sonnix.functional.check_loss_reduction_strategy` is
  `validate_loss_reduction_strategy`, `check_moment_order`
  (`sonnix.functional.loss.relative`) is `validate_moment_order`, and
  `check_abs_range`/`check_frequency` (`sonnix.modules.numerical._utils`) are
  `validate_abs_range` and `validate_frequency`. `VALID_REDUCTIONS`
  moves to `sonnix.validation.loss`.

- The `input_size` and `output_size` properties of the numerical encoders are
  now `n_features` and `feature_size`, like their representation.
- The `prediction` and `target` arguments (and attributes) of `TransformedLoss`
  are now `prediction_transform` and `target_transform`.

### Changed (keyword-only options)

- The options are keyword-only: `preserve_gradient` (`safe_exp`, `safe_log`,
  `SafeExp`, `SafeLog`), `bias` (`NLinear`, `ExU`), `device` and `dtype`
  (`NLinear`, `ExU`, `ScaleAndShift`, the dynamic norms, the alpha activations),
  `learnable` (the alpha activations, the numerical encoders and their
  `create_*` class methods). `SafeExp(20.0, True)`
  is now `SafeExp(20.0, preserve_gradient=True)`.

### Snake

- `Snake` is a `BaseAlphaActivation` with a learnable `alpha` (per channel with
  `num_parameters`), as in the paper, and supports `device` and `dtype`. The
  `frequency` argument is replaced by `init`, and `Snake(frequency=2.0)` is now
  `Snake(init=2.0, learnable=False)`. The forward is
  `x + sin(alpha * x) ** 2 / alpha`, which has the same values
  with fewer operations.

### Wrapper modules

- Add `ToDtype(dtype)`. `ToFloat` and `ToLong` are now `ToDtype` subclasses
  (`float32` and `int64`).
- `ToCategoricalLabel` has a `dim` argument (default `-1`).
- `Square` is a `Pow(exponent=2)` subclass (it is no longer a separate
  implementation) and is listed in `sonnix.modules.activation.math.__all__` with
  `Pow`. Its output on integer tensors is now floating point, like `Pow`.

### Normalization layers

- `BaseDynamicNorm` extends `ScaleAndShift` and is abstract (`forward` is an
  abstract method). The subclasses call `ScaleAndShift.forward` for the affine
  part. `alpha` is registered after `weight` and `bias`, so the order of the
  keys of the state dict changes (the names do not).
- `ScaleAndShift`, `DynamicTanh` and `DynamicAsinh` accept `bias=False` (no
  shift, `bias` is `None`). `bias` is part of their representation.

### Fusion modules

- Add `BaseFusion`: it checks that there is at least one input (the check was
  repeated in three modules) and calls the abstract method `fuse`.
  `MultiplicationFusion`, `SumFusion`, `AverageFusion` and
  `ConcatFusion` extend it.
- `SumFusion` has no `normalized` argument and `AverageFusion` no longer extends
  `SumFusion`: `SumFusion(normalized=True)` is now `AverageFusion()`.

### Error functions

- `absolute_error`, `absolute_relative_error` and
  `symmetric_absolute_relative_error` raise a `ValueError` when `prediction` and
  `target` do not have the same shape, as their documentation says, instead of
  broadcasting. This is the same policy as `relative_loss`.

### Shape modules

- `View` uses `reshape`, so it works with non-contiguous inputs (it was
  `Tensor.view`, which raised).
- `Squeeze` accepts a tuple of dimensions as `torch.Tensor.squeeze` does.

### Utils

- `sonnix.utils` stays a documented toolbox (every module has a page in
  the API reference).
- `get_named_modules` uses a single traversal for every `depth`.
- `module_mode` restores the mode by module and not by name: a module added,
  replaced or shared under several names in the context no longer breaks the
  restoration (it raised a `KeyError` for an added module).

### Numerical encoders

- `AsinhNumericalEncoder(scale, learnable=True)` copies `scale` (the parameter
  aliased the tensor of the caller, so an in-place change leaked into the
  module). `CosSinNumericalEncoder` already did.
- `prepare_tensor_param`, `check_frequency` and `check_abs_range` move from
  `sonnix.modules.numerical.sine` to the private module
  `sonnix.modules.numerical._utils`. They are no longer in the `__all__` of
  `sine`. `check_frequency` and `check_abs_range` use `coola.validation`, so the
  message of the errors for `num_frequencies`, `min_frequency` and
  `min_abs_value` changes (for example `min_frequency must be greater than 0`).

- The six class methods
  `CosSinNumericalEncoder.create_{rand,linspace,logspace}_{frequency,value_range}`
  are replaced by `CosSinNumericalEncoder.create`:

  ```python
  CosSinNumericalEncoder.create(
      spacing, num_frequencies, *, frequency_range=None, value_range=None, learnable=False
  )
  ```

  For example `create_logspace_value_range(5, 0.1, 10.0)` is
  `create("logspace", 5, value_range=(0.1, 10.0))`. It also works for
  `AsinhCosSinNumericalEncoder` and returns the subclass.

- `CosSinNumericalEncoder.create` accepts `generator` (a `torch.Generator` or a
  seed, used by `spacing="rand"`), `device` and `dtype`, so the initialization
  can be reproduced without touching the global random number generator.
- Add `PiecewiseLinearNumericalEncoder.from_quantiles(x, n_bins)`: the bins are
  the empirical quantiles of the data of each feature.

### Alpha activations

- The alpha activations (`ExpSin`, `Gaussian`, `Laplacian`, `MultiQuadratic`,
  `Quadratic`, `Snake`) have a `channel_dim` argument (default `-1`, the
  previous behaviour): the dimension of the input that a per-channel `alpha` is
  broadcast over. `channel_dim=1` matches `torch.nn.PReLU` for an `(N, C, L)`
  input. It is shown in the representation when `num_parameters > 1`.
- `Gaussian`, `Laplacian` and `Snake` reject `init=0` (they divide by `alpha`).
- The docstrings document `device`, `dtype` and `channel_dim`.

### Structure

- `Clamp` moves from `sonnix.modules.fusion.clamp` to
  `sonnix.modules.activation.clamp` (next to `ReLUn`; it has a single input, so
  it is not a fusion module). `sonnix.modules.Clamp` and
  `sonnix.modules.activation.Clamp` still work.
- Every base class is part of the API and is exported from `sonnix.modules`:
  `BaseLoss`, `BaseBinaryFocalLoss`, `BaseBinaryPoly1Loss`, `BaseFusion` and
  `BaseDynamicNorm` (which is also in `sonnix.modules.activation.__all__`), next
  to `BaseAlphaActivation` and `BaseRelativeIndicator`.
- The loss modules import their functions from `sonnix.functional` (`asinh`,
  `indicators`, `poisson` and `quantile` used the file path or
  `sonnix.functional.loss`). No change for callers.
- `validate_focal_params` moves from `sonnix.functional.loss.focal` to
  `sonnix.validation` (and `sonnix.validation.loss`), with the other validators,
  and is exported there. It now has tests.
- Add the `LogCoshLoss(scale=1.0, *, reduction="mean")` and
  `MSLELoss(*, reduction="mean")` modules, the counterparts of
  `log_cosh_loss` and `msle_loss`.
- `PiecewiseLinearNumericalEncoder.forward` clamps with two precomputed
  non-persistent buffers (`lower`, `upper`) instead of building a list and
  calling `cat` on every call. The output and the state dict are unchanged.
