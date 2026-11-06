# `sonnix`: target state and the changes to reach it

Date: 2026-11-06, on `main` at `fc04eeb`.

This document describes the intended design of the `sonnix` package and the steps to get from the
current `main` branch to it. It follows the format of
[`coola/review/recursive.md`](https://github.com/durandtibo/coola/blob/main/review/recursive.md):
constraints, properties, findings, architecture, public API, plan, tests, decisions.

It replaces the plan of `review/suggestions.md` (most of its items are done: #247-#409). That file is
still the reference for the history of the numerical fixes.

Scope: all of `src/` (7.8k lines, read in full), `pyproject.toml`, the import-linter contracts, `docs/`,
`README.md`, `CLAUDE.md`, `tests/` layout. Items marked **[probe]** were reproduced today by running
code (nothing in the repo was changed except this file). The others are read from the source.

## Overview

`sonnix` is a library of PyTorch building blocks that `torch.nn` does not provide: losses, activations,
numerical-feature encoders, a few layers. The code is in good shape: every public object has a
docstring and an example, the numerical edge cases of the previous review are fixed, and the tests are
registry-driven. What is left is not a bug list, it is **inconsistency** and **surface area**:

- the same rule is applied in some places and not in others (shape check, validation of a
  hyperparameter, positional vs keyword-only, `x` vs `input`);
- about a quarter of the public classes are compositions of other public classes
  (`AsinhMSELoss` = `TransformedLoss(MSELoss, Asinh, Asinh)`; seven indicator modules that only call
  a function), which each cost a function, a module, a docstring, a test file and a registry entry;
- the package ships things that are not nn building blocks (dataset/optimizer factories, a training
  loop helper, a dataset) next to the library, with no layering rule for them.

The target is a smaller package where each rule below is true **everywhere**, enforced by a test, and
where each public name earns its place.

## Constraints

1. **Simple and easy to maintain:** delete code instead of adding to it. A public object that is a
   one-line composition of two other public objects is removed and documented as a recipe.
2. **Breaking changes are fine:** the version is `0.0.1a0`. No compatibility shim, no deprecation
   period. Each breaking change is listed in `CHANGELOG.md` (section "Breaking changes") with the PR
   number.
3. **Same numerics:** no loss or activation changes its value, except where a finding below says it
   is a bug. Every numerical change comes with a regression test that fails without the fix.
4. **Follow `CLAUDE.md` and `docs/docs/conventions.md`:** `coola.validation` for argument checks,
   keyword-only options, buffers for frozen tensors, `device`/`dtype` for modules that create tensors,
   docstring and test rules.
5. **`torch` is the only hard dependency besides `coola`:** no other library is imported at module
   level.
6. **No slower than today:** the forward passes are the same tensor operations. The only added work
   is shape/argument checks, which are Python-level and `torch.compile`-friendly (no data-dependent
   branch).

## 1. Properties of the package

Each property is a rule that holds for **every** public object, the object that enforces it, and the
test that proves it. A property without a test is a wish.

| #   | Property                           | Rule                                                                                                                                                                                                                                   | Enforced by                                                                   |
| --- | ---------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------- |
| P1  | **Function/module duality**        | Every loss and every parametrised op with numerical content exists as a stateless function in `sonnix.functional` and as a module in `sonnix.modules` that stores the hyperparameters and calls the function. The module has no logic. | registry test: module output equals function output for the same inputs       |
| P2  | **Validate once, fail early**      | The function validates all its arguments (`coola.validation` or a `sonnix.validation` helper). The module calls the **same** helper in `__init__`, so a bad value fails at construction with the same error as the function.           | parity test over the registry (exists: `test_loss_validation_parity`)         |
| P3  | **Uniform call signature**         | Losses: `(prediction, target, *, <options>, reduction)`. Single-tensor ops: `(input, *, <options>)`. All options are keyword-only in functions; in modules the one defining hyperparameter may be positional (as in `torch.nn`).       | `test_keyword_only_options` (exists), extended to the functions               |
| P4  | **Strict shapes**                  | A function with two tensors of the same role (prediction/target, loss/indicator) requires equal shapes and raises `ValueError` naming both shapes. Nothing broadcasts silently.                                                        | one `validate_same_shape` call per function; parametrised test                |
| P5  | **One reduction contract**         | Every loss accepts `reduction` in `none`/`mean`/`sum`/`batchmean`, validated by `reduce_loss`; default `mean`; `none` keeps the shape of the elementwise loss.                                                                         | `reduce_loss`; registry test over all losses                                  |
| P6  | **Numerically safe**               | Outputs are finite for finite inputs of any magnitude that the dtype can represent, in `float16`/`bfloat16`/`float32`; gradients are finite (no `nan` at 0 or at a clamp). Documented exceptions are in the docstring `Note`.          | half-precision test, `gradcheck`, finite-gradient test (exist)                |
| P7  | **Pure**                           | A function or a forward never mutates its inputs and has no hidden state. A module's only state is its parameters and buffers.                                                                                                         | in-place-mutation test over the registry (new)                                |
| P8  | **Frozen = buffer**                | Tensors that are not trained are buffers (`register_buffer`), never `Parameter(requires_grad=False)`. `learnable=False` gives a buffer with the same name, so the `state_dict` keys do not depend on the flag.                         | registry test (`parameters()` vs `buffers()`)                                 |
| P9  | **Device and dtype**               | A module that creates its own tensors takes `device` and `dtype`; it follows `.to()`/`.half()`; the output dtype is the input dtype.                                                                                                   | dtype/device tests (exist)                                                    |
| P10 | **Export and compile**             | Every module in the registry passes `torch.export` and `torch.compile(fullgraph=True)` on CPU. No `.item()`, no data-dependent Python branch in a forward.                                                                             | registry tests (exist)                                                        |
| P11 | **Introspectable**                 | The hyperparameters are public attributes with the name of the constructor argument; `repr` shows them (`extra_repr`).                                                                                                                 | `test_hyperparameters` (exists)                                               |
| P12 | **One public path**                | The public API of a package is what its `__init__.py` exports (`sonnix.functional`, `sonnix.modules`, `sonnix.utils`, `sonnix.validation`, `sonnix.factory`, `sonnix.testing`). Docs, docstrings and examples import only from there.  | `test_imports`; a docstring-import lint in the doctest job                    |
| P13 | **Layered imports**                | `modules → functional → validation`, `modules → factory`; `utils` and `factory` depend on no other sonnix package; `testing` depends on all and nothing depends on it. No cycle.                                                       | import-linter contracts (extended in step 1)                                  |
| P14 | **Configuration is trusted input** | `sonnix.factory` imports and **calls** a `_target_` before checking the type of the result. Any argument typed `Module \| dict` carries the same warning. `is_*_config` is the check that does not call the target.                    | docstring `Warning` (exists); one test that every `Module \| dict` arg has it |
| P15 | **Optional means lazy**            | Only `torch` and `coola` are imported at module level. Anything else is guarded.                                                                                                                                                       | `package_checks` job (exists)                                                 |

Not guaranteed: bit-exact equality across torch versions, `float64` stability beyond what the tests
cover, behaviour on non-CPU devices in CI (no GPU job; the CUDA tests run locally before a release),
learnable `alpha` that crosses `0` (documented in the numerical-stability guide).

## 2. What differs today

The previous review's numerical findings are fixed. The findings below are about rules that are not
applied everywhere (P1-P4, P12, P13) and about surface area.

| #   | Finding                                                                                                                                                                                                                                                                                                                                                                 | Evidence                                                               |
| --- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------- |
| F1  | **Shape policy is inconsistent (P4).** Two tensors of different shape give a `ValueError` in `relative_loss` and in the error functions, a bare torch `RuntimeError` in `log_cosh_loss`, the indicators, the fusion modules and the encoders, a `UserWarning` plus a broadcast result in `asinh_mse_loss`, and **a silent wrong result** in `quantile_regression_loss`. | [probe] `quantile_regression_loss(x[4,3], y[1,3])` returns a scalar    |
| F2  | **Hyperparameters are not validated everywhere (P2).** `BinaryPoly1Loss(alpha=nan)` and `alpha=-1` are accepted; `SafeExp(max_val=-1)` and `max_val=nan` are accepted (`safe_log` validates `min_val`, `safe_exp` does not); `Squeeze(dim=1.5)` and `View(("a",))` are accepted; `Pow("a")` fails with a message from `float()`.                                        | [probe]                                                                |
| F3  | **Positional vs keyword-only is inconsistent (P3).** `log_cosh_loss(p, t, scale)`, `binary_focal_loss(p, t, alpha, gamma)`, `quantile_regression_loss(p, t, q)`, `poisson_regression_loss(p, t, eps)`, `relative_loss(l, i, eps)` take options positionally, while `reduction` is keyword-only. `conventions.md` says "Options are keyword-only".                       | `src/sonnix/functional/loss/*.py`                                      |
| F4  | **`x` vs `input` vs `scalar` vs `scores` for the one tensor of a forward.** 24 forwards use `input` (each with `# noqa: A002`, 29 in total), 7 use `x` (`NLinear`, `ScaleAndShift`, `DynamicTanh`, `DynamicAsinh`, `View`), the four encoders use `scalar`, the two label modules `scores`. `conventions.md` documents `input` for activations only.                    | `grep "def forward" src`                                               |
| F5  | **Compositions are public classes.** `AsinhMSELoss`, `AsinhSmoothL1Loss`, `MSLELoss`, `RelativeMSELoss`, `RelativeSmoothL1Loss` and their functions are `TransformedLoss`/`RelativeLoss` over a `torch` criterion. Each costs a function, a module, a docstring with `Args`, a registry entry, a parity test.                                                           | 5 modules + 3 functions, ~700 lines with docs                          |
| F6  | **Eight indicator modules call one function each** (`BaseRelativeIndicator` + 7 classes, 266 lines). They have no parameters (except `k`), no state, and are only consumed by `RelativeLoss`, which could take a callable.                                                                                                                                              | `modules/loss/indicators.py`                                           |
| F7  | **`ScaleAndShift` is inherited only to reuse two lines.** `BaseDynamicNorm(ScaleAndShift)` needs `ScaleAndShift.reset_parameters(self)` called explicitly in `__init__` (with a comment) because the subclass overrides `reset_parameters`. `DynamicTanh` _is not_ a `ScaleAndShift` in any useful sense (different forward, extra parameter).                          | `modules/scale_and_shift.py:76-86`                                     |
| F8  | **Not nn building blocks, no layering.** `sonnix.factory.dataset`/`optimizer` are one-line wrappers of `coola.factory.resolve_object` that nothing in `src` uses; `sonnix.utils.loss` (`is_loss_decreasing*`) is a test helper; `DummyDataset` is in `testing`. `modules` imports `factory`, but the import-linter contract does not know `factory`.                    | README "Training and testing helpers (not used by the modules...)"     |
| F9  | **Dead or single-use utilities.** `utils.iterator.get_named_modules` (yields a `"[root]"` pseudo-name) and `utils.random.resolve_torch_generator` are used by no `src` module except one private helper; `top_module_mode` is used by nothing but its test; `get_module_devices` and `is_module_on_device` only by tests.                                               | greps (src / tests / docs)                                             |
| F10 | **Two public import paths per class.** `sonnix.modules.ClassicalRelativeIndicator` and `sonnix.modules.loss.ClassicalRelativeIndicator` are both exported and both used in docstrings; the docs reference only the flat path. The sub-packages' `__all__` duplicate the 80-name list of `modules/__init__.py`.                                                          | `modules/__init__.py`, `modules/loss/__init__.py`                      |
| F11 | **The reduction values are written twice.** `Reduction = Literal[...]` in `functional/reduction.py` and `VALID_REDUCTIONS = (...)` in `validation/loss.py`. Adding a strategy means changing both, and nothing checks that they agree.                                                                                                                                  | read from code                                                         |
| F12 | **`p_t` / `alpha_t` are recomputed in four functions** (focal ×2, Poly-1 ×2), each with the same 3-line comment about soft targets.                                                                                                                                                                                                                                     | `functional/loss/focal.py`, `poly.py`                                  |
| F13 | **The three `create` class methods repeat the range logic.** `AsinhNumericalEncoder.create` and `CosSinNumericalEncoder.create` each re-implement "exactly one of `*_range` / `value_range`", the `1/value_range` conversion and the min/max check. `on_duplicate` is typed `str` where a `Literal` validated by `validate_in` is meant.                                | `numerical/asinh.py`, `numerical/sine.py`, `numerical/piecewise.py`    |
| F14 | **The encoders do not check the input shape.** `NLinear` raises a `ValueError` naming `n`; the encoders let torch raise a broadcast `RuntimeError` when the last dimension is not `n_features`, and silently broadcast when `n_features == 1`.                                                                                                                          | [probe] `AsinhNumericalEncoder(ones(2,3))(rand(5,4))` → `RuntimeError` |
| F15 | **`View` is not a view.** It calls `reshape` on the **whole** tensor, including the batch dimension, so `View((4, 8))` fails on any other batch size. `nn.Flatten`/`nn.Unflatten` cover the useful cases.                                                                                                                                                               | `modules/shape.py`                                                     |
| F16 | **`MulticlassFlatten` is a loss wrapper in `shape.py`.** It wraps a criterion exactly like `TransformedLoss`, but lives with the shape helpers and is listed under "Layers" in the docs.                                                                                                                                                                                | `modules/shape.py`, `docs/docs/refs/modules/layers.md`                 |
| F17 | **`Clamp` raises by hand** (`min_val > max_val`) and `ToBinaryLabel(threshold)` / `Pow(exponent)` / `ReLUn(max_val)` / `SafeExp(max_val)` check nothing or call `float()` only. The rule "validate with `coola.validation`" is applied to roughly half of the constructors.                                                                                             | `activation/clamp.py`, `activation/math.py`, `classification.py`       |
| F18 | **`BaseAlphaActivation` encodes validity in two class flags** (`allow_zero_init`, `allow_negative_init`) read in `__init__`. It works, but the rule is invisible in the subclass signature and the docstrings repeat it.                                                                                                                                                | `activation/alpha.py`                                                  |
| F19 | **Docstring `Args:` blocks are copied** into every subclass (focal ×3, Poly-1 ×3, relative ×3, alpha ×6). The copies drift (`Snake`'s `init`).                                                                                                                                                                                                                          | read from code                                                         |

## 3. Architecture

### 3.1 Packages and dependencies

```text
sonnix
├── validation/   argument validators and the Reduction type        (depends on: coola)
├── functional/   stateless ops: activation, error, loss, reduction (depends on: validation)
├── factory/      resolve_module, is_module_config, create_sequential (depends on: coola, torch)
├── modules/      nn.Module wrappers and layers                    (depends on: functional, factory, validation)
│   ├── activation/  loss/  numerical/  fusion/                       (topic folders, not public paths)
├── utils/        module inspection helpers                         (depends on: torch)
└── testing/      DummyDataset, is_loss_decreasing*                 (depends on: anything, nothing depends on it)
```

Import-linter contracts after the change:

```toml
[[tool.importlinter.contracts]]
name = "Layers"
type = "layers"
layers = ["sonnix.testing", "sonnix.modules", "sonnix.functional", "sonnix.validation"]

[[tool.importlinter.contracts]]
name = "Independent packages"
type = "independence"
modules = ["sonnix.utils", "sonnix.factory", "sonnix.validation"]
```

`functional` must not import `factory` or `modules`; `utils` and `factory` import no other sonnix
package. This also makes the (currently untested) `modules → factory` edge explicit (F8).

### 3.2 The function/module pair (P1-P3)

A loss function:

```python
def log_cosh_loss(
    prediction: Tensor, target: Tensor, *, scale: float = 1.0, reduction: Reduction = "mean"
) -> Tensor:
    validate_gt(scale, 0, name="scale")
    validate_same_shape(prediction, target, name1="prediction", name2="target")
    ...
    return reduce_loss(loss, reduction)
```

Its module:

```python
class LogCoshLoss(BaseLoss):
    def __init__(self, scale: float = 1.0, *, reduction: Reduction = "mean") -> None:
        super().__init__(reduction=reduction)  # validates the reduction
        validate_gt(scale, 0, name="scale")  # same check as the function
        self.scale = float(scale)

    def forward(self, prediction: Tensor, target: Tensor) -> Tensor:
        return log_cosh_loss(prediction, target, scale=self.scale, reduction=self.reduction)
```

Rules:

- The function validates everything, in this order: hyperparameters, then tensors.
- A check shared by a function and a module lives in `sonnix.validation` (`validate_focal_params`,
  `validate_general_robust_params`, ...). A check used by one function only is inline
  (`validate_gt(scale, 0, name="scale")`). A validator that is one `coola.validation` call is not
  created (`CLAUDE.md`).
- The module's `forward` is one call. If a `forward` needs a second line, the logic belongs in the
  function.
- `extra_repr` shows the hyperparameters in the order of the constructor.

### 3.3 Shapes (P4)

`validate_same_shape(a, b, name1=..., name2=...)` is called by every function that takes
`(prediction, target)`, `(loss, indicator)` or two tensors of the same role. It is already used by the
error functions and `relative_loss`; the target adds it to the 14 functions in F1. The exception is
the criterion of `TransformedLoss`/`RelativeLoss` (a `torch` loss, whose rules are not ours).

`NLinear` and the encoders validate the last dimensions of their input and name the argument
(`n`, `n_features`) in the message:

```text
ValueError: Expected an input of shape (*, n_features=3) but received (5, 4)
```

`n_features == 1` does **not** broadcast: `create(..., n_features=k)` builds the shared encoding
explicitly (F14).

### 3.4 Losses

A loss exists as a function **and** a module only if it has numerical content that is not the
composition of a torch criterion and a transform.

| Loss                                     | Function                              | Module                                               | Notes                                                      |
| ---------------------------------------- | ------------------------------------- | ---------------------------------------------------- | ---------------------------------------------------------- |
| Log-cosh                                 | `log_cosh_loss`                       | `LogCoshLoss`                                        | stable `softplus` form                                     |
| General robust (Barron)                  | `general_robust_regression_loss`      | `GeneralRobustRegressionLoss`                        | `alpha` ∈ ℝ ∪ {-inf}, `scale > 0`, `max_val > 0` or `None` |
| Quantile                                 | `quantile_regression_loss`            | `QuantileRegressionLoss`                             | `q` ∈ [0, 1]                                               |
| Poisson                                  | `poisson_regression_loss`             | `PoissonRegressionLoss`                              | `log_input` option                                         |
| Binary focal (probabilities / logits)    | `binary_focal_loss[_with_logits]`     | `BinaryFocalLoss[WithLogits]`                        | soft targets supported                                     |
| Binary Poly-1 (probabilities / logits)   | `binary_poly1_loss[_with_logits]`     | `BinaryPoly1Loss[WithLogits]`                        | `alpha` finite (may be negative: it is the Poly-1 `ε`)     |
| Relative (criterion / indicator)         | `relative_loss` + indicator functions | `RelativeLoss`                                       | indicator is a callable, see 3.5                           |
| Composition: transform then criterion    | none                                  | `TransformedLoss`                                    | replaces `AsinhMSELoss`, `AsinhSmoothL1Loss`, `MSLELoss`   |
| Composition: flatten the class dimension | none                                  | `MulticlassFlatten` (moved to `sonnix.modules.loss`) | wraps a criterion, like `TransformedLoss`                  |

Removed (F5): `asinh_mse_loss`, `asinh_smooth_l1_loss`, `msle_loss`, `AsinhMSELoss`,
`AsinhSmoothL1Loss`, `MSLELoss`, `RelativeMSELoss`, `RelativeSmoothL1Loss`. The recipes go in the
losses guide (`docs/docs/guide/losses.md`, section "Composing losses"):

```python
from torch import nn
from sonnix.modules import Asinh, Log1p, RelativeLoss, TransformedLoss

asinh_mse = TransformedLoss(nn.MSELoss(), prediction_transform=Asinh(), target_transform=Asinh())
msle = TransformedLoss(nn.MSELoss(), prediction_transform=Log1p(), target_transform=Log1p())
relative_mse = RelativeLoss(nn.MSELoss(reduction="none"), indicator=classical_relative_indicator)
```

`BaseLoss(*, reduction)` stays: it validates and stores `reduction` and gives the shared `extra_repr`.
`TransformedLoss` stays a plain `nn.Module` (it has no `reduction`: its criterion owns it).

`sonnix.functional.loss.*` gets one private module `_binary.py` with `_probability_of_target(prob,
target)` and `_class_weight(alpha, target)` used by the focal and Poly-1 functions (F12). Four copies
of the soft-target comment become one.

### 3.5 Relative indicators

The indicator is a function `(prediction, target) -> Tensor` that returns a non-negative tensor of
the same shape. The seven functions stay in `sonnix.functional` (they have numerical content: the
scaled `moment_mean_indicator`, the `safe_pow` geometric mean). The seven module classes and
`BaseRelativeIndicator` are deleted (F6). `RelativeLoss` takes a callable or a configuration:

```python
RelativeLoss(
    criterion=nn.SmoothL1Loss(reduction="none"),
    indicator=geometric_mean_indicator,  # default: classical_relative_indicator
)
RelativeLoss(criterion, indicator=partial(moment_mean_indicator, k=2))
```

`RelativeLoss.indicator` is stored as a plain attribute (a function has no parameters, so it does not
need to be a submodule). It is validated with `callable(...)`: `validate_callable` if `coola` has one,
otherwise `validate_isinstance(indicator, Callable)`. This is decision D2.

### 3.6 Activations and layers

- **Elementwise math modules** (`Asinh`, `Exp`, `Expm1`, `Log`, `Log1p`, `Sin`, `Sinh`, `Square`, `Pow`):
  stay. They are two lines each and exist because `torch.nn` has no module for them, which
  `nn.Sequential` and configurations need. `Square` stays an independent class (it already is).
  `Pow(exponent)` validates with `validate_finite`.
- **`SafeExp` / `SafeLog`:** the functions validate `max_val` (finite) and `min_val` (> 0); the
  modules call the same validators (F2).
- **Alpha activations:** keep `BaseAlphaActivation` and the six classes. Replace the two class flags
  by one overridable method `_validate_init(init)` called by `__init__` (default: finite). `Gaussian`
  and `Snake` reject `0`, `Laplacian` rejects `<= 0`. The rule is then next to the class that needs it
  and the error message is specific (F18).
- **`ScaleAndShift` / `DynamicTanh` / `DynamicAsinh`:** `BaseDynamicNorm` stops inheriting from
  `ScaleAndShift`. It owns `self.affine = ScaleAndShift(...)` and one `alpha` parameter; `forward` is
  `self.affine(self._activation(x * self.alpha))`. `_affine` and the explicit
  `ScaleAndShift.reset_parameters(self)` disappear (F7). `state_dict` keys change from `weight` to
  `affine.weight` (breaking, accepted).
- **`ExU`, `NLinear`, `ResidualBlock`:** unchanged except `forward(self, input)`.
- **Fusion:** unchanged (`BaseFusion.fuse(inputs)`), `AverageFusion.fuse` returns
  `SumFusion().fuse(inputs) / len(inputs)` through a shared private helper, so the sum loop exists once.
- **Shape and dtype:** `Squeeze` validates `dim` (int, tuple of ints or `None`); `View` is deleted
  (F15, decision D3); `ToDtype`, `ToFloat`, `ToLong` stay (`ToFloat`/`ToLong` exist because a
  configuration cannot write a `torch.dtype`); `ToBinaryLabel`, `ToCategoricalLabel` stay and take
  `input`.
- **`Clamp`:** keeps its manual `min_val <= max_val` check, since `coola.validation` cannot express a
  relation between two arguments. It is the one allowed hand-written check and it has a test.

### 3.7 Numerical encoders

One private helper in `numerical/_utils.py`:

```python
def resolve_range(
    name: str, range_: tuple[float, float] | None, value_range: tuple[float, float] | None
) -> tuple[float, float]:
    """Return ``(min, max)`` of the parameter, from its own range or from the value range."""
```

It raises the "exactly one of" error, validates `value_range` with `validate_abs_range`, converts
`value_range` to `(1 / max, 1 / min)`, and checks `0 < min <= max` (F13). Both `create` methods call
it. `on_duplicate` becomes `Literal["warn", "ignore"]`. The four encoders share `n_features`,
`feature_size`, `learnable`, `extra_repr`, and the input check of 3.3 through a small private base
class `_NumericalEncoder(nn.Module)` that implements `forward` as
`self._encode(self._check(input))`. Public classes are unchanged:

| Class                             | Parameters (buffer or parameter)          | Output shape                             |
| --------------------------------- | ----------------------------------------- | ---------------------------------------- |
| `AsinhNumericalEncoder`           | `scale (n_features, feature_size)`        | `(*, n_features, feature_size)`          |
| `CosSinNumericalEncoder`          | `frequency`, `phase_shift` (same shape)   | `(*, n_features, 2 * n_frequencies)`     |
| `AsinhCosSinNumericalEncoder`     | as `CosSin`                               | `(*, n_features, 2 * n_frequencies + 1)` |
| `PiecewiseLinearNumericalEncoder` | `edges`, `width` (+ non-persistent masks) | `(*, n_features, n_bins)`                |

### 3.8 `sonnix.utils`, `sonnix.factory`, `sonnix.testing`

| Package             | Target content                                                                                                                                                                                                                               | Removed / moved                                                                                                                                                                             |
| ------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `sonnix.utils`      | `get_module_device`, `get_module_devices`, `is_module_on_device`; `module_mode`; `num_parameters`, `num_learnable_parameters`, `has_parameters`, `has_learnable_parameters`, `freeze_module`, `unfreeze_module`; `load_state_dict_to_module` | `top_module_mode`, `iterator.get_named_modules`, `random.resolve_torch_generator` (private copy in `numerical/_utils.py`), `find_module_state_dict` (made private), `loss.*` (to `testing`) |
| `sonnix.factory`    | `resolve_module`, `is_module_config`, `create_sequential`                                                                                                                                                                                    | `dataset.py`, `optimizer.py` (use `coola.factory.resolve_object` directly)                                                                                                                  |
| `sonnix.testing`    | `DummyDataset`, `is_loss_decreasing`, `is_loss_decreasing_with_adam`, `is_loss_decreasing_with_sgd`                                                                                                                                          | —                                                                                                                                                                                           |
| `sonnix.validation` | the validators of today, plus `Reduction` and `VALID_REDUCTIONS` defined **once** (`VALID_REDUCTIONS = typing.get_args(Reduction)`), re-exported by `sonnix.functional` (F11)                                                                | `validate_moment_order` is kept only if `moment_mean_indicator` and a module both need it; with the indicator modules gone it is inlined (`validate_int` + `validate_ge`)                   |

`load_state_dict_to_module` and its prefix search are kept: it is the only piece of `utils` that solves
a problem `torch` does not (checkpoints wrapped in a dict, `module.` / `_orig_mod.` prefixes). The
`find_module_state_dict` / `_strip_prefix` pair is a private implementation detail.

`is_loss_decreasing*` moves to `testing` because it is a test of a module, not an operation on one.
`testing` is then "everything a user needs to test their own module"; it is the right home for the
README's "Training and testing helpers" paragraph, which can be deleted from the Overview.

### 3.9 Public paths (P12)

- `sonnix.modules.__all__` is the only public list of module classes. The `__init__.py` of
  `modules/activation`, `modules/loss`, `modules/numerical`, `modules/fusion` keep their imports (they
  are needed to build the flat namespace) but have **no `__all__`** and are not documented. Docstrings
  and examples import from `sonnix.modules` / `sonnix.functional`.
- `sonnix.functional.__all__` stays flat for the same reason.
- `tests/unit/test_imports.py` additionally asserts that every name in `sonnix.modules.__all__` is
  defined in `sonnix.modules` and that no doctest imports from a sub-package.

## 4. Public API after the change

```python
# sonnix.functional
Reduction, reduce_loss
absolute_error, absolute_relative_error, symmetric_absolute_relative_error
safe_exp, safe_log, safe_pow, rectifier_asinh_unit
binary_focal_loss, binary_focal_loss_with_logits
binary_poly1_loss, binary_poly1_loss_with_logits
general_robust_regression_loss, log_cosh_loss, poisson_regression_loss, quantile_regression_loss
relative_loss
(
    arithmetical_mean_indicator,
    classical_relative_indicator,
    geometric_mean_indicator,
)
maximum_mean_indicator, minimum_mean_indicator, moment_mean_indicator, reversed_relative_indicator

# sonnix.modules
# losses
(
    BaseLoss,
    BinaryFocalLoss,
    BinaryFocalLossWithLogits,
    BinaryPoly1Loss,
    BinaryPoly1LossWithLogits,
)
(
    GeneralRobustRegressionLoss,
    LogCoshLoss,
    PoissonRegressionLoss,
    QuantileRegressionLoss,
)
RelativeLoss, TransformedLoss, MulticlassFlatten
# activations
(
    Asinh,
    Clamp,
    Exp,
    Expm1,
    Log,
    Log1p,
    Pow,
    ReLUn,
    RectifierAsinhUnit,
    SafeExp,
    SafeLog,
)
(
    Sin,
    Sinh,
    Square,
    SquaredReLU,
    BaseAlphaActivation,
    ExpSin,
    Gaussian,
    Laplacian,
)
MultiQuadratic, Quadratic, Snake, BaseDynamicNorm, DynamicAsinh, DynamicTanh
# layers
ExU, NLinear, ResidualBlock, ScaleAndShift
# fusion
BaseFusion, AverageFusion, ConcatFusion, MultiplicationFusion, SumFusion
# encoders
(
    AsinhNumericalEncoder,
    AsinhCosSinNumericalEncoder,
    CosSinNumericalEncoder,
    PiecewiseLinearNumericalEncoder,
)
# shape, dtype, labels
Squeeze, ToDtype, ToFloat, ToLong, ToBinaryLabel, ToCategoricalLabel

# sonnix.factory:    resolve_module, is_module_config, create_sequential
# sonnix.validation: Reduction, VALID_REDUCTIONS, validate_loss_reduction_strategy, validate_same_shape,
#                    validate_focal_params, validate_general_robust_params, validate_abs_range, validate_frequency
# sonnix.utils:      see 3.8
# sonnix.testing:    see 3.8
```

Counts: `modules.__all__` 80 → 63 names; `functional.__all__` 28 → 25; `factory` 7 → 3;
`utils` 20 functions → 14.

### Migration table

| Before                                                           | After                                                                                   |
| ---------------------------------------------------------------- | --------------------------------------------------------------------------------------- |
| `log_cosh_loss(p, t, 2.0)`                                       | `log_cosh_loss(p, t, scale=2.0)` (every function option is keyword-only)                |
| `AsinhMSELoss()`                                                 | `TransformedLoss(nn.MSELoss(), prediction_transform=Asinh(), target_transform=Asinh())` |
| `AsinhSmoothL1Loss(beta=0.5)`                                    | the same with `nn.SmoothL1Loss(beta=0.5)`                                               |
| `MSLELoss()`                                                     | the same with `Log1p()`                                                                 |
| `RelativeMSELoss(indicator=GeometricMeanIndicator())`            | `RelativeLoss(nn.MSELoss(reduction="none"), indicator=geometric_mean_indicator)`        |
| `MomentMeanIndicator(k=2)`                                       | `partial(moment_mean_indicator, k=2)`                                                   |
| `sonnix.modules.loss.X`                                          | `sonnix.modules.X`                                                                      |
| `DynamicTanh(...).weight` / state dict key `weight`              | `.affine.weight` / key `affine.weight`                                                  |
| `forward(x=...)` / `forward(scalar=...)` / `forward(scores=...)` | `forward(input=...)` (positional calls are unchanged)                                   |
| `View(shape)`                                                    | `nn.Flatten`, `nn.Unflatten`, or `x.reshape` in the model's `forward`                   |
| `sonnix.utils.loss.is_loss_decreasing*`                          | `sonnix.testing.is_loss_decreasing*`                                                    |
| `sonnix.factory.resolve_dataset` / `resolve_optimizer`           | `coola.factory.resolve_object(config, Dataset)` / `(config, Optimizer)`                 |
| `utils.iterator.get_named_modules`, `top_module_mode`            | removed (`module.named_modules()`, `module_mode`)                                       |
| `utils.random.resolve_torch_generator`                           | `coola.random.make_torch_generator`                                                     |
| `BinaryPoly1Loss(alpha=nan)`, `SafeExp(max_val=-1)`, ...         | `ValueError`                                                                            |
| `quantile_regression_loss(p[4,3], t[1,3])`                       | `ValueError` (shapes differ); broadcast explicitly with `expand`                        |

## 5. Dependents

| Consumer                      | Effect                                                                                                                                                                                                                                               |
| ----------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `tests/registry.py`           | 8 entries removed (compositions) + 7 indicator entries; new entries for none (the registry shape is unchanged).                                                                                                                                      |
| `docs/docs/refs/*`            | `modules/loss.md` loses 7 sections and the indicator table; `layers.md` loses `View` and `MulticlassFlatten` (moved to loss.md); `utils.md` loses `iterator` and `random`; `factory.md` loses two sections; `testing.md` gains the training helpers. |
| `docs/docs/guide/losses.md`   | the "Composing losses" section gets the recipes of 3.4 and 3.5.                                                                                                                                                                                      |
| `docs/docs/conventions.md`    | adds P3 (function options keyword-only, module defining hyperparameter positional), `input` for every single-tensor forward, P4.                                                                                                                     |
| `README.md`                   | the Overview drops the paragraph "Training and testing helpers (not used by the modules...)" and the list of removed classes.                                                                                                                        |
| `notebooks/activations.ipynb` | uses activations only; unaffected (verify with the notebook CI job).                                                                                                                                                                                 |
| `CHANGELOG.md`                | one "Breaking changes" entry per step of section 7, with the PR number.                                                                                                                                                                              |

## 6. Evidence

Numbers from `main` at `fc04eeb` (counts from the source, probes from today):

| Measure                                                        | Today                                                            | Target (estimate) |
| -------------------------------------------------------------- | ---------------------------------------------------------------- | ----------------- |
| Public classes in `sonnix.modules`                             | 72 (80 names incl. base/alias)                                   | ~55               |
| Public functions in `sonnix.functional`                        | 27                                                               | 24                |
| `# noqa: A002` in `src`                                        | 29                                                               | 0 (D4) or 29      |
| Functions of two same-role tensors without a shape check       | 14                                                               | 0                 |
| Constructors that take a numeric option and do not validate it | ≥ 9 (`Pow`, `SafeExp`, `Poly1`, `Squeeze`, `ToBinaryLabel`, ...) | 0                 |
| Lines of `src`                                                 | 7 815                                                            | ~6 600            |
| Lines of `tests`                                               | 13 537                                                           | ~12 000           |

No benchmark gate is needed: the forward passes are unchanged tensor operations. The one performance
risk is the extra shape check, a Python-level comparison of two `torch.Size`, a few hundred
nanoseconds. `tests/benchmark/test_nlinear.py` stays the only benchmark; step 3 adds one for
`log_cosh_loss` on a small tensor to show the check is not measurable.

## 7. Changes to reach the target state

One branch and one pull request per step, small and reviewable on its own. Steps 2-5 are independent
of each other and can be opened from `main` in parallel; 6 depends on 4; 7 and 8 on 2 and 5. Each
step updates `CHANGELOG.md` (with the PR number), the docs it touches and the tests, and runs
`ruff`, `inv types.check --complete`, `pytest tests`, `xdoctest src/sonnix all`.

1. **Properties and layering (non-breaking).** Add `docs/docs/conventions.md` section "Design rules"
   (P1-P15, shortened). Extend import-linter with `factory` and the independence contract (3.1). Add
   the P14 test (every `Module | dict` argument has the warning) and the in-place-mutation registry
   test (P7). _Why first: it makes the rules checkable before the code changes._
2. **Validation gaps and shape policy (breaking, bug-level).** F1, F2, F17: `validate_same_shape` in
   the 14 functions; `validate_finite` on `alpha` (Poly-1), `max_val` (`SafeExp`), `exponent` (`Pow`),
   `threshold` (`ToBinaryLabel`); `validate_int` on `Squeeze.dim`; modules call the same validators.
   Regression tests that fail without the fix (`git stash push src`): one per row of F1 and F2, driven
   by the registry so the function and the module are checked together.
3. **Keyword-only function options (breaking).** F3: add `*` after the tensors in the 10 functions;
   extend `test_keyword_only_options` to `sonnix.functional` with `inspect.signature`.
4. **Delete compositions (breaking).** F5, F6, F16: remove the 8 losses/functions of 3.4, the 7
   indicator modules and `BaseRelativeIndicator`; `RelativeLoss` takes a callable; move
   `MulticlassFlatten` to `modules/loss/`; add the recipes to the guide. Largest deletion; no new code
   except the callable check.
5. **`input` everywhere (breaking in keyword calls).** F4: rename the parameter of `NLinear`,
   `ScaleAndShift`, `DynamicTanh/Asinh`, the four encoders, the two label modules. Decision D4 on
   the `A002` setting is applied here.
6. **Dynamic norms by composition (breaking: state dict).** F7: `BaseDynamicNorm` owns a
   `ScaleAndShift`. Delete `_affine` and the explicit `reset_parameters` call. Add a test that
   `DynamicTanh().state_dict().keys()` equals the documented set.
7. **Encoders (non-breaking except F14 error type).** F13, F14: `resolve_range`, `Literal` for
   `on_duplicate`, the `_NumericalEncoder` base with the input check; tests for the new `ValueError`
   and the `n_features == 1` no-broadcast rule.
8. **Packages (breaking).** F8-F11: move `utils.loss` to `testing`; delete `factory.dataset`,
   `factory.optimizer`, `utils.iterator`, `top_module_mode`; privatise `resolve_torch_generator`
   and `find_module_state_dict`; define `Reduction` once in `validation`; remove the sub-package
   `__all__` lists (3.9); update the docs navigation and the README.
9. **Alpha activations and small cleanups.** F18, F12, F19: `_validate_init` hook; `_binary.py`
   helpers for focal/Poly-1; `View` deleted (D3).
10. **Documentation and migration notes.** Final pass on `README.md`, the guide pages, the
    `CHANGELOG.md` "Breaking changes" summary with the migration table of section 4.

Steps 2 and 3 are the only ones that change behaviour users can hit with correct code; the rest only
removes names.

## 8. Tests and gates

- **Parity (P1, P2):** for every loss in the registry, the module output equals the function output,
  and the same invalid value raises the same exception type and message in both (exists; extended to
  the new validators).
- **Shapes (P4):** one parametrised test over all two-tensor functions: mismatched shapes raise
  `ValueError` with `match=r"must have the same shape"`; matched shapes do not.
- **Signatures (P3):** `inspect.signature` check that every public function has no
  positional-or-keyword parameter after the tensor arguments; every module's options after the first
  are keyword-only.
- **Reduction (P5):** over all losses: `none` returns the elementwise shape, `sum` equals
  `none.sum()`, `mean` equals `none.mean()`, `batchmean` equals `none.sum() / N`.
- **Numerics (P6):** the half-precision value test and the finite-gradient test cover every new or
  changed function; a regression test per bug of step 2.
- **State (P7, P8):** inputs are unchanged after forward (clone and compare); `learnable=False`
  gives buffers with the same state-dict keys as `learnable=True`.
- **Export/compile (P10):** unchanged registry tests; they run for the dynamic norms after step 6.
- **Imports (P12, P13):** `test_imports` walks every module; `lint-imports` runs the extended
  contracts.
- **Docs:** `xdoctest src/sonnix all`, `mkdocs build --strict`, the notebook job.
- **Unit-test rules of `CLAUDE.md`** apply to everything added: one behaviour per test, `match` on
  every `pytest.raises`, seeded data, no local imports, 100 % coverage of the lines added or changed,
  a regression test for every bug fix, and an integration test for every patched behaviour.

## 9. Decisions for the maintainers

| #   | Decision                                                                                                                                              | Recommendation                                                                                                                                                                                                                                                                                      |
| --- | ----------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| D1  | Delete the composition losses (`AsinhMSELoss`, `AsinhSmoothL1Loss`, `MSLELoss`, `RelativeMSELoss`, `RelativeSmoothL1Loss`) and their functions (3.4)? | **Yes.** They are `TransformedLoss`/`RelativeLoss` with a torch criterion. Keep them only if the library's identity is a catalogue of named losses; then keep the _modules_ and drop the _functions_.                                                                                               |
| D2  | Indicators as plain callables instead of modules (3.5)?                                                                                               | **Yes.** They have no parameters. The cost is that `RelativeLoss.indicator` no longer appears as a submodule in `repr`, and a configuration needs `functools.partial` for `k`.                                                                                                                      |
| D3  | Delete `View`?                                                                                                                                        | **Yes.** It reshapes the batch dimension too, so it is only correct for one batch size. If a reshape module is wanted, add `Unflatten`-style `Reshape(dim, sizes)` instead.                                                                                                                         |
| D4  | Set `lint.flake8-builtins.builtins-ignorelist = ["input"]` to remove the 29 `# noqa: A002`?                                                           | **Yes**, once, in `pyproject.toml`: `input` is the `torch.nn` name for the argument, and the same argument is the only builtin shadowed. This is a changed rule set, which `CLAUDE.md` says is not a fix for a lint error; it needs the maintainers' explicit approval, otherwise keep the `noqa`s. |
| D5  | Strict equal shapes for all two-tensor functions (P4)?                                                                                                | **Yes.** `(N,)` vs `(N, 1)` is the most common silent bug in regression code. A caller who wants broadcasting calls `expand`.                                                                                                                                                                       |
| D6  | Function options keyword-only, module defining hyperparameter positional (P3)?                                                                        | **Yes.** It matches `torch.nn` for modules (`nn.LeakyReLU(0.1)`) and `conventions.md` for options.                                                                                                                                                                                                  |
| D7  | Keep `sonnix.factory` at all, or inline `coola.factory.resolve_object` in the 4 wrapper modules?                                                      | **Keep** the 3 functions: the trust warning (P14) lives in one place and `is_module_config` has no `torch`-free equivalent.                                                                                                                                                                         |
| D8  | Move `is_loss_decreasing*` and `DummyDataset` to `testing`, delete the dataset/optimizer factories?                                                   | **Yes.** They are consumed only by tests and docs.                                                                                                                                                                                                                                                  |
| D9  | Composition vs inheritance for `DynamicTanh`/`DynamicAsinh`                                                                                           | **Composition.** It deletes a documented workaround; the cost is a changed state-dict key.                                                                                                                                                                                                          |

## 10. Not covered

- New losses and layers (`Tweedie`, `Dice`/`Tversky`, `SwiGLU`/`GeGLU`): the target describes the
  rules a new object must follow, not which objects to add. Each one follows P1-P12 and adds one
  registry entry.
- A softplus or clamped parametrisation of a learnable `alpha` (the `Gaussian`/`Laplacian` division
  by zero): needs an experiment, not a rule.
- Packaging and CI (28 workflow files, `coola` pre-release pin, a GPU job): tracked in
  `review/suggestions.md` section 7.
- Lazy top-level imports (`import sonnix; sonnix.modules.X`): `import sonnix.modules` is explicit
  today and keeps import time predictable.
- Replacing `docformatter` with `ruff` docstring formatting.
- `torch.nn.utils.parametrize`-based alternatives to the `learnable` flag.

## Appendix A: size of the package

Estimated from the current file sizes; docstrings and examples included.

| Area                                | Lines today | Lines after | Change                                                                             |
| ----------------------------------- | ----------- | ----------- | ---------------------------------------------------------------------------------- |
| `functional/loss`                   | 1 100       | ~950        | −3 functions (asinh ×2, msle); `_binary.py` +40; −duplicated comments              |
| `modules/loss`                      | 1 400       | ~800        | −`indicators.py` (266), −`asinh.py` (94), −RelativeMSE/SmoothL1 (~90), −MSLE (~45) |
| `modules/activation`                | 1 080       | ~1 040      | `_validate_init` hook; `BaseDynamicNorm` composition (−15)                         |
| `modules/numerical`                 | 860         | ~800        | `resolve_range` and `_NumericalEncoder` replace repeated code                      |
| `modules/shape.py`, `dtype.py`, ... | 480         | ~400        | −`View`, `MulticlassFlatten` moved                                                 |
| `factory`                           | 306         | ~140        | −`dataset.py` (81), −`optimizer.py` (84)                                           |
| `utils`                             | 896         | ~620        | −`iterator`, −`random`, −`top_module_mode`; `loss.py` (220) moves                  |
| `testing`                           | 61          | ~285        | +`loss.py`                                                                         |
| `validation`                        | 251         | ~230        | `Reduction` defined once; `validate_moment_order` inlined                          |
| **Total `src`**                     | **7 815**   | **~6 600**  |                                                                                    |

## Appendix B: the rules as a checklist for a new public object

1. Is it a composition of existing public objects? If yes, it is a recipe in the guide, not a class.
2. Function in `sonnix.functional` first (if it has numerical content); module second, calling it (P1).
3. Hyperparameters validated in the function; the module calls the same validator (P2).
4. Signature `(prediction, target, *, options, reduction)` or `(input, *, options)` (P3).
5. Two same-role tensors: `validate_same_shape` (P4). Loss: `reduce_loss` (P5).
6. Check large, small, zero and half-precision inputs and the gradient at the boundaries (P6).
7. Frozen tensors are buffers; `device`/`dtype` if the module creates tensors (P8, P9).
8. `extra_repr` and public attributes named like the arguments (P11).
9. One registry entry: export, compile, dtype and parity tests come for free (P10).
10. Export it from the flat `__init__.py` only; document it in the matching `docs/docs/refs` page (P12).
