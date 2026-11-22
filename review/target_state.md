# `sonnix`: target state and the changes to reach it

Date: 2026-11-22, on `main` at `5003fc7`. This is the second revision. The first
one (twelve steps, #413-#440) is fully merged; its history is in
[Appendix C](#appendix-c-history-of-the-first-plan) and in `CHANGELOG.md`.

This document describes the design `sonnix` should have and the steps to get
there from `main`. It follows the format of
[`coola/review/recursive.md`](https://github.com/durandtibo/coola/blob/main/review/recursive.md):
properties, constraints, findings, architecture, public API, dependents,
evidence, plan, tests, decisions. The properties come first: every choice below
exists to make one of them true **by construction** (one helper, one registry)
instead of by the discipline of each class.

Scope: all of `src/` (6.8k lines, read in full), `tests/` layout and registry,
`docs/`, `README.md`, `CLAUDE.md`, `pyproject.toml` (import-linter). Items marked
**[probe]** were reproduced by running code today; **[reading]** comes from the
source. Nothing in the repository was changed except this file.

## 1. Properties of the package

`sonnix` is a library of PyTorch building blocks that `torch.nn` does not
provide: losses, activations, numerical-feature encoders and a few layers. Its
public API has three tiers: `sonnix.functional` (stateless operations),
`sonnix.modules` (the `torch.nn.Module` wrappers and layers) and the support
packages (`validation`, `factory`, `utils`, `testing`) that the first two use.

Each property is a rule that holds for **every** public object, with the
mechanism that makes it true and the test that proves it. A property without a
test is a wish. P1-P15 exist in `docs/docs/conventions.md`; the revision
sharpens P2, P3, P8, P11 and P12 (the evidence of section 3 shows that they were
not true as written) and adds P16.

| #   | Property                            | Rule                                                                                                                                                                                                                                                                                                                                                                                          | Enforced by                                                                                  |
| --- | ----------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------- |
| P1  | **Function/module duality**         | Every loss, and every op that needs numerical care (`safe_exp`, `safe_log`, `safe_pow`), is a stateless function in `sonnix.functional` **and** a module (when it has hyperparameters) in `sonnix.modules` that stores the hyperparameters and calls the function. A module with learnable state or without numerical care (an activation, a layer) has no function. A `forward` is one call. | registry parity test: module output equals function output (exists)                          |
| P2  | **Validate early and completely**   | Every hyperparameter is checked for **type** (a `bool` is not an `int`), **finiteness** and **range**, before it is stored or converted, with `coola.validation` or a `sonnix.validation` helper. A float is finite unless its docstring says otherwise (`alpha=-inf`, `max_val=None`). The function checks at call time; the module checks in `__init__` with the same helper and message.   | `test_loss_validation` parity (exists), extended by a registry table of invalid values (new) |
| P3  | **Uniform call signature**          | Losses: `(prediction, target, *, <options>, reduction)`. Single-tensor ops: `(input, *, <options>)`. Options of a function are keyword-only. A module takes **positionally** its structural arguments (sizes, wrapped modules, tensors) and at most **one** hyperparameter (the defining one, like `nn.LeakyReLU(0.1)`); the rest is keyword-only. The tensor of a `forward` is `input`.      | `ModuleCase.positional` test over the registry (new); `inspect.signature` test on functions  |
| P4  | **Strict shapes**                   | Two tensors of the same role must have the same shape (`ValueError` naming both). A layer or an encoder checks the dimension it consumes and names the argument (`n`, `n_features`). Nothing broadcasts silently.                                                                                                                                                                             | `validate_same_shape`; parametrised shape tests (exist)                                      |
| P5  | **One reduction contract**          | Every loss accepts `none`/`mean`/`sum`/`batchmean`, validated by `reduce_loss`; default `mean`; `none` keeps the shape of the elementwise loss; `reduction` is keyword-only.                                                                                                                                                                                                                  | `reduce_loss`; registry test over all losses                                                 |
| P6  | **Numerically safe**                | Outputs and gradients are finite for finite inputs the dtype can represent, in `float16`/`bfloat16`/`float32`, and for every **valid** hyperparameter. Exceptions are documented in the docstring.                                                                                                                                                                                            | half-precision, `gradcheck` and finite-gradient tests (exist)                                |
| P7  | **Pure**                            | A function or a `forward` never mutates its inputs and keeps no hidden state.                                                                                                                                                                                                                                                                                                                 | in-place-mutation registry test (exists)                                                     |
| P8  | **Complete state, by construction** | The `state_dict` holds the parameters and buffers that define the behaviour, and **nothing derived**. A tensor derived from persistent state is non-persistent and recomputed after `load_state_dict`. A frozen tensor is a buffer with the same name as the learnable one (`register_tensor`). Round trip: `m2.load_state_dict(m1.state_dict())` makes `m2` behave as `m1`.                  | `register_tensor`; round-trip registry test with a second, different build (new)             |
| P9  | **Device and dtype**                | A module that creates tensors takes `device` and `dtype`, follows `.to()`/`.half()`, and the output dtype is the input dtype.                                                                                                                                                                                                                                                                 | dtype/device registry tests (exist)                                                          |
| P10 | **Export and compile**              | Every registered module passes `torch.export` and `torch.compile(fullgraph=True)` on CPU: no `.item()`, no data-dependent Python branch in a `forward`.                                                                                                                                                                                                                                       | registry tests (exist)                                                                       |
| P11 | **Introspectable**                  | Every hyperparameter, **including a callable or a submodule**, is a public attribute named like the constructor argument and is shown by `repr`.                                                                                                                                                                                                                                              | `test_hyperparameters` (exists), extended to `repr` (new)                                    |
| P12 | **One public path**                 | `sonnix.{functional,modules,validation,factory,utils,testing}` each export their API from `__init__.py` (`__all__`), and nothing else is public. The sub-folders (`functional.loss`, `modules.loss`, ...) have no `__all__`. Docs, docstrings and examples import from the flat path. Every exported name is in `docs/docs/refs` and has an `Example` that runs.                              | `test_imports`, docs-coverage test (new), `xdoctest`                                         |
| P13 | **Layered imports**                 | `testing → modules → functional → validation`; `utils`, `factory` and `validation` import no other sonnix package; no cycle.                                                                                                                                                                                                                                                                  | import-linter contracts (exist)                                                              |
| P14 | **Configuration is trusted input**  | `sonnix.factory` imports and **calls** a `_target_` before it checks the type of the result. An argument typed `Module \| dict` says so in a `Warning`.                                                                                                                                                                                                                                       | docstring test (exists)                                                                      |
| P15 | **Optional means lazy**             | Only `torch` and `coola` are imported at module level.                                                                                                                                                                                                                                                                                                                                        | `package_checks` job (exists)                                                                |
| P16 | **Earned names**                    | A public name is a loss, a layer, an activation, an encoder, an op with numerical care, or a helper that these use or that tests them. An object that is a fixed-argument composition of other public objects or of a `torch.nn` object is a **recipe** in the guide, not a class, unless users look for it by name (`ReLUn`, `Square`, `ToFloat`): then a thin wrapper is allowed and its docstring names the composition. A base class is public only if users are meant to subclass it.                                             | review checklist (Appendix B); a class without a registry entry fails                        |

Not guaranteed, and documented as such: bit-exact results across torch
versions; behaviour on GPU in CI (no GPU job; the CUDA tests run locally before
a release); a learnable `alpha` that crosses `0` (documented in the
numerical-stability guide); mixed-dtype inputs (torch promotes them).

## 2. Constraints

1. **Simple and easy to maintain.** Delete code instead of adding next to it.
   Each new helper must remove more code than it adds.
2. **Breaking changes are fine.** The version is `0.0.1a0`: no shim, no
   deprecation period. Each breaking change is listed in `CHANGELOG.md`
   ("Breaking changes" and the migration table) with the PR number.
3. **Same numerics.** No valid call changes its value. The only behaviour
   changes are the ones a finding below names, each with a regression test that
   fails without the fix (`git stash push src`).
4. **No slower than today.** A forward pass stays the same tensor operations;
   the added checks run at construction or are Python-level comparisons. The
   piecewise encoder is the one place where this decides the design (4.4).
5. **Follow `CLAUDE.md` and `docs/docs/conventions.md`.**
6. **`torch` and `coola` are the only hard dependencies.**

## 3. What differs today

The baseline is green: `ruff check` clean, 8,921 tests passed (3 skipped),
`xdoctest` 118/118, `lint-imports` 3 contracts kept. `inv types.check` and
`mkdocs build --strict` were not run for this revision. What is left is
**properties that are not true as written** and **surface area**, not bugs in
the numerics.

| #   | Finding                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       | Evidence                                                         |
| --- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------- |
| F1  | **Non-finite hyperparameters are accepted (P2, P6).** `validate_gt(x, 0)` and `validate_ge` accept `+inf`. `log_cosh_loss(scale=inf)` and `poisson_regression_loss(eps=inf)` return **`nan`**; `binary_focal_loss(gamma=inf)`, `general_robust_regression_loss(scale=inf)` and `absolute_relative_error(eps=inf)` return a silent `0`; `SafeLog(min_val=inf)` outputs `inf`; `ReLUn(max_val=inf)` is a no-op; `Clamp(nan, nan)` outputs `nan`.                                                                                                                                                                                                                                                                                                                                | [probe]                                                          |
| F2  | **Arguments checked late, by hand, or not at all (P2).** `Clamp('a', 1)` raises the message of `>` between `str` and `int`; `ConcatFusion(dim=1.5)` and `ToCategoricalLabel(dim='a')` are accepted; `DynamicTanh(alpha_init_value=nan)` is accepted and `'a'` fails inside `fill_`; `create(num_frequencies=2.5)`, `create(feature_size=2.5)` and `from_quantiles(n_bins=2.5)` fail inside `torch.linspace` (the validators use `validate_ge` without `validate_int`); `GeneralRobustRegressionLoss(alpha='a')` calls `float()` **before** validating (`ValueError`, while the function raises `TypeError`); `PoissonRegressionLoss(log_input=...)` stores `bool(...)`; `AsinhNumericalEncoder` and `CosSinNumericalEncoder` accept `nan` tensors (the piecewise one checks). | [probe]                                                          |
| F3  | **The `state_dict` is not complete (P8).** `PiecewiseLinearNumericalEncoder` stores `edges` and `width` (derived from `edges`) as persistent buffers, and `is_step` (derived from `width`) as a non-persistent one that is never recomputed. After `a.load_state_dict(b.state_dict())`, `a(x)` differs from `b(x)`: `[1.0, 0.5, 0.25]` instead of `[1.0, 1.0, 0.25]` for `b` built with the duplicated edge `[0, 1, 1, 3]`.                                                                                                                                                                                                                                                                                                                                                   | [probe]                                                          |
| F4  | **P3 is not met and is not enforced.** The module rule ("one defining hyperparameter may be positional") is broken by `BinaryFocalLoss`, `BinaryFocalLossWithLogits` (`alpha, gamma`), `GeneralRobustRegressionLoss` (`alpha, scale, max_val`) and `RelativeLoss` (`indicator, eps`). `test_option_is_keyword_only` checks a hand-picked list of 27 pairs, so a new class is never checked. Two functions break the function rule: `reduce_loss(tensor, reduction)` and `safe_pow(x, exponent)` (`x` and `tensor` instead of `input`).                                                                                                                                                                                                                                        | [probe] `inspect.signature` over `sonnix.modules.__all__`        |
| F5  | **P12 is not true.** `sonnix.utils` and `sonnix.testing` export nothing (`__all__` is empty, the docs import from `sonnix.utils.params`, ...). `sonnix.functional.loss` exports 16 names while `sonnix.modules.loss` exports none, and `docs/docs/refs/functional.md` documents the indicators under `sonnix.functional.loss.*`. `docs/docs/refs/utils.md` renders `sonnix.testing.loss` (copy of `testing.md`). `rectifier_asinh_unit` is not listed in the functional reference.                                                                                                                                                                                                                                                                                            | [probe] + docs                                                   |
| F6  | **Compositions and non-blocks are still public (P16).** `ToFloat`, `ToLong` = `ToDtype(torch.float32/int64)`; `ReLUn(n)` = `Clamp(0, n)` (= `nn.Hardtanh(0, n)`); `Square` = `Pow(2)`; `absolute_relative_error` and `symmetric_absolute_relative_error` are **bit-identical** to `relative_loss(absolute_error(p, t), <indicator>(p, t), reduction="none")` with `classical_relative_indicator` and `arithmetical_mean_indicator`; `absolute_error` is `(t - p).abs()`; none of the three errors is used by a loss or a module; `rectifier_asinh_unit` is `input.clamp(min=0).asinh()`.                                                                                                                                                                                      | [probe] equality checked on a random `(5, 4)` pair               |
| F7  | **Public base classes that are not extension points.** `BaseBinaryFocalLoss` and `BaseBinaryPoly1Loss` only share an `__init__`. The numerical encoders share a base class named `NumericalEncoder` (not `Base*`, unlike `BaseLoss`, `BaseFusion`, `BaseAlphaActivation`, `BaseDynamicNorm`) that lives in `_utils.py` with a public `__all__` and is not exported, so `isinstance(m, NumericalEncoder)` needs a private import.                                                                                                                                                                                                                                                                                                                                              | [probe] `hasattr(sonnix.modules, "NumericalEncoder")` is `False` |
| F8  | **Support code that nothing in the library uses (P16).** `sonnix.utils` has 11 functions (542 lines); only `module_mode` is used (by `testing`). `get_module_device(s)`, `is_module_on_device`, `has_*`, `num_*`, `freeze_module`, `unfreeze_module` are one-liners of `torch`; `load_state_dict_to_module` is a 206-line heuristic. `DummyDataset` is used by no test outside its own. `is_loss_decreasing*` **is** used (21 test files) and stays.                                                                                                                                                                                                                                                                                                                          | grep over `src`, `tests`, `docs`, `notebooks`                    |
| F9  | **Validators that duplicate each other or are used once.** `validate_abs_range`, `validate_frequency` and `validate_moment_order` each have **one** call site. `resolve_range` already checks `0 < min <= max` (since #439), so `validate_frequency` repeats the check on the same numbers and `validate_abs_range` repeats it on the `value_range` before the conversion. `CLAUDE.md`: no validator that is one comparison.                                                                                                                                                                                                                                                                                                                                                  | [reading] `grep validate_ src`                                   |
| F10 | **"Parameter or buffer" is written three times** (`BaseAlphaActivation`, `AsinhNumericalEncoder`, `CosSinNumericalEncoder._register_tensor`). P8 holds only because each copy is right.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       | [reading]                                                        |
| F11 | **`n_bins` is off by one.** `from_quantiles(x, n_bins=3)` builds **2** segments, `feature_size == n_bins - 1`, and the argument `bins` of the constructor holds **edges**. The paper (and a reader) calls the segments bins.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  | [probe] 4 edges → `feature_size` 3                               |
| F12 | **`RelativeLoss.repr` hides the indicator (P11).** The indicator is a plain attribute (a function is not a submodule) and `extra_repr` does not print it.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     | [probe] `reduction=mean, eps=1e-08, detach_indicator=False`      |
| F13 | **Small duplications.** `AverageFusion.fuse` repeats the sum loop of `SumFusion`; the `init == 0` check is written three times (`Gaussian`, `Snake`, `Laplacian`, which adds a second message for `< 0`); `Snake` is alone in `snake.py`, while its five siblings are in `alpha.py`.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          | [reading]                                                        |
| F14 | **Dtype statements that the code does not enforce.** `absolute_error`, `absolute_relative_error` and `symmetric_absolute_relative_error` document "same shape and data type"; `float32` against `float64` is accepted and promoted. The four binary losses raise a raw torch `RuntimeError` for an integer or `bool` target (labels from a `DataLoader` are often `long`).                                                                                                                                                                                                                                                                                                                                                                                                    | [probe]                                                          |
| F15 | **`# noqa: A002` appears 44 times** (the `input` argument, decision D4 of the first revision, never applied: it needs the maintainers' approval because `CLAUDE.md` says a changed rule set is not a fix).                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    | `grep -c`                                                        |

What is already right and must not regress: parity of every module with its
function, strict shapes in all losses (P4), keyword-only `reduction` and
function options, the composition of the dynamic norms, the shared range logic
(`resolve_range`), the registry-driven export/compile/dtype/purity tests, the
import-linter layering, and the 100 % coverage rule.

## 4. Architecture

### 4.1 Packages and dependencies

```text
sonnix
├── validation/   argument validators, Reduction                    (imports: coola)
├── functional/   stateless ops: losses, indicators, safe_*, reduce (imports: validation)
├── factory/      resolve_module, is_module_config, create_sequential (imports: coola, torch)
├── utils/        register_tensor, module_mode                       (imports: torch)
├── modules/      nn.Module wrappers, layers, encoders              (imports: functional, factory, utils, validation)
│   └── activation/ loss/ numerical/ fusion/                        (folders, not public paths)
└── testing/      is_loss_decreasing*                               (imports: anything; nothing imports it)
```

The import-linter contracts stay as they are. Folders under `functional` and
`modules` have a docstring-only `__init__.py` (P12): the flat package builds the
namespace.

### 4.2 The function/module pair and the validators (P1, P2)

The shape of a loss does not change (first revision, section 3.2). What changes
is **what a check is**. One helper in `sonnix.validation` makes P2 true for the
most common case, a strictly positive finite number:

```python
def validate_positive(value: float, *, name: str) -> None:
    """Validate that ``value`` is a finite number strictly greater than 0."""
    validate_finite(value, name=name)  # type (not a bool) and finiteness
    validate_gt(value, 0, name=name)
```

It replaces `validate_gt(x, 0, ...)` where `x` is `scale` (log-cosh, general
robust), `eps` (Poisson, relative loss, `RelativeLoss`), `min_val` (`safe_log`)
and `max_val` (general robust): about nine call sites, in the functions and
in the modules. The other rules use `coola.validation` directly:

| Argument                                                         | Check                                                                       |
| ---------------------------------------------------------------- | --------------------------------------------------------------------------- |
| `gamma` (focal), `exponent` (`safe_pow`)                         | `validate_finite` then `validate_ge(.., 0)`                                 |
| `q`, `alpha` (focal)                                             | `validate_range` (rejects `nan` and `inf`)                                  |
| `alpha` (Poly-1), `max_val` (`SafeExp`), `threshold`, `init`     | `validate_finite`                                                           |
| `alpha` (general robust)                                         | `validate_number`, then the custom check (`nan` and `+inf` rejected)        |
| `min_val`, `max_val` (`Clamp`)                                   | `None` or `validate_finite`; then the relation `min_val <= max_val` by hand |
| `dim`, `num_frequencies`, `feature_size`, `n_bins`, `n_features` | `validate_int` **then** `validate_ge`                                       |
| `log_input`, `learnable`, `detach_indicator`, `bias`             | `validate_isinstance(.., bool)`; no `bool(...)` conversion                  |
| `indicator`                                                      | `callable(...)`: `coola` has no `validate_callable`; custom `TypeError`     |

Rules:

- A module validates **before** it converts or stores (`float(alpha)` after the
  check, never before): the function and the module raise the same exception.
- `validate_abs_range`, `validate_frequency` and `validate_moment_order` are
  deleted (F9): `resolve_range` checks the range once, `create` calls
  `validate_int` + `validate_ge` on its own count, and `moment_mean_indicator`
  inlines `validate_int` + `validate_ge` for `k`.
- A constructor that takes a **tensor** (the encoders) checks that it is finite,
  as the piecewise encoder does today.
- If `coola.validation` gains a `validate_positive`, delete ours (D5).

### 4.3 Signatures (P3)

Functions: no change except `reduce_loss(loss, *, reduction="mean")` (the
operand is the loss, `reduction` is an option, like every other loss function)
and `safe_pow(input, exponent)`. `exponent` stays positional: it is the second
operand of a binary op, as in `torch.pow(input, exponent)`. It is the only
exception, and the signature test lists it.

Modules: a class takes positionally its **structural** arguments and at most
one hyperparameter.

| Class                                                    | Positional (target)                             | Keyword-only                                        |
| -------------------------------------------------------- | ----------------------------------------------- | --------------------------------------------------- |
| `BinaryFocalLoss`, `BinaryFocalLossWithLogits`           | `alpha`                                         | `gamma`, `reduction`                                |
| `GeneralRobustRegressionLoss`                            | `alpha`                                         | `scale`, `max_val`, `reduction`                     |
| `RelativeLoss`                                           | `criterion` (a module)                          | `indicator`, `eps`, `reduction`, `detach_indicator` |
| `Clamp`                                                  | `min_val`, `max_val` (a pair, as `nn.Hardtanh`) | none                                                |
| alpha activations, `NLinear`, `ExU`, encoders, `*Fusion` | unchanged (sizes, tensors, `init`, `dim`)       | unchanged                                           |

`ModuleCase` of the registry gets a `positional` field (the names allowed to be
positional, reviewed in the PR); one test compares it with
`inspect.signature`. A new class without an entry already fails
`test_registry_covers_every_module`, so P3 is checked for every class and not
for a hand-picked list (F4).

### 4.4 State (P8)

**`register_tensor`**, in `sonnix.utils` (it is shared by `modules` and has no
sonnix dependency):

```python
def register_tensor(module: nn.Module, name: str, tensor: Tensor, *, learnable: bool) -> None:
    """Register ``tensor`` as a parameter if ``learnable``, else as a buffer, under ``name``."""
    if learnable:
        module.register_parameter(name, nn.Parameter(tensor.detach().clone()))
    else:
        module.register_buffer(name, tensor.detach().clone())
```

It replaces the three copies (F10). `BaseAlphaActivation`,
`AsinhNumericalEncoder` and `CosSinNumericalEncoder` call it; the `learnable`
properties stay (`isinstance(self.alpha, nn.Parameter)`).

**`PiecewiseLinearNumericalEncoder`** keeps **one** persistent buffer, `edges`
(`(n_features, n_edges)`, sorted). `width`, `is_step`, `lower` and `upper` are
non-persistent and built by one private method `_derive()`, called by
`__init__` and by a load hook:

```python
self.register_buffer("edges", edges)
self._derive()  # width, is_step, lower, upper: persistent=False
self.register_load_state_dict_post_hook(lambda module, _: module._derive())
```

Deriving them inside `forward` would also remove the stale state, but a
prototype measured it 1.29x slower on a small input (54 → 69 µs, 8 features ×
8 edges × batch 256) and 1.06x on a larger one (1.45 → 1.53 ms, 100 × 16 ×
1024), so constraint 4 rejects it. The hook keeps `forward` unchanged.
`register_load_state_dict_post_hook` exists in the supported torch (≥ 2.4).

Naming (F11): the constructor argument `bins` becomes `edges` and
`from_quantiles(x, n_bins)` makes `n_bins` segments from `n_bins + 1` quantiles,
so `feature_size == n_bins`. `n_bins >= 1`, and an encoder needs at least 2
edges. A single edge is rejected: today it is a special case with
`feature_size == 1`, which breaks the documented rule `feature_size == edges - 1`.

### 4.5 Surface area (P16)

> **Decision of the maintainers (#451):** `ToFloat`, `ToLong`, `ReLUn`, `Square`, the
> three errors and `rectifier_asinh_unit` are **kept**, because users expect them. Only the
> private bases, the `ToDtype` names and the small de-duplications of this section were
> done. D2 and D3 are answered "keep".

| Removed                                                                          | Replacement                                                                                                                                                                |
| -------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `ToFloat`, `ToLong`                                                              | `ToDtype("float32")`, `ToDtype("int64")`: `dtype` accepts a `torch.dtype` or its name (a configuration cannot write a `torch.dtype`, which was their only reason to exist) |
| `ReLUn(n)`                                                                       | `Clamp(0.0, n)`                                                                                                                                                            |
| `Square`                                                                         | `Pow(2)`                                                                                                                                                                   |
| `absolute_error`, `absolute_relative_error`, `symmetric_absolute_relative_error` | `(target - prediction).abs()`; `relative_loss(abs_error, indicator, reduction="none")` (recipes in the losses guide)                                                       |
| `rectifier_asinh_unit` (function)                                                | `RectifierAsinhUnit` stays: the module is the public op; the function was one line without numerical care (P1)                                                             |
| `BaseBinaryFocalLoss`, `BaseBinaryPoly1Loss` (public)                            | private `_BinaryFocalLoss`, `_BinaryPoly1Loss` (documented, not exported)                                                                                                  |

`Snake` moves into `alpha.py`. `AverageFusion` calls the same private sum as
`SumFusion` (`BaseFusion._sum(inputs)`). The alpha activations keep the
`_validate_init` hook (default `validate_finite`), but write each rule once:
`Gaussian` and `Snake` share one private function `_validate_non_zero(init)` in
`alpha.py`, and `Laplacian` calls `validate_gt(init, 0, name="init")` (one
message instead of two).

### 4.6 Encoders

- `NumericalEncoder` becomes **`BaseNumericalEncoder`** in
  `modules/numerical/base.py` and is exported from `sonnix.modules` (F7).
  `_utils.py` keeps the helpers (`create_spaced_values`, `prepare_tensor_param`,
  `resolve_range`) without `__all__`.
- The two `create` class methods keep their parameters. They validate
  `feature_size` / `num_frequencies` with `validate_int` + `validate_ge` and
  `n_features` likewise (F2).
- The table of the four encoders and their output shapes does not change,
  except `PiecewiseLinearNumericalEncoder`: `(*, n_features, n_bins)`.

### 4.7 Support packages (F8)

| Package             | Target content                                                                                                                                                             | Removed                                                                                                                                                                                                                       |
| ------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `sonnix.utils`      | `module_mode`, `register_tensor`                                                                                                                                           | `get_module_device`, `get_module_devices`, `is_module_on_device`, `has_parameters`, `has_learnable_parameters`, `num_parameters`, `num_learnable_parameters`, `freeze_module`, `unfreeze_module`, `load_state_dict_to_module` |
| `sonnix.testing`    | `is_loss_decreasing`, `is_loss_decreasing_with_adam`, `is_loss_decreasing_with_sgd` (used by 21 test files)                                                                | `DummyDataset`                                                                                                                                                                                                                |
| `sonnix.factory`    | `resolve_module`, `is_module_config`, `create_sequential` (the guide on configurations uses them)                                                                          | —                                                                                                                                                                                                                             |
| `sonnix.validation` | `Reduction`, `VALID_REDUCTIONS`, `validate_loss_reduction_strategy`, `validate_same_shape`, `validate_focal_params`, `validate_general_robust_params`, `validate_positive` | `validate_abs_range`, `validate_frequency`, `validate_moment_order`                                                                                                                                                           |

This is the one removal of working code that no sonnix object needs. It is a
separate, last step (step 7) so that it can be dropped without touching the
rest (decision D1). `utils` and `testing` export their names (P12), and
`docs/docs/refs/utils.md` stops rendering `sonnix.testing.loss`.

### 4.8 Public paths (P12)

- Every package of 4.1 has an `__all__` in its `__init__.py`; folders do not.
- `docs/docs/refs/functional.md` documents the indicators as
  `sonnix.functional.<name>`.
- A **docs-coverage test** walks the six `__all__` lists and checks that every
  name appears in `docs/docs/refs` (the probe found one miss:
  `rectifier_asinh_unit`, which disappears with F6).

## 5. Public API after the change

```python
# sonnix.functional (25 → 21)
Reduction, reduce_loss  # reduce_loss(loss, *, reduction="mean")
safe_exp, safe_log, safe_pow
binary_focal_loss, binary_focal_loss_with_logits
binary_poly1_loss, binary_poly1_loss_with_logits
general_robust_regression_loss, log_cosh_loss, poisson_regression_loss, quantile_regression_loss
relative_loss
arithmetical_mean_indicator, classical_relative_indicator, geometric_mean_indicator
maximum_mean_indicator, minimum_mean_indicator, moment_mean_indicator, reversed_relative_indicator

# sonnix.modules (58 → 53)
# losses (12)
BaseLoss, BinaryFocalLoss, BinaryFocalLossWithLogits, BinaryPoly1Loss, BinaryPoly1LossWithLogits
GeneralRobustRegressionLoss, LogCoshLoss, PoissonRegressionLoss, QuantileRegressionLoss
RelativeLoss, TransformedLoss, MulticlassFlatten
# activations (23)
(
    Asinh,
    Clamp,
    Exp,
    Expm1,
    Log,
    Log1p,
    Pow,
    RectifierAsinhUnit,
    SafeExp,
    SafeLog,
    Sin,
    Sinh,
    SquaredReLU,
)
BaseAlphaActivation, ExpSin, Gaussian, Laplacian, MultiQuadratic, Quadratic, Snake
BaseDynamicNorm, DynamicAsinh, DynamicTanh
# layers (4), fusion (5), encoders (5), shape / dtype / labels (4)
ExU, NLinear, ResidualBlock, ScaleAndShift
BaseFusion, AverageFusion, ConcatFusion, MultiplicationFusion, SumFusion
BaseNumericalEncoder, AsinhNumericalEncoder, AsinhCosSinNumericalEncoder
CosSinNumericalEncoder, PiecewiseLinearNumericalEncoder
Squeeze, ToDtype, ToBinaryLabel, ToCategoricalLabel

# sonnix.validation (9 → 7)   sonnix.factory (3, unchanged)
# sonnix.utils (11 → 2)       sonnix.testing (4 → 3)
```

### Migration table

| Before                                                                            | After                                                                                                   |
| --------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------- |
| `log_cosh_loss(p, t, scale=inf)`, `Clamp(nan, nan)`, `ReLUn(inf)`, ...            | `ValueError` (P2)                                                                                       |
| `ConcatFusion(dim=1.5)`, `create(num_frequencies=2.5)`, `from_quantiles(.., 2.5)` | `TypeError` naming the argument                                                                         |
| `BinaryFocalLoss(0.25, 2.0)`                                                      | `BinaryFocalLoss(0.25, gamma=2.0)`                                                                      |
| `GeneralRobustRegressionLoss(2.0, 1.0, 5.0)`                                      | `GeneralRobustRegressionLoss(2.0, scale=1.0, max_val=5.0)`                                              |
| `RelativeLoss(criterion, indicator, 1e-6)`                                        | `RelativeLoss(criterion, indicator=indicator, eps=1e-6)`                                                |
| `reduce_loss(x, "sum")`; `safe_pow(x=t, exponent=2)`                              | `reduce_loss(x, reduction="sum")`; `safe_pow(input=t, exponent=2)`                                      |
| `ToFloat()`, `ToLong()`                                                           | `ToDtype("float32")`, `ToDtype("int64")`                                                                |
| `ReLUn(6.0)`, `Square()`                                                          | `Clamp(0.0, 6.0)`, `Pow(2)`                                                                             |
| `absolute_error(p, t)` and the two relative errors                                | the recipes of the losses guide                                                                         |
| `rectifier_asinh_unit(x)`                                                         | `RectifierAsinhUnit()(x)` or `x.clamp(min=0).asinh()`                                                   |
| `BaseBinaryFocalLoss`, `BaseBinaryPoly1Loss`                                      | subclass `BaseLoss`                                                                                     |
| `sonnix.modules.numerical._utils.NumericalEncoder`                                | `sonnix.modules.BaseNumericalEncoder`                                                                   |
| `PiecewiseLinearNumericalEncoder(bins=...)`, `from_quantiles(x, n_bins=k)`        | `PiecewiseLinearNumericalEncoder(edges=...)`, `from_quantiles(x, n_bins=k-1)` (same encoder)            |
| state dict key `width` of the piecewise encoder                                   | removed (derived); checkpoints with `width` load with `strict=False`                                    |
| `sonnix.utils.params.*`, `sonnix.utils.device.*`, `load_state_dict_to_module`     | `sum(p.numel() for p in m.parameters())`, `p.requires_grad_(False)`, `next(m.parameters()).device` (D1) |
| `sonnix.testing.DummyDataset`                                                     | a local `torch.utils.data.TensorDataset` (D1)                                                           |

## 6. Dependents

| Consumer                                 | Effect                                                                                                                                                                                                      |
| ---------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `tests/registry.py`                      | −`ReLUn`, `Square`, `ToFloat`, `ToLong` entries; `ModuleCase` gets `positional` and `other` (a second build with different values, for the round trip).                                                     |
| `docs/docs/refs/*`                       | `functional.md` loses `Errors` and the `.loss.` paths; `utils.md` is fixed and shrinks; `testing.md` loses `dummy`; `modules/activation.md` loses `ReLUn`, `Square`; `layers.md` loses `ToFloat`, `ToLong`. |
| `docs/docs/guide/losses.md`              | gains the recipes: relative error, relative-mean error, absolute error.                                                                                                                                     |
| `docs/docs/guide/numerical_stability.md` | `Square` → `Pow(2)`.                                                                                                                                                                                        |
| `docs/docs/conventions.md`               | the design-rules table gets the revised P2, P3, P8, P11, P12 and P16.                                                                                                                                       |
| `README.md`, `docs/docs/index.md`        | the Overview drops `ReLUn`, the `utils` description ("inspect and train modules") and `DummyDataset`.                                                                                                       |
| `notebooks/activations.ipynb`            | uses `snn.ReLUn(m)`: change to `Clamp(0.0, m)`; verify with the notebook CI job.                                                                                                                            |
| `CHANGELOG.md`                           | one "Breaking changes" entry per step, with the PR number, and the migration table above.                                                                                                                   |
| `pyproject.toml`                         | import-linter: unchanged. `ruff`: unchanged unless D4 is approved.                                                                                                                                          |

## 7. Evidence

Counts from `main` at `5003fc7`; probes ran today in a scratch script (outside
the repository).

| Measure                                                           | Today               | Target                    |
| ----------------------------------------------------------------- | ------------------- | ------------------------- |
| Names in `sonnix.modules.__all__`                                 | 58                  | 53                        |
| Names in `sonnix.functional.__all__`                              | 25                  | 21                        |
| Public functions in `sonnix.utils`                                | 11 (542 lines)      | 2 (~100)                  |
| Validators used once or redundant                                 | 3                   | 0                         |
| Float hyperparameters that accept `inf` or `nan` (probe, 8 found) | 8                   | 0                         |
| Classes with more than one positional hyperparameter (P3)         | 4 (+ `Clamp`)       | 0 (+ `Clamp`, documented) |
| Copies of "parameter or buffer"                                   | 3                   | 1                         |
| Persistent buffers derived from another buffer                    | 1 (`width`)         | 0                         |
| `src` lines                                                       | 6,835               | ~6,000                    |
| `tests` lines                                                     | 12,635              | ~11,800                   |
| Tests / doctests / lint on `main`                                 | 8,921 / 118 / clean | same or more              |

Piecewise encoder, forward cost of deriving the state on every call (CPU,
`float32`, best of 5 × 200 runs): 1.29x slower for 8 features × 8 edges × batch
256, 1.06x for 100 × 16 × 1024. The load hook (4.4) keeps today's speed, so no
benchmark gate is needed; `tests/benchmark/test_nlinear.py` stays the only
benchmark.

## 8. Changes to reach the target state

One branch and one pull request per step (`CLAUDE.md`), small and reviewable on
its own. Every step updates `CHANGELOG.md`, the docs it touches and the tests,
and runs `ruff`, `inv types.check --complete`, `pytest tests` and
`xdoctest src/sonnix all`. Steps 2, 3 and 4 are independent and can start from
`main` in parallel; 5 needs 2 (`ToDtype` validation); 6 needs 5 (the names are
settled); 7 is independent; 8 needs 4.

1. **Properties and docs (non-breaking).** Update `docs/docs/conventions.md`
   with the revised rules; fix `docs/docs/refs/utils.md` (F5); add the
   docs-coverage test (names must be listed explicitly in `docs/docs/refs`; it
   passes once `rectifier_asinh_unit` is added to `functional.md`). _Why
   first: the rules are written before the code changes, and the one failing
   check is a docs typo._
2. **Validation (breaking: values that were accepted).** F1, F2, F9: add
   `validate_positive`; apply the table of 4.2 to the functions and the modules;
   delete `validate_abs_range`, `validate_frequency`, `validate_moment_order`;
   check tensor arguments of the encoders; stop `bool(...)` and `float()`
   coercions. Regression tests, driven by one registry table
   `INVALID_ARGUMENTS = {class_or_function: [(kwargs, exception, match), ...]}`
   so the function and the module are checked together, with `nan`, `inf` and a
   wrong type per float argument.
3. **State (breaking: state-dict keys).** F3, F10, F11: add
   `sonnix.utils.register_tensor`; use it in the alpha activations and the
   encoders; rework `PiecewiseLinearNumericalEncoder` (`edges`, `_derive()` and
   the load hook, `n_bins` = segments). Add the round-trip registry test
   (`ModuleCase.other`); it fails on `main` for the piecewise encoder
   (`git stash push src`). The single-edge case is rejected here.
4. **Signatures (breaking in positional calls).** F4: the four classes of 4.3,
   `reduce_loss`, `safe_pow(input, exponent)`; `ModuleCase.positional` and the
   signature test; delete `test_option_is_keyword_only` (replaced by the
   registry-driven test).
5. **Surface area (breaking).** F6, F7 (private bases), F13: delete the names of
   4.5; `ToDtype` accepts a name; `Snake` into `alpha.py`; shared fusion sum;
   one `init` message per rule; recipes in `guide/losses.md`; update the notebook
   and the registry.
6. **Public paths and encoder base (breaking: import paths).** F5, F7
   (`BaseNumericalEncoder`): `__init__.py` exports for `utils` and `testing`;
   docstring-only `__init__` in `functional/loss` and `functional/activation`;
   the indicators documented as `sonnix.functional.<name>`; `test_imports`
   asserts the `__all__` of the six packages and rejects an `__all__` in a
   folder.
7. **Support packages (breaking).** F8, decision D1: delete the ten helpers of
   `sonnix.utils` and `DummyDataset` with their tests; update the README and
   `docs/docs/refs`. _Last on purpose: if the maintainers keep these helpers,
   skip this step; nothing else depends on it._
8. **Small consistency (non-breaking).** F12, F14: `RelativeLoss.extra_repr`
   prints the indicator (`indicator=classical_relative_indicator` or
   `functools.partial(...)`); the four binary losses cast integer and `bool`
   targets to the prediction dtype (D7); the docstrings say what the code
   checks. Extend `test_hyperparameters` to `repr`.
9. **Migration notes (docs only).** Fill the `CHANGELOG.md` migration table with
   the rows of section 5 and the PR numbers; update `README.md` and
   `docs/docs/index.md`.

Steps 2 and 3 are the only ones that change the behaviour of valid code
(a checkpoint of the piecewise encoder, `n_bins`); the others remove names or
make an invalid call fail earlier.

## 9. Tests and gates

The unit-test rules of `CLAUDE.md` apply to everything added (one behaviour per
test, `match` on every `pytest.raises`, seeded data, no local imports, literal
expected values, a regression test for each bug, an integration test for each
patched behaviour, 100 % coverage of the lines added or changed).

- **Validation (P2).** `INVALID_ARGUMENTS`: for each float argument `nan`,
  `inf` and `"a"`; for each int argument `2.5`, `True`, `0`; for each flag
  `1`. The function and the module raise the same type and message
  (parity); the loss is not `nan` for any **valid** value (P6).
- **Signatures (P3).** `ModuleCase.positional` equals the positional parameters
  of the constructor; the functions have only tensor operands (and `exponent`)
  positional.
- **State (P8).** For every module with state: the `state_dict` keys are the
  same for `learnable=True` and `False`; `m2.load_state_dict(m1.state_dict())`
  gives `m2(x) == m1(x)` for `m2` built from `ModuleCase.other`; no buffer is
  persistent and derived (a named list in the registry).
- **Piecewise (F3, F11).** The literal output of the duplicated-edge encoder
  after a load; `from_quantiles(x, n_bins=3).feature_size == 3`; a `width` key
  in an old checkpoint is reported by `strict=False` as unexpected.
- **Surface (P16).** `test_registry_covers_every_module` (exists) fails for a
  new class without an entry; the removed names are absent from every
  `__all__`.
- **Public paths (P12).** `test_imports`: the six `__all__` lists, no `__all__`
  in a folder, no docstring import from a folder; the docs-coverage test.
- **Numerics (P6) and shapes (P4).** The half-precision, `gradcheck` and shape
  tests of today stay; `RelativeLoss` with each indicator and a `partial`
  indicator for the new `repr`.
- **Docs.** `xdoctest src/sonnix all`, `mkdocs build --strict`, the notebook
  job.

## 10. Decisions for the maintainers

| #   | Decision                                                                                                         | Recommendation                                                                                                                                                                                                                                                                        |
| --- | ---------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| D1  | Delete the ten generic helpers of `sonnix.utils` and `DummyDataset` (4.7, step 7)?                               | **Yes.** Nothing in the library uses them, `torch` has a one-liner for nine of them, and `load_state_dict_to_module` is 206 lines of heuristic that solves a checkpoint problem, not a building-block one. Keep them only if there are known external users; the step is independent. |
| D2  | Delete the three error functions and `rectifier_asinh_unit` (F6)?                                                | **Yes.** Two errors are `relative_loss` with an indicator (bit-identical, probed), the third is `(t - p).abs()`; none is a loss or a layer. Keep them only if metrics are in scope, then move them to a `sonnix.metrics` package, not `functional`.                                   |
| D3  | Delete `ReLUn`, `Square`, `ToFloat`, `ToLong`, and let `ToDtype` take a name?                                    | **Yes.** Same rule as the first revision's deletion of `AsinhMSELoss`: a fixed-argument composition is a recipe. The cost is one line in the docs and the notebook per name.                                                                                                          |
| D4  | `lint.flake8-builtins.builtins-ignorelist = ["input"]` to remove the 44 `# noqa: A002`?                          | **Yes**, once, in `pyproject.toml`: `input` is the `torch.nn` name and the only builtin shadowed. Still needs explicit approval (`CLAUDE.md`: a changed rule set is not a fix). Otherwise keep the `noqa`s.                                                                           |
| D5  | `validate_positive` in `sonnix.validation` or upstream in `coola`?                                               | **Here now**, a few lines; delete it when `coola.validation` has it.                                                                                                                                                                                                                  |
| D6  | Rename `bins` → `edges` and make `n_bins` the number of segments in the piecewise encoder (F11)?                 | **Yes.** It removes an off-by-one that every reader of `from_quantiles(x, n_bins=3)` hits, at the price of a changed meaning of `n_bins` (documented in the migration table).                                                                                                         |
| D7  | Cast integer and `bool` targets of the four binary losses to the prediction dtype (F14)?                         | **Yes.** `DataLoader` labels are `long`; the raw `RuntimeError` of torch is the only outcome today, and the cast is free. A **float** target out of `[0, 1]` is still the caller's problem (not checked: data-dependent).                                                             |
| D8  | `Clamp(min_val, max_val)` as the one two-hyperparameter positional exception (P3)?                               | **Yes**, like `nn.Hardtanh(min_val, max_val)`; the registry lists it. The alternative (`Clamp(*, min_val, max_val)`) breaks every call for a rule that torch itself does not follow.                                                                                                  |
| D9  | Keep `sonnix.factory` (3 functions)?                                                                             | **Keep.** Four modules use `resolve_module`; the trust warning (P14) lives in one place.                                                                                                                                                                                              |
| D10 | Move the top-level module files (`exu.py`, `nlinear.py`, ...) into a folder that mirrors the docs page "Layers"? | **No.** It moves files and tests for no behaviour change; the flat namespace (P12) is what users see.                                                                                                                                                                                 |

## 11. Not covered

- New losses and layers (`Tweedie`, `Dice`/`Tversky`, `SwiGLU`/`GeGLU`, a
  multiclass focal loss): the target gives the rules a new object follows (P1-P16,
  Appendix B), not which objects to add.
- Sample weights, masks and `ignore_index` in the losses: a feature, with its own
  design (shape of the weights against `reduction`).
- A softplus or clamped parametrisation of a learnable `alpha`: needs an
  experiment, not a rule.
- Packaging and CI (the workflows, a GPU job): tracked in `review/suggestions.md`.
- Lazy top-level imports (`import sonnix; sonnix.modules.X`): `import
sonnix.modules` stays explicit and keeps the import time predictable.
- Replacing `docformatter` with `ruff` docstring formatting.

## Appendix A: size of the package

| Area                                          | Lines today | Lines after | Change                                                          |
| --------------------------------------------- | ----------- | ----------- | --------------------------------------------------------------- |
| `utils`                                       | 542         | ~100        | −`device`, `params`, `state_dict`; +`register_tensor` (~40)     |
| `testing`                                     | 278         | 220         | −`dummy.py`                                                     |
| `validation`                                  | 227         | ~175        | −`numerical.py`, −`validate_moment_order`, +`validate_positive` |
| `functional` (errors, `rectifier_asinh_unit`) | 158         | 0           | −`error.py`, −`activation/hyperbolic.py`                        |
| `modules` (`dtype`, `relu`, `Square`)         | ~215        | ~130        | −`ToFloat`, `ToLong`, `ReLUn`, `Square`                         |
| `modules/numerical`                           | 860         | ~840        | `register_tensor`, `_derive()`, `base.py`                       |
| rest of `src`                                 | ~4,550      | ~4,535      | `register_tensor` in the alpha activations, shared fusion sum   |
| **Total `src`**                               | **6,835**   | **~6,000**  |                                                                 |

## Appendix B: checklist for a new public object

1. Is it a fixed-argument composition of existing public objects or of a
   `torch.nn` object? Then it is a recipe in the guide (P16).
2. Function first if it is a loss or needs numerical care; the module calls it
   (P1).
3. Every hyperparameter: type, finiteness, range, before it is stored; the module
   uses the same helper (P2).
4. Signature `(prediction, target, *, options, reduction)` or `(input, *,
options)`; a module's positional arguments are listed in its registry entry
   (P3).
5. Two same-role tensors: `validate_same_shape` (P4). Loss: `reduce_loss` (P5).
6. Large, small, zero and half-precision inputs; the gradient at the boundaries;
   every valid hyperparameter (P6).
7. A trained tensor that can be frozen: `register_tensor`; anything derived is
   non-persistent and rebuilt after a load (P8). `device` and `dtype` if the
   module creates tensors (P9).
8. Public attributes named like the arguments, shown in `repr` (P11).
9. One registry entry (with `positional` and `other`): export, compile, dtype,
   purity, parity and round-trip tests come for free (P10).
10. Export it from the flat `__init__.py`, document it in the matching
    `docs/docs/refs` page with an example that runs (P12).

## Appendix C: history of the first plan

The first revision (2026-11-06, `fc04eeb`) had 12 steps, all merged. Its
findings (shape policy, validation of the hyperparameters, keyword-only function
options, `input` for every forward, composition losses, dynamic norms by
composition, encoder range logic and input check, package layout, alpha
`_validate_init`, copied `Args:` blocks) are closed; the details are in
`CHANGELOG.md`.

| Step | Pull request | Summary                                                      |
| ---- | ------------ | ------------------------------------------------------------ |
| 1    | #413         | properties, layering contracts, purity and trust tests       |
| 2    | #414         | validation gaps and strict shapes                            |
| 3    | #415         | keyword-only function options                                |
| 4    | #417         | composition losses and indicator modules deleted             |
| 5    | #418         | `input` for every single-tensor `forward`                    |
| 6    | #419         | dynamic norms built by composition                           |
| 7    | #420         | encoder range logic and input check                          |
| 8    | #422         | packages reorganized                                         |
| 9    | #423         | `_validate_init` hook, shared binary helpers, `View` removed |
| 10   | #424         | migration notes and conventions                              |
| 11   | #439         | `resolve_range` checks `0 < min <= max`                      |
| 12   | #440         | copied `Args:` blocks removed                                |

## Appendix D: status of this revision

| Step | Pull request | Notes   |
| ---- | ------------ | ------- |
| 1    | #446         | open    |
| 2    | #447         | open    |
| 3    | #449         | open    |
| 4    | #450         | open    |
| 5    | #451         | open    |
| 6    | #452         | open    |
| 7-9  | —            | planned |
