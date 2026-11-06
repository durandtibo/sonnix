# Sonnix codebase review

Date of review: 2026-11-06, on `main` at `bb26f91`.

Scope: all of `src/` (read in full, about 6.8k lines), plus `pyproject.toml`, `.github/`,
`.pre-commit-config.yaml`, `docs/`, `README.md`, `CONTRIBUTING.md`, `notebooks/` and the layout of
`tests/` (about 9.7k lines). I did **not** audit the tests one by one: section 6 comes from the
layout and from greps, not from reading every test. Breaking changes are acceptable, and the
package is still `0.0.1a0`.

Items marked **[verified]** were reproduced by running code (scripts were run in a scratch
directory, nothing in the repo was changed except this file). The others come from reading the
source.

Priority legend: **P0** = broken on `main` (tooling, wrong results), **P1** = bug or API issue
that is cheap to fix now and costly after a release, **P2** = quality, **P3** = nice to have.

## 0. Baseline on `main`

| Check                     | Result                                                         |
| ------------------------- | -------------------------------------------------------------- |
| `pytest tests`            | 6626 passed, 3 skipped                                         |
| `ruff check src tests`    | clean                                                          |
| `pyright src`             | 0 errors                                                       |
| `ty check`                | clean                                                          |
| `xdoctest src/sonnix all` | **3 failed**, 124 passed (see 1.1)                             |
| `lint-imports`            | **fails**: `Could not read any configuration.` (see 1.3)       |
| `mkdocs build --strict`   | **fails**: `sonnix.utils.imports could not be found` (see 1.2) |

### Done since the previous review

The previous version of this file listed 1.1-1.10, 2.1-2.6, 3.1-3.8, 3.10 (partly) and 4.x. Those
were merged in #247-#285: `NLinear` init, stable `log_cosh_loss`, `ExpSin`, `x or default` on modules,
shared sin/cos parameters, buffers for frozen tensors, hyperparameter validation, `batchmean`,
`BaseLoss` and the `Reduction` literal, the with/without-logits pairs sharing a base, `alpha: float | None`,
`min_val`/`max_val`, keyword-only `reduction`, `ValueError` instead of `RuntimeError`, the objectory
removal (`coola.factory`), the `numpy` extra, the classifier, `__version__`, `is_loss_decreasing`
`max_norm`, `DummyDataset`, `MulticlassFlatten`, deterministic `get_module_devices`. I re-checked that
they hold. The remaining and new points are below.

---

## 1. Broken on `main` (P0)

### 1.1 Three doctests fail since #275 [verified]

`src/sonnix/modules/loss/relative.py`, the `RelativeLoss`, `RelativeMSELoss` and
`RelativeSmoothL1Loss` examples.

`extra_repr` now prints `detach_indicator=False`, but the docstrings still show
`eps=1e-08, reduction=mean`. `inv test.doctest` (the `Doctest` CI job) runs `xdoctest`, so that job
is red on `main`. Fix the three expected reprs. While there, `RelativeSmoothL1Loss` stores neither
`beta` nor shows it in the repr, so the repr hides a hyperparameter (see 3.2).

### 1.2 The documentation build is broken [verified]

`docs/docs/refs/utils.md` has `::: sonnix.utils.imports`, but `sonnix/utils/imports.py` was removed
with the objectory shim. `mkdocs build --strict` aborts with `BuildError`, so the `build` job of
`ci-doctest.yaml` and `release-docs*` fail. Also:

- `sonnix.utils.factory` has no page, although `resolve_module` and `create_sequential` are the
  main entry points for config-driven models.
- `refs/testing.md` documents `sonnix.testing`, which now contains only `DummyDataset`.
- No CI job would have caught this earlier, since nothing builds the docs on pull requests
  other than that job. Make sure it is a required check.

### 1.3 `lint-imports` has no configuration [verified]

`.pre-commit-config.yaml` runs `uv run lint-imports` ("Enforces the package layering contract
defined in pyproject.toml"), but `pyproject.toml` has no `[tool.importlinter]` section. The hook
exits non-zero. In addition `import-linter` is only installed for Python >= 3.14 in the dev group.
Either add the contract or drop the hook. Suggested contract (it matches the real imports I read):

```toml
[tool.importlinter]
root_package = "sonnix"

[[tool.importlinter.contracts]]
name = "Layers"
type = "layers"
layers = ["sonnix.modules", "sonnix.functional", "sonnix.utils"]
```

`sonnix.testing` should be independent. Note that `sonnix.modules.loss.relative` currently imports
`BaseRelativeIndicator` from the package root `sonnix.modules.loss` under `TYPE_CHECKING`; import it
from `sonnix.modules.loss.indicators` so the contract is clean.

---

## 2. Correctness and numerical issues (P1)

### 2.1 `load_state_dict_to_module(..., strict=False)` silently loads nothing [verified]

`src/sonnix/utils/state_dict.py`

The "find the matching sub-dict" fallback only runs when `load_state_dict` raises `RuntimeError`.
With `strict=False` torch never raises for missing or unexpected keys, so a wrapped checkpoint
(`{"model": {...}}`) loads **nothing**, and the function logs `The weights are loaded in the module.`
(I checked: the weights are unchanged). Fix: when `strict=False`, inspect the
`_IncompatibleKeys` result, and if every key is missing, run the search. Also log the
missing/unexpected keys, and do not log "loaded" in that case.

Smaller points in the same file:

- `find_module_state_dict` reassigns its parameter `state_dict` inside the loop
  (`state_dict = find_module_state_dict(value, ...)`). It works, but it is hard to follow. Use a
  new name.
- An empty `module_keys` (a module without state, such as `ReLU`) matches the first empty dict and
  returns `{}`, which the caller treats as "not found". Document or special-case it.
- There is no cycle protection for self-referencing containers.

### 2.2 The device helpers ignore buffers, which matters since #252 [verified]

`src/sonnix/utils/device.py`

Frozen encoders now hold their tensors as buffers. `get_module_device(AsinhNumericalEncoder(...).to("meta"))`
returns `cpu`, and `is_module_on_device(module, meta)` is vacuously `True` for any module with
only buffers. Use `itertools.chain(module.parameters(), module.buffers())` in
`get_module_device`, `get_module_devices` and `is_module_on_device`, and document that. The same
applies to `has_parameters`/`num_parameters` only if you want buffers counted, which you
probably do not: keep those parameter-only, but say so.

### 2.3 Focal and Poly-1 give different values for soft targets [verified]

`src/sonnix/functional/loss/focal.py`, `poly.py`

#261 derives `p_t` as `exp(-bce)` in the `*_with_logits` variants. That identity only holds for
binary targets. For `target = rand(6)`, `binary_focal_loss(sigmoid(x), t)` and
`binary_focal_loss_with_logits(x, t)` differ by up to 0.07 (focal) and 0.19 (Poly-1). The
docstring says "targets are expected to be binary", but the probability variants accept soft
targets (label smoothing, mixup) and produce a different, arguably more meaningful value. Either:

- compute `p_t = sigmoid(x) * t + (1 - sigmoid(x)) * (1 - t)` in both (one extra op, same
  results), or
- validate/document the binary requirement in both variants and in the module docstrings.

I recommend the first, since it makes the pair interchangeable (which is what the shared base class
in #280 promises) and adds a test `with_logits(x, t) == probs(sigmoid(x), t)` for soft `t`.

### 2.4 `NaN` gradients at exactly 0 [verified]

- `geometric_mean_indicator` and `moment_mean_indicator` (k >= 2) have a `NaN` gradient where
  `prediction == 0` (`sqrt(0)`, `0 ** (1/k)`). `relative_loss(..., geometric_mean_indicator)` with
  a prediction of exactly 0 (any ReLU output) gives `grad = [nan, ...]`. Fix: compute
  `sqrt(max(|t| * |p|, eps))`, or use `exp(0.5 * (log|t| + log|p|))` with a clamp. The
  indicator is later clamped to `eps` in `relative_loss`, but the clamp does not help because
  `NaN * 0` is `NaN`.
- `binary_focal_loss_with_logits` with `gamma < 1` has a `NaN` gradient when `p_t == 1`
  (`(1 - p_t) ** gamma` with `gamma = 0.5` at 0). Validate `gamma >= 1` or compute the weight with
  a guarded `pow`. `gamma in (0, 1)` is unusual but accepted by the validator.

### 2.5 `quantile_regression_loss` at `q=0.5` is half of the MAE [verified]

The docstrings of `quantile_regression_loss` and `QuantileRegressionLoss` say "`q=0.5` is
equivalent to the Mean Absolute Error". It returns `0.5008` where `l1_loss` returns `1.0016`. The
pinball loss is `max(q * d, (q - 1) * d)`, so at `q=0.5` it is `0.5 * |d|`. Fix the docstring (or
rescale by `2`, but that is not the standard definition).

### 2.6 Other numerical points

- `ExU.forward` computes `weight.exp()` in the input dtype. In `float16` it overflows for
  `weight > 11`. The init (`N(4, 0.5)`) is far from it, but nothing bounds it during training.
  Document, or add a clamp option as in the NAM reference implementations.
- `PiecewiseLinearNumericalEncoder` with duplicate bin edges (`bins = [0, 0, 1]`) sets the zero
  width to `1.0`, so the first segment contributes `0.5` for `x = 0.5` instead of saturating at
  `1` [verified]. The paper's reference drops duplicate edges. Either dedupe the edges per feature
  (needs ragged handling) or set the width to `eps` so the segment saturates, and warn.
- `msle_loss` silently returns `nan` below `-1`. `Pow(exponent=0.5)` has the same issue for negative
  inputs. Both are documented in the first case, not in the second.

### 2.7 Unvalidated or inconsistently validated arguments [verified]

The modules and the functions do not validate the same things:

| Call                                                          | Result                                                   |
| ------------------------------------------------------------- | -------------------------------------------------------- |
| `GeneralRobustRegressionLoss(scale=0)`                        | `ValueError`                                             |
| `general_robust_regression_loss(..., scale=0)`                | returns `nan`                                            |
| `PoissonRegressionLoss(eps=-1)` / `RelativeLoss(..., eps=-1)` | accepted                                                 |
| `AsinhSmoothL1Loss(beta=-1)`                                  | accepted, fails in `forward` with a torch `RuntimeError` |
| `ReLUn(max_val=-1.0)`                                         | accepted, returns `-1` everywhere                        |
| `Clamp(min_val=2.0, max_val=-2.0)`                            | accepted, returns `-2`                                   |
| `MultiQuadratic(num_parameters=0)`                            | accepted                                                 |
| `SafeLog(min_val=-1.0)`                                       | accepted                                                 |

Put the validation in one place per hyperparameter: a `validate_*` helper in the functional module
(as `validate_focal_params` already does) called by both the function and the module constructor,
using `coola.validation` (`validate_gt`, `validate_ge`, `validate_range`) as the project rules
require. Do the same for `eps` in `absolute_relative_error`, `symmetric_absolute_relative_error`,
`poisson_regression_loss` and `relative_loss`, and for `beta` in the two `smooth_l1` losses.

---

## 3. API design (P1, breaking changes welcome)

### 3.1 Hyperparameters are still not readable [verified]

`ReLUn`, `Clamp`, `Snake`, `Pow`, `View`, `ConcatFusion`, `AsinhSmoothL1Loss`, `BinaryFocalLoss`,
`QuantileRegressionLoss`, ... store `self._max`, `self._alpha`, `self._beta`, ... and expose nothing
(`hasattr(ReLUn(), "max_val")` is `False`), while `reduction`, `ExU.in_features` and
`ToBinaryLabel.threshold` are public. Users cannot read or log a config, and `copy`/`torch.save` of
a module are inconsistent in what they expose. Pick public attributes (or read-only properties)
for every constructor argument, named like the argument, and keep `_`-prefixed names only for
derived state. This was 3.4 in the previous review and was not done.

### 3.2 `RelativeSmoothL1Loss` hides `beta`

`beta` goes into `nn.SmoothL1Loss(reduction="none", beta=beta)`, and the class stores nothing, so
the repr shows `(criterion): SmoothL1Loss()` (torch prints no `beta`). Store `beta` and show it in
`extra_repr`, and validate it (`beta >= 0`). `RelativeLoss` also stores `eps` without `float()`
(unlike every other module).

### 3.3 Shared `extra_repr` in `BaseLoss`

Every loss repeats `extra_repr`, and the order is inconsistent: `reduction=mean, beta=1.0`
for `AsinhSmoothL1Loss`, `alpha=..., reduction=...` for the focal losses. Let `BaseLoss.extra_repr`
return `reduction=...` and have subclasses extend it (`f"{super().extra_repr()}, beta={...}"`),
or generate the repr from a tuple of hyperparameter names.

### 3.4 Inconsistent names across layers

- `ExU`, `NLinear` use `in_features`/`out_features`; the encoders use `input_size`/`output_size`;
  `ScaleAndShift` and the dynamic norms use `normalized_shape`. Pick `in_features`/`out_features`
  for anything that maps a feature dimension, and `n_features`/`feature_size` for encoders (the
  `extra_repr` already prints `n_features=..., feature_size=...`, so the properties disagree with
  the repr).
- `TransformedLoss(criterion, prediction, target)`: `prediction` and `target` are the
  _transform modules_, the same words as the forward arguments. Rename to `prediction_transform` and
  `target_transform` (breaking).
- Loss function arguments are `(prediction, target)` while torch uses `(input, target)`, and the
  activation functions use `input`. That is a legitimate choice, but state it in the docs.

### 3.5 Options must be keyword-only

`CLAUDE.md` says new options are keyword-only, but the following accept them positionally:
`SafeExp(max_val, preserve_gradient)`, `SafeLog`, `safe_exp`, `safe_log`, `NLinear(..., bias, device, dtype)`,
`ExU`, `BaseAlphaActivation(num_parameters, init, learnable, device, dtype)`, `ScaleAndShift`,
`BaseDynamicNorm(..., device, dtype)`, the encoders' `learnable`, `create_*` class methods.
`SafeExp(20.0, True)` is unreadable. Add `*` before `preserve_gradient`, `learnable`, `device`,
`dtype`, `bias`.

### 3.6 `Snake` is not the paper's Snake

The paper's `a` is a learnable parameter (per-channel). `Snake(frequency)` is a fixed float with no
`device`/`dtype`. Make it a `BaseAlphaActivation` (which already has `learnable`,
`num_parameters`, `device`, `dtype`), with the forward `x + sin(a x)^2 / a`. Note the formula
currently computed (`x - cos(2 a x) / (2 a) + 1 / (2 a)`) is equivalent but costs more ops.

### 3.7 Trivial wrapper modules

(Open from the previous review, 3.6.) `Exp`, `Expm1`, `Log`, `Log1p`, `Sin`, `Sinh`, `Asinh`,
`Square`, `ToFloat`, `ToLong`, and the eight indicator classes are one-liners with 25-30 lines of
docs each, which is a lot of text for no logic. Options:

1. A single `Elementwise(name: Literal["exp", "log", ...])` module plus a `ToDtype(dtype)` (this
   replaces `ToFloat`/`ToLong`, which hard-code `float32`/`int64`).
2. Keep the names for config files but generate them from one base (`fn` class attribute).

Also: `Square` duplicates `Pow(2)`, and `ToCategoricalLabel` has no `dim` argument.

### 3.8 `ScaleAndShift` and `BaseDynamicNorm` duplicate each other

Same `normalized_shape` parsing, same weight/bias creation, same `reset_parameters` (plus `alpha`).
`DynamicTanh`/`DynamicAsinh` are `ScaleAndShift` with an inner nonlinearity and a scalar `alpha`.
Make `BaseDynamicNorm` extend `ScaleAndShift`. `BaseDynamicNorm` also has no `forward` and does not
use `ABC`/`abstractmethod` (the indicators do since #275), and the subclasses' docstrings do not
mention `device`/`dtype`. Neither module has an option to drop the bias.

### 3.9 `AverageFusion(SumFusion)`

It subclasses `SumFusion` only to hard-code `normalized=True` (and its own `__init__` takes no
argument). Make them thin wrappers of one private `_ReduceFusion`, or drop `normalized`
from `SumFusion`. `MultiplicationFusion`, `SumFusion`, `ConcatFusion` repeat the same
`if not inputs: raise ValueError` block three times.

### 3.10 `functional.error` and `relative_loss`

- `absolute_error`, `absolute_relative_error` and `symmetric_absolute_relative_error` say "same
  shape" but silently broadcast [verified: `(3,1)` and `(1,4)` give `(3,4)`]. `relative_loss`
  _does_ check shapes. Pick one policy and document it.
- These three functions have no module counterparts, while every loss function has one except
  `log_cosh_loss` and `msle_loss`, which also lack modules (see section 8).

### 3.11 `View` should use `reshape`

`View.forward` uses `x.view(*shape)`, which fails on non-contiguous input [verified]. #283 fixed
this for `MulticlassFlatten`; do the same here (or name the module `Reshape` and keep `View` as an
alias). `Squeeze` accepts only an `int`, while `Tensor.squeeze` takes a tuple since torch 2.0.

### 3.12 `utils` public API with no user in the package

Nothing in `src/` uses `resolve_torch_generator`, `get_named_modules`, `top_module_mode`,
`get_module_devices`, `is_module_on_device`, `resolve_dataset`, `resolve_optimizer`,
`is_dataset_config`, `is_optimizer_config`, `create_sequential`, or `find_module_state_dict`
(only tests and docs reference them). The library describes itself as "a library of PyTorch
modules", so decide what `sonnix.utils` is: either a documented, supported toolbox (then give it a
docs page per module, see 1.2) or an internal package with `_` names. In particular, the dataset and
optimizer config helpers have nothing to do with modules or losses and could be dropped.
`get_named_modules(depth=...)` also has two code paths (`depth == 1` vs `> 1`, the second counts dots
in names); a single bounded traversal is simpler. `module_mode` keys the saved modes by name, which
is wrong for shared submodules that appear under two names (the last one wins).

### 3.13 Numerical encoders

(Open from the previous review, 3.9.)

- `create_rand_frequency` / `create_rand_value_range` use the global RNG: no `generator`/`seed`
  argument, and no `dtype`/`device`. `sonnix.utils.random.resolve_torch_generator` exists for
  exactly this and is unused.
- Six `create_*` class methods (`rand`/`linspace`/`logspace` x `frequency`/`value_range`) differ in
  one line each. One `create(spacing: Literal["rand", "linspace", "logspace"], *, frequency_range=None,
value_range=None)` removes six near-identical docstrings and signatures.
- `prepare_tensor_param`, `check_frequency` and `check_abs_range` are in `sine.py`'s `__all__` and
  imported from there by `asinh.py` and `piecewise.py`. `check_frequency`/`check_abs_range` are
  hand-written range checks that `coola.validation.validate_ge`/`validate_gt` already cover (project
  rule). Move the shared helpers to a private `numerical/_utils.py`.
- `learnable=False` registers a clone of the tensor, `learnable=True` registers the parameter
  from a clone in `CosSin` but from the original in `AsinhNumericalEncoder` (`nn.Parameter(scale)`
  aliases the caller's tensor, so in-place changes of `scale` leak into the module). Clone in both.
- Add `PiecewiseLinearNumericalEncoder.from_quantiles(x, n_bins)`; it is the paper's main use case
  and the current constructor requires users to compute the bins themselves.

### 3.14 Alpha activations

- Per-channel `alpha` broadcasts over the **last** dimension (`nn.PReLU` uses dim 1). Document it,
  or add a `channel_dim` argument. For an `(N, C, L)` input a `Gaussian(num_parameters=C)` raises a
  shape error [verified].
- `init` and `num_parameters` are not validated (`num_parameters=0`, `init=0` for `Gaussian` give a
  degenerate module).
- `ExpSin`, `Gaussian`, ... docstrings document 3 constructor arguments but the class takes 5
  (`device`, `dtype` are inherited and undocumented).

---

## 4. Packaging and dependencies (P2)

- `torch>=2.4,<3.0`: the CI matrix should include the oldest supported torch. I did not find a
  job pinning torch 2.4 (`ci-test-deps.yaml` has no `torch` reference); verify and add one, or
  raise the floor to what is tested. The environment used for this review has torch 2.15.0, so
  nothing in this review says anything about 2.4.
- `environment.yaml` (conda) duplicates the dependency list. Generate it or delete it, along with
  the `make conda` instructions (see 5.2).
- `[tool.pyright]` hard-codes `venvPath = "."` and `venv = ".venv"`, which breaks for contributors
  with another environment layout. Prefer running `uv run pyright`.
- Two type checkers run side by side (`pyright` and `ty`). `ty` is pre-1.0 (`>=0.0.97`). Gate on
  one and keep the other advisory, otherwise every `ty` release can turn CI red.
- `Changelog` URL points at GitHub Releases, and there is no `CHANGELOG.md`. CLAUDE.md says to
  update "any changelog": there is none to update. Since this review recommends several breaking
  changes (3.1, 3.4, 3.5, 3.7), start one now.

---

## 5. Documentation (P1-P2)

### 5.1 README and home page

- README badges point to workflows that do not exist: `nightly-tests.yaml`, `docs.yaml`,
  `docs-dev.yaml` [verified against `.github/workflows/`]. The "Code style: black" badge is wrong
  (the repo uses `ruff format`).
- README is a badge wall plus "A library of PyTorch modules". `docs/docs/index.md` has
  `## Overview` and `## Motivation` both set to `TODO`. Add: install, a short example for a loss, a
  module and an encoder, what the library contains, and why use it instead of plain PyTorch.

### 5.2 Stale instructions [verified]

- `docs/docs/get_started.md`: still describes `poetry`, "Python 3.8+" and `make conda`. The repo
  uses `uv`, requires Python >= 3.11, and `inv` for tasks.
- `CONTRIBUTING.md` says `make unit-test-cov`; no such target exists (`grep` over the `Makefile`
  and `.make/` finds only the CONTRIBUTING mention). Document the real commands
  (`inv test.unit`, `inv test.all`, `inv test.doctest`, ... see `inv --list`).
- `mkdocs.yml` has the whole User Guide nav commented out. Write a short guide (losses and
  reductions, activations, numerical encoders, building models from configs) or delete the comment.
- The API reference is one `::: package` block per sub-package, so `sonnix.modules` renders as one
  page with 64 classes. Split by sub-module (activation, loss, numerical, fusion) and add a
  summary table per page.
- `notebooks/activations.ipynb` is not linked from the docs and not executed in CI.

### 5.3 Docstring errors still present [verified by reading]

- `general_robust_regression_loss`: "If `max` is set" (the argument is `max_val`), "adaptative"
  typo and a mis-indented `Note:`; `GeneralRobustRegressionLoss` has the same typo. The new `alpha`
  documentation says the limit cases are used "when `alpha` is within `1e-5`" but the signature
  is `alpha: float`, so the tensor support mentioned in #259 is not typed. A 0-d tensor works
  [verified], any other shape cannot (the code branches with Python `abs(alpha - 2) < eps`).
  Either type it as `float` and remove the claim, or implement the tensor path with `torch.where`.
- `PoissonRegressionLoss` doctest feeds `torch.randn` predictions although the docstring says the
  loss is only meaningful for positive predictions (the functional example was fixed in #260, the
  module example was not). The functional docs also say the targets "must be positive (zero is
  allowed)".
- `BinaryFocalLoss` / `BinaryFocalLossWithLogits` class docs describe `reduction` with the older
  three-value wording; use the same paragraph as everywhere else (the paragraph is duplicated in
  about 20 docstrings, see 7.2).
- `MultiQuadratic`/`Quadratic`: "Formula" is correct, but the `learnable` docs say "the parameters are
  learnt", while `num_parameters` text says "there is only two values are legitimate" (grammar).

---

## 6. Tests (P2)

Status: done in #337 (registry, `export`, `compile`), #339 (dtypes), #340 (`gradcheck`), #341
(equivalence), #342 (reduction registry) and #336 (GPU note). The `hypothesis` and
`pytest-benchmark` dependencies are still unused. The `compile` test uses the `eager` backend with
`fullgraph=True` (no C++ compiler needed). `float16` is not tested.

From the layout and greps (not a test-by-test audit):

- **No `gradcheck`, no `hypothesis`, no `torch.compile`/`torch.export` test** anywhere in `tests/`,
  although `hypothesis` is declared in the `property` group, and `pytest-benchmark` in `dev` has no
  benchmark. I ran `torch.export.export` by hand on `NLinear`, `ExU`, `CosSinNumericalEncoder` and
  `PiecewiseLinearNumericalEncoder`; all four export [verified]. A parametrized smoke test over every
  module in `sonnix.modules.__all__` (build with a minimal config, run `export` and `compile` on
  CPU) would keep it that way, and it is the right place to catch the Python-side control flow in
  `PiecewiseLinear...forward`.
- `float64` appears in 8 places and `bfloat16` in 3, `float16` nowhere. Add a dtype-parametrized
  forward/backward test over all modules and functions (`float32`, `float64`, `bfloat16`), and a
  test that `module.to(dtype)` keeps buffers and parameters consistent.
- Add `gradcheck` (in `float64`) for every functional loss and activation. It would have found 2.4
  (NaN gradient at 0), which none of the current `loss.backward()` smoke tests detect, because they
  only check that the backward runs, not that gradients are finite.
- Add **equivalence tests** against PyTorch or against another variant: `with_logits` vs probability
  focal/Poly-1 for soft targets (2.3), `NLinear` vs `n` separate `nn.Linear`, `quantile(q=0.5)` vs
  `0.5 * l1_loss` (2.5), `general_robust(alpha=2)` vs `mse_loss`, `poisson_regression_loss` vs
  `poisson_nll_loss(log_input=False)` up to the eps.
- Regression tests for 2.1 (`strict=False` with a wrapped checkpoint) and 2.2 (buffers and
  devices) should be written first, to verify they fail before the fix.
- The unit tests of the `*_invalid_reduction`, `*_extra_repr`, `*_reduction_*` kind are repeated for
  each loss. One parametrized test over a registry of `(function, module, kwargs)` would check all
  of them, and a new loss would be covered by registering it. The registry can also drive the
  gradcheck, dtype and export tests above, so the cost is paid once.
- CI never runs a GPU or MPS job (`torch_cuda_available` tests are skipped on every runner). Say so
  in `CONTRIBUTING.md` and run the CUDA tests locally before a release.

---

## 7. Tooling and CI (P2)

### 7.1 Workflow and lint configuration

- 28 workflow files for a 6.8k-line library, including a stale-run canceller
  (`bot-cancel-stale-queued-runs`, which `concurrency: cancel-in-progress` covers), three
  `lib-get-*` reusable workflows and four `nightly-*` ones. Merge what can be merged. Check that
  every workflow sets `concurrency` and minimal `permissions` (the ones I read do set
  `permissions: contents: read`).
- `E501` and `W505` are ignored while `line-length = 100` and `pycodestyle.max-doc-length = 72` are
  set, so both limits are no-ops for ruff (`ruff format` and `docformatter` do the wrapping). Remove the
  dead settings or enforce them. `docformatter` and `ruff` docstring rules overlap and need
  `D205`/`D209` disabled to not fight each other; recent `ruff format` formats docstring code, so
  one of them could go.
- `G004` (f-strings in logging) is ignored globally, and `utils/loss.py` and `utils/state_dict.py`
  log with f-strings. Prefer `%s` lazy logging in new code.
- `.benchmarks/`, `htmlcov/`, `coverage.xml` and `.coverage` are in the working tree. They are
  ignored by git (no tracked file matches), so this is only about `inv clean.all` covering them.

### 7.2 Repeated documentation blocks

The five-line `reduction` paragraph appears in about 20 docstrings (every loss function and
module). It is the reason `reduction` documentation drifts (see 5.3). Define it once, for example
as a module-level constant used by a docstring decorator, or refer the reader to
`sonnix.functional.reduce_loss` and keep one line per docstring. `mkdocstrings` supports
`--8<--` snippets for the markdown pages, but not for docstrings, so the reference-by-link approach
is the pragmatic one.

---

## 8. Structure and missing features (P2-P3)

### 8.1 Structure

Status: `Clamp` moved (#329), base classes exported (#330), loss modules import from
`sonnix.functional` (#331), `validate_focal_params` moved to `sonnix.validation` (#332).
`check_moment_order` and `check_loss_reduction_strategy` had already been handled (validators in
`sonnix.validation`); the stale mention in `CLAUDE.md` is fixed in #333.

- Every class has two or three import paths: `sonnix.modules.X`, `sonnix.modules.<sub>.X` and the
  file path. `Clamp` is imported in `sonnix/modules/__init__.py` from `fusion.clamp` but is not in
  `fusion/__all__`, and it is not a fusion module (it has a single input); move it next to
  `ReLUn` in `modules/activation/` (a `ReLUn` is `Clamp(0, n)`).
- `BaseDynamicNorm`, `BaseBinaryFocalLoss`, `BaseBinaryPoly1Loss`, `BaseLoss` are public in their
  sub-package but not in `sonnix.modules.__all__` (while `BaseAlphaActivation` and
  `BaseRelativeIndicator` are). Decide which base classes are API.
- `modules/loss/poisson.py` and `quantile.py` import from `sonnix.functional.loss`, while the other
  loss modules import from `sonnix.functional`. Use one.
- `validate_focal_params` and `check_moment_order` are public (no underscore, as the project rule
  asks) but not in `__all__`, and `modules/loss/focal.py` imports one of them by file path.
  Export them or move them to `sonnix.functional.validation`.
- `sonnix.functional.__init__` re-exports `check_loss_reduction_strategy`; callers should use
  `validate_in` from `coola`, or `reduce_loss` directly (which already checks). Keep only
  `reduce_loss` and `Reduction` public.

### 8.2 Missing counterparts

Status: `LogCoshLoss` and `MSLELoss` added (#334). The learnable `Snake` was done with 3.6. The
other candidates (`Tweedie`, `Huber` with scale, `Dice`/`Tversky`, `SwiGLU`/`GeGLU`, the
`NumericalEmbedding` docs example) are new features, not fixes, and are not done.

- No `LogCoshLoss` and no `MSLELoss` module although `log_cosh_loss` and `msle_loss` exist.
  `AsinhMSELoss` and `AsinhSmoothL1Loss` have both.
- Candidates that fit the scope: `Tweedie` and `Huber`-with-scale losses, `Dice`/`Tversky` for
  segmentation, `SwiGLU`/`GeGLU`, a learnable `Snake` (3.6), and a `NumericalEmbedding`
  composition (encoder + `NLinear`) in the docs, which shows why `NLinear` and the encoders are in
  the same library.

### 8.3 Performance (P3)

Status: the piecewise encoder is done (#335). `NLinear`: benchmarked on CPU (4 shapes, forward and
forward + backward) `reshape + bmm + transpose`, `einsum("bni,nio->bno")` and `baddbmm`: none is
consistently faster than the current code (`einsum` is 10-80% slower in forward on 3 of 4 shapes,
`baddbmm` is up to 87% slower in forward + backward), so `NLinear.forward` is unchanged. `ExU` is
unchanged, as the review said.

- `PiecewiseLinearNumericalEncoder.forward` builds a Python list with a conditional splat and a
  `cat` on every call. Precomputing per-position `lo`/`hi` buffers (`lo[0] = -inf`, `hi[-1] = inf`)
  gives one `clamp(min=lo, max=hi)` and no data-dependent Python branches.
- `NLinear.forward` does reshape + transpose + `bmm` + transpose + add + reshape. A single
  `torch.baddbmm` or `einsum("bni,nio->bno")` avoids the two transposes. Benchmark it first
  (`pytest-benchmark` is already a dev dependency).
- `ExU.forward` recomputes `weight.exp()` every call; with `torch.compile` this is fused, in eager
  it costs one extra elementwise op. Not worth changing.

---

## 9. Suggested order of work

One pull request per line, each with a test that fails before the fix.

1. **Unbreak `main` (P0):** 1.1 (doctest reprs), 1.2 (docs reference), 1.3 (`import-linter`
   config or remove the hook). These are small, and they make three CI checks green again.
2. **Correctness (P1):** 2.1 (`strict=False` loading), 2.2 (buffers in device helpers), 2.3
   (soft-target focal/Poly-1), 2.4 (NaN gradients of the indicators), 2.5 (quantile docstring), then 2.7
   (one validation point per hyperparameter).
3. **API clean-up before the first beta (P1):** 3.1 (public hyperparameters), 3.5 (keyword-only
   options), 3.4 (names), 3.2/3.3 (`BaseLoss.extra_repr`, `beta`), 3.6 (`Snake`), 3.11 (`View`). Add a
   `CHANGELOG.md` with this PR, since most of these break callers.
4. **Docs (P1-P2):** 5.1-5.3, and the docs build as a required check.
5. **Tests (P2):** a registry of losses/modules driving reduction, dtype, `gradcheck`, `export` and
   equivalence tests (section 6), then delete the per-loss duplicates they replace.
6. **Structure and extras (P2-P3):** 3.7-3.10, 3.12-3.14, sections 4, 7 and 8.
