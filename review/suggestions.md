# Sonnix codebase review

Scope: everything tracked in the repo (`src/`, `tests/`, `docs/`, `notebooks/`, `dev/`, `.github/`,
`.make/`, `pyproject.toml`, README/CONTRIBUTING). Breaking changes are acceptable, and the package is
still `0.0.1a0`, so this is the right time to make them.

State at review time: 6518 tests pass (21 skipped), 100% line coverage, 134 doctests pass, `ruff`
clean. `pyright` reports **1 error** and `lint-imports` **fails** (see 6.1 and 6.2).

Items marked **[verified]** were reproduced by running code. The others come from reading the source.

Priority legend: **P0** = wrong results or broken tooling, **P1** = API or design issue that is cheap
to fix now and costly after release, **P2** = quality, **P3** = nice to have.

---

## 1. Correctness bugs (P0)

### 1.1 `NLinear` is initialised ~8x too small [verified]

`src/sonnix/modules/nlinear.py:92-96`

The weight has shape `(n, in, out)`. `init.kaiming_uniform_` and `_calculate_fan_in_and_fan_out`
treat it like a conv kernel: dim 0 is the output maps, dim 1 the input maps, dim 2 the receptive
field. So `fan_in = in * out` instead of `in`. For `NLinear(3, 16, 64)` I measured a weight std of
0.018, against 0.144 for `nn.Linear(16, 64)`. The factor is `sqrt(out_features)`, which grows with
layer width. The bias bound is wrong in the same way (0.031 instead of 0.25).

Fix: compute the bound explicitly, e.g. `bound = 1 / math.sqrt(in_features)`, and use
`init.uniform_(self.weight, -bound, bound)` for both parameters. Also add a test that compares the
std to `nn.Linear`.

Related issues in the same file:

- `x.view(-1, ...)` fails on non-contiguous input. Use `reshape`.
- `x @= self.weight` and `x += self.bias` look like in-place ops but are not clearly so, and `+=`
  mutates an intermediate. A single `torch.baddbmm(bias.unsqueeze(1), x, weight)` or
  `torch.einsum("bni,nio->bno", ...)` is clearer and avoids the transposes.
- Add `device`/`dtype` factory kwargs to the docs (they exist but are undocumented in `Args`).
- The docstring comment `# Example with 1 feature` is wrong.

### 1.2 `log_cosh_loss` overflows for |error| > ~89 (fp32) [verified]

`src/sonnix/functional/loss/log.py:51`

`cosh(x).log()` returns `inf` for `x=100`, and the gradient is `nan`. Use the stable form
`|x| + softplus(-2|x|) - log(2)` (or `x + softplus(-2x) - log 2`).

The `scale` parameter divides the error but does not rescale the loss, so `scale` changes the loss
magnitude as well as its shape. Document this, or multiply by `scale` so the loss is comparable to
Huber.

### 1.3 `ExpSin` docstring disagrees with the implementation [verified]

`src/sonnix/modules/activation/alpha.py:103`

The docstring says `exp(-sin(alpha * x))`, which matches the paper. The code computes
`exp(+sin(alpha * x))`: `ExpSin()(tensor([1.0]))` gives 2.3198 where the formula gives 0.4311.
Decide which one is intended and fix the other.

### 1.4 `ExU` bias sign and shape

`src/sonnix/modules/exu.py:79`

The paper defines `h(x) = f(exp(w) * (x - b))`. The code does `x + b`. This is only a
reparametrisation, but it matters when loading paper weights. The `extra_repr` line is also over
100 chars. Document the sign convention or follow the paper.

### 1.5 `ResidualBlock` / `TransformedLoss` / `RelativeLoss` use `x or default` on modules [verified]

`src/sonnix/modules/residual.py:53`, `modules/loss/transform.py:55-56`, `modules/loss/relative.py:73`

`skip or nn.Identity()` treats any falsy module as "not provided". `nn.Sequential()` defines
`__len__`, so an empty `Sequential` is falsy. `ResidualBlock(..., skip=nn.Sequential())` silently
becomes `Identity` (confirmed). An empty config dict does the same. Use `if skip is None`.

### 1.6 `CosSinNumericalEncoder` doubles its learnable parameters [verified]

`src/sonnix/modules/numerical/sine.py:85-89`

`frequency.repeat(1, 2)` creates one parameter of width `feature_size`, so the sin half and the cos
half are optimised independently. With `learnable=True` the sin and cos features no longer share a
frequency after the first step. That contradicts the docstring (`feature_size // 2` frequencies) and
changes what is being learned. Store the half-width parameter and concatenate in `forward`
(`torch.cat((sin(z), cos(z)), -1)` with `z = x * freq + phase`). This is breaking for state dicts,
which is fine at this version.

### 1.7 Frozen parameters registered as `nn.Parameter` [verified]

`AsinhNumericalEncoder`, `CosSinNumericalEncoder` (`learnable=False` default)

A frozen tensor stored as `nn.Parameter(requires_grad=False)` appears in `module.parameters()`.
`num_parameters(AsinhNumericalEncoder(torch.ones(2, 3)))` returns 6 and the optimizer receives it.
Register it as a buffer when `learnable=False` (or make `learnable` structural: buffer vs
parameter). `BaseAlphaActivation` has the same pattern.

### 1.8 Division by zero and unvalidated hyperparameters

- `MomentMeanIndicator(k=0)` raises a raw `ZeroDivisionError` in `forward` [verified]
  (`functional/loss/relative.py:233`). Validate `k >= 1` in the constructor and the function.
- `BinaryFocalLoss(alpha=5.0)` is accepted [verified], although the docs say `[0, 1]`. `gamma` is
  validated in the module but not in the functional API.
- `QuantileRegressionLoss(q=...)` does not check `0 <= q <= 1`.
- `Snake(frequency=0)` divides by zero. `Gaussian`/`Laplacian` give `inf`/`nan` when `alpha` reaches
  0 during training. Consider a clamp or an `eps`, or document it.
- `reduce_loss(torch.tensor(1.), "batchmean")` raises `IndexError` on 0-d input [verified].

### 1.9 `reduction="batchmean"` is accepted inconsistently [verified]

`check_loss_reduction_strategy` accepts `batchmean`, but every docstring says only
`mean`/`none`/`sum`. `BinaryFocalLoss(reduction="batchmean")` is accepted. `asinh_mse_loss(...,
reduction="batchmean")` fails inside PyTorch with a different error and message, and `AsinhMSELoss`
would pass `__init__` validation and then fail in `forward`. Either support it everywhere (route
the asinh/msle losses through `reduce_loss` too) or remove it. Same for `RelativeLoss`.

### 1.10 Docstrings that describe the wrong behaviour

- `binary_poly1_loss*` and `BinaryPoly1Loss*`: `alpha` is documented as "in `[0, 1]`, ignored if
  negative", copied from the focal loss. The code uses `ce + alpha * (1 - p_t)` with no such rule.
  The module docs also list a `gamma` argument that does not exist
  (`modules/loss/poly.py:26,104`).
- `safe_exp`/`SafeExp`: "higher than the specified _minimum_ value" should read "maximum".
- `Pow`: described as "natural logarithm of the input" (copy-paste).
- `is_dataset_config` / `is_module_config` / `is_optimizer_config`: all say "configuration for a
  `torch.nn.Module`".
- `is_loss_decreasing_with_sgd` example calls `is_loss_decreasing_with_adam`.
- `is_loss_decreasing` example uses `SGD` without importing it, and `relative_loss` uses
  `classical_relative_indicator` without importing it. They pass under xdoctest only because it
  shares the module namespace. Add the imports so the rendered docs are copy-pasteable.
- `ScaleAndShift`, `BaseDynamicNorm`, `DynamicAsinh`, `DynamicTanh`: "willcnormalize" typo.
  `BaseDynamicNorm`'s example instantiates `DynamicTanh`, and the class has no `forward`.
- `get_random_seed` says the range is up to `2**63 - 1`, but `randint`'s upper bound is exclusive.

---

## 2. Numerical robustness (P1)

- **`clamp` kills gradients.** `safe_exp`, `safe_log`, `general_robust_regression_loss(max=...)`,
  `poisson_regression_loss`, `relative_loss` (eps), `absolute_relative_error` all use `clamp`, which
  gives zero gradient outside the range. This is often the intent, but it is not documented. Offer a
  gradient-preserving variant (e.g. `safe_log(x) = log(x + eps)` or a straight-through clamp) and
  document the difference.
- **Barron loss** (`general_robust_regression_loss`): the `alpha == 2` / `alpha == 0` branches use
  exact float equality. `alpha=1e-3` falls into the generic formula, which is numerically unstable
  near 0 (the paper's reference code uses a stabilised form). Add `alpha = -inf` and handle the
  neighbourhood of 0 and 2 with an epsilon switch. It also does not support a tensor `alpha`, so the
  "adaptive" half of the paper cannot be built on it. Accepting `alpha: float | Tensor` would make a
  learnable-`alpha` module a small addition.
- **Poisson loss** duplicates `torch.nn.functional.poisson_nll_loss(log_input=False, eps=...)`. If
  you keep it, document the difference (no Stirling term; `clamp` vs `+ eps`). The docs say the
  inputs "must be positive" but the example uses `randn`.
- **Focal / Poly1 `*_with_logits`** compute `sigmoid` and a separate BCE. The focal weight can be
  derived from the BCE (`p_t = exp(-ce)`, as torchvision does) to save an op and remove the
  duplicated `p_t` code. Note that `binary_cross_entropy` (the probability variants) raises under
  `torch.autocast`; document that the `*_with_logits` variants are the AMP-safe ones.
- **`msle_loss`** returns `nan` for inputs `< -1`, silently. Document, or add a `clamp(min=...)`
  option.
- **Mixed precision.** Buffers/params created in `float32` (`BaseAlphaActivation` hard-codes
  `dtype=torch.float`, `PiecewiseLinearNumericalEncoder` buffers, the numerical encoders) ignore the
  `dtype=` convention that `NLinear` and `ExU` follow. Pick one policy: accept
  `device`/`dtype` everywhere, or never.

---

## 3. API design (P1, breaking changes welcome)

### 3.1 One loss base class, one reduction type

Almost every loss module repeats the same `check_loss_reduction_strategy` + `self.reduction`
boilerplate (about 15 classes), plus identical `extra_repr`/`forward` shapes. Introduce:

```python
Reduction = Literal["none", "mean", "sum"]  # (+ "batchmean" if kept)


class BaseLoss(nn.Module):
    def __init__(self, reduction: Reduction = "mean"): ...
```

or subclass `torch.nn.modules.loss._Loss`, which already provides `reduction`. Typing `reduction`
as `Literal` gives editor and type-checker support, and `str` currently lets `reduction="meen"`
through the type checker.

### 3.2 Remove duplicated module pairs

`BinaryFocalLoss`/`BinaryFocalLossWithLogits` and `BinaryPoly1Loss`/`BinaryPoly1LossWithLogits` are
line-for-line copies. The same applies to the functional pairs. Options: a `from_logits: bool`
flag, or a shared private base that holds validation and `extra_repr`. The same applies to
`AsinhMSELoss`/`AsinhSmoothL1Loss`.

### 3.3 Sentinel values in public signatures

- Focal `alpha < 0` means "disabled". Use `alpha: float | None = None`.
- `ReLUn(max=...)`, `Clamp(min=..., max=...)`, `SafeExp(max)`, `SafeLog(min)`, and
  `GeneralRobustRegression(max=...)` shadow builtins and need `# noqa: A002`. The PyTorch names
  `min_val`/`max_val` (as in `nn.Hardtanh`) avoid the noqa and the shadowing.
- Functions that take `input` (also shadows a builtin) are fine for PyTorch parity, but the losses
  use `(prediction, target)` and `torch` uses `(input, target)`. Pick one and say so in the docs.

### 3.4 Inconsistent argument order and attribute visibility

- `log_cosh_loss(prediction, target, reduction, scale)` but `general_robust_regression_loss(...,
alpha, scale, max, reduction)` and `quantile_regression_loss(..., q, reduction)` vs
  `QuantileRegressionLoss(reduction, q)`. Make `reduction` always last and keyword-only
  (`*, reduction=`).
- Hyperparameters are private (`self._alpha`) in most modules but public (`self.alpha`, `self.reduction`)
  in others. Private attributes with no property mean users cannot read or tune them, and
  `torch.save`/`copy` see an inconsistent shape. Pick public read-only properties.
- `ExU`/`NLinear` expose `in_features`/`out_features`, but `ScaleAndShift` and the dynamic norms use
  `normalized_shape`.
- `input_size` / `output_size` on the numerical encoders, while the rest of the library uses
  `in_features` / `out_features`.

### 3.5 `utils.factory`

- `setup_object_typed` only **logs a warning** when the object is not an instance of `cls`
  (`factory.py:411-412`) and returns it anyway, so `setup_module({"_target_": "torch.optim.SGD",
...})` returns an optimizer typed as `Module`. This also causes the single pyright error. Raise
  `TypeError`.
- `setup_object` and `setup_object_typed` log f-strings at INFO (`G004` is ignored in ruff config).
  Prefer lazy `%s` logging, and `debug` level for per-object construction, which is noisy for deep
  configs.
- The module relies on a `fallback/objectory.py` shim whose three functions all raise. A simpler
  design is a lazy import inside the three functions that need it, with one clear error. This
  removes `utils/fallback/`, `check_objectory`, `objectory_available`, `raise_error_objectory_missing`
  and half of `utils/imports.py`.
- `utils/imports.py::objectory_available` is a decorator that silently returns `None` when the
  package is missing. That is a trap. Its name also collides with the pytest marker of the same name
  in `testing/fixtures.py`. Drop one.
- `is_dataset_config` / `setup_dataset` / `setup_optimizer` / `is_optimizer_config` have nothing to do
  with "a library of PyTorch modules". Consider moving the factory helpers to a separate package, or
  trimming to `setup_module` + `create_sequential`.

### 3.6 Trivial wrapper modules

`Exp`, `Expm1`, `Log`, `Log1p`, `Sin`, `Sinh`, `Asinh`, `Square`, `ToFloat`, `ToLong`,
`ClassicalRelativeIndicator`, ... are one-liners with ~30 lines of docs each. Two options:

1. Keep them (they are useful for config-driven models) but generate them from one `Elementwise`
   base with a `fn` attribute, plus `ToDtype(dtype)` replacing `ToFloat`/`ToLong`.
2. Add a generic `Lambda(fn)` / `Elementwise("exp")` module and deprecate the individual classes.

Either way, `ToFloat`/`ToLong` hard-code `float32`/`int64`, so a `ToDtype(torch.float64)` /
`ToDtype("bfloat16")` module is the natural generalisation.

### 3.7 Indicators

`BaseRelativeIndicator.forward` raises `NotImplementedError`. Use `abc.ABC` + `@abstractmethod` (or
delete the class and use plain callables). The eight indicator classes and eight functions differ
only in one expression. Consider one `RelativeIndicator(kind: Literal[...], k=1)` or registering
functions in a dict. Also:

- `RelativeLoss` docstring says `indicator` is "the name of the indicator function", but it takes a
  module or config.
- The denominator is not detached, so gradients flow through the indicator
  (e.g. `ReversedRelativeIndicator` is `|prediction|`, which rewards shrinking the prediction).
  Add `detach_indicator: bool` and document the default.
- `RelativeSmoothL1Loss` does not expose `beta`.
- `relative_loss` raises `RuntimeError` for a shape mismatch. `ValueError` is the right type.

### 3.8 Error types

Argument validation raises `RuntimeError` in `numerical/sine.py`, `fusion/*`, and
`functional/loss/relative.py`, and `ValueError` elsewhere. Use `ValueError`/`TypeError` for bad
arguments, keep `RuntimeError` for state errors. This is a breaking change, but the right one to
make before 1.0.

### 3.9 Numerical encoders

- `create_rand_frequency` / `create_rand_value_range` use the global RNG and cannot be seeded locally.
  The library already has `sonnix.utils.random.get_torch_generator`, but nothing in `src/` uses
  it. Add a `generator`/`seed` argument.
- Six `create_*` class methods (`rand`/`linspace`/`logspace` x `frequency`/`value_range`) could be one
  `create(spacing=..., range=...)`, or the `*_value_range` ones could be a flag.
- `prepare_tensor_param`, `check_frequency`, and `check_abs_range` are in `sine.py`'s public
  `__all__`, and `asinh.py`/`piecewise.py` import from `sine.py`. Move them to a private
  `numerical/_utils.py`.
- `PiecewiseLinearNumericalEncoder`: add a `from_quantiles(x, n_bins)` constructor (the paper's main
  use case). Bin duplicates are silently turned into width 1 (`width[width == 0] = 1.0`); log or
  raise instead.
- Add a learnable-embedding follow-up (the paper pairs these encoders with a per-feature linear
  layer), which is exactly what `NLinear` provides. A short `NumericalEmbedding` composition
  example in the docs would show the intent of this library.

### 3.10 Smaller API points

- `Snake`: the paper's `a` is learnable (and per-channel). Here it is a fixed float. Subclass
  `BaseAlphaActivation`, or add `learnable`.
- `BaseAlphaActivation.num_parameters` per-channel broadcasts over the **last** dim, while
  `nn.PReLU` uses dim 1. State this explicitly in the docs.
- `ScaleAndShift` and `BaseDynamicNorm` duplicate weight/bias creation and `reset_parameters`. Share a
  helper. They also lack `device`/`dtype`.
- `MulticlassFlatten` uses `.view` (fails on non-contiguous inputs) and returns `Any`. Use `reshape`,
  and type the return as `Tensor`.
- `Squeeze`, `View` mirror `nn.Flatten`/`nn.Unflatten`. Fine to keep, but consider `Reshape` with
  `reshape` instead of `view`.
- `AverageFusion(SumFusion)` takes a `normalized` constructor arg it then ignores. A subclass that
  hard-codes a parent flag is a Liskov smell: make both thin wrappers of one `ReduceFusion(op)`.
- `get_module_devices` returns `list({...})` from a set, so ordering is nondeterministic. Sort it or
  return a `set`.
- `find_module_state_dict` reuses the parameter name `state_dict` as its loop variable, which makes
  the control flow hard to follow, and `load_state_dict_to_module` only retries on `RuntimeError`.
  That is fine, but the retry path never honours `strict=False` the way a user would expect.
- `get_named_modules(depth=...)`: depth semantics differ between `depth == 1` (children) and `> 1`
  (uses `named_modules` and counts dots). Rewrite it as a single bounded traversal.
- `is_loss_decreasing` hard-codes `clip_grad_norm_(max_norm=1)`, which changes what is being tested.
  Make it a parameter. The import alias `random_seed as torch_seed` is misleading.
- `DummyDataset.__init__` converts `feature_size`/`num_examples` with `int(...)` but then passes the
  unconverted values to `torch.randn`. Its docstring omits `rng_seed`.
- `functional.error.*` accept broadcastable inputs although the docs say "same shape". Either check or
  document broadcasting.

---

## 4. Packaging and dependencies (P1)

- **`sonnix.testing` imports `pytest` at import time** (`testing/fixtures.py:11`), but `pytest` is
  not a runtime dependency. `import sonnix.testing` (and the `testing` docs page) fails for users
  without it. Guard the import, or move the markers to a `pytest` plugin / `tests/` helper.
- **`numpy` extra is unused**: nothing in `src/` imports numpy, yet `pyproject.toml` declares a
  `numpy` extra and `docs/get_started.md` says `pip install 'sonnix[all]'` "installs NumPy and
  PyTorch". Remove the extra, or use it.
- `coola` is a **hard dependency** but is used for exactly two things: `package_available` (a
  one-liner with `importlib.util.find_spec`) and `random_seed` in `utils/loss.py`. Dropping it
  removes a runtime dependency for a numerical library. (If `coola` stays only for tests, move it to
  the dev group.)
- `torch>=2.4,<3.0`: verify the lower bound in CI (the `ci-test-deps` matrix should include the
  oldest supported torch). The upper bound `<3.0` is fine.
- Version is `0.0.1a0` but the classifier is `Development Status :: 4 - Beta`. Use `3 - Alpha`, or
  bump the version. There is no changelog (`Changelog` URL points to GitHub releases only).
- `__version__` falls back to `"0.0.0"`, and `tests/package_checks.py` asserts it is not `0.0.0`.
  Fine, but prefer `importlib.metadata` only and let it raise in dev installs.

---

## 5. Tests (P2)

The tests are plentiful, but several kinds are missing:

- **No dtype coverage**: no test uses `float64` / `float16` / `bfloat16` (grep over `tests/`). The
  `NLinear` and `PiecewiseLinear...` buffers/params are not tested after `.to(dtype)`.
- **No numerical-gradient checks**: `torch.autograd.gradcheck` on the functional losses and
  activations (in `float64`) would have caught 1.2 and similar issues. Many tests only check
  `loss.backward()` runs.
- **No property-based tests** even though `hypothesis` is declared in the `property` group and the
  comment in `pyproject.toml` explains why. Good targets: `reduce_loss` invariants, indicator
  ordering `min <= geometric <= arithmetic <= max`, `ToBinaryLabel` idempotence, `quantile` with
  `q=0.5 == 0.5 * L1`.
- **No equivalence tests against PyTorch / reference implementations** (the strongest guard against
  silent bugs like 1.1): `NLinear` vs `n` separate `nn.Linear`; `binary_focal_loss_with_logits` vs
  `torchvision.ops.sigmoid_focal_loss` (optional dep); `quantile(q=0.5)` vs `l1_loss / 2`;
  `asinh_mse_loss` vs manual; `general_robust(alpha=2)` vs `mse_loss`; `general_robust(alpha=1)` vs
  the Charbonnier loss.
- **No `torch.compile` / TorchScript / `torch.export` smoke tests.** Several modules use in-place ops
  (`NLinear`), Python control flow on shape (`Piecewise`), or `float()` conversions.
- **No state-dict round-trip tests** for the modules with buffers/parameters (confirms 1.7).
- **No CUDA/MPS runs in CI**: `torch_cuda_available` tests are skipped on every runner. Fine for GitHub
  runners, but say so in CONTRIBUTING.
- **Test layout mirrors `src/` but duplicates `functional` and `modules` cases.** Parametrize
  one reduction test over all losses (a registry of `(functional, module, kwargs)`) instead of ~15
  near-identical `test_*_invalid_reduction` / `test_*_extra_repr` / `test_*_reduction_*` cases.
  This would remove a large part of the ~9.4k lines of tests, and new losses would get full
  coverage by registration.
- 100% line coverage hides the issues above. Consider enabling mutation testing (e.g. `mutmut`) on
  `functional/` as a periodic job.
- `coverage.xml`, `htmlcov/`, `.coverage` exist in the working tree. They are git-ignored, but
  `make clean` should also be part of `inv` tasks.

---

## 6. Tooling, CI, and docs (P1-P2)

### 6.1 `import-linter` is configured nowhere [verified]

`.pre-commit-config.yaml` runs `uv run lint-imports` with the comment "Enforces the package layering
contract defined in pyproject.toml", but `pyproject.toml` has no `[tool.importlinter]` section.
Running it prints `Could not read any configuration.` and exits non-zero. Either add the contract
(suggested layers: `sonnix.modules` -> `sonnix.functional` -> `sonnix.utils`; `sonnix.testing`
independent) or remove the hook. Note also that `import-linter` is only installed for Python

> = 3.14 in the dev group, so the hook cannot work on 3.11-3.13 dev environments.

### 6.2 `pyright` reports an error [verified]

`utils/factory.py:320` (`setup_object_typed` returns `Any | object` for a `T`). The last commit is
titled "Fix typing" but the checker still fails. Fixing 3.5 (raise `TypeError`, then
`cast(T, obj)`) resolves it. Also note two type checkers run side by side (`pyright` and `ty`):
`ty` is still pre-1.0 and its rules are in flux. Either gate on only one, or mark `ty` advisory.

### 6.3 README and docs are stale

- README badges point to workflows that do not exist: `nightly-tests.yaml`, `docs.yaml`,
  `docs-dev.yaml` (actual files: `nightly-test-package*.yaml`, `release-docs*.yaml`). A "Code style:
  black" badge is shown but the repo uses `ruff format`.
- README is a badge wall plus one sentence. Add: install, a 10-line example for modules/losses/
  encoders, the list of what is in the library, and a comparison with PyTorch's built-ins (why
  use sonnix).
- `docs/docs/get_started.md` describes installing from source with **poetry**, **Python 3.8+**,
  `make conda`: none of which exist any more (the repo uses `uv`, Python >= 3.11, and `inv`).
- `docs/docs/index.md`, user-guide nav entries are commented out in `mkdocs.yml`. Either write a
  short User Guide (activations, losses, numerical encoders, config-driven construction) or delete
  the commented-out nav.
- `CONTRIBUTING.md` mentions `make unit-test-cov`, but I found no such target in the `Makefile` or
  `.make/*.mk` (it may come from `invoke`). Verify the documented commands.
- API docs are one `::: package` block per sub-package, so every class lands on a giant page. Use
  per-submodule pages (activation, loss, numerical, ...) and a summary table.
- One notebook (`notebooks/activations.ipynb`) exists but is not referenced from the docs or tested.
  Run it in CI with `nbmake`/`papermill` or link it from the docs, and add notebooks for losses and
  encoders.
- Docstring `pycon` examples show reprs and values that are only partially deterministic
  (`DummyDataset(... rng_seed=14700295087918620795)`). They pass today but will break on RNG changes
  across torch releases. Use `...` more.

### 6.4 CI/workflow volume

There are 28 workflow files for a 6.7k-line library, including bots, nightly matrices, and three
`lib-get-*` reusable workflows. This is a lot to maintain. Consider: merge the `nightly-test-*` trio,
and drop `bot-cancel-stale-queued-runs` (GitHub's `concurrency:` + `cancel-in-progress` does this).
Check that every workflow sets `concurrency` and minimal `permissions` (the main one does).
Add `pip-audit`/`osv-scanner` or rely on Dependabot (already present).

### 6.5 Lint configuration

- `E501`/`W505` are ignored while `pycodestyle.max-doc-length = 72` and `line-length = 100` are set.
  The ignore makes the limit a no-op. Remove one or the other.
- `PLR0913` is globally ignored (many args). Fine for losses, but the `S` ruleset is enabled for
  `src/` and `logging` f-strings (`G004`) are ignored: those cost lazy-evaluation perf in hot paths
  (`factory.py`).
- `docformatter` and `ruff` docstring rules overlap (D205/D209 are disabled to avoid fighting).
  Newer `ruff format` has docstring formatting, so one of them could go.
- `[tool.pyright]` sets `venvPath = "."`, `venv = ".venv"`, which breaks for contributors who use
  another env location. Prefer `uv run pyright`.
- `environment.yaml` (conda) duplicates the dependency list. If conda is still supported, generate
  it from `pyproject.toml`; if not, remove it.

---

## 7. Structure and layering (P2)

- `sonnix.modules.__init__` re-exports 80 names flat, but sub-packages (`modules.activation`,
  `modules.loss`, `modules.numerical`, `modules.fusion`) also export them, so every class has two or
  three import paths and `Clamp` is imported from `fusion.clamp` while `fusion/__init__.py`
  doesn't list it. Pick one canonical path per symbol and enforce it in docs. `BaseDynamicNorm` is
  public in `dynamic.py.__all__` but not re-exported.
- `Clamp` is in `modules/fusion/` but is not a fusion module (single input). Move it to
  `modules/activation/`, next to `ReLUn`, which is `clamp(0, max)`.
- `sonnix.functional` mixes activations, losses, errors, and reduction helpers. `reduction.py` and
  `error.py` sit beside the `loss/` and `activation/` sub-packages. Fine, but `functional/__init__`
  re-exports `check_loss_reduction_strategy`/`reduce_loss`, which are internal helpers. Make them
  `_reduction` or keep them out of the root namespace.
- `modules/loss/poisson.py` and `quantile.py` import from `sonnix.functional.loss`, while the other
  loss modules import from `sonnix.functional`. Pick one.
- `RelativeLoss` imports `BaseRelativeIndicator` only under `TYPE_CHECKING` from
  `sonnix.modules.loss` (the package root), which hints at an import-cycle workaround. With the
  import-linter contract from 6.1 this would be visible.
- `utils/fallback/`, `utils/imports.py`, `utils/factory.py` form an optional-dependency layer that is
  larger than what it protects (see 3.5).

---

## 8. Performance (P3)

- `NLinear`: replace transpose + matmul + transpose with `einsum`/`baddbmm`, and benchmark against
  `torch.nn.utils.parametrizations`-free alternatives. `pytest-benchmark` is already a dev
  dependency but there are no benchmarks. A `benchmarks/` folder (NLinear vs loop of `nn.Linear`,
  encoders at large batch) would document the cost.
- `PiecewiseLinearNumericalEncoder.forward` builds a Python list with conditional splats and a
  `cat` every call. A single `x.clamp(min, max)` with per-position bounds (precomputed
  `lo`/`hi` buffers, `lo[0] = -inf`, `hi[-1] = inf`) is one fused op and compile-friendly.
- Focal/Poly1 `*_with_logits` compute `sigmoid` and BCE separately (see section 2).
- `module_mode` stores modes by name and re-walks `named_modules()` in `finally`. Fine, but keying by
  the module object would avoid problems with parameter-sharing aliasing.

---

## 9. Missing features worth considering (P3)

Things users of a "library of PyTorch modules" with this scope would expect:

- Losses: `huber`-with-scale, `Tweedie`, `LogCoshLoss` module (functional exists, module missing,
  like `msle_loss`), `Dice`/`Tversky`, and class-weighted CE variants. Note the functional
  `log_cosh_loss` and `msle_loss` have no module counterparts, while every other functional loss has one
  (`MSLELoss`, `LogCoshLoss`). This asymmetry is probably accidental.
- Same for `absolute_error`, `absolute_relative_error`, `symmetric_absolute_relative_error`: no module
  wrappers, which is fine if they are metrics, but then document them as such.
- Activations: `SwiGLU`/`GeGLU`, learnable `Snake`, `PositiveSoftplus`.
- A `compile`-friendly, documented list of what is `torch.compile`/`export` safe.
- `__repr__` for `Parameter`-based hyperparameters; `ScaleAndShift.extra_repr` only prints the
  shape.

---

## 10. Suggested order of work

1. **Bugs (P0):** 1.1 (NLinear init), 1.2 (log-cosh), 1.3 (ExpSin), 1.5 (`or` on modules), 1.6 (sin/cos
   params), 1.8 (`k=0`), 6.1 (import-linter), 6.2 (pyright). Each is a few lines plus a test, and 1.1
   and 1.6 change numerical behaviour or state dicts, so call them out in release notes.
2. **API clean-up before the first beta (P1):** 3.1-3.5 (BaseLoss + `Literal` reductions, drop
   duplicated pairs, `alpha: float | None`, `TypeError` in `setup_object_typed`, consistent error types),
   1.7 (buffers), 1.9 (`batchmean`), section 4 (`pytest` import, `numpy` extra, `coola`).
3. **Docs (P1-P2):** fix the docstring errors in 1.10, README, `get_started.md`, broken badges.
4. **Tests (P2):** add an equivalence/gradcheck/dtype layer (section 5) and collapse duplicated tests
   into registry-driven parametrisations.
5. **Structure and extras (P2-P3):** section 7, 8, 9.
