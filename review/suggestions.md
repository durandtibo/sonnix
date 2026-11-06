# Sonnix codebase review

Date of review: 2026-11-06, on `main` at `c467900`.

Scope: all of `src/` (7.3k lines, read in full), `pyproject.toml`, `.github/`, `.pre-commit-config.yaml`,
`docs/`, `README.md`, `CHANGELOG.md`, `CONTRIBUTING.md`, `notebooks/`, and `tests/` (11.9k lines). I did
**not** read the tests one by one: section 5 comes from the layout, `ast` counts, greps and a coverage run.
Breaking changes are acceptable, and the package is still `0.0.1a0`.

Items marked **[verified]** were reproduced by running code in a scratch directory (nothing in the repo
was changed except this file). The others come from reading the source.

Priority legend: **P0** = broken on `main`, **P1** = bug or API issue that is cheap to fix now and costly
after a release, **P2** = quality, **P3** = nice to have.

## 0. Baseline on `main`

| Check                            | Result                                                                               |
| -------------------------------- | ------------------------------------------------------------------------------------ |
| `import sonnix.modules`          | **fails**: `cannot import name 'get_torch_generator' from 'coola.random'` (1.1)      |
| `pytest tests`                   | **54 collection errors** (same cause). With the import fixed: 8261 passed, 3 skipped |
| `xdoctest src/sonnix all`        | **68 failed**, 65 passed (same cause). With the import fixed: 133 / 133 passed       |
| `ruff check src tests`           | clean                                                                                |
| `inv types.check --complete`     | clean (`ty` and `pyright`, 0 errors)                                                 |
| `lint-imports`                   | 2 contracts kept                                                                     |
| `mkdocs build --strict`          | passes                                                                               |
| Coverage (with the import fixed) | 99 %, one missing line: `Pow.extra_repr` (5.3)                                       |
| GitHub Actions on `main`         | every run since #345 is `queued`, none completed (1.2)                               |

"With the import fixed" means: I ran everything with a `sitecustomize.py` in the scratchpad that aliases
`coola.random.get_torch_generator` to `make_torch_generator`. Nothing in the repo was patched.

### Done since the previous review

The previous version of this file (sections 1-8) was implemented in #247-#352: doctests, docs build,
`lint-imports` contract, `strict=False` loading, buffers in the device helpers, soft-target focal/Poly-1,
NaN gradients (`safe_pow`), the quantile docstring, validation of the hyperparameters, public
hyperparameters, keyword-only options, `BaseLoss.extra_repr`, `BaseFusion`, `ToDtype`, `Elementwise`
simplifications, the `create` class method and `from_quantiles` of the encoders, `channel_dim`, the
`sonnix.validation` package, `sonnix.factory`, `LogCoshLoss`/`MSLELoss`, the test registry (`export`,
`compile`, dtypes, `gradcheck`, equivalence), lazy logging, and the single `reduce_loss` paragraph. I
re-checked that those fixes hold (for example `load_state_dict_to_module(strict=False)` now searches the
wrapped checkpoint). Everything below is new or still open.

---

## 1. Broken on `main` (P0)

### 1.1 `sonnix.modules` cannot be imported with the locked `coola` [verified]

`coola 2.0.0a2` (the version in `uv.lock`, released today, required since #345) renamed
`coola.random.get_torch_generator` to `make_torch_generator(seed, *, device="cpu")`. Two files still use the
old name: `src/sonnix/utils/random.py` and `src/sonnix/testing/dummy.py`. Since `sonnix.modules.numerical.sine`
imports `sonnix.utils.random`, **the whole `sonnix.modules` package fails to import** (`sonnix.functional`
still works). The two test files that import the old name (`tests/unit/utils/test_random.py` and
`tests/unit/modules/numerical/test_sine.py`) fail at collection too, like every test that imports
`sonnix.modules`.

Fix: use `make_torch_generator` in the two modules and in the tests. `coola` is pre-release, so also:

- Cap the pin (`coola>=2.0.0a2,<2.1`) until `coola` 2.0 is stable, since an `a` release can rename public
  functions (this one did).
- Add a smoke test that imports every public module (`pkgutil.walk_packages(sonnix.__path__)`), so an
  import-time failure is reported by one test with a clear name rather than 54 collection errors.

### 1.2 CI gave no signal on `main` [verified with `gh run list`]

At the time of the review, every `CI`, `Documentation (unstable)` and `Verify Workflows` run on `main` since
#345 (about 12 runs) is `queued`; only the `CodeQL` runs completed (cancelled). So #345 (the `coola` bump that
caused 1.1) and the five PRs after it (#346-#349, #352) were merged without a completed check. Check the runner availability or
quota, and make the `CI` workflow a required status check so a PR cannot merge without a result.

`dev/config/package_versions.json` also has `"coola": []`: the dependency matrix tests `torch` 2.4-2.15 but
never tests `coola` at all, neither the minimum nor the latest. Add the minimum (`2.0.0a2`) and the latest,
and a nightly job that runs `uv lock --upgrade-package coola` and the tests, which would have caught 1.1
before it reached `main`.

### 1.3 One line is not covered: `Pow.extra_repr`

`src/sonnix/modules/activation/math.py:209`. `CLAUDE.md` requires 100 % coverage of the code in `src/`.
`Square` overrides `extra_repr`, and no unit test calls `repr(Pow(...))`. Add the test.

---

## 2. Correctness and numerical issues (P1)

### 2.1 The relative indicators overflow in half precision and for large `k` [verified]

`src/sonnix/functional/loss/relative.py`. `geometric_mean_indicator` computes `sqrt(|t| * |p|)` and
`moment_mean_indicator` computes `((|t|^k + |p|^k) / 2) ^ (1 / k)`. The intermediate value overflows long
before the result does:

| Input (`p=300`, `t=400`)       | float16 | bfloat16 | float32 |
| ------------------------------ | ------- | -------- | ------- |
| `geometric_mean_indicator`     | `inf`   | 346      | 346.41  |
| `moment_mean_indicator`, `k=2` | `inf`   | 354      | 353.55  |

For float32, `|x| = 3e19` with `k=2` gives `inf`, and `p=50, t=60` with `k=100` gives `inf` (the exact value
is 59.59). An `inf` indicator makes `relative_loss` return `0`, so the loss silently vanishes instead of
failing. Fix with a scaled form and keep the zero-safe gradient of `safe_pow`:

```python
m = torch.maximum(t.abs(), p.abs()).clamp(min=torch.finfo(m.dtype).tiny)  # scale
moment = m * safe_pow(((t.abs() / m) ** k + (p.abs() / m) ** k) * 0.5, 1 / k)
geometric = safe_pow(t.abs(), 0.5) * safe_pow(p.abs(), 0.5)  # not sqrt(t * p)
```

I checked the scaled `moment` in float16 (`353.5`) and `k=100` (`59.5855`). Do not use a plain `sqrt` for the
geometric mean: its gradient at `0` is `nan` (this is why `safe_pow` exists). Add float16/bfloat16 value tests
against a float32 reference (5.2).

### 2.2 `general_robust_regression_loss` accepts invalid `alpha` and `max_val` [verified]

- `alpha=+inf` and `alpha=nan` return `nan` (`b = inf`, `inf / 2 * 0`). Barron's limit cases are `-inf`, `0`
  and `2`; `+inf` has no finite limit in this parametrization. Validate that `alpha` is not `nan` and not
  `+inf`.
- `max_val=-1.0` is accepted by the function and by `GeneralRobustRegressionLoss`: the loss is the constant
  `-1`. Validate `max_val > 0` (or `>= 0`) in both.

### 2.3 Validation is done by the module only in some losses [verified]

Same pattern as before: the module validates, the function it wraps does not, so the function (the
documented entry point) is the weaker one.

| Call                                        | Result                |
| ------------------------------------------- | --------------------- |
| `LogCoshLoss(scale=0)`                      | `ValueError`          |
| `log_cosh_loss(x, y, scale=0.0)`            | `nan`                 |
| `log_cosh_loss(x, y, scale=-1.0)`           | negative loss (-0.27) |
| `GeneralRobustRegressionLoss(max_val=-1.0)` | accepted (see 2.2)    |

Put one `validate_*` call in the function and let the module call the same helper (or only construct
the arguments), as `validate_focal_params` already does.

### 2.4 `poisson_regression_loss` is unbounded below for negative predictions [verified]

The loss is `p - t * log(clamp(p, eps))`: the clamp protects only the `log`, the linear term `p` is not
clamped. For `t=1`: `p=-5` gives 13.4, `p=-50` gives -31.6 and `p=-500` gives -481.6. A model with a free
output can drive the loss to `-inf` instead of being pushed back to positive values. The docstring only says
the loss "is not meaningful" for negative predictions.

Options (cheapest first): clamp both terms (`p_c = p.clamp(min=eps)`, loss `p_c - t * log(p_c)`); and add a
`log_input: bool = False` option like `torch.nn.functional.poisson_nll_loss` (`exp(p) - t * p`), which is what
most models want (the output is a log-rate and needs no positivity constraint).

### 2.5 `PiecewiseLinearNumericalEncoder` input handling [verified]

- Integer bins crash with `RuntimeError: value cannot be converted to type int64_t without overflow`
  (`lower[0] = -torch.inf` on an integer tensor). Cast the bins to the default float dtype if they are not
  floating point.
- Empty bins (`torch.zeros(1, 0)`) raise `IndexError` in `forward`; `NaN` bins give `NaN` outputs without a
  message. Validate `n_bins >= 1` and finite values (`validate_ge`, `torch.isfinite`).
- `from_quantiles` on a discrete or skewed feature (many equal quantiles, very common for tabular data)
  always emits the duplicate-edge `UserWarning` and builds step segments. The reference implementation
  (`rtdl-num-embeddings`) drops the duplicate edges. Either dedupe (needs a ragged layout or padding) or add a
  `from_quantiles(..., on_duplicate="warn" | "ignore")` option, and mention it in the docstring.
- `__init__` computes `is_duplicate`, overwrites it with zeros in the `else` branch, then mutates `width` in
  place. Compute `is_step` once from `width == 0` and the number of bins.

### 2.6 Constructor arguments that are not validated [verified]

The messages come from torch (`RuntimeError`, `TypeError`, `IndexError`), not from the library, and some
values give a silently wrong module.

| Call                                         | Result                                                             |
| -------------------------------------------- | ------------------------------------------------------------------ |
| `NLinear(0, 4, 5)`                           | accepted (empty weight)                                            |
| `NLinear(-1, 4, 5)`, `ExU(-1, 3)`            | torch `RuntimeError: ... negative dimension`                       |
| `NLinear(3, 4, 5)(x)` with `x.shape == (4,)` | `IndexError: tuple index out of range`                             |
| `NLinear(3, 4, 5)(x)`, `x.shape == (2,2,4)`  | `RuntimeError` from `bmm` (does not say `n` is 3)                  |
| `DynamicTanh(normalized_shape=-1)`           | torch `RuntimeError`                                               |
| `Laplacian(init=-1.0)(tensor([5.0]))`        | `148.4`: `exp(+\|x\|)` explodes instead of decaying                |
| `Gaussian(init=float("nan"))`                | accepted                                                           |
| `Gaussian(num_parameters=2.5)` / `True`      | torch `TypeError` from `torch.full`                                |
| `MomentMeanIndicator(k=True)`, `k=1.5`       | accepted although `k` is typed `int` (`validate_ge` ignores types) |

Use `coola.validation` (`validate_int`, `validate_gt`, `validate_ge`), which `BaseAlphaActivation` already
uses for `channel_dim`. `NLinear.forward` should check `x.ndim >= 2` and `x.shape[-2] == n` and name `n` in the
message. For the alpha activations decide whether a negative `init` is legal: `Laplacian` needs `alpha > 0`
(it computes `exp(-|x| / alpha)`), `Gaussian` and `Snake` are fine with either sign. For `k`, either type it
`float` and say so, or call `validate_int`.

### 2.7 Smaller numerical points

- `safe_log(x, preserve_gradient=True)` is described as straight-through, but the backward still multiplies by
  `1 / clamped`: at `x=0` the gradient is `1 / min_val = 1e8` [verified]. State it in the `Note`, or clamp the
  gradient too.
- `safe_exp(nan)` returns `nan` (`clamp` propagates it), which is the right behaviour but the docstring says
  "leads to an output tensor without Inf" only; mention `nan` is not removed.
- `ExU` and `msle_loss` document their overflow/`nan` behaviour; `Pow(exponent=0.5)` and `Gaussian`/`Laplacian`
  with a learnable `alpha` that reaches `0` do too. Nothing to do, but a learnable `alpha` that crosses `0`
  is a training failure that is hard to debug: consider an optional `min_alpha` clamp or a `softplus`
  parametrization for the three modules that divide by `alpha`.

### 2.8 `resolve_*` runs the `_target_` before the type check [verified]

`resolve_module({"_target_": "os.getpid"})` calls `os.getpid()` and **then** raises
`TypeError: Received object is not a Module instance (received: <class 'int'>)`. `is_module_config` does not
call the target (it only reads the return type hint), but `resolve_module`, `resolve_dataset`,
`resolve_optimizer`, and every module that accepts a configuration (`ResidualBlock`, `TransformedLoss`,
`RelativeLoss`, `MulticlassFlatten`, `create_sequential`) go through `resolve_object`. A configuration read
from an untrusted file is therefore arbitrary code execution (`{"_target_": "os.system", "command": ...}`).
This is the same trust model as Hydra, but it is not written anywhere. Document it in `sonnix.factory`
and in the docstring of each `Module | dict` argument ("the configuration must come from a trusted source"),
and consider an `is_*_config` check before the call.

---

## 3. API design (P1-P2, breaking changes welcome)

### 3.1 `BaseLoss(reduction)` is positional, its subclasses are keyword-only

`src/sonnix/modules/loss/base.py`: `def __init__(self, reduction: Reduction = "mean")`. Every concrete loss
declares `*, reduction`, so `AsinhMSELoss("sum")` is a `TypeError` and `BaseLoss("sum")` works. Make it
keyword-only in `BaseLoss`; then `AsinhMSELoss.__init__`, `MSLELoss.__init__` (which exist only to force
keyword-only) and `AsinhMSELoss.extra_repr` (a copy of the base method) can be deleted.

### 3.2 Subclasses that do not fit their parent, and explicit-class calls

- `Square(Pow)`: `Square().exponent` exists, `Square(exponent=3)` is a `TypeError`, and `extra_repr` is
  overridden to return `""`. It is not substitutable for a `Pow`. Make it an independent one-line module.
- `DynamicTanh/DynamicAsinh.forward` call `ScaleAndShift.forward(self, ...)` and `ScaleAndShift.__init__`
  calls `ScaleAndShift.reset_parameters(self)` explicitly (with a comment saying why). It works, but it is a
  workaround for the inheritance. Add a private `_affine(x)` method to `ScaleAndShift` that the subclasses
  call, or use composition.
- `BaseDynamicNorm` documents a user-written subclass that does the same call; the example would be simpler
  with the helper.

### 3.3 Public names that are not part of the API

- `safe_pow` is documented under `sonnix.functional.power.safe_pow` but is not exported from
  `sonnix.functional`, and its docstring has no `Example` (project rule).
- `ALPHA_EPS` (`functional/loss/general_robust.py`) and `SPACINGS` (`modules/numerical/sine.py`) are public
  constants that are not in `__all__`.
- `BaseAlphaActivation.broadcast_alpha` is public but is an implementation detail of the subclasses.

Decide per name: export it (and document it) or prefix it with `_`.

### 3.4 Remaining inconsistencies in the attributes

- `ToBinaryLabel` stores `_threshold` and exposes a property, while `ToCategoricalLabel.dim`
  and the other modules store a public attribute (as `CHANGELOG.md` says all modules do).
- `NLinear.n/in_features/out_features` are properties derived from `weight.shape` (no docstring, which the
  project rule requires), `ExU` stores plain ints. Pick one.
- `forward` methods: `ToBinaryLabel.forward` and the focal/Poly-1 forwards carry a full docstring, almost
  every other `forward` has none. Follow the rule in `CLAUDE.md` (document an override only when it adds
  something) in both directions.

### 3.5 `utils` options and behaviour

- `load_state_dict_to_module(state_dict, module, strict=True)`: make `strict` keyword-only. The search for the
  sub-dict compares key sets exactly, so checkpoints saved from `DataParallel` (`module.` prefix) or
  `torch.compile` (`_orig_mod.` prefix) are never found. Strip a common prefix before comparing.
- `is_loss_decreasing*`: `num_iterations`, `random_seed` and `lr` are positional; make them keyword-only. The
  seed is applied only to the two evaluation passes, not to the training steps, although the docstring says it
  makes the function deterministic "if the module contains randomness" (read from the code; a 5-run dropout
  experiment gave the same answer every time, so I could not show a difference). Seed the loop or reword.
- `sonnix.__version__ = importlib.metadata.version("sonnix")` makes `import sonnix` raise
  `PackageNotFoundError` when the source is used without being installed (a vendored copy, `PYTHONPATH`).
  Fall back to `"0.0.0+unknown"`.

### 3.6 Scope: modules that nothing in the library uses

`sonnix.factory.dataset`, `sonnix.factory.optimizer`, `sonnix.testing.DummyDataset` and `sonnix.utils.loss`
are not used by `sonnix.modules` or `sonnix.functional` (only by tests and docs). The README describes the
package as "a library of PyTorch modules, losses and functional operations". Either keep them and say in the
README that `sonnix` also ships training/testing helpers, or move them to a separate package; the
`ResidualBlock`/`TransformedLoss` use of `resolve_module` is the part that belongs here.

### 3.7 Features that fit the scope (P3)

- `AsinhNumericalEncoder.create(...)`, like `CosSinNumericalEncoder.create` (the asinh encoder has no factory).
- `create` builds an encoder for **one** feature (`n_features=1`); add `n_features` so it is usable
  without repeating tensors by hand.
- A `NumericalEmbedding` composition (encoder + `NLinear`) in the docs: it shows why both are in the library.
- `Tweedie` and `Huber`-with-scale losses, `Dice`/`Tversky`, `SwiGLU`/`GeGLU` (not done before).
- A `log_input` option for the Poisson loss (2.4).

---

## 4. Code health (P2-P3)

### 4.1 Third-party imports placed before `__all__` in 10 files

`CLAUDE.md` says to match the surrounding code. Most files put `__all__` right after
`from __future__ import annotations`; these ten put `from coola.validation import ...` before it:
`functional/activation/math.py`, `functional/error.py`, `functional/loss/asinh.py`,
`functional/loss/general_robust.py`, `functional/loss/poisson.py`, `modules/activation/alpha.py`,
`modules/activation/math.py`, `modules/activation/relu.py`, `modules/loss/asinh.py`,
`modules/loss/poisson.py`. It looks like an automated edit that added the import at the top. Move them
below `__all__` with the other third-party imports (`ruff check --select I` will accept it).

### 4.2 Duplicated class documentation

The `Args:` block of the constructor is copied into every subclass: six alpha activations (about 25 lines
each, identical except `Snake`), three focal classes, three Poly-1 classes, the three relative losses. The
project rule asks for the arguments in the class docstring, so the copy is intentional, but the copies
already drifted (`Snake` documents `init` differently, `BaseBinaryFocalLoss` has no example). Two options:
keep the copies and add a test that compares the `Args:` of the siblings; or document only in the base class
and let `mkdocstrings` show the inherited members. The `griffe-inherited-docstrings` dependency is in the `docs`
group but is not referenced in `docs/mkdocs.yml` (nor is `mkdocs-include-markdown-plugin`): enable it
(`extensions: [griffe_inherited_docstrings]`) so the "an override inherits the parent docstring" rule of
`CLAUDE.md` is true in the rendered documentation, or drop the two dependencies.

### 4.3 Thin wrappers

Every loss exists twice (function + module) and every indicator three times (function, module, registry entry).
The registry in `tests/registry.py` keeps them in sync, so the cost is only reading. I would **not**
generate the modules: explicit classes are what configuration files and the docs need. If the next loss makes
the boilerplate painful, add a `make_loss_module(fn, **defaults)` helper then.

### 4.4 Performance

Nothing to fix: `NLinear` and `ExU` were benchmarked in the previous round, the piecewise encoder is a single
`clamp`, `safe_pow` is two `where`. The fusion modules loop in Python over the inputs, which is fine for the
two or three inputs they are used with (not benchmarked).

---

## 5. Tests (P2)

The suite is large (1008 test functions, 99 % line coverage, `gradcheck`, `export`, `compile`, dtype and
equivalence tests) and the registry approach from the previous round works well. What is left:

### 5.1 The "one behaviour per test" rule is broken in 127 test functions [verified with `ast`]

`CLAUDE.md`: a test has one `assert` or one `pytest.raises`. 127 of 1008 functions have several (one has a
loop). Worst files: `unit/modules/activation/test_dynamic.py` (21), `unit/modules/numerical/test_sine.py`
(14), `unit/modules/numerical/test_piecewise.py` (12), `unit/modules/test_scale_and_shift.py` (10),
`unit/modules/activation/test_alpha.py` (8), `unit/modules/fusion/test_exu.py` (7), `unit/utils/test_mode.py`
(6, one test has 9 asserts), `unit/modules/test_nlinear.py` (6). Split them, or parametrize when the asserts are
the same check on several values. Do it file by file, one PR per file, since a split changes no behaviour.

### 5.2 Missing numerical checks

- Value tests in `float16`/`bfloat16` (2.1 would have been found by a comparison against a float32 reference
  on inputs of magnitude 300). The dtype tests only check that forward/backward run and the dtype is kept.
- Regression tests for 2.1-2.6, each shown to fail without the fix (`git stash push src`).
- A test per validator that the function and the module raise the same error for the same value (2.3), driven
  by the registry.

### 5.3 Layout and dependencies

- `tests/unit/modules/fusion/test_exu.py` and `test_dtype.py` test `modules/exu.py` and `modules/dtype.py`:
  the mirror rule (`tests/unit/<package>/test_<module>.py`) puts them in `tests/unit/modules/`.
- `Pow.extra_repr` is not tested (1.3).
- `hypothesis` (group `property`) and `pytest-benchmark` (group `dev`) are declared but used by no test. Use
  them (a property test that `with_logits(x, t) == probs(sigmoid(x), t)` for random soft `t`; a benchmark that
  records the `NLinear` shapes) or remove them.
- Two unit tests in `tests/unit/modules/numerical/test_sine.py` patch `torch.rand`
  (`create("rand", ...)`); `CLAUDE.md` requires an integration test for every patched behaviour, and
  `tests/integration/modules/` has no `numerical` directory. Add one that calls `create("rand", ...)` with
  a seed and checks the range and the reproducibility (the same seed gives the same frequencies).
- CI never runs a GPU/MPS job (already documented in `CONTRIBUTING.md`). Float16 on CPU does not exercise the
  same kernels as CUDA; run `pytest -m` for the CUDA tests locally before a release.

---

## 6. Documentation (P2)

- `docs/site/` is not ignored (`.gitignore` has `/site`, which only matches the repository root), so
  `mkdocs build -f docs/mkdocs.yml` leaves an untracked directory [verified]. Add `docs/site/`.
- `mkdocs build` prints the MkDocs 2.0 warning of `mkdocs-material`. Pin `mkdocs>=1.6,<2.0` in the `docs`
  group so a MkDocs 2 release cannot break the documentation job.
- `notebooks/activations.ipynb` has 46 cells and **no stored outputs**: on GitHub and in the docs it shows no
  plot. Execute it and store the outputs (or render it in the docs with `mkdocs-jupyter`), and run it in CI
  with `nbmake` or `jupyter nbconvert --execute` so it cannot rot.
- Add user-guide pages (the nav only has Home, Get started, Conventions, Reference): losses and the
  reduction semantics, numerical encoders and `NLinear` as an embedding, building a model from
  configurations (with the trust note of 2.8), numerical-stability notes (2.1, 2.4, 2.7).
- The reference has one page per sub-package, but `sonnix.modules.numerical` and `sonnix.modules.loss` each
  list 10-30 objects without a summary table. Add a table (name, one-line description) at the top of each
  page.
- `CHANGELOG.md` has an `Unreleased` section; add the entries for the changes of this review as they land
  (1.1 is a "Fixed", 3.1 and 3.2 are "Breaking changes"), with the PR number.

---

## 7. Packaging, CI and tooling (P2-P3)

- `coola>=2.0.0a2,<3.0` follows the pre-release line (1.1, 1.2).
- 28 workflow files for a 7k-line library. `bot-cancel-stale-queued-runs` is only useful while runs queue for
  hours (1.2); once the queue is fixed, `concurrency: cancel-in-progress` covers it. The three `lib-get-*`
  and four `nightly-*` workflows could be merged into one reusable workflow with a matrix.
- `pre-commit` runs `ty` and `lint-imports` but not `pyright`, while CI and `CLAUDE.md` require both type
  checkers. Add `pyright` as a local hook (or drop one checker).
- `ruff` ignores `D102`, `D105`, `D107`, `D205` and `D209` because of `docformatter`; recent `ruff format`
  formats docstring code and `D205`/`D209` can be enforced, so `docformatter` could go. Worth a trial run
  on the repository, not urgent.

---

## 8. Suggested order of work

One pull request per line, each with a test that fails before the fix (stacked PRs only when needed).

1. **Unbreak `main` (P0):** 1.1 (`make_torch_generator` and the `coola` cap), then 1.2 (CI queue,
   required checks, `coola` in the matrix), then 1.3.
2. **Correctness (P1):** 2.1 (indicator overflow), 2.2 and 2.3 (`alpha`, `max_val`, `scale` validation in the
   functions), 2.4 (Poisson), 2.5 (piecewise encoder), 2.6 (constructor validation), 2.8 (document the trust
   model of `resolve_*`).
3. **API (P1-P2):** 3.1 (`BaseLoss` keyword-only, delete the redundant `__init__`/`extra_repr`), 3.2
   (`Square`, `_affine`), 3.3 and 3.4 (public names, attributes), 3.5 (`utils` options, prefix stripping,
   `__version__` fallback). Add the `CHANGELOG.md` entries in the same PRs.
4. **Tests (P2):** 5.2 (half-precision value tests and the function/module validation tests), then 5.1 file
   by file, then 5.3.
5. **Docs (P2):** 6 (ignore `docs/site/`, pin `mkdocs`, notebook outputs, user guide, summary tables) and 4.2
   (enable or drop the unused docs dependencies).
6. **Code health and extras (P2-P3):** 4.1 (import order), 3.6 (scope decision), 3.7 (features), 7.
