# `sonnix`: target state and the changes to reach it

Date: 2026-11-22, on `main` at `55919a9`. This is the third revision. The first one (twelve steps,
#413-#440) and the second one (nine steps, #446-#455) are merged; their history is in
[Appendix C](#appendix-c-history-of-the-previous-revisions) and in `CHANGELOG.md`.

This document describes the design `sonnix` should have and the steps to get there from `main`. It
follows the format of
[`coola/review/recursive.md`](https://github.com/durandtibo/coola/blob/main/review/recursive.md):
properties, constraints, findings, design, public API, dependents, evidence, plan, tests,
decisions. The properties come first: every choice below exists to make one of them true **by
construction** (one helper, one registry test) instead of by the discipline of each class.

What this revision is about. The first two revisions fixed the structure: shapes, validation,
signatures, state, public paths. What is left is not visible by reading the code. It appears when
the properties are **run** on edge inputs (zeros, tiny and large values, `float16`) and when the
documentation is compared with the code. Of the sixteen properties in `docs/docs/conventions.md`,
four are stated and **not true** (P1, P4, P6, P12), one is imprecise (P9), and one is missing
(P17: the documentation says what the code does).

Scope: all of `src/` (7.1k lines, read in full), the test registry and the meta-tests, `docs/`,
`README.md`, `CLAUDE.md`, `pyproject.toml` (import-linter), `review/`. Items marked **[probe]** were
reproduced today by running code in the scratchpad (outside the repository); **[prototype]** means
the proposed fix was run in a throw-away script and gave the stated result; **[reading]** comes from
the source. Nothing in the repository was changed except this file.

Baseline on `main`: `ruff check` clean; `inv types.check --complete` clean (`ty` and `pyright`, 0
errors); `lint-imports` 3 contracts kept; `xdoctest` 121/121; `pytest tests` 9,882 passed, 3
skipped, in 20 s. `mkdocs build --strict` was not run.

## 1. Properties of the package

`sonnix` is a library of PyTorch building blocks that `torch.nn` does not provide: losses,
parametric activations, numerical-feature encoders and a few layers. It is used by other codebases
(decision #453), so a public name is useful if a user needs it, not only if `sonnix` calls it. The
API has three tiers: `sonnix.functional` (stateless operations), `sonnix.modules` (the
`torch.nn.Module` wrappers and layers) and the support packages (`validation`, `factory`, `utils`,
`testing`).

Each property is a rule that holds for **every** public object, with the mechanism that makes it
true and the test that proves it. A property without a test is a wish. The column "Today" is the
result of this review.

| #   | Property                            | Rule                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                | Enforced by                                                                                                  | Today                         |
| --- | ----------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------ | ----------------------------- |
| P1  | **Function/module duality**         | Every loss, and every op that needs numerical care, is a stateless function in `sonnix.functional` **and** a module in `sonnix.modules` that stores the hyperparameters and calls the function: the losses, `safe_exp`/`SafeExp`, `safe_log`/`SafeLog`, `safe_pow`/`SafePow`. A module with learnable state or without numerical care (an activation, a layer) has no function. A function that users look for by name (`rectifier_asinh_unit`, the errors, the indicators) may have no module (P16). A `forward` is one call.      | parity registry test: module output equals function output (exists for the losses; new for `SafePow`)        | **not true** (F7)             |
| P2  | **Validate early and completely**   | Every hyperparameter is checked for **type** (a `bool` is not an `int`), **finiteness** and **range**, before it is stored or converted, with `coola.validation` or a `sonnix.validation` helper. A float is finite unless its docstring says otherwise. The function checks at call time; the module checks in `__init__` with the same helper and message.                                                                                                                                                                        | `tests/integration/test_invalid_arguments.py` (exists)                                                       | true [probe]                  |
| P3  | **Uniform call signature**          | Losses: `(prediction, target, *, <options>, reduction)`. Single-tensor ops: `(input, *, <options>)`. Options of a function are keyword-only. A module takes positionally its structural arguments (sizes, wrapped modules, tensors) and at most one hyperparameter; the rest is keyword-only. The tensor of a `forward` is `input`.                                                                                                                                                                                                 | `ModuleCase.positional` compared with `inspect.signature` (exists)                                           | true                          |
| P4  | **Strict shapes**                   | Two tensors of the same role must have the same shape (`ValueError` naming both): the prediction and the target, the inputs of a sum, average or product fusion, the two branches of a residual block. A layer or an encoder checks the dimension it consumes and names the argument (`n`, `n_features`, `normalized_shape`, `channel_dim`). Nothing broadcasts silently: a size-1 dimension is not expanded. A wrapper of an arbitrary criterion (`TransformedLoss`, `MulticlassFlatten`) leaves the policy to the wrapped module. | `validate_same_shape`; parametrised shape tests (exist); registry test `wrong_inputs` for every module (new) | **not true** (F5)             |
| P5  | **One reduction contract**          | Every loss accepts `none`/`mean`/`sum`/`batchmean`, validated by `reduce_loss`; default `mean`; `none` keeps the shape of the elementwise loss; `reduction` is keyword-only.                                                                                                                                                                                                                                                                                                                                                        | `reduce_loss`; registry test over all losses (exists)                                                        | true                          |
| P6  | **Numerically safe**                | For `float16`, `bfloat16`, `float32` and `float64`: wherever the `float32` result and its gradient are representable in the dtype, the output and the gradient are finite. A **default** hyperparameter is valid in every dtype (a guard `eps` or `min_val` is raised to the smallest normal number of the dtype; the `max_val` of `safe_exp` is lowered so that `exp` is finite). An unbounded input is squared or exponentiated in at least `float32`. A hyperparameter that can still overflow (an exponent) is documented.      | registry test `test_finite_wherever_float32_fits` on zeros, tiny, large and negative-large inputs (new)      | **not true** (F1-F4)          |
| P7  | **Pure**                            | A function or a `forward` never mutates its inputs and keeps no hidden state.                                                                                                                                                                                                                                                                                                                                                                                                                                                       | in-place-mutation registry test (exists)                                                                     | true                          |
| P8  | **Complete state, by construction** | The `state_dict` holds the parameters and buffers that define the behaviour, and **nothing derived**. A tensor derived from persistent state is non-persistent and recomputed after `load_state_dict`. A frozen tensor is a buffer with the same name as the learnable one (`register_tensor`). Round trip: `m2.load_state_dict(m1.state_dict())` makes `m2` behave as `m1`.                                                                                                                                                        | `register_tensor`; round-trip registry test with a second build (exists)                                     | true                          |
| P9  | **Device and dtype**                | A module that creates tensors takes `device` and `dtype`. After `module.to(dtype)` every floating tensor of the module has `dtype` and the output has the dtype of the input. A module whose dtype differs from the dtype of the input follows the promotion rules of `torch` (`NLinear` raises, like `torch.nn.Linear`): that case is not guaranteed.                                                                                                                                                                              | dtype/device registry tests, with `float16` added (new)                                                      | imprecise, `float16` untested |
| P10 | **Export and compile**              | Every registered module passes `torch.export` (also with a dynamic batch dimension) and `torch.compile(fullgraph=True)` on CPU: no `.item()`, no data-dependent Python branch in a `forward`. A shape check on static dimensions is allowed.                                                                                                                                                                                                                                                                                        | registry tests (exist); dynamic batch (new, [prototype] passes)                                              | true, batch not tested        |
| P11 | **Introspectable**                  | Every hyperparameter, including a callable or a submodule, is a public attribute named like the constructor argument and is shown by `repr`.                                                                                                                                                                                                                                                                                                                                                                                        | `test_hyperparameters` (exists)                                                                              | true                          |
| P12 | **One public path**                 | Every public object has **one** public path. `sonnix.{functional,modules,validation,factory,utils,testing}` each export their API from `__init__.py` (`__all__`); a name is exported by **one** package; the sub-folders have no `__all__`. Docs, docstrings and examples import from the flat path. Every exported name is listed under **its own package** in `docs/docs/refs` and has an `Example` that runs.                                                                                                                    | `test_imports`, `test_docs` (exist, the second too weak), `xdoctest`, no-duplicate test (new)                | **not true** (F8)             |
| P13 | **Layered imports**                 | `testing → modules → functional → validation`; `utils`, `factory` and `validation` import no other sonnix package (`utils` is a leaf that every layer may import); no cycle.                                                                                                                                                                                                                                                                                                                                                        | import-linter contracts (exist)                                                                              | true                          |
| P14 | **Configuration is trusted input**  | `sonnix.factory` imports and **calls** a `_target_` before it checks the type of the result. An argument typed `Module \| dict` says so in a `Warning`.                                                                                                                                                                                                                                                                                                                                                                             | `test_config_warning` (exists)                                                                               | true                          |
| P15 | **Optional means lazy**             | Only `torch` and `coola` are imported at module level.                                                                                                                                                                                                                                                                                                                                                                                                                                                                              | `package_checks` job (exists)                                                                                | true                          |
| P16 | **Earned names**                    | A public name is a loss, a layer, an activation, an encoder, an op with numerical care, or a helper that these use, that tests them, or that users need. A fixed-argument composition is a recipe in the guide, unless users look for it by name (`ReLUn`, `Square`, `ToFloat`): then a thin wrapper is allowed and its docstring names the composition. A base class is public only if users are meant to subclass it.                                                                                                             | review checklist (Appendix B); a class without a registry entry fails                                        | true                          |
| P17 | **Honest documentation** (new)      | A docstring says what the code does: the `Args:` block lists exactly the arguments of the signature, a `Raises:` block names what the code raises, and a guarantee stated in a docstring or in the guide ("no `inf`", "raises `ValueError`") has a test. `CLAUDE.md` points only to documents that are true; an obsolete review file is deleted.                                                                                                                                                                                    | signature-vs-`Args:` test over the six `__all__` (new), `xdoctest` (exists)                                  | **not true** (F9, F10)        |

Not guaranteed, and documented as such: bit-exact results across `torch` versions; behaviour on GPU
in CI (no GPU job; the CUDA tests run locally before a release); a learnable `alpha` that crosses `0`
(decision D9: documented, not changed); a result that the dtype cannot represent (`Square` of
`1000` in `float16`, `exp(100)` in `float32`); the probability variants of the binary losses with a
prediction of exactly `0` or `1` in `float16` (the `binary_cross_entropy` of `torch` divides by `0`
and raises under `autocast`; use the `*_with_logits` variants); a module fed a dtype other than its
own (P9).

## 2. Constraints

1. **Simple and easy to maintain.** A property is made true by **one** helper or **one** registry
   test, not by a check written in each class. Delete code when a change allows it (section 4.6).
   Net size is in Appendix A: this revision adds about 125 lines of `src`, because it makes untrue
   properties true, and removes 590 lines of obsolete documentation.
2. **Breaking changes are fine.** The version is `0.0.1a0`: no shim, no deprecation period. Each
   breaking change is listed in `CHANGELOG.md` ("Breaking changes" and the migration table) with the
   PR number.
3. **Same numerics for valid calls** in `float32`, `float64` and `bfloat16`. A behaviour changes only
   where a finding of section 3 names it, and each such change has a regression test that fails
   without the fix (`git stash push src`). The `float16` results change (that is the point of F1-F4).
4. **No slower than today.** A forward pass stays the same tensor operations. The new checks compare
   static shapes in Python (they pass `torch.export`, [prototype]). The validation of the options
   costs `0.14` µs (`validate_positive`) and `0.33` µs (`validate_same_shape`) per call [probe], so
   no check is moved out of the functions to save time. `NLinear` with `einsum` runs at the speed of
   `bmm` (4.6).
5. **Closed decisions are not reopened.** `sonnix` is a library: the helpers of `sonnix.utils`,
   `DummyDataset`, the named wrappers (`ToFloat`, `ToLong`, `ReLUn`, `Square`), the three errors and
   `rectifier_asinh_unit` are **kept** (#451, #453). This revision adds no removal of a public name.
6. **Follow `CLAUDE.md` and `docs/docs/conventions.md`.**
7. **`torch` and `coola` are the only hard dependencies.**

## 3. What differs today

| #   | Finding                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      | Status              |
| --- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------- |
| F1  | **The default guards underflow in `float16` (P6).** `1e-8` is `0.0` in `float16` (smallest normal: `6.1e-5`). `safe_log(0)` returns `-inf` (`-18.4` in `bfloat16`); `poisson_regression_loss(0, 0)` returns `nan` and `(0, 1)` returns `inf`; `relative_loss(0, 0)`, `absolute_relative_error(0, 0)` and `symmetric_absolute_relative_error(0, 0)` return `nan`; `SafeLog` and `RelativeLoss` follow. `bfloat16` and `float32` are fine.                                                                                                                                                                                                                                     | [probe]             |
| F2  | **`safe_exp` does not guarantee a finite output (P6).** The default `max_val=20` gives `exp(20) = 4.9e8`, which is `inf` in `float16` for every input above `11.09`; `max_val=100` gives `inf` in `float32`. The guide (`numerical_stability.md`) says "so the output has no `inf`".                                                                                                                                                                                                                                                                                                                                                                                         | [probe]             |
| F3  | **`general_robust_regression_loss` overflows in `float16` where the loss fits.** `((p - t) / scale)²` is `inf` for `\|p - t\| / scale > 255`. For `p - t = 300`: `alpha=1` gives `inf` with a `nan` gradient (value in `float32`: `299`); `alpha=0` gives `inf` (value: `10.7`). A robust loss exists for outliers. The tests use inputs in `±30` (`_scaled`) and never see it.                                                                                                                                                                                                                                                                                              | [probe]             |
| F4  | **`Gaussian` returns a `nan` gradient for `alpha` in `float16`.** For `\|x\| >= 256`, `x.pow(2).mul(-0.5).div(alpha.pow(2)).exp()` computes `-inf` (output `0`, correct), and the backward of the division multiplies `-inf` by `0`. `exp(-0.5 * (x / alpha)²)` has the same value in `float32` and a gradient of `0` in `float16`. A learnable `alpha` is destroyed by one large input.                                                                                                                                                                                                                                                                                     | [probe] [prototype] |
| F5  | **Silent broadcasting (P4).** `ScaleAndShift(5)(x[3, 1])` and `DynamicTanh(5)(x[3, 1])` return `(3, 5)`; `SumFusion`, `AverageFusion` and `MultiplicationFusion` accept `(4, 3)` with `(4, 1)` and with `(3,)`; `ResidualBlock(nn.Linear(3, 1))(x[4, 3])` adds a `(4, 1)` branch to a `(4, 3)` skip; `Gaussian(num_parameters=3)(x[4, 1])` broadcasts its three `alpha` over a size-1 channel. `Gaussian(3, channel_dim=1)(x[5])` raises a bare `IndexError`. The error of the encoders and of `NLinear` is already strict and names the argument.                                                                                                                           | [probe]             |
| F6  | **`safe_pow` hides an invalid input.** `safe_pow([-4, 0, 4], 0.5)` returns `[0, 0, 2]` (`torch.pow` gives `nan` for `-4`) and `safe_pow(-2, 2)` returns `0` (the power is `4`). The docstring says "non-negative input" and the code masks `input > 0`. The wanted property is only a finite gradient **at zero**: mask `input != 0`.                                                                                                                                                                                                                                                                                                                                        | [probe] [prototype] |
| F7  | **P1 is not true for `safe_pow`.** `safe_exp` and `safe_log` have a module (`SafeExp`, `SafeLog`); `safe_pow` has none, and the docstring of `Pow` warns that its gradient is infinite at `0` for an exponent in `(0, 1)`, which `safe_pow` fixes.                                                                                                                                                                                                                                                                                                                                                                                                                           | [reading]           |
| F8  | **`Reduction` has two public paths (P12).** `sonnix.validation.Reduction is sonnix.functional.Reduction`; both are in `__all__`, it is documented once (`refs/validation.md`). `test_docs` passes because its regex accepts any package (`sonnix(\.\w+)+\.{name}`).                                                                                                                                                                                                                                                                                                                                                                                                          | [probe]             |
| F9  | **The docstrings drift (P17).** `Pow` has no `Args:` block (`exponent` is undocumented); `ToBinaryLabel` says it raises when `dim` is not an integer (there is no `dim`); the guide says to make Poisson predictions positive with `SafeExp`, `Square` or a softplus although `log_input=True` exists; F2. A probe over the 105 public classes and functions found no other mismatch, and every one has an `Example`.                                                                                                                                                                                                                                                        | [probe]             |
| F10 | **`review/suggestions.md` is obsolete and `CLAUDE.md` points to it.** The 505-line review of 2026-11-06 lists as open the crash of `import sonnix.modules` (1.1), the indicator overflow (2.1), the unvalidated arguments (2.2-2.6), `BaseLoss` positional (3.1), ...: the code items are fixed since (#413-#455; the CI items of its section 1.2 and 7 are tracked in `.github/GUIDELINES.md`). `CLAUDE.md` still says it "lists known issues and priorities".                                                                                                                                                                                                              | [reading]           |
| F11 | **The tests do not run the properties at the edges.** `test_dtype.py` uses `float32`, `float64`, `bfloat16`: no `float16`, although the 51 registered modules all run in `float16` on CPU [probe]. `test_half_precision.py` has 14 hand-picked functions with inputs in `±300`, `±30` or `[1, 301]`: never `0`, never tiny, and `float16` is compared at a tolerance. F1-F4 are all caught by one rule on four constant inputs (section 9).                                                                                                                                                                                                                                  | [reading] [probe]   |
| F12 | **Logic written twice.** `binary_focal_loss` and `binary_focal_loss_with_logits` differ by two lines (`binary_cross_entropy` against `..._with_logits`, and `sigmoid`); so do the two Poly-1 functions. The fix of the integer targets (#454) had to be applied to the four functions. `NLinear.forward` reshapes, transposes and calls `bmm` (6 lines) where `einsum("...ni,nio->...no")` gives the same result at the same speed (`43` against `47` µs at `256 × 8 × 16 × 32`; `1,407` against `1,391` µs at `1,024 × 32 × 64 × 64`) for any number of leading dimensions. `NLinear.reset_parameters` has a branch `if in_features > 0 else 0` that validation makes dead. | [reading] [probe]   |
| F13 | **The count prefix is not uniform.** `n_features`, `n_bins`, `n` (`NLinear`) against `num_frequencies` and `num_parameters`. `num_parameters` is the name of `torch.nn.PReLU` and stays; `num_frequencies` is the only other `num_`.                                                                                                                                                                                                                                                                                                                                                                                                                                         | [reading]           |
| F14 | **Carried over.** 45 `# noqa: A002` (the `input` argument): decision D4 of the second revision, never answered (`CLAUDE.md` says a changed rule set is not a fix, so it needs approval). `pre-commit` runs `ty` and `import-linter` but not `pyright`, which `CLAUDE.md` requires (`inv types.check` runs both).                                                                                                                                                                                                                                                                                                                                                             | [reading]           |

What is already right and must not regress: the validation of every hyperparameter (a fuzz of the
constructors of all 57 exported classes with `nan`, `inf`, `-inf`, `True`, `"a"`, `2.5`, `0` and
`-1` accepts only legitimate values, for example `init=0` for `Quadratic` or `q=0`); keyword-only
options and the registry check of the positional arguments; the state round trip and the derived
buffers of the piecewise encoder; strict shapes in all the losses and in the encoders; the
function/module parity of the losses; the registry-driven export, compile, purity, dtype and
hyperparameter tests; the import-linter layering; the 100 % coverage rule. `bfloat16` has no failure
at all [probe]: every failure of this review is `float16`.

## 4. Design

### 4.1 Packages and dependencies

```text
sonnix
├── validation/   argument validators, Reduction (defined, not exported)  (imports: coola)
├── functional/   stateless ops: losses, indicators, safe_*, reduce_loss, Reduction  (imports: validation, utils)
├── factory/      resolve_module, is_module_config, create_sequential     (imports: coola, torch)
├── utils/        register_tensor, resolve_eps, resolve_max_exponent, module_mode, ...  (imports: torch)
├── modules/      nn.Module wrappers, layers, encoders                    (imports: functional, factory, utils, validation)
│   └── activation/ loss/ numerical/ fusion/                              (folders, not public paths)
└── testing/      is_loss_decreasing*, DummyDataset                       (imports: anything; nothing imports it)
```

The import-linter contracts do not change: `functional` may import `utils` (a leaf, it imports only
`torch`); it still may not import `modules` or `factory`.

### 4.2 Dtype-aware guards (P6; F1, F2)

A guard is a number that must not underflow or overflow **in the dtype of the tensor**. Two helpers
in `sonnix.utils` (next to `register_tensor`) make the rule true in one place:

```python
def resolve_eps(eps: float, tensor: Tensor) -> float:
    """Return ``eps``, raised to the smallest positive normal number of the dtype of ``tensor``."""
    return max(eps, torch.finfo(_float_dtype(tensor)).tiny)


def resolve_max_exponent(max_val: float, tensor: Tensor) -> float:
    """Return ``max_val``, lowered so that ``exp`` of it is finite in the dtype of ``tensor``."""
    return min(max_val, math.log(torch.finfo(_float_dtype(tensor)).max) - 1.0)


def _float_dtype(tensor: Tensor) -> torch.dtype:  # the dtype itself, or the default one for an int
    return tensor.dtype if tensor.is_floating_point() else torch.get_default_dtype()
```

The margin of `1.0` is needed: `log(65504) = 11.0899` rounds up to `11.09375` in `float16`, and
`exp(11.09375)` is `inf` [prototype]. With the margin the cap is `10.09` (`float16`), `87.7`
(`float32`), `708.8` (`float64`); `bfloat16` has the range of `float32`.

| Site                                                           | Today                       | Target                                                  |
| -------------------------------------------------------------- | --------------------------- | ------------------------------------------------------- |
| `safe_log(input, min_val)`                                     | `input.clamp(min=min_val)`  | `input.clamp(min=resolve_eps(min_val, input))`          |
| `safe_exp(input, max_val)`                                     | `input.clamp(max=max_val)`  | `input.clamp(max=resolve_max_exponent(max_val, input))` |
| `poisson_regression_loss` (`log_input=False`)                  | `prediction.clamp(min=eps)` | `prediction.clamp(min=resolve_eps(eps, prediction))`    |
| `relative_loss`                                                | `indicator.clamp(min=eps)`  | `indicator.clamp(min=resolve_eps(eps, indicator))`      |
| `absolute_relative_error`, `symmetric_absolute_relative_error` | `.clamp(min=eps)`           | `.clamp(min=resolve_eps(eps, target))`                  |

The docstrings of `SafeLog`, `SafeExp` and `eps` say "raised to the smallest normal number of the
dtype" and "lowered so that the output is finite". The attributes and the `repr` keep the value
the user passed (P11); the effective value depends on the input, like a `torch.finfo`.

Two more changes, both **[prototype]**: results in the table of section 7.

- **`general_robust_regression_loss` computes in at least `float32`.** `dtype = torch.promote_types(prediction.dtype, torch.float32)`;
  the difference is squared and scaled in `dtype`; the loss is cast back to the input dtype. For
  `p - t = 300` in `float16`: `alpha=1` gives `299.0` and a gradient of `1.0` (today `inf` and `nan`);
  `alpha=0` gives `10.711` (today `inf`). With `alpha=2` the loss is `9e4`, which `float16` cannot
  hold: `inf` stays, and the gradient is finite (`600`). No other function needs it: `log_cosh_loss`,
  the indicators and `quantile` and the `*_with_logits` losses are finite on the same inputs
  (zeros, tiny, `±3000`), except for the two `eps` cases of F1 [probe].
- **`Gaussian.forward` is `input.div(alpha).pow(2).mul(-0.5).exp()`** (F4): one operation fewer and
  no division of `-inf` in the backward. `Laplacian`, `Snake`, `Quadratic`, `MultiQuadratic` and
  `ExpSin` have no failure [probe].

### 4.3 Strict shapes (P4; F5)

The check is a comparison of static shapes in the `forward`; it passes `torch.export` with a dynamic
batch and `torch.compile(fullgraph=True)` [prototype]. Each message names the argument.

| Module                                               | Check                                                                                                                                                                        | Message names                    |
| ---------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------- |
| `ScaleAndShift`, `DynamicTanh`, `DynamicAsinh`       | `input.shape[-len(normalized_shape):] == normalized_shape`, like `LayerNorm`; `(3, 1)` is no longer expanded                                                                 | `normalized_shape` and the shape |
| `SumFusion`, `AverageFusion`, `MultiplicationFusion` | all the inputs have the shape of the first one: a class attribute `BaseFusion._same_shape = True`, `ConcatFusion` sets it to `False` (`torch.cat` checks its own dimensions) | the position and both shapes     |
| `ResidualBlock`                                      | `validate_same_shape(skip_output, residual_output, name1="skip", name2="residual")` before the sum                                                                           | both shapes                      |
| alpha activations with `num_parameters > 1`          | `-input.ndim <= channel_dim < input.ndim` and `input.shape[channel_dim] == num_parameters`, in `_broadcast_alpha`                                                            | `channel_dim`, `num_parameters`  |

`TransformedLoss` and `MulticlassFlatten` wrap an arbitrary criterion; the shape policy is the one of
that criterion (P4 says so). The registry gets a field `wrong_inputs: Callable | None` (inputs with a
wrong shape) and one test asserts `ValueError` for every case that has one; a case without it is
listed in a named set with the reason, like `NOT_REGISTERED` (the elementwise modules, which accept
any shape).

### 4.4 `safe_pow` and its module (P1; F6, F7)

```python
def safe_pow(input, exponent):  # exponent >= 0, finite (unchanged)
    if exponent == 0:
        return torch.ones_like(input)
    is_nonzero = input != 0
    return torch.where(is_nonzero, torch.where(is_nonzero, input, 1.0).pow(exponent), 0.0)
```

`safe_pow` is `torch.pow` with a gradient of `0` at `0` and nothing else: a negative input with a
fractional exponent is `nan`, and `safe_pow(-2, 2)` is `4` [prototype]. The relative indicators call
it on `abs()` values and do not change. `SafePow(exponent)` is the module, completing the trio
`SafeExp`, `SafeLog`, `SafePow`; its docstring and that of `Pow` point to each other (`Pow` is the
plain `torch.pow`; the negative exponents stay there).

### 4.5 One public path for `Reduction` (P12; F8)

`Reduction` and `VALID_REDUCTIONS` are defined once, in `sonnix.validation.loss` (the lowest layer,
which `validate_loss_reduction_strategy` needs). `sonnix.functional` exports `Reduction` (it is the
type of the `reduction` argument of every function, next to `reduce_loss`); `sonnix.validation`
keeps `VALID_REDUCTIONS` and `validate_loss_reduction_strategy` and **stops exporting**
`Reduction`. `refs/validation.md` and `refs/functional.md` are updated. Two tests: `test_docs`
requires `sonnix.<package>.<name>` for the package that exports the name, and a new test asserts
that no object is exported by two packages (`id` of the objects of the six `__all__`).

### 4.6 Simplifications (F12)

- `NLinear.forward`: after the shape check, `out = torch.einsum("...ni,nio->...no", input, self.weight)`
  and `out + self.bias`. No reshape, no transpose, any number of leading dimensions; export and
  compile pass [prototype]; `tests/benchmark/test_nlinear.py` is the gate. The dead branch of
  `reset_parameters` is removed.
- Focal and Poly-1: a private core per loss, `_focal(ce_loss, p_t, target, alpha, gamma, reduction)`
  and `_poly1(ce_loss, p_t, alpha, reduction)` in `functional/loss/`; the four public functions
  validate, cast the target, compute `ce_loss` and `p_t`, and call it. The probability and logits
  variants cannot drift again. No behaviour change.
- The `target` paragraph of the four binary losses (copied eight times with the modules) is written
  once in each function and referred to by the modules (`See sonnix.functional.<name>`), like the
  `reduction` paragraph.

### 4.7 Names (F13)

`create(..., num_frequencies=...)` of `CosSinNumericalEncoder` and `AsinhCosSinNumericalEncoder`
becomes `n_frequencies` (D6). The rule in `conventions.md`: counts are `n_<things>`; `num_parameters`
is the name of `torch.nn.PReLU` and stays.

### 4.8 Documentation and tooling (P17; F9, F10, F14)

- **Delete `review/suggestions.md`** and point `CLAUDE.md` to `review/target_state.md`.
- **`test_docstring_args_match_signature`**: for the objects of the six `__all__`, the names of
  `Args:` equal the parameters of the signature (`__init__` for a class). A class that refers to its
  base (`ExpSin`, `Gaussian`, ... "the arguments are the ones of `BaseAlphaActivation`") is listed
  with the base as the reference. A second check reads the `Raises:` blocks: every lowercase
  identifier in double backticks must be a parameter of the signature. Together they fail today on
  `Pow` (no `Args:`) and on `ToBinaryLabel` (`dim`) (F9).
- **`docs/docs/index.md` includes `README.md`** (the `pymdownx.snippets` extension is on): the two
  files share 88 lines today and each edit is made twice.
- **`numerical_stability.md`** is rewritten to what the code does: `safe_exp` is finite in every
  dtype; `safe_log` and the `eps` are raised to the dtype; Poisson with `log_input=True` for a model
  output without positivity constraint; a half-precision section that lists the unrepresentable
  cases of section 1.
- `pre-commit` gets a `pyright` hook (or `inv types.check`); D4 of the second revision is asked
  again (D7).

## 5. Public API after the change

```python
# sonnix.functional (25, unchanged)  Reduction stays here
# sonnix.modules    (57 -> 58)       + SafePow
# sonnix.validation (7 -> 6)         - Reduction (VALID_REDUCTIONS and the validators stay)
# sonnix.utils      (12 -> 14)       + resolve_eps, resolve_max_exponent
# sonnix.factory (3), sonnix.testing (4): unchanged
# total 108 -> 109
```

### Migration table

| Before                                                                                             | After                                                                                |
| -------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------ |
| `safe_log(0)`, `SafeLog()(0)` in `float16`: `-inf`                                                 | `-9.7` (`eps` raised to the smallest normal number of the dtype)                     |
| `poisson_regression_loss`, `relative_loss`, `absolute_relative_error` on `0` in `float16`: `nan`   | finite (same rule)                                                                   |
| `safe_exp(x)` in `float16` for `x > 11.09`: `inf`; `safe_exp(x, max_val=100)` in `float32`: `inf`  | finite: `max_val` is lowered to `log(finfo.max) - 1`                                 |
| `general_robust_regression_loss` in `float16`, `\|p - t\| / scale > 255`: `inf` and `nan` gradient | computed in `float32`, cast back                                                     |
| `Gaussian` gradient of `alpha` in `float16` for large `\|x\|`: `nan`                               | `0`                                                                                  |
| `safe_pow(negative, 0.5)`: `0`; `safe_pow(-2, 2)`: `0`                                             | `nan` (like `torch.pow`); `4`                                                        |
| `ScaleAndShift(5)(x[3, 1])`, `DynamicTanh`, `DynamicAsinh`                                         | `ValueError` naming `normalized_shape`; use `x.expand(3, 5)` to broadcast on purpose |
| `SumFusion()(a[4, 3], b[4, 1])`, `AverageFusion`, `MultiplicationFusion`                           | `ValueError` naming both shapes; `expand` first                                      |
| `ResidualBlock` with branches of different shapes                                                  | `ValueError` naming `skip` and `residual`                                            |
| `Gaussian(3)(x[4, 1])`, `Gaussian(3, channel_dim=1)(x[5])` (broadcast, `IndexError`)               | `ValueError` naming `channel_dim` and `num_parameters`                               |
| `from sonnix.validation import Reduction`                                                          | `from sonnix.functional import Reduction`                                            |
| `CosSinNumericalEncoder.create("linspace", num_frequencies=8, ...)`                                | `create("linspace", n_frequencies=8, ...)` (D6)                                      |

## 6. Dependents

| Consumer                                                            | Effect                                                                                                                                                                       |
| ------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `tests/registry.py`                                                 | `RelativeLoss` is built with `nn.L1Loss(reduction="none")` (`MSELoss` overflows in `float16` on its own); fields `wrong_inputs`, `fixed` (the label inputs); `SafePow` entry |
| `tests/integration/modules/test_dtype.py`, `test_export_compile.py` | `float16` in `DTYPES`; export with a dynamic batch dimension; the new edge-input test                                                                                        |
| `tests/unit/functional/test_half_precision.py`                      | adds zeros, tiny and large inputs; `general_robust` on `±1000`; the `_scaled` workaround is removed                                                                          |
| `tests/unit/test_docs.py`, `test_imports.py`                        | per-package name; no object exported twice; `Args:` against the signature                                                                                                    |
| `docs/docs/refs/*`                                                  | `Reduction` moves to `functional.md`; `SafePow` in `modules/activation.md`; `utils.md` gets two helpers                                                                      |
| `docs/docs/conventions.md`                                          | the table of section 1 (P1, P4, P6, P9, P10, P12, P17), the rule on counts (4.7)                                                                                             |
| `docs/docs/guide/numerical_stability.md`, `losses.md`               | 4.8; `losses.md` mentions that `eps` is raised to the dtype                                                                                                                  |
| `docs/docs/index.md`, `README.md`                                   | `index.md` includes `README.md`; the overview lists `SafePow`                                                                                                                |
| `CLAUDE.md`                                                         | the pointer to `review/suggestions.md` becomes `review/target_state.md`                                                                                                      |
| `CHANGELOG.md`                                                      | one "Breaking changes" entry per step, the migration rows above, with PR numbers                                                                                             |
| `pyproject.toml`                                                    | import-linter: unchanged. `ruff`: unchanged unless D7 is approved                                                                                                            |

## 7. Evidence

Probes and prototypes ran in the scratchpad; the commands are not in the repository. `float16` and
`bfloat16` on CPU, `torch 2.15`.

| Case                                                        | `main`, `float16` | Prototype, `float16`           | `float32`         |
| ----------------------------------------------------------- | ----------------- | ------------------------------ | ----------------- |
| `safe_log(0)`                                               | `-inf`            | `-9.703`                       | `-18.42`          |
| `safe_exp(1000)`                                            | `inf`             | `24192` (margin `1.0`)         | `4.85e8`          |
| `poisson_regression_loss(0, [0, 1])`                        | `[nan, inf]`      | `[6.1e-5, 9.703]`              | `[1e-8, 18.42]`   |
| `relative_loss(0, 0)`                                       | `nan`             | `0`                            | `0`               |
| `general_robust`, `alpha=1`, `p - t = 300`: value, gradient | `inf`, `nan`      | `299.0`, `1.0`                 | `299.0`, `1.0`    |
| `general_robust`, `alpha=0`, `p - t = 300`: value, gradient | `inf`, `0`        | `10.711`, `0.007`              | `10.714`, `0.007` |
| `Gaussian`, `x = 1000`: gradient of `alpha`                 | `nan`             | `0`                            | `0`               |
| `safe_pow([-4, 0, 4], 0.5)`: values, gradient at `0`        | `[0, 0, 2]`, `0`  | `[nan, 0, 2]`, `0` (`float32`) | same              |

The failures of the 51 registered modules on zeros, tiny, `1e3` and `-1e3` inputs (finite in
`float32`, not in `float16`; `bfloat16` has none): `SafeLog`, `SafeExp`, `PoissonRegressionLoss`,
`RelativeLoss`, `GeneralRobustRegressionLoss`, `Gaussian` (gradient), and the results that `float16`
cannot represent (`Pow`, `Square`, `SquaredReLU`, `ExU`, the `MSELoss` inside `RelativeLoss`), which
the rule of section 9 excludes by construction.

| Measure                                                          | Today                                            | Target                            |
| ---------------------------------------------------------------- | ------------------------------------------------ | --------------------------------- |
| Properties of section 1 that are true                            | 11 of 17 (P2, P3, P5, P7, P8, P10, P11, P13-P16) | 17 of 17                          |
| Public names (`functional` / `modules` / `validation` / `utils`) | 25 / 57 / 7 / 12                                 | 25 / 58 / 6 / 14                  |
| Registered modules tested in `float16`                           | 0 of 51                                          | 51 of 51                          |
| Functions with an edge-input test in half precision              | 0 (hand-picked inputs)                           | all the registered losses and ops |
| Public objects whose `Args:` is checked against the signature    | 0                                                | 105                               |
| `src` lines                                                      | 7,124                                            | about 7,250                       |
| `tests` lines                                                    | 13,806                                           | about 14,200                      |
| Lines of obsolete or duplicated documentation                    | 505 (`suggestions.md`) + 88 (`index.md`)         | 0                                 |
| Tests / doctests / lint / types on `main`                        | 9,882 / 121 / clean / clean                      | same or more                      |

## 8. Changes to reach the target state

One branch and one pull request per step (`CLAUDE.md`), small and reviewable on its own. Every step
updates `CHANGELOG.md`, the docs it touches and the tests, and runs `ruff`,
`inv types.check --complete`, `pytest tests`, `xdoctest src/sonnix all` and, for a step that
changes `docs/`, `mkdocs build --strict`. Steps 1, 2, 3 and 4 are independent and can start from
`main` in parallel; 5 needs 2 (same files in `functional/`); 6 is last.

1. **Honest documentation (non-breaking).** F9, F10, F14: delete `review/suggestions.md`; fix
   `CLAUDE.md`; fix the `Args:` of `Pow` and the `Raises:` of `ToBinaryLabel`; add
   `test_docstring_args_match_signature`; make `test_docs` per package (`sonnix.functional.Reduction`
   is documented only under `validation` until step 4, so the test lists it as a known exception
   that step 4 removes); `index.md` includes `README.md`; the `pyright` hook; update
   `conventions.md` with the table of section 1. _Why first: the rules are written before the code
   changes, and the failing checks are typos._
2. **Numerics (breaking in `float16` and for `safe_pow`).** F1-F4, F6, F11: `resolve_eps` and
   `resolve_max_exponent` in `sonnix.utils` with their tests; the five sites of 4.2; the `float32`
   computation of `general_robust`; the `Gaussian` formula; `safe_pow` on `input != 0`;
   `float16` in `DTYPES`; the registry rule `test_finite_wherever_float32_fits` (section 9); the
   half-precision unit tests extended; the guide rewritten. Each of F1-F4 and F6 has a regression
   test that fails on `main` (`git stash push src`).
3. **Strict shapes (breaking: values that were accepted).** F5: the four checks of 4.3,
   `ModuleCase.wrong_inputs` and the registry test. The tests that rely on broadcasting
   (`grep` for `(4, 1)` in the fusion and scale tests) are changed to `expand`.
4. **One path and `SafePow` (breaking: an import path).** F7, F8: `SafePow` with a registry entry
   (parity with `safe_pow`, export, compile, dtype); `Reduction` exported by `functional` only; the
   no-duplicate test; the exception of step 1 removed; the refs pages.
5. **Simplifications (non-breaking).** F12: `NLinear` with `einsum` (benchmark before and after in
   the PR), the dead branch, the focal and Poly-1 cores, the `target` paragraph. _Skip any item
   whose benchmark or export test regresses._
6. **Names and migration notes (breaking, small).** F13 (D6): `n_frequencies`; fill the migration
   table of `CHANGELOG.md` with the rows of section 5 and the PR numbers; update `README.md` and
   `docs/docs/index.md`.

Steps 2 and 3 are the only ones that change the behaviour of calls that worked (a `float16` result
that was `inf` or `nan`; a broadcast that was silent). Steps 1, 4, 5 and 6 remove or rename a name,
or change nothing visible.

## 9. Tests and gates

The unit-test rules of `CLAUDE.md` apply to everything added (one behaviour per test, `match` on
every `pytest.raises`, seeded data, no local imports, literal expected values, a regression test for
each bug, an integration test for each patched behaviour, 100 % coverage of the lines added or
changed). The tests added by this revision are **registry-driven**: a new class is covered by adding
one `ModuleCase`.

- **Numerics (P6)**: `test_finite_wherever_float32_fits(case, edge, dtype)` for the 51 cases, the
  edges `zeros`, `tiny` (`1e-10`), `large` (`1e3`), `negative_large` (`-1e3`) applied to the floating
  inputs (`ModuleCase.fixed` keeps the labels of the binary losses), and the dtypes `float16` and
  `bfloat16`. The rule: run in `float32`; for the output and for each gradient whose `float32` value
  is finite and below `finfo(dtype).max / 4`, assert the value in `dtype` is finite. It excludes by
  construction what the dtype cannot represent (a `Square` of `1000` in `float16`). It fails on
  `main` for `SafeLog`, `SafeExp`, `PoissonRegressionLoss`, `RelativeLoss`,
  `GeneralRobustRegressionLoss` and `Gaussian`. The functions without a module (the errors, the
  indicators, `safe_pow`, `rectifier_asinh_unit`) go through a short list `FUNCTION_CASES` with the
  same rule. Unit tests with literal values for `resolve_eps` and `resolve_max_exponent` (`float16`,
  `bfloat16`, `float32`, `float64`, an integer tensor).
- **Shapes (P4)**: `test_wrong_inputs_raise_value_error` over the cases with `wrong_inputs`; the
  literal messages of the four checks; `x[3, 1]` into `ScaleAndShift(5)` is the regression test.
- **P1**: `SafePow` against `safe_pow` in the parity test; `safe_pow` literal values for negative
  inputs and the gradient at `0`.
- **P9, P10**: `float16` in `DTYPES`; `torch.export` with `dynamic_shapes` on the batch dimension for
  the registry.
- **P12, P17**: per-package reference test; no object exported twice; `Args:` against the signature.
- **Simplifications**: the existing equivalence and gradcheck tests pass unchanged (step 5 is a
  refactor); `tests/benchmark/test_nlinear.py` before and after.
- **Docs**: `xdoctest src/sonnix all`, `mkdocs build --strict`, the notebook job.

## 10. Decisions for the maintainers

| #   | Decision                                                                                            | Recommendation                                                                                                                                                                                                                                                    |
| --- | --------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| D1  | `resolve_eps` and `resolve_max_exponent` public in `sonnix.utils` (4.2)?                            | **Yes.** `CLAUDE.md`: shared helpers are public and tested; users who write their own safe ops need them (the library rule of #453). A private copy in `functional` would be imported by `modules` across a layer.                                                |
| D2  | Add `SafePow` (4.4), or reword P1 so that `safe_pow` is function-only?                              | **Add it.** About 70 lines with the docstring, completes `SafeExp`/`SafeLog`/`SafePow`, makes P1 literally true. The alternative costs nothing and leaves the one safe op that cannot be used in a `Sequential`.                                                  |
| D3  | `safe_pow` as `torch.pow` with a finite gradient at zero (negative input gives `nan` or the power)? | **Yes.** Today it returns a wrong number silently (F6). A data-dependent error is not possible under `torch.export`, and `nan` is what `torch.pow` returns.                                                                                                       |
| D4  | Where does `Reduction` live (4.5)?                                                                  | **`sonnix.functional`**, next to `reduce_loss`; `sonnix.validation` keeps the validators. The other way (validation) is as short but users look for the type where they find the function.                                                                        |
| D5  | Strict shapes for the three elementwise fusions and the residual block (4.3)?                       | **Yes** (P4). It breaks `[B, F] * [B, 1]`, which is one `expand` or one `*` away. A `broadcast=True` option is rejected: it brings back the silent case for the one who sets it by habit.                                                                         |
| D6  | `num_frequencies` becomes `n_frequencies` (4.7)?                                                    | **Yes**, last and independent (step 6): it is a rename with no behaviour change. Drop it if the churn is not wanted.                                                                                                                                              |
| D7  | `lint.flake8-builtins.builtins-ignorelist = ["input"]` to remove the 45 `# noqa: A002` (F14)?       | **Yes**, once, in `pyproject.toml`: `input` is the `torch.nn` name and the only shadowed builtin. It is asked again because `CLAUDE.md` says a changed rule set is not a fix, so it needs explicit approval.                                                      |
| D8  | `float16` is a supported dtype of P6, tested on CPU (`DTYPES`, edge rule)?                          | **Yes.** All 51 modules run in `float16` on CPU, the `ExU` docstring and the half-precision tests already treat it as supported, and every real failure of this review is `float16`. The alternative (only `bfloat16`) makes F1-F4 "documented" instead of fixed. |
| D9  | A learnable `alpha` that crosses `0` (`Gaussian`, `Laplacian`, `Snake`): parametrise it?            | **No change.** It needs an experiment (a `softplus` or `abs` parametrisation changes the model), not a rule. It stays documented in the guide and in P6.                                                                                                          |

Closed decisions (answered by the maintainers, not reopened): keep the named wrappers `ToFloat`,
`ToLong`, `ReLUn`, `Square`, the three errors and `rectifier_asinh_unit` (#451); keep the helpers of
`sonnix.utils` and `DummyDataset` (#453); `validate_positive` in `sonnix.validation`, `edges` and
`n_bins` as segments, integer targets cast, `Clamp(min_val, max_val)` positional, `sonnix.factory`
kept, no folder move for the top-level modules (second revision, #447-#455).

## 11. Not covered

- New losses and layers (`Tweedie`, `Dice`/`Tversky`, `SwiGLU`/`GeGLU`, a multiclass focal loss): the
  target gives the rules a new object follows (P1-P17, Appendix B), not which objects to add.
- Sample weights, masks and `ignore_index` in the losses: a feature with its own design (the shape of
  the weights against `reduction`).
- A parametrisation of a learnable `alpha` (D9).
- **A stable release.** `coola>=2.0.0a5` is a pre-release; the latest stable `coola` on PyPI is
  `1.1.10` (checked today). `sonnix` cannot leave `0.0.1a0` before `coola` 2.0 is stable, or before
  the dependency is relaxed.
- Packaging and CI (the workflows, a GPU job, `docformatter` against `ruff` docstring formatting):
  tracked in `.github/GUIDELINES.md`.
- Lazy top-level imports (`import sonnix; sonnix.modules.X`): `import sonnix.modules` stays explicit
  and keeps the import time predictable.
- Fusing the focal and Poly-1 operations to cut the memory of the intermediates (`binary_focal_loss`
  is `27` µs against `4` µs for `binary_cross_entropy` on a `16 × 4` tensor [probe]): a performance
  topic, to do with a benchmark that shows a need.

## Appendix A: size of the package

| Area                                            | Lines today | Lines after | Change                                                                              |
| ----------------------------------------------- | ----------- | ----------- | ----------------------------------------------------------------------------------- |
| `utils`                                         | 627         | ~680        | `resolve_eps`, `resolve_max_exponent` (~50 with docstrings and examples)            |
| `validation`                                    | 217         | ~214        | `Reduction` no longer exported                                                      |
| `functional`                                    | 1,493       | ~1,470      | five guards (+10), `general_robust` (+6), focal and Poly-1 cores (−40)              |
| `modules` (`SafePow`)                           | 4,340       | ~4,440      | `SafePow` (+70), the four checks (+35), `NLinear` (−8), `Gaussian` (0)              |
| `factory`, `testing`                            | 430         | 430         | none                                                                                |
| **Total `src`** (with `sonnix/__init__.py`, 17) | **7,124**   | **~7,250**  | +~125                                                                               |
| `tests`                                         | 13,806      | ~14,200     | the registry rules replace hand-picked cases; +~400                                 |
| `review/`, `docs/index.md`                      | 1,100 + 145 | ~540 + 57   | −`suggestions.md` (505), this file rewritten, `index.md` includes `README.md` (−88) |

## Appendix B: checklist for a new public object

1. Is it a fixed-argument composition of existing public objects or of a `torch.nn` object? Then it is
   a recipe in the guide, unless users look for it by name (P16).
2. Function first if it is a loss or needs numerical care; the module calls it (P1).
3. Every hyperparameter: type, finiteness, range, before it is stored; the module uses the same
   helper (P2).
4. Signature `(prediction, target, *, options, reduction)` or `(input, *, options)`; a module's
   positional arguments are listed in its registry entry (P3).
5. Two same-role tensors: `validate_same_shape`; a layer checks the dimension it consumes and names the
   argument; nothing broadcasts (P4). Loss: `reduce_loss` (P5).
6. A guard (`eps`, `min_val`, `max_val`) goes through `resolve_eps` / `resolve_max_exponent`; an
   unbounded square or exponential is computed in at least `float32`; the registry edge rule passes in
   `float16` (P6).
7. A trained tensor that can be frozen: `register_tensor`; anything derived is non-persistent and
   rebuilt after a load (P8). `device` and `dtype` if the module creates tensors (P9).
8. Public attributes named like the arguments, shown in `repr` (P11).
9. One registry entry (with `positional`, `other`, `wrong_inputs`): export, compile, dtype, purity,
   parity, round-trip, shape and edge tests come for free (P10).
10. Export it from the flat `__init__.py` of **one** package, document it under that package in
    `docs/docs/refs` with an `Args:` that matches the signature and an example that runs (P12, P17).

## Appendix C: history of the previous revisions

First revision (2026-11-06, `fc04eeb`, twelve steps, all merged). Closed: shape policy, validation
of the hyperparameters, keyword-only function options, `input` for every forward, composition losses,
dynamic norms by composition, encoder range logic and input check, package layout, alpha
`_validate_init`, copied `Args:` blocks.

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

Second revision (2026-11-22, nine steps). Closed: non-finite hyperparameters, the state of the
piecewise encoder (`edges`, `register_tensor`), uniform signatures, the public paths, the private
bases, integer targets in the binary losses, the `repr` of `RelativeLoss`.

| Step | Pull request | Summary                                                                            |
| ---- | ------------ | ---------------------------------------------------------------------------------- |
| 1    | #446         | properties and docs                                                                |
| 2    | #447         | validation of every hyperparameter, `validate_positive`                            |
| 3    | #449         | complete state: `register_tensor`, `edges`, load hook                              |
| 4    | #450         | uniform signatures                                                                 |
| 5    | #451         | surface area: private bases, `ToDtype` accepts a name; the named wrappers are kept |
| 6    | #452         | one public path for every package                                                  |
| 7    | #453         | dropped: `sonnix` is a library, the helpers are kept                               |
| 8    | #454         | integer targets, `RelativeLoss` repr                                               |
| 9    | #455         | migration table                                                                    |

## Appendix D: status of this revision

| Step | Pull request | Notes       |
| ---- | ------------ | ----------- |
| 1    |              | not started |
| 2    |              | not started |
| 3    |              | not started |
| 4    |              | not started |
| 5    |              | not started |
| 6    |              | not started |
