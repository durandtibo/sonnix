# `sonnix`: target state and the changes to reach it

Date: 2026-11-22, on `main` at `fa8e72a`. This is the fourth revision. The first one (twelve steps,
#413-#440), the second one (nine steps, #446-#455) and the third one (six steps, #459-#464) are
merged; their history is in [Appendix C](#appendix-c-history-of-the-previous-revisions) and in
`CHANGELOG.md`.

This document describes the design `sonnix` should have and the steps to get there from `main`. It
follows the format of
[`coola/review/recursive.md`](https://github.com/durandtibo/coola/blob/main/review/recursive.md):
properties, constraints, findings, design, public API, dependents, evidence, plan, tests, decisions.
The properties come first: every choice below exists to make one of them true **by construction**
(one helper, one registry test, one signature-driven test) instead of by the discipline of each
class.

What this revision is about. The first three revisions fixed the structure (shapes, validation,
signatures, state, public paths) and the numerics on edge inputs. The seventeen properties hold on
the surface the tests reach. This revision runs them on the surface the tests do not reach, and
finds four things: a special-case formula that is discontinuous (F1), the gradient-preserving mode
of `safe_exp` and `safe_log` that returns wrong values (F2), a wrapper that reshapes without
checking (F3), and two packages (`utils`, `testing`) that the validation fuzz of the second revision
never covered (F4). The common cause of the last one is that the checks which keep the properties
true are partly **hand-written tables**, so the next object can escape them silently. That is the
new property **P18, easy to maintain**, and the reason why most steps below replace a table by a
rule derived from the signature or the registry.

Scope: all of `src/` (7,438 lines in 68 files; about 1,700 lines of code once docstrings, comments
and blank lines are removed), `tests/` (15,124 lines, 11,143 tests), the documentation (613 lines of
markdown with `README.md`), `CLAUDE.md`, `pyproject.toml`, the workflow names, `review/`. Items
marked **[probe]** were reproduced today by running code in the scratchpad (outside the repository);
**[prototype]** means the proposed fix was run in a throw-away script and gave the stated result;
**[reading]** comes from the source. Nothing in the repository was changed except this file.

Baseline on `main`: `ruff check` clean; `inv types.check --complete` clean (`ty` and `pyright`, 0
errors); `lint-imports` 3 contracts kept; `xdoctest` 126/126; `pytest tests` 11,143 passed, 3
skipped, in 22 s; `mkdocs build --strict` passes (its only message: Material for MkDocs reaches end
of life on 2027-05-05). The `python` and `pycon` examples of `README.md` and of the guides run
[probe], but no CI job runs them (F7).

## 1. Properties of the package

`sonnix` is a library of PyTorch building blocks that `torch.nn` does not provide: losses,
parametric activations, numerical-feature encoders and a few layers. It is used by other codebases
(decision #453), so a public name is useful if a user needs it, not only if `sonnix` calls it. The
API has three tiers: `sonnix.functional` (stateless operations), `sonnix.modules` (the
`torch.nn.Module` wrappers and layers) and the support packages (`validation`, `factory`, `utils`,
`testing`). 110 public names: 25 / 58 / 6 / 14 / 3 / 4.

Each property is a rule that holds for **every** public object, with the mechanism that makes it
true and the test that proves it. A property without a test is a wish. The column "Today" is the
result of this review. P1-P17 keep their number; P2, P4, P6, P11 and P17 are made more precise; P18
and P19 are new.

| #   | Property                            | Rule                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   | Enforced by                                                                                                                                       | Today                           |
| --- | ----------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------- |
| P1  | **Function/module duality**         | Every loss, and every op that needs numerical care, is a stateless function in `sonnix.functional` **and** a module in `sonnix.modules` that stores the hyperparameters and calls the function: the losses, `safe_exp`/`SafeExp`, `safe_log`/`SafeLog`, `safe_pow`/`SafePow`. A module with learnable state or without numerical care (an activation, a layer) has no function. A function that users look for by name (`rectifier_asinh_unit`, the errors, the indicators) may have no module (P16). A `forward` is one call.                                                                                                                                                                                                                                                                                         | parity registry test: module output equals function output                                                                                        | true                            |
| P2  | **Validate early and completely**   | Every scalar hyperparameter of **every** public object (functions, modules, `utils`, `testing`, `create` class methods) is checked for **type** (a `bool` is not an `int`), **finiteness** and **range**, before it is stored or converted, with `coola.validation` or a `sonnix.validation` helper. A float is finite unless its docstring says otherwise. The function checks at call time; the module checks in `__init__` with the same helper and message. A helper that is called after its caller validated (`resolve_eps`, `resolve_max_exponent`) documents the precondition instead of validating twice.                                                                                                                                                                                                     | signature-driven invalid-argument test over the six `__all__`: a scalar parameter without a check fails (new, 4.5)                                | **not true** (F4, F5)           |
| P3  | **Uniform call signature**          | Losses: `(prediction, target, *, <options>, reduction)`. Single-tensor ops: `(input, *, <options>)`. Options of a function are keyword-only. A module takes positionally its structural arguments (sizes, wrapped modules, tensors) and at most one hyperparameter; the rest is keyword-only. The tensor of a `forward` is `input`.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    | `ModuleCase.positional` compared with `inspect.signature`                                                                                         | true                            |
| P4  | **Strict shapes**                   | Two tensors of the same role must have the same shape (`ValueError` naming both): the prediction and the target, the inputs of a sum, average or product fusion, the two branches of a residual block. A layer or an encoder checks the dimension it consumes and names the argument. Nothing broadcasts silently. A wrapper that **reshapes** (`MulticlassFlatten`) checks the shapes it relies on; a wrapper that only forwards (`TransformedLoss`) leaves the policy to the wrapped module.                                                                                                                                                                                                                                                                                                                         | `validate_same_shape`; registry test `wrong_inputs` for every module                                                                              | **not true** (F3)               |
| P5  | **One reduction contract**          | Every loss accepts `none`/`mean`/`sum`/`batchmean`, validated by `reduce_loss`; default `mean`; `none` keeps the shape of the elementwise loss; `reduction` is keyword-only.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           | `reduce_loss`; registry test over all losses                                                                                                      | true                            |
| P6  | **Numerically safe**                | For `float16`, `bfloat16`, `float32` and `float64`: wherever the `float32` result and its gradient are representable in the dtype, the output and the gradient are finite, **for every option** (`preserve_gradient` included). A **default** hyperparameter is valid in every dtype (a guard is raised to the smallest normal number of the dtype; the `max_val` of `safe_exp` is lowered so that `exp` is finite). An unbounded input is squared or exponentiated in at least `float32`. A hyperparameter that can still overflow (an exponent) is documented.                                                                                                                                                                                                                                                       | registry test `test_finite_wherever_float32_fits`; the option `preserve_gradient` is added to its axes (new)                                      | **not true** (F2)               |
| P7  | **Pure**                            | A function or a `forward` never mutates its inputs and keeps no hidden state.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          | in-place-mutation registry test                                                                                                                   | true                            |
| P8  | **Complete state, by construction** | The `state_dict` holds the parameters and buffers that define the behaviour, and **nothing derived**. A tensor derived from persistent state is non-persistent and recomputed after `load_state_dict`. A frozen tensor is a buffer with the same name as the learnable one (`register_tensor`). Round trip: `m2.load_state_dict(m1.state_dict())` makes `m2` behave as `m1`.                                                                                                                                                                                                                                                                                                                                                                                                                                           | `register_tensor`; round-trip registry test                                                                                                       | true                            |
| P9  | **Device and dtype**                | A module that creates tensors takes `device` and `dtype`. After `module.to(dtype)` every floating tensor of the module has `dtype` and the output has the dtype of the input. A module whose dtype differs from the dtype of the input follows the promotion rules of `torch` (`NLinear` raises): that case is not guaranteed.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         | dtype/device registry tests, `float16` included                                                                                                   | true                            |
| P10 | **Export and compile**              | Every registered module passes `torch.export` (also with a dynamic batch dimension) and `torch.compile(fullgraph=True)` on CPU: no `.item()`, no data-dependent Python branch in a `forward`. A shape check on static dimensions is allowed.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           | registry tests                                                                                                                                    | true                            |
| P11 | **Introspectable**                  | Every hyperparameter, including a callable or a submodule, is a public attribute named like the constructor argument and is shown by `repr`. A scalar hyperparameter is stored as a plain Python value of its declared type (a `float` argument is a `float`).                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         | `test_hyperparameters`; the type is added (new)                                                                                                   | **not true** (F11)              |
| P12 | **One public path**                 | Every public object has **one** public path. `sonnix.{functional,modules,validation,factory,utils,testing}` each export their API from `__init__.py` (`__all__`); a name is exported by **one** package; the sub-folders have no `__all__`. Docs, docstrings, examples **and imports of another package** use the flat path. Every exported name is listed under its own package in `docs/docs/refs` and has an `Example` that runs.                                                                                                                                                                                                                                                                                                                                                                                   | `test_imports`, `test_docs`; flat cross-package imports (new, F8)                                                                                 | **not true** (F8)               |
| P13 | **Layered imports**                 | `testing → modules → functional → validation`; `utils`, `factory` and `validation` import no other sonnix package; no cycle.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           | import-linter contracts                                                                                                                           | true                            |
| P14 | **Configuration is trusted input**  | `sonnix.factory` imports and **calls** a `_target_` before it checks the type of the result. An argument typed `Module \| dict` says so in a `Warning`.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                | `test_config_warning`                                                                                                                             | true                            |
| P15 | **Optional means lazy**             | Only `torch` and `coola` are imported at module level.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 | `package_checks` job                                                                                                                              | true                            |
| P16 | **Earned names**                    | A public name is a loss, a layer, an activation, an encoder, an op with numerical care, or a helper that these use, that tests them, or that users need. A fixed-argument composition is a recipe in the guide, unless users look for it by name (`ReLUn`, `Square`, `ToFloat`). A base class is public only if users are meant to subclass it.                                                                                                                                                                                                                                                                                                                                                                                                                                                                        | review checklist (Appendix B); a class without a registry entry fails                                                                             | true                            |
| P17 | **Honest documentation**            | A docstring says what the code does: `Args:` lists exactly the arguments of the signature, `Raises:` names what the code raises, and a guarantee stated in a docstring or in the guide has a test. **Every example shown to a reader (docstring, `README.md`, guide) is executed by a test.**                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          | signature-vs-`Args:` test, `xdoctest`, markdown examples test (new, F7)                                                                           | **not true** (F7)               |
| P18 | **Easy to maintain** (new)          | A rule lives in one place and an object is added or changed in the fewest places. (1) **One mechanism per rule**: validation, reduction, shape check, dtype guard, state registration and a core shared by two variants are one helper, never a copy in a class. (2) **One source per fact**: a description, a rule or a number is written once; the other places are generated or checked against it by a test. (3) **A new object is one source entry, one `__all__` line, one reference directive and one registry entry**; every other test is derived from the registry or from the signature, and a missing entry fails a test. (4) **No suppression** (`noqa`, `type: ignore`, `ty: ignore`, `pragma: no cover`) in `src`. (5) **Small surface**: two hard dependencies, every declared `torch` version tested. | registry completeness tests; signature-driven tests (new); suppression and flat-import tests (new); import-linter; the CI matrix `torch` 2.4-2.15 | **partly** (F4-F6, F8, F9, F12) |
| P19 | **Matches its definition** (new)    | Every loss and op equals its reference on typical inputs (the `torch` equivalent or a naive formula in `float64`). A formula with limit cases (`alpha=2`, `alpha=0`, `alpha=-inf` of the robust loss) is **continuous** across the switch. An option that is only about the gradient (`preserve_gradient`) does not change the value.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  | reference tests (exist for most); continuity test at the branch points and value test of `preserve_gradient` (new)                                | **not true** (F1, F2)           |

Not guaranteed, and documented as such: bit-exact results across `torch` versions; behaviour on GPU
in CI (no GPU job; the CUDA tests run locally before a release); a learnable `alpha` that crosses
`0` (decision D9 of the third revision: documented, not changed); a result that the dtype cannot
represent (`Square` of `1000` in `float16`, `exp(100)` in `float32`); the probability variants of
the binary losses with a prediction of exactly `0` or `1` in `float16` (use the `*_with_logits`
variants); a module fed a dtype other than its own (P9); the gradient at a non-finite input.

## 2. Constraints

1. **Simple and easy to maintain (P18).** A property is made true by **one** helper, one registry
   rule or one signature-driven test, not by a check written in each class or a row written by hand.
   A step that adds a rule also removes the table it replaces. Net size is in Appendix A.
2. **Breaking changes are fine.** The version is `0.0.1a0`: no shim, no deprecation period. Each
   breaking change is listed in `CHANGELOG.md` (migration table) with the PR number.
3. **Same numerics for valid calls**, except where a finding names it (F1: the default `alpha=2`
   loss halves; F2: `preserve_gradient=True` on large inputs). Each such change has a regression
   test that fails without the fix (`git stash push src`).
4. **No slower than today.** The default path of every function is unchanged. The new validation is
   on `utils` and `testing`, not on a forward pass; `resolve_eps` and `resolve_max_exponent` (on the
   forward path) get no validation. The `preserve_gradient=True` path adds one `where` and one
   `detach`.
5. **Closed decisions are not reopened.** The helpers of `sonnix.utils`, `DummyDataset`, the named
   wrappers, the three errors and `rectifier_asinh_unit` are **kept** (#451, #453). This revision
   removes and adds no public name.
6. **Follow `CLAUDE.md` and `docs/docs/conventions.md`.**
7. **`torch` and `coola` are the only hard dependencies.**

## 3. What differs today

| #   | Finding                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     | Status              |
| --- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------- |
| F1  | **`general_robust_regression_loss(alpha=2)` is discontinuous (P19).** The general branch is Barron's `ρ`, whose limit at `alpha=2` is `0.5 * (x/c)²`; the limit branch of `alpha=2` returns `(x/c)²`, twice as much. At `x=1`: `alpha=1.99` gives `0.491`, `alpha=2.0001` gives `0.5002`, `alpha=2` gives `1.0`. At `x=3`: `4.371`, `4.502`, `9.0`. The other limits (`alpha=0`, `-inf`) are right. The default `alpha` is `2.0`, so the default loss is twice the value of its own neighbours. The tests pin `(x/c)²` (`ones` against `-ones` gives `4.0`) and never compare two branches. | [probe]             |
| F2  | **`preserve_gradient=True` changes the value (P19, P6).** `input + (clamped - input).detach()` loses `clamped` by cancellation when `\|input\|` is large. `safe_exp(1e8, preserve_gradient=True)` is `8.9e6` (`exp(16)`, expected `4.85e8`); `safe_exp(1e30, ...)` is `1.0`; `safe_log(-1e30, ...)` is `-inf` (expected `-18.4`); `safe_exp(inf, ...)` and `safe_log(-inf, ...)` are `nan` (without the option: finite). The guide promises a finite output.                                                                                                                                | [probe] [prototype] |
| F3  | **`MulticlassFlatten` does not check what it reshapes (P4).** `prediction (2, 6, 3)` with `target (4, 3)` has the same number of elements as the flattened pair, so it is accepted and the loss pairs unrelated rows (`torch.Size([12])` returned). `prediction (4, 3, 5)` with `target (4, 2)` raises a bare `RuntimeError: shape '[8, 5]' is invalid for input of size 60`.                                                                                                                                                                                                               | [probe]             |
| F4  | **P2 stops at `modules` and `functional`.** The fuzz of the second revision covered the 57 classes. `sonnix.testing`: `is_loss_decreasing*` accepts `num_iterations=0`, `-1`, `True`, `lr=nan`, `0`, `inf`, `True`, `max_norm=nan`, `-1`, `0`; `DummyDataset` accepts `feature_size=0`, `2.5` and `True` (silently `int()`-converted). `sonnix.utils`: `load_state_dict_to_module(strict="yes")` and `register_tensor(learnable="yes")` are accepted (truthy). 118 scalar parameters are public in total; the ones above are the only ones that accept an invalid value.                    | [probe]             |
| F5  | **The invalid-argument tests are a hand-written table** (`tests/integration/test_invalid_arguments.py`, 428 lines). A function, a module or an argument without a row is not detected, which is how F4 went unnoticed. Type and finiteness rows are the same for every argument of the same type and can be derived from the signature.                                                                                                                                                                                                                                                     | [reading]           |
| F6  | **58 rows of the reference tables copy the first line of a docstring** (`refs/modules/*.md`). Two already differ (single against double backticks: `MulticlassFlatten`, `AsinhCosSinNumericalEncoder`). Nothing checks them.                                                                                                                                                                                                                                                                                                                                                                | [probe]             |
| F7  | **No job runs the examples of the markdown** (`README.md`: 16 `>>>` lines; 5 `python` blocks in the guides). They pass today [probe]; the next edit of a signature breaks them silently (P17). `xdoctest src/sonnix` covers only the docstrings.                                                                                                                                                                                                                                                                                                                                            | [probe]             |
| F8  | **12 cross-package imports use a module path instead of the package (P12)**: 8 `from sonnix.functional.reduction import Reduction` in `modules`, 3 `from sonnix.utils.tensor import register_tensor`, 1 `from sonnix.utils.mode import module_mode` in `testing`. A move of a file inside a package breaks another package. (The thirteenth, `functional` importing `Reduction` from `sonnix.validation.loss`, is deliberate: it is defined in the lowest layer and exported by `functional`.)                                                                                              | [reading]           |
| F9  | **46 `# noqa: A002`** (the argument `input`), the only suppressions in `src`. Decision D7 of the second and third revisions was asked twice and not answered; `CLAUDE.md` says a changed rule set is not a fix, so it needs approval (D6).                                                                                                                                                                                                                                                                                                                                                  | [reading]           |
| F10 | **The count prefix is still not uniform.** The rule says counts are `n_<things>`; `DummyDataset(num_examples)`, `is_loss_decreasing*(num_iterations)` and the private `create_spaced_values(num_values)` use `num_`. `num_parameters` is the name of `torch.nn.PReLU`, and the helpers `num_parameters`, `num_learnable_parameters` follow it: they stay.                                                                                                                                                                                                                                   | [probe]             |
| F11 | **Four scalar attributes keep the value as given** while every other hyperparameter is `float(...)`: `Clamp.min_val`, `Clamp.max_val`, `GeneralRobustRegressionLoss.max_val`, `BaseDynamicNorm.alpha_init_value` (`Clamp(0, 1).min_val` is an `int`). `repr` and the registry test of attributes cannot tell.                                                                                                                                                                                                                                                                               | [probe]             |
| F12 | **The rules are written three times**: `docs/docs/conventions.md` (P1-P17), the table of this document, and the sentence "properties P1-P17" in `CLAUDE.md`. Adding P18 and P19 means three edits, and nothing checks that they agree.                                                                                                                                                                                                                                                                                                                                                      | [reading]           |

What is already right and must not regress: the validation of every hyperparameter of the 57 classes
and of the functions; keyword-only options and the registry check of positional arguments; the state
round trip and the derived buffers of the piecewise encoder; strict shapes in the losses, fusions,
residual block and encoders; function/module parity; the registry-driven export, compile, purity,
dtype, edge-input and hyperparameter tests; the dtype-aware guards; the import-linter layering;
`mkdocs build --strict`; the CI matrix over every `torch` from 2.4 to 2.15. The 52 registered
modules also run, with the same values, on an **empty batch**, after `copy.deepcopy`, after
`torch.save`/`torch.load`, under CPU `bfloat16` autocast and on a non-contiguous input [probe]: no
failure.

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

Nothing moves. The import-linter contracts do not change. A merge of the three files that hold
`safe_exp`, `safe_log` and `safe_pow` was considered and rejected (D8): the layout is not public,
each file is one concept, and a move hides the diff of F2.

### 4.2 The robust loss is continuous (P19; F1)

```python
if abs(alpha - 2) < _ALPHA_EPS:
    loss = squared_error.mul(0.5)  # Barron: rho(x, 2, c) = 0.5 * (x / c) ** 2
```

The docstring says `alpha=2` is `0.5 * x^2`; the guide says that the default loss is half the
squared error. Values [prototype of the formula]:

| `x` (`scale=1`) | `alpha=1.99` | `alpha=2.0001` | `alpha=2` today | `alpha=2` target |
| --------------- | ------------ | -------------- | --------------- | ---------------- |
| `1`             | `0.4909`     | `0.5002`       | `1.0`           | `0.5`            |
| `3`             | `4.3712`     | `4.5023`       | `9.0`           | `4.5`            |

Tests (P19): a naive Barron formula in `float64` against the function for `alpha` in
`{-inf, -2, 0, 1, 2, 3}` and three values of `x`; continuity: the loss at `alpha=2` is within `1e-2`
of the loss at `2 ± 1e-3`, and the same at `0` and at `-1e3` against `-inf`. The two tests that pin
`4.0` and `1.0` become `2.0` and `0.5`. Migration: the value and the gradient of the default loss
halve; `scale=s / sqrt(2)` reproduces the old `alpha=2` value.

### 4.3 `preserve_gradient` keeps the value (P19, P6; F2)

One private helper in `functional/activation/math.py`, used by `safe_exp` and `safe_log` (no other
module imports it):

```python
def _clamp(input, *, min_val=None, max_val=None, preserve_gradient):
    clamped = input.clamp(min=min_val, max=max_val)
    if not preserve_gradient:
        return clamped
    # The value of ``clamped`` and the gradient of ``input`` (identity); ``input - input.detach()``
    # is 0 for a finite input and ``nan`` for an infinite one, which is replaced by 0.
    return clamped.detach() + torch.where(input.isfinite(), input - input.detach(), 0.0)
```

[prototype] `1e8`, `1e30` and `inf` clamp to `20.0` (today `exp(16)`, `1.0`, `nan` after `exp`);
`-1e30` and `-inf` clamp to `1e-8` (today `-inf`, `nan` after `log`). The gradient is `1` for a
finite input and equal to today's on `[-3, 0, 5, 25, 100]`; at an infinite input it is `0` (not
guaranteed, section 1). `torch.export` and `torch.compile(fullgraph=True)` pass. The default path
(`preserve_gradient=False`) is unchanged. The guide states that the value does not depend on the
option.

### 4.4 `MulticlassFlatten` checks its shapes (P4; F3)

```python
if prediction.shape[:-1] != target.shape:
    msg = f"prediction must have the shape of target plus a class dimension, got {tuple(prediction.shape)} and {tuple(target.shape)}"
    raise ValueError(msg)
```

A static-shape comparison: it passes `torch.export`. The module gets a `wrong_inputs` entry in the
registry. `TransformedLoss` is unchanged (it only forwards).

### 4.5 Validation of the whole surface and a signature-driven test (P2, P18; F4, F5, F11)

| Site                                                      | Argument                                 | Check                                                        |
| --------------------------------------------------------- | ---------------------------------------- | ------------------------------------------------------------ |
| `is_loss_decreasing`, `_with_adam`, `_with_sgd`           | `num_iterations`                         | `validate_int`, `validate_ge(..., 1)`                        |
| `is_loss_decreasing_with_adam`, `_with_sgd`               | `lr`                                     | `validate_positive`                                          |
| `is_loss_decreasing*`                                     | `max_norm`                               | `None` or `validate_positive`                                |
| `DummyDataset`                                            | `feature_size`, `num_examples`           | `validate_int`, `validate_ge(..., 1)`; no `int()` conversion |
| `load_state_dict_to_module`, `register_tensor`            | `strict`, `learnable`                    | `validate_isinstance(..., bool)`                             |
| `Clamp`, `GeneralRobustRegressionLoss`, `BaseDynamicNorm` | `min_val`, `max_val`, `alpha_init_value` | stored as `float(...)` (`None` stays `None`)                 |

The hand-written table becomes a rule. A registry of callables with a valid base call
(`MODULE_CASES` and `FUNCTION_CASES` already hold them; a short `UTILITY_CASES` adds `utils`,
`testing` and the `create` class methods) feeds **one** test that walks the signature:

- an `int` parameter rejects `2.5`, `True` and `"a"` with a `TypeError` that names the parameter;
- a `float` parameter rejects `"a"` (`TypeError`) and `nan`, `inf` (`ValueError`), unless it is in
  `ACCEPTS_NON_FINITE` with a reason (`alpha` of the robust loss accepts `-inf`);
- a `bool` parameter rejects `1` with a `TypeError`;
- the other parameters (tensors, modules, strings, dtypes, callables) are exempt as a class; their
  checks are the shape, state and factory tests.

A scalar parameter that is not validated fails at once, so adding an option cannot escape P2. The
test covers the 118 scalar parameters of the six `__all__` [probe]. The range rows (`gamma >= 0`,
`q` in `[0, 1]`) stay in the table, which shrinks by the type and finiteness rows. A second test
asserts that every `float`/`int`/`bool` parameter of the six `__all__` is reached by a case
(completeness).

### 4.6 Gates that keep P12, P17 and P18 true (F6, F7, F8, F9, F12)

All in the existing files `tests/unit/test_docs.py` and `tests/unit/test_imports.py` (they mirror no
module, like the tests that are already there):

- **Markdown examples run**: `doctest.testfile` on `README.md` and on the guides, and `exec` of each
  `python` block of `docs/docs`. 10 lines, no new dependency.
- **Reference tables follow the docstrings**: every row `| [`Name`](#...) | description |` equals
  the first line of the docstring of `Name` (backticks normalised). The table stays (it is a useful
  index); it can no longer drift (D5).
- **Flat cross-package imports**: an `ast` walk of `src/sonnix`; a `from sonnix.<a>.<module> import`
  in package `<b>` with `<a> != <b>` fails, except `sonnix.validation.loss` in
  `functional/reduction.py` (named, with the reason). The 12 sites of F8 are changed to
  `from sonnix.functional import Reduction`, `from sonnix.utils import register_tensor`,
  `module_mode`.
- **No suppression**: `src` contains no `noqa`, `type: ignore`, `ty: ignore` or `pragma: no cover`.
  With D6 approved the 46 `# noqa: A002` go away through
  `lint.flake8-builtins.builtins-ignorelist = ["input"]`; without it the test allows `A002` and
  nothing else.
- **One source for the rules**: `conventions.md` holds the full text of the rules (it is published);
  this document keeps the table (its format asks for the properties first). A test compares the
  numbers and the bold titles of both tables. `CLAUDE.md` says "properties P1-P19".

### 4.7 Names (F10)

`DummyDataset(n_examples=...)` and `is_loss_decreasing*(n_iterations=...)`; the private
`create_spaced_values(n_values=...)`. The rule in `conventions.md`: arguments that count are
`n_<things>`; `num_parameters` (the `torch.nn.PReLU` argument) and the helpers `num_parameters`,
`num_learnable_parameters` stay (D3).

## 5. Public API after the change

```python
# sonnix.functional 25, modules 58, validation 6, utils 14, factory 3, testing 4: no name added or removed
# total 110 -> 110
```

### Migration table

| Before                                                                                               | After                                                                                 |
| ---------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------- |
| `general_robust_regression_loss(p, t)` / `GeneralRobustRegressionLoss()` (`alpha=2`): `(x/c)²`       | `0.5 * (x/c)²`, continuous with `alpha != 2`; `scale=s / sqrt(2)` gives the old value |
| `safe_exp(1e8, preserve_gradient=True)`: `8.9e6`; `safe_log(-1e30, ...)`: `-inf`; `inf` input: `nan` | the value of `safe_exp(x)` / `safe_log(x)`; gradient `1` (`0` at an infinite input)   |
| `MulticlassFlatten(c)(p[2, 6, 3], t[4, 3])` (silent); `(p[4, 3, 5], t[4, 2])` (`RuntimeError`)       | `ValueError` naming both shapes                                                       |
| `is_loss_decreasing*(num_iterations=0)`, `lr=nan`, `max_norm=-1`, `DummyDataset(feature_size=2.5)`   | `ValueError` / `TypeError` naming the argument                                        |
| `load_state_dict_to_module(..., strict="yes")`, `register_tensor(..., learnable="yes")`              | `TypeError`                                                                           |
| `Clamp(0, 1).min_val` (`int`)                                                                        | `0.0` (`float`)                                                                       |
| `DummyDataset(num_examples=8)`, `is_loss_decreasing*(num_iterations=2)`                              | `n_examples=8`, `n_iterations=2` (D3)                                                 |

## 6. Dependents

| Consumer                                                                                              | Effect                                                                                                                                   |
| ----------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------- |
| `tests/unit/functional/loss/test_general_robust.py`, `tests/unit/modules/loss/test_general_robust.py` | the `alpha=2` expected values halve; reference and continuity tests added                                                                |
| `tests/unit/functional/activation/test_math.py`                                                       | `preserve_gradient` value tests on large and infinite inputs                                                                             |
| `tests/registry.py`                                                                                   | `preserve_gradient=True` entries for `SafeExp`, `SafeLog`; `wrong_inputs` for `MulticlassFlatten`; `UTILITY_CASES`, `ACCEPTS_NON_FINITE` |
| `tests/integration/test_invalid_arguments.py`                                                         | type and finiteness rows removed; the signature-driven test added; range rows stay                                                       |
| `tests/unit/test_docs.py`, `tests/unit/test_imports.py`                                               | markdown examples, reference-table rows, flat imports, suppressions, rule-table parity                                                   |
| `src/sonnix/modules/**`, `src/sonnix/testing/loss.py`                                                 | the 12 imports of F8                                                                                                                     |
| `docs/docs/conventions.md`, `docs/docs/guide/numerical_stability.md`, `losses.md`                     | P18, P19, the rule on counts; the value of `preserve_gradient`; the robust loss at `alpha=2`                                             |
| `CLAUDE.md`                                                                                           | "properties P1-P19"                                                                                                                      |
| `CHANGELOG.md`                                                                                        | one "Breaking changes" entry per step, the migration rows above, with PR numbers                                                         |
| `pyproject.toml`                                                                                      | `builtins-ignorelist = ["input"]` only if D6 is approved                                                                                 |

## 7. Evidence

Probes and prototypes ran in the scratchpad; the commands are not in the repository. `torch 2.15.0`,
CPU, Python 3.14.

| Case                                                                                                  | `main`                            | Prototype / target          |
| ----------------------------------------------------------------------------------------------------- | --------------------------------- | --------------------------- |
| robust loss at `x=1`, `alpha` 1.99 / 2.0001 / 2                                                       | `0.4909` / `0.5002` / `1.0`       | `0.4909` / `0.5002` / `0.5` |
| `safe_exp([1e8, 1e30, inf], preserve_gradient=True)`                                                  | `[8.9e6, 1.0, nan]` (after `exp`) | `[4.85e8, 4.85e8, 4.85e8]`  |
| `safe_log([-1e30, -inf, 0], preserve_gradient=True)`                                                  | `[-inf, nan, -18.42]`             | `[-18.42, -18.42, -18.42]`  |
| gradient of the helper on `[-3, 0, 5, 25, 100]`                                                       | `[0.05, 1, 148, 4.85e8, 4.85e8]`  | identical                   |
| `MulticlassFlatten` `p[2, 6, 3]`, `t[4, 3]`                                                           | returns `torch.Size([12])`        | `ValueError`                |
| the 52 registered modules: empty batch, `deepcopy`, `torch.save`, `bfloat16` autocast, non-contiguous | no failure                        | same                        |

| Measure                                              | Today                                         | Target                                 |
| ---------------------------------------------------- | --------------------------------------------- | -------------------------------------- |
| Properties of section 1 that are true                | 12 of 19 (not P2, P4, P6, P11, P17, P18, P19) | 19 of 19                               |
| Public names                                         | 110                                           | 110                                    |
| Scalar parameters whose validation is tested by rule | 0 (hand-written rows)                         | 118 of 118                             |
| `# noqa` in `src`                                    | 46                                            | 0 with D6, else 46 (A002 only, tested) |
| Cross-package module-path imports                    | 12                                            | 0                                      |
| Documentation examples run by a test                 | docstrings only                               | docstrings, README, guides             |
| Hand-copied descriptions without a check             | 58                                            | 0                                      |
| `src` lines                                          | 7,438                                         | about 7,500 (estimate)                 |
| `tests` lines                                        | 15,124                                        | about 15,250 (estimate)                |
| Tests / doctests / lint / types on `main`            | 11,143 / 126 / clean / clean                  | same or more                           |

## 8. Changes to reach the target state

One branch and one pull request per step (`CLAUDE.md`), small and reviewable on its own. Every step
updates `CHANGELOG.md`, the docs it touches and the tests, and runs `ruff`,
`inv types.check --complete`, `pytest tests`, `xdoctest src/sonnix all` and, for a step that changes
`docs/`, `mkdocs build --strict`. Steps 1, 2, 3 and 4 are independent and can start from `main` in
parallel; 5 needs 3 (same files in `testing/`).

1. **Gates and one source (non-breaking).** F6-F9, F12: the five tests of 4.6 and the 12 imports
   they require; P18 and P19 in `conventions.md` and `CLAUDE.md`; `builtins-ignorelist` if D6 is
   approved. _Why first: the rules are written before the code changes._ P19 is written as "not
   true" until step 2.
2. **Two bugs (breaking numerics).** F1, F2: 4.2 and 4.3 with their regression tests (each fails on
   `main` with `git stash push src`), the migration rows, the guide.
3. **Validate the whole surface (breaking: invalid values raise).** F4, F5, F11: the table and the
   signature-driven test of 4.5; the range-independent rows of the hand-written table are deleted in
   the same PR.
4. **`MulticlassFlatten` (breaking: mismatched shapes raise).** F3: 4.4.
5. **Names (breaking, small).** F10: 4.7 (D3), `conventions.md`, migration rows.

Steps 2, 3 and 4 are the only ones that change the behaviour of calls that worked. Steps 1 and 5
change no value.

## 9. Tests and gates

The unit-test rules of `CLAUDE.md` apply to everything added (one behaviour per test, `match` on
every `pytest.raises`, seeded data, no local imports, literal expected values, a regression test for
each bug, 100 % coverage of the lines added or changed). The tests added by this revision are
**derived**: a new object or option is covered by one registry entry, and a missing entry fails.

- **P19**: reference formula in `float64` for the robust loss; continuity at `alpha = 2, 0, -inf`;
  `safe_exp`/`safe_log` with `preserve_gradient=True` equal the plain value on `1e8`, `1e30`, `inf`,
  `-inf`; the same cases added to `test_finite_wherever_float32_fits` (P6).
- **P2**: `test_invalid_scalar_argument(callable, parameter, value)` over the signatures;
  completeness test; literal messages for the new checks of 4.5; `float` attributes of `Clamp`,
  `GeneralRobustRegressionLoss`, `BaseDynamicNorm` (P11).
- **P4**: `wrong_inputs` for `MulticlassFlatten`; the two shapes of F3 are the regression tests.
- **P12, P17, P18**: the five gates of 4.6.
- **Docs**: `xdoctest src/sonnix all`, `mkdocs build --strict`, the notebook job.

## 10. Decisions for the maintainers

| #   | Decision                                                                                                | Recommendation                                                                                                                                                              |
| --- | ------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| D1  | `alpha=2` of the robust loss is `0.5 * (x/c)²` (4.2), halving the default loss?                         | **Yes.** It is Barron's definition and the limit of the general branch; the alternative leaves a factor-2 jump at the default. `scale / sqrt(2)` reproduces the old values. |
| D2  | `preserve_gradient`: fix (4.3) or remove the option?                                                    | **Fix.** It is documented and tested, the fix is one helper and the default path is untouched. Removing it breaks users of the straight-through clamp for no gain.          |
| D3  | Rename `num_examples` and `num_iterations` to `n_...` (4.7)?                                            | **Yes**, last (step 5); drop it if the churn is not wanted. `num_parameters` and its two helpers stay (the `torch.nn` name).                                                |
| D4  | `DummyDataset` requires `feature_size >= 1` and `num_examples >= 1`?                                    | **Yes.** An empty dataset is useless as a test fixture, and `feature_size=0` builds an empty target.                                                                        |
| D5  | Reference tables: test against the docstring (4.6), generate them, or delete them?                      | **Test.** It adds no dependency and keeps the index; generation needs a plugin (`mkdocs-macros`), deletion loses the overview.                                              |
| D6  | `lint.flake8-builtins.builtins-ignorelist = ["input"]` to remove the 46 `# noqa: A002` (F9)?            | **Yes**, once, in `pyproject.toml`: `input` is the `torch.nn` name and the only shadowed builtin. Asked for the third time; without it P18 (4) carries a named exception.   |
| D7  | One source for the rules: `conventions.md` has the text, this document the table, a test compares them? | **Yes.** The format needs the properties at the top of this document; the published page needs them too; the test removes the drift.                                        |
| D8  | Merge `functional/activation/math.py` and `functional/power.py` into one `safe` module?                 | **No.** The layout is not public, each file is one concept, and the move costs test renames and hides the diff of F2.                                                       |

Closed decisions (answered by the maintainers, not reopened): keep the named wrappers `ToFloat`,
`ToLong`, `ReLUn`, `Square`, the three errors and `rectifier_asinh_unit` (#451); keep the helpers of
`sonnix.utils`, `load_state_dict_to_module` and `DummyDataset` (#453); `validate_positive` in
`sonnix.validation`, `edges` and `n_bins` as segments, integer targets cast,
`Clamp(min_val, max_val)` positional, `sonnix.factory` kept, no folder move for the top-level
modules (second revision); `float16` is a supported dtype, `Reduction` is exported by `functional`,
`SafePow` exists, strict shapes in the fusions (third revision).

## 11. Not covered

- New losses and layers (`Tweedie`, `Dice`/`Tversky`, `SwiGLU`/`GeGLU`, a multiclass focal loss):
  the target gives the rules a new object follows (P1-P19, Appendix B), not which objects to add.
- Sample weights, masks and `ignore_index` in the losses: a feature with its own design (the shape
  of the weights against `reduction`).
- A parametrisation of a learnable `alpha` that crosses `0`: it needs an experiment, not a rule.
- **A stable release.** `coola>=2.0.0a5` is a pre-release; `sonnix` cannot leave `0.0.1a0` before
  `coola` 2.0 is stable, or before the dependency is relaxed.
- **The documentation toolchain.** Material for MkDocs reaches end of life on 2027-05-05 (the build
  prints it); its successor is a tooling migration, tracked outside this document.
- The size of some test files (`test_nlinear.py` has 1,657 tests, a five-axis cross-product, for a
  130-line module): the whole suite runs in 22 s, so no action; revisit if it passes a minute.
- Packaging and CI (the workflows, a GPU job): tracked in `.github/GUIDELINES.md`.

## Appendix A: size of the package

| Area                                | Lines today | Lines after | Change                                                         |
| ----------------------------------- | ----------- | ----------- | -------------------------------------------------------------- |
| `functional`                        | 1,603       | ~1,620      | `_clamp` helper (+14), `alpha=2` (0), docstrings               |
| `modules`                           | 4,464       | ~4,470      | `MulticlassFlatten` check (+6), `float(...)` (+4), imports (0) |
| `utils`                             | 709         | ~715        | `learnable` and `strict` checks                                |
| `testing`                           | 295         | ~325        | validation of 5 arguments, `n_...` names                       |
| `validation`, `factory`, `__init__` | 367         | 367         | none                                                           |
| **Total `src`**                     | **7,438**   | **~7,500**  | +~60 (estimate)                                                |
| `tests`                             | 15,124      | ~15,250     | +~250 gates and rules, −~120 hand-written rows (estimate)      |
| `review/`                           | 538         | ~330        | this file replaces the previous revision                       |

Lines of code without docstrings, comments and blank lines: about 1,700 in 68 files.

## Appendix B: checklist for a new public object

1. Is it a fixed-argument composition of existing public objects or of a `torch.nn` object? Then it
   is a recipe in the guide, unless users look for it by name (P16).
2. Function first if it is a loss or needs numerical care; the module calls it (P1).
3. Every scalar hyperparameter: type, finiteness, range, before it is stored, stored as its declared
   type; the module uses the same helper (P2, P11). The signature-driven test fails if one is
   missing.
4. Signature `(prediction, target, *, options, reduction)` or `(input, *, options)`; a module's
   positional arguments are listed in its registry entry (P3).
5. Two same-role tensors: `validate_same_shape`; a layer checks the dimension it consumes; a wrapper
   that reshapes checks the shapes; nothing broadcasts (P4). Loss: `reduce_loss` (P5).
6. A guard goes through `resolve_eps` / `resolve_max_exponent`; an unbounded square or exponential
   is computed in at least `float32`; every option passes the edge rule in `float16` (P6).
7. A formula with limit cases is checked against a reference and for continuity at the switch (P19).
8. A trained tensor that can be frozen: `register_tensor`; anything derived is non-persistent (P8).
   `device` and `dtype` if the module creates tensors (P9).
9. Public attributes named like the arguments, shown in `repr` (P11).
10. One registry entry (`positional`, `other`, `wrong_inputs`): export, compile, dtype, purity,
    parity, round-trip, shape, edge and invalid-argument tests come for free (P10, P18).
11. Export it from the flat `__init__.py` of **one** package, import it from the flat path
    elsewhere, document it under that package in `docs/docs/refs` with an `Args:` that matches the
    signature and an example that runs (P12, P17).

## Appendix C: history of the previous revisions

First revision (2026-11-06, twelve steps, all merged): #413 properties and layering, #414 validation
gaps and strict shapes, #415 keyword-only options, #417 composition losses deleted, #418 `input` for
every forward, #419 dynamic norms by composition, #420 encoder range logic, #422 packages
reorganized, #423 `_validate_init` hook, #424 migration notes, #439 `resolve_range`, #440 copied
`Args:` removed.

Second revision (2026-11-22, nine steps, all merged): #446 properties and docs, #447 validation of
every hyperparameter, #449 complete state (`register_tensor`, `edges`), #450 uniform signatures,
#451 surface area, #452 one public path, #453 helpers kept, #454 integer targets, #455 migration
table.

Third revision (2026-11-22, six steps, all merged): #459 honest documentation, #460 dtype-aware
guards and the `float16` failures, #461 strict shapes in the layers, fusions and residual block,
#462 `SafePow` and `Reduction` exported by `functional` only, #463 one core for the probability and
logits variants of the focal and Poly-1 losses (`einsum` skipped: slower than `bmm`), #464
`n_frequencies` and the migration table.

## Appendix D: status of this revision

| Step | Pull request | Notes   |
| ---- | ------------ | ------- |
| 1    |              | pending |
| 2    |              | pending |
| 3    |              | pending |
| 4    |              | pending |
| 5    |              | pending |
