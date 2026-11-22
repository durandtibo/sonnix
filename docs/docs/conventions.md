# Conventions

## Argument names

- The loss functions and the loss modules take `(prediction, target)`, while
  PyTorch uses `(input, target)`. This is a deliberate choice: the first
  argument of a loss is the output of a model. Every module with a single tensor
  input names it `input` in `forward`, like `torch.nn`.
- Layers that map a feature dimension (`NLinear`, `ExU`) use
  `in_features` and `out_features`.
- The numerical encoders expose `n_features` (the size of the last input
  dimension) and `feature_size` (the size of the encoding of one feature), as
  shown in their representation.
- `normalized_shape` follows `torch.nn.LayerNorm`.
- The hyperparameters of a module are public attributes named like
  the constructor arguments.
- Options of the functions are keyword-only. A module takes positionally its
  structural arguments (sizes, wrapped modules, tensors) and at most one
  hyperparameter, the defining one (`LogCoshLoss(2.0)`, like
  `nn.LeakyReLU(0.1)`); the other options are keyword-only.
- `TransformedLoss` takes `prediction_transform` and `target_transform`: the
  modules applied to the prediction and to the target before the criterion.

## Scope of the public API

`sonnix` is a library that is used by other codebases. A public name is useful
if a user needs it, not only if `sonnix` itself calls it: a helper that nothing
in the library uses (`sonnix.utils`, `sonnix.testing`) stays public. A name is
removed only when it is wrong or when it duplicates another name without helping
the user.

## Design rules

Each rule holds for every public object. The target design is described in
`review/target_state.md`. The tests listed in that document enforce them.

| #   | Rule                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| --- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| P1  | **Function/module duality.** Every loss, and every op that needs numerical care (`safe_exp`, `safe_log`, `safe_pow`), is a stateless function in `sonnix.functional` **and** a module in `sonnix.modules` that stores the hyperparameters and calls the function: the losses, `safe_exp`/`SafeExp`, `safe_log`/`SafeLog`, `safe_pow`/`SafePow`. A module with learnable state or without numerical care (an activation, a layer) has no function. A function that users look for by name (`rectifier_asinh_unit`, the errors, the indicators) may have no module (P16). A `forward` is one call. |
| P2  | **Validate early and completely.** Every hyperparameter is checked for type (a `bool` is not an `int`), finiteness and range, before it is stored or converted, with `coola.validation` or a `sonnix.validation` helper. A float is finite unless its docstring says otherwise. The function checks at call time; the module checks in `__init__` with the same helper and message.                                                                                                                        |
| P3  | **Uniform call signature.** Losses: `(prediction, target, *, <options>, reduction)`. Single-tensor ops: `(input, *, <options>)`. Options of a function are keyword-only. A module takes positionally its structural arguments and at most one hyperparameter (the defining one); the rest is keyword-only. The tensor of a `forward` is `input`.                                                                                                                                                           |
| P4  | **Strict shapes.** Two tensors of the same role must have the same shape (`ValueError` naming both): the prediction and the target, the inputs of a sum, average or product fusion, the two branches of a residual block. A layer or an encoder checks the dimension it consumes and names the argument (`n`, `n_features`, `normalized_shape`, `channel_dim`). Nothing broadcasts silently: a size-1 dimension is not expanded. A wrapper of an arbitrary criterion (`TransformedLoss`, `MulticlassFlatten`) leaves the policy to the wrapped module. |
| P5  | **One reduction contract.** Losses accept `none`/`mean`/`sum`/`batchmean`, validated by `reduce_loss`; the default is `mean`; `none` keeps the shape of the elementwise loss; `reduction` is keyword-only.                                                                                                                                                                                                                                                                                                 |
| P6  | **Numerically safe.** For `float16`, `bfloat16`, `float32` and `float64`: wherever the `float32` result and its gradient are representable in the dtype, the output and the gradient are finite. A default hyperparameter is valid in every dtype (a guard `eps` or `min_val` is raised to the smallest normal number of the dtype; the `max_val` of `safe_exp` is lowered so that `exp` is finite). An unbounded input is squared or exponentiated in at least `float32`. A hyperparameter that can still overflow (an exponent) is documented. |
| P7  | **Pure.** A function or a forward never mutates its inputs and has no hidden state.                                                                                                                                                                                                                                                                                                                                                                                                                        |
| P8  | **Complete state, by construction.** The `state_dict` holds the parameters and buffers that define the behaviour, and nothing derived. A tensor derived from persistent state is non-persistent and recomputed after `load_state_dict`. A frozen tensor is a buffer with the same name as the learnable one, never `Parameter(requires_grad=False)`. Round trip: `m2.load_state_dict(m1.state_dict())` makes `m2` behave as `m1`.                                                                          |
| P9  | **Device and dtype.** A module that creates tensors takes `device` and `dtype`. After `module.to(dtype)` every floating tensor of the module has `dtype` and the output has the dtype of the input. A module whose dtype differs from the dtype of the input follows the promotion rules of `torch` (`NLinear` raises, like `torch.nn.Linear`): that case is not guaranteed. |
| P10 | **Export and compile.** Every registered module passes `torch.export` (also with a dynamic batch dimension) and `torch.compile(fullgraph=True)` on CPU: no `.item()`, no data-dependent Python branch in a `forward`. A shape check on static dimensions is allowed. |
| P11 | **Introspectable.** Every hyperparameter, including a callable or a submodule, is a public attribute named like the constructor argument and is shown by `repr`.                                                                                                                                                                                                                                                                                                                                           |
| P12 | **One public path.** Every public object has one public path. `sonnix.{functional,modules,validation,factory,utils,testing}` each export their API from `__init__.py` (`__all__`); a name is exported by one package; the sub-folders have no `__all__`. Docs, docstrings and examples import from the flat path. Every exported name is listed under its own package in `docs/docs/refs` and has an `Example` that runs. |
| P13 | **Layered imports.** `testing → modules → functional → validation`; `utils`, `factory` and `validation` import no other sonnix package; no cycle. Enforced by import-linter.                                                                                                                                                                                                                                                                                                                               |
| P14 | **Configuration is trusted input.** `sonnix.factory` imports and calls a `_target_` before it checks the type of the result. An argument typed `Module \| dict` says so in a `Warning`.                                                                                                                                                                                                                                                                                                                    |
| P15 | **Optional means lazy.** Only `torch` and `coola` are imported at module level.                                                                                                                                                                                                                                                                                                                                                                                                                            |
| P16 | **Earned names.** A public name is a loss, a layer, an activation, an encoder, an op with numerical care, or a helper that these use, that tests them, or that users of the library need. A fixed-argument composition of other public objects or of a `torch.nn` object is a recipe in the guide, unless users look for it by name (`ReLUn`, `Square`, `ToFloat`): then a thin wrapper is allowed and its docstring names the composition. A base class is public only if users are meant to subclass it. |
| P17 | **Honest documentation.** A docstring says what the code does: the `Args:` block lists exactly the arguments of the signature, a `Raises:` block names what the code raises, and a guarantee stated in a docstring or in the guide ("no `inf`", "raises `ValueError`") has a test. `CLAUDE.md` points only to documents that are true; an obsolete review file is deleted. |
