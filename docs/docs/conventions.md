# Conventions

## Argument names

- The loss functions and the loss modules take `(prediction, target)`, while PyTorch uses
  `(input, target)`. This is a deliberate choice: the first argument of a loss is the output of a
  model. The activation functions and modules use `input`, like `torch.nn.functional`.
- Layers that map a feature dimension (`NLinear`, `ExU`) use `in_features` and `out_features`.
- The numerical encoders expose `n_features` (the size of the last input dimension) and
  `feature_size` (the size of the encoding of one feature), as shown in their representation.
- `normalized_shape` follows `torch.nn.LayerNorm`.
- The hyperparameters of a module are public attributes named like the constructor arguments.
- Options are keyword-only.
- `TransformedLoss` takes `prediction_transform` and `target_transform`: the modules applied to the
  prediction and to the target before the criterion.

## Design rules

Each rule holds for every public object. The target design is described in
`review/target_state.md`; a rule marked _planned_ is not yet true everywhere.

| #   | Rule                                                                                                                                                                                |
| --- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| P1  | **Function/module duality.** A loss or a parametrised op with numerical content is a function in `sonnix.functional` and a module in `sonnix.modules` that only calls the function. |
| P2  | **Validate once, fail early.** The function validates all its arguments; the module calls the same validator in `__init__`.                                                         |
| P3  | **Uniform call signature.** Losses: `(prediction, target, *, <options>, reduction)`. Single-tensor ops: `(input, *, <options>)`. Options are keyword-only (_planned_ for functions). |
| P4  | **Strict shapes.** Two tensors of the same role must have the same shape (`ValueError` naming both shapes). Nothing broadcasts silently (_planned_).                                |
| P5  | **One reduction contract.** Losses accept `none`/`mean`/`sum`/`batchmean`, validated by `reduce_loss`; the default is `mean`.                                                       |
| P6  | **Numerically safe.** Finite outputs and gradients for finite inputs in `float16`/`bfloat16`/`float32`; exceptions are documented in the docstring.                                 |
| P7  | **Pure.** A forward never mutates its inputs and has no hidden state.                                                                                                               |
| P8  | **Frozen = buffer.** Tensors that are not trained are buffers, never `Parameter(requires_grad=False)`.                                                                              |
| P9  | **Device and dtype.** A module that creates tensors takes `device` and `dtype` and keeps the input dtype.                                                                           |
| P10 | **Export and compile.** Every registered module passes `torch.export` and `torch.compile(fullgraph=True)`.                                                                          |
| P11 | **Introspectable.** Hyperparameters are public attributes named like the constructor arguments and are shown by `repr`.                                                             |
| P12 | **One public path.** The public API is what the `__init__.py` of `sonnix.{functional,modules,utils,validation,factory,testing}` exports.                                            |
| P13 | **Layered imports.** `testing → modules → functional → validation`; `utils`, `factory` and `validation` are independent of each other. Enforced by import-linter.                   |
| P14 | **Configuration is trusted input.** An argument typed `Module \| dict` calls its `_target_`; its docstring carries a `Warning`.                                                     |
| P15 | **Optional means lazy.** Only `torch` and `coola` are imported at module level.                                                                                                     |
