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
