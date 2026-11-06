# Changelog

All notable changes to this project are documented in this file. The package is still in alpha
(`0.0.1a0`), so breaking changes can happen in any release.

## Unreleased

### Breaking changes

- The hyperparameters of the modules are public attributes named like the constructor arguments
  (`ReLUn.max_val`, `Clamp.min_val`/`max_val`, `Snake.frequency`, `Pow.exponent`,
  `SafeExp.max_val`/`preserve_gradient`, `SafeLog.min_val`/`preserve_gradient`, `Squeeze.dim`,
  `View.shape`, `ConcatFusion.dim`, `SumFusion.normalized`, `ScaleAndShift.normalized_shape`,
  `DynamicTanh.alpha_init_value`, `MomentMeanIndicator.k`, and the `alpha`, `gamma`, `beta`, `q`,
  `eps`, `scale`, `max_val`, `detach_indicator` attributes of the losses). The previous private
  attributes (`_max`, `_alpha`, ...) are removed.

### Fixed

- `RelativeSmoothL1Loss` stores `beta` and shows it in its representation.

### Changed

- `BaseLoss.extra_repr` returns `reduction=...` and the losses extend it, so the representation of
  every loss starts with `reduction=...` followed by its hyperparameters.

### Renamed

- The `input_size` and `output_size` properties of the numerical encoders are now `n_features` and
  `feature_size`, like their representation.
- The `prediction` and `target` arguments (and attributes) of `TransformedLoss` are now
  `prediction_transform` and `target_transform`.
