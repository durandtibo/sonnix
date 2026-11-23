# sonnix

<p align="center">
    <a href="https://github.com/durandtibo/sonnix/actions/workflows/ci.yaml">
        <img alt="CI" src="https://img.shields.io/github/actions/workflow/status/durandtibo/sonnix/ci.yaml?branch=main&label=CI&logo=github"></a>
    <a href="https://github.com/durandtibo/sonnix/actions/workflows/nightly-package.yaml">
        <img alt="Nightly Package Tests" src="https://img.shields.io/github/actions/workflow/status/durandtibo/sonnix/nightly-package.yaml?label=nightly&logo=github"></a>
    <a href="https://codecov.io/gh/durandtibo/sonnix">
        <img alt="Codecov" src="https://img.shields.io/codecov/c/github/durandtibo/sonnix?logo=codecov"></a>
    <a href="https://durandtibo.github.io/sonnix/">
        <img alt="Documentation (stable)" src="https://img.shields.io/badge/docs-stable-blue?logo=readthedocs&logoColor=white"></a>
    <br/>
    <a href="https://pypi.org/project/sonnix/">
        <img alt="PyPI version" src="https://img.shields.io/pypi/v/sonnix?logo=pypi&logoColor=white"></a>
    <a href="https://pypi.org/project/sonnix/">
        <img alt="Python" src="https://img.shields.io/pypi/pyversions/sonnix?logo=python&logoColor=white"></a>
    <a href="https://opensource.org/licenses/BSD-3-Clause">
        <img alt="BSD-3-Clause" src="https://img.shields.io/pypi/l/sonnix"></a>
    <br/>
    <a href="https://github.com/astral-sh/ruff">
        <img alt="Ruff" src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json"></a>
    <a href="https://google.github.io/styleguide/pyguide.html#s3.8-comments-and-docstrings">
        <img alt="Doc style: google" src="https://img.shields.io/badge/docstyle-google-3666d6"></a>
    <a href="https://scorecard.dev/viewer/?uri=github.com/durandtibo/sonnix">
        <img alt="OpenSSF Scorecard" src="https://img.shields.io/ossf-scorecard/github.com/durandtibo/sonnix?label=openssf%20scorecard"></a>
    <br/>
    <a href="https://pepy.tech/project/sonnix">
        <img alt="Downloads" src="https://img.shields.io/pepy/dt/sonnix"></a>
    <a href="https://pepy.tech/project/sonnix">
        <img alt="Monthly downloads" src="https://img.shields.io/pepy/dm/sonnix"></a>
    <a href="https://github.com/durandtibo/sonnix/stargazers">
        <img alt="GitHub stars" src="https://img.shields.io/github/stars/durandtibo/sonnix?logo=github"></a>
</p>

A library of PyTorch modules, losses and functional operations.

## Overview

`sonnix` is a library of [PyTorch](https://pytorch.org/) building blocks that
are missing from `torch.nn`. It has three layers: `sonnix.functional` (stateless
operations), `sonnix.modules` (the `torch.nn.Module` wrappers) and
`sonnix.utils` (helpers to inspect and train modules). It also ships helpers for
training and testing, listed at the end. `sonnix` is a library that other
codebases depend on, so a public name is not removed because the library itself
does not use it. It contains:

- **Losses**: robust and asymmetric regression losses (general robust, quantile,
  Poisson, relative losses), binary focal and Poly-1 losses, and
  `TransformedLoss`. They share the same `reduction` options (`none`,
  `mean`, `sum`, `batchmean`).
- **Activations**: parametric activations (`Snake`, `Gaussian`, `Laplacian`,
  `Quadratic`, `MultiQuadratic`, `ExpSin`), `ReLUn`, `SquaredReLU`, `Asinh` and
  safe `exp`/`log`/`pow` modules.
- **Numerical encoders**: `CosSinNumericalEncoder`,
  `AsinhCosSinNumericalEncoder`, `AsinhNumericalEncoder` and
  `PiecewiseLinearNumericalEncoder` to embed numerical features.
- **Layers**: `NLinear`, `ExU`, `ResidualBlock`, `ScaleAndShift`, `DynamicTanh`,
  `DynamicAsinh`, the fusion modules and the shape and dtype helpers.
- **Helpers**: `sonnix.factory` builds modules from a configuration
  (`ResidualBlock` and `TransformedLoss` use its `resolve_module`),
  `sonnix.utils` inspects, freezes and loads modules (`num_parameters`,
  `freeze_module`, `load_state_dict_to_module`), and `sonnix.testing` provides
  `DummyDataset` and `is_loss_decreasing` to test your own modules.

## Installation

```shell
pip install sonnix
```

`sonnix` requires Python 3.11+ and PyTorch.

## Quick start

A loss:

```pycon
>>> import torch
>>> from sonnix.modules import QuantileRegressionLoss
>>> loss = QuantileRegressionLoss(q=0.9)
>>> loss
QuantileRegressionLoss(reduction=mean, q=0.9)
>>> loss(torch.tensor([[1.0], [2.0]]), torch.tensor([[2.0], [1.0]]))
tensor(0.5000)

```

A module:

```pycon
>>> import torch
>>> from sonnix.modules import Snake
>>> m = Snake()
>>> m(torch.zeros(2, 3)).shape
torch.Size([2, 3])

```

A numerical encoder, which maps each feature to a vector:

```pycon
>>> import torch
>>> from sonnix.modules import CosSinNumericalEncoder
>>> m = CosSinNumericalEncoder.create("linspace", n_frequencies=5, frequency_range=(0.1, 1.0))
>>> m(torch.randn(4, 1)).shape
torch.Size([4, 1, 10])

```

The notebook
[`notebooks/activations.ipynb`](https://github.com/durandtibo/sonnix/blob/main/notebooks/activations.ipynb)
plots the activations.

## Motivation

`torch.nn` covers the standard layers. Research code keeps re-implementing the
same extras (a robust loss, a periodic activation, a numerical embedding), often
with a different handling of the reduction, the numerical stability or the
device. `sonnix` provides them once, with a consistent API:

- the same `reduction` options for every loss;
- validated arguments and readable hyperparameters (`repr` shows them);
- numerically stable implementations, tested for overflow and finite gradients;
- `device` and `dtype` support, and frozen tensors stored as buffers so
  `module.to(...)` and `state_dict` work as expected.

## API stability

:warning: While `sonnix` is in development stage, no API is guaranteed to be
stable from one release to the next. In fact, it is very likely that the API
will change multiple times before a stable 1.0.0 release. In practice, this
means that upgrading `sonnix` to a new version will possibly break any code that
was using the old version of `sonnix`.

## License

`sonnix` is licensed under BSD 3-Clause "New" or "Revised" license available
in [LICENSE](https://github.com/durandtibo/sonnix/blob/main/LICENSE) file.
