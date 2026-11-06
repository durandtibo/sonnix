# sonnix

<p align="center">
    <a href="https://github.com/durandtibo/sonnix/actions/workflows/ci.yaml">
        <img alt="CI" src="https://github.com/durandtibo/sonnix/actions/workflows/ci.yaml/badge.svg">
    </a>
    <a href="https://github.com/durandtibo/sonnix/actions/workflows/nightly-package.yaml">
        <img alt="Nightly Package Tests" src="https://github.com/durandtibo/sonnix/actions/workflows/nightly-package.yaml/badge.svg">
    </a>
    <a href="https://codecov.io/gh/durandtibo/sonnix">
        <img alt="Codecov" src="https://codecov.io/gh/durandtibo/sonnix/branch/main/graph/badge.svg">
    </a>
    <br/>
    <a href="https://google.github.io/styleguide/pyguide.html#s3.8-comments-and-docstrings">
        <img  alt="Doc style: google" src="https://img.shields.io/badge/%20style-google-3666d6.svg">
    </a>
    <a href="https://github.com/astral-sh/ruff">
        <img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json" alt="Ruff" style="max-width:100%;">
    </a>
    <a href="https://github.com/guilatrova/tryceratops">
        <img  alt="Doc style: google" src="https://img.shields.io/badge/try%2Fexcept%20style-tryceratops%20%F0%9F%A6%96%E2%9C%A8-black">
    </a>
    <br/>
    <a href="https://pypi.org/project/sonnix/">
        <img alt="PYPI version" src="https://img.shields.io/pypi/v/sonnix">
    </a>
    <a href="https://pypi.org/project/sonnix/">
        <img alt="Python" src="https://img.shields.io/pypi/pyversions/sonnix.svg">
    </a>
    <a href="https://opensource.org/licenses/BSD-3-Clause">
        <img alt="BSD-3-Clause" src="https://img.shields.io/pypi/l/sonnix">
    </a>
    <br/>
    <a href="https://pepy.tech/project/sonnix">
        <img  alt="Downloads" src="https://static.pepy.tech/badge/sonnix">
    </a>
    <a href="https://pepy.tech/project/sonnix">
        <img  alt="Monthly downloads" src="https://static.pepy.tech/badge/sonnix/month">
    </a>
    <br/>
</p>

A library of PyTorch modules, losses and functional operations.

## Overview

`sonnix` is a library of [PyTorch](https://pytorch.org/) building blocks that are missing from
`torch.nn`. It has three layers: `sonnix.functional` (stateless operations), `sonnix.modules` (the
`torch.nn.Module` wrappers) and `sonnix.utils` (helpers to inspect and train modules). It also ships
helpers for training and testing, listed at the end. It contains:

- **Losses**: robust and asymmetric regression losses (general robust, quantile, Poisson, relative
  and asinh losses), binary focal and Poly-1 losses, and `TransformedLoss`. They share the same
  `reduction` options (`none`, `mean`, `sum`, `batchmean`).
- **Activations**: parametric activations (`Snake`, `Gaussian`, `Laplacian`, `Quadratic`,
  `MultiQuadratic`, `ExpSin`), `ReLUn`, `SquaredReLU`, `Asinh` and safe `exp`/`log` modules.
- **Numerical encoders**: `CosSinNumericalEncoder`, `AsinhCosSinNumericalEncoder`,
  `AsinhNumericalEncoder` and `PiecewiseLinearNumericalEncoder` to embed numerical features.
- **Layers**: `NLinear`, `ExU`, `ResidualBlock`, `ScaleAndShift`, `DynamicTanh`, `DynamicAsinh`,
  the fusion modules and the shape and dtype helpers.
- **Training and testing helpers** (not used by the modules and the functions above): `sonnix.factory`
  builds modules, datasets and optimizers from a configuration (`ResidualBlock` and `TransformedLoss`
  use its `resolve_module`), `sonnix.utils` inspects, freezes and checks modules (`num_parameters`,
  `freeze_module`, `is_loss_decreasing`, `load_state_dict_to_module`), and `sonnix.testing` provides
  `DummyDataset`.

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
>>> m = CosSinNumericalEncoder.create("linspace", num_frequencies=5, frequency_range=(0.1, 1.0))
>>> m(torch.randn(4, 1)).shape
torch.Size([4, 1, 10])

```

The notebook
[`notebooks/activations.ipynb`](https://github.com/durandtibo/sonnix/blob/main/notebooks/activations.ipynb)
plots the activations.

## Motivation

`torch.nn` covers the standard layers. Research code keeps re-implementing the same extras (a
robust loss, a periodic activation, a numerical embedding), often with a different handling of the
reduction, the numerical stability or the device. `sonnix` provides them once, with a consistent
API:

- the same `reduction` options for every loss;
- validated arguments and readable hyperparameters (`repr` shows them);
- numerically stable implementations, tested for overflow and finite gradients;
- `device` and `dtype` support, and frozen tensors stored as buffers so `module.to(...)` and
  `state_dict` work as expected.
