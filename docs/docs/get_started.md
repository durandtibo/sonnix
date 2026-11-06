# Get Started

It is highly recommended to install in
a [virtual environment](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/)
to keep your system in order.

## Installing with `pip` (recommended)

The following command installs the latest version of the library:

```shell
pip install sonnix
```

`sonnix` has no optional dependency: `pip install sonnix` installs `coola` and PyTorch.

## Installing from source

`sonnix` requires Python 3.11 or later. [`uv`](https://docs.astral.sh/uv/) is used to manage the
virtual environment and the dependencies. If `uv` is not installed on your machine, follow
the [installation guide](https://docs.astral.sh/uv/getting-started/installation/).

First, clone the git repository:

```shell
git clone git@github.com:durandtibo/sonnix.git
cd sonnix
```

Then, create a virtual environment in `.venv` and install the package (editable) with the
development dependencies:

```shell
uv sync --group dev --group docs
```

Activate the environment, or prefix the commands with `uv run`:

```shell
source .venv/bin/activate
```

The tasks of the project are run with [`invoke`](https://www.pyinvoke.org/). `inv --list` lists
them. You can test the installation with the following commands:

```shell
inv test.unit
inv test.integration
inv test.doctest
```
