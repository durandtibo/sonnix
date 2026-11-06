# Build a model from configurations

A configuration is a dictionary with a `_target_` key (the import path of a class or a function)
and the arguments of the target. `sonnix.factory` turns configurations into objects, and the
modules that wrap other modules (`ResidualBlock`, `TransformedLoss`, `RelativeLoss`,
`MulticlassFlatten`) accept either a module or its configuration:

```python
from sonnix.factory import create_sequential, resolve_module
from sonnix.modules import ResidualBlock

linear = resolve_module({"_target_": "torch.nn.Linear", "in_features": 4, "out_features": 4})

model = create_sequential(
    [
        {"_target_": "torch.nn.Linear", "in_features": 4, "out_features": 8},
        {"_target_": "torch.nn.ReLU"},
        {"_target_": "sonnix.modules.NLinear", "n": 3, "in_features": 8, "out_features": 2},
    ]
)

block = ResidualBlock(
    residual={"_target_": "torch.nn.Linear", "in_features": 4, "out_features": 4}
)
```

`is_module_config` checks that a dictionary describes a module without calling the target.

!!! warning "Only use trusted configurations"

    `resolve_*` imports the `_target_` and **calls it before** the type of the result is checked.
    A configuration can therefore run any importable callable, for example
    `{"_target_": "os.system", "command": "..."}`. This is the same trust model as Hydra: never
    load a configuration from an untrusted file or message. `is_*_config` imports the target but
    does not call it, so it can be used to check a configuration first (a callable without a
    return type hint is reported as invalid).

See the [reference](../refs/factory.md) for the list of functions.
