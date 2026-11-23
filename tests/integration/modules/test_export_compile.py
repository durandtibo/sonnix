from __future__ import annotations

import inspect

import pytest
import torch
from coola.equality import objects_are_allclose
from torch.export import Dim

from sonnix import modules
from tests.registry import MODULE_CASES, ModuleCase

# The modules that are not instantiated by the registry.
NOT_REGISTERED = {
    "BaseAlphaActivation",
    "BaseDynamicNorm",
    "BaseFusion",
    "BaseLoss",
    "BaseNumericalEncoder",
    "MulticlassFlatten",
}

##################################
#     Tests for the registry     #
##################################


def test_registry_covers_every_module() -> None:
    registered = {case.name for case in MODULE_CASES}
    assert set(modules.__all__) - registered == NOT_REGISTERED


def test_registry_has_unique_names() -> None:
    names = [case.name for case in MODULE_CASES]
    assert len(names) == len(set(names))


##################################
#     Tests for torch.export     #
##################################


@pytest.mark.parametrize("case", MODULE_CASES, ids=repr)
def test_export_matches_eager(case: ModuleCase) -> None:
    module = case.build().eval()
    inputs = case.make_inputs()
    exported = torch.export.export(module, inputs)
    assert objects_are_allclose(exported.module()(*inputs), module(*inputs))


def _dynamic_batch_shapes(module: torch.nn.Module, inputs: tuple[torch.Tensor, ...]) -> tuple:
    # The first dimension of every input is dynamic. The inputs of a module with a variadic
    # ``forward`` (the fusions) are one tuple.
    batch = Dim("batch", min=2)
    shapes = tuple({0: batch} if x.ndim > 0 else None for x in inputs)
    parameters = inspect.signature(module.forward).parameters.values()
    if any(p.kind is inspect.Parameter.VAR_POSITIONAL for p in parameters):
        return (shapes,)
    return shapes


@pytest.mark.parametrize("case", MODULE_CASES, ids=repr)
def test_export_dynamic_batch_matches_eager_on_another_batch_size(case: ModuleCase) -> None:
    module = case.build().eval()
    inputs = case.make_inputs()
    exported = torch.export.export(
        module, inputs, dynamic_shapes=_dynamic_batch_shapes(module, inputs)
    )
    larger = tuple(torch.cat([x, x]) if x.ndim > 0 else x for x in inputs)
    assert objects_are_allclose(exported.module()(*larger), module(*larger))


###################################
#     Tests for torch.compile     #
###################################


@pytest.mark.parametrize("case", MODULE_CASES, ids=repr)
def test_compile_without_graph_break_matches_eager(case: ModuleCase) -> None:
    module = case.build().eval()
    inputs = case.make_inputs()
    compiled = torch.compile(module, backend="eager", fullgraph=True)
    assert objects_are_allclose(compiled(*inputs), module(*inputs))
