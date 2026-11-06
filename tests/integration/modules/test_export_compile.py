from __future__ import annotations

import pytest
import torch
from coola.equality import objects_are_allclose

from sonnix import modules
from tests.registry import MODULE_CASES, ModuleCase

# The modules that are not instantiated by the registry.
NOT_REGISTERED = {
    "BaseAlphaActivation",
    "BaseBinaryFocalLoss",
    "BaseBinaryPoly1Loss",
    "BaseDynamicNorm",
    "BaseFusion",
    "BaseLoss",
    "BaseRelativeIndicator",
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


###################################
#     Tests for torch.compile     #
###################################


@pytest.mark.parametrize("case", MODULE_CASES, ids=repr)
def test_compile_without_graph_break_matches_eager(case: ModuleCase) -> None:
    module = case.build().eval()
    inputs = case.make_inputs()
    compiled = torch.compile(module, backend="eager", fullgraph=True)
    assert objects_are_allclose(compiled(*inputs), module(*inputs))
