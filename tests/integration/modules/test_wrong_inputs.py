from __future__ import annotations

import pytest

from tests.registry import MODULE_CASES, ModuleCase

# The modules without ``wrong_inputs``, with the reason (property P4).
NO_WRONG_INPUTS = {
    "Asinh": "elementwise: accepts any shape",
    "Clamp": "elementwise: accepts any shape",
    "Exp": "elementwise: accepts any shape",
    "Expm1": "elementwise: accepts any shape",
    "Log": "elementwise: accepts any shape",
    "Log1p": "elementwise: accepts any shape",
    "Pow": "elementwise: accepts any shape",
    "ReLUn": "elementwise: accepts any shape",
    "RectifierAsinhUnit": "elementwise: accepts any shape",
    "SafeExp": "elementwise: accepts any shape",
    "SafeLog": "elementwise: accepts any shape",
    "Sin": "elementwise: accepts any shape",
    "Sinh": "elementwise: accepts any shape",
    "Square": "elementwise: accepts any shape",
    "SquaredReLU": "elementwise: accepts any shape",
    "ExpSin": "num_parameters=1 accepts any shape; num_parameters > 1 has its own tests",
    "Gaussian": "num_parameters=1 accepts any shape; num_parameters > 1 has its own tests",
    "Laplacian": "num_parameters=1 accepts any shape; num_parameters > 1 has its own tests",
    "MultiQuadratic": "num_parameters=1 accepts any shape; num_parameters > 1 has its own tests",
    "Quadratic": "num_parameters=1 accepts any shape; num_parameters > 1 has its own tests",
    "Snake": "num_parameters=1 accepts any shape; num_parameters > 1 has its own tests",
    "ExU": "wraps torch.nn.functional.linear, which raises a RuntimeError like torch.nn.Linear",
    "ResidualBlock": "the shapes of the branches depend on the two modules, not on the input",
    "Squeeze": "accepts any shape (torch.squeeze ignores a dimension that is not 1)",
    "ToBinaryLabel": "elementwise: accepts any shape",
    "ToCategoricalLabel": "accepts any shape",
    "ToDtype": "elementwise: accepts any shape",
    "ToFloat": "elementwise: accepts any shape",
    "ToLong": "elementwise: accepts any shape",
    "ConcatFusion": "torch.cat checks its own dimensions (a RuntimeError)",
    "TransformedLoss": "the shape policy is the one of the wrapped criterion",
}
CASES_WITH_WRONG_INPUTS = [case for case in MODULE_CASES if case.wrong_inputs is not None]

##################################
#     Tests for the registry     #
##################################


def test_registry_lists_every_module_without_wrong_inputs() -> None:
    names = {case.name for case in MODULE_CASES if case.wrong_inputs is None}
    assert names == set(NO_WRONG_INPUTS)


def test_no_wrong_inputs_names_are_registered() -> None:
    assert set(NO_WRONG_INPUTS) <= {case.name for case in MODULE_CASES}


############################################
#     Tests for the wrong input shapes     #
############################################


@pytest.mark.parametrize("case", CASES_WITH_WRONG_INPUTS, ids=repr)
def test_wrong_inputs_raise_value_error(case: ModuleCase) -> None:
    module = case.build()
    with pytest.raises(ValueError, match=r"shape"):
        module(*case.wrong_inputs())
