from __future__ import annotations

import importlib
import logging

import torch
from coola.equality import objects_are_equal

import sonnix
from sonnix.functional import absolute_error
from sonnix.modules import Exp
from sonnix.utils.params import has_parameters

logger: logging.Logger = logging.getLogger(__name__)


def check_version() -> None:
    logger.info("Checking __version__...")
    assert sonnix.__version__ != "0.0.0"


def check_packages() -> None:
    logger.info("Checking packages...")
    packages_to_import = [
        "sonnix.functional",
        "sonnix.modules",
        "sonnix.utils",
    ]
    for package in packages_to_import:
        module = importlib.import_module(package)
        assert module is not None


def check_functional() -> None:
    logger.info("Checking 'functional' package...")
    assert objects_are_equal(
        absolute_error(torch.ones(2, 3), torch.ones(2, 3)),
        torch.zeros(2, 3),
    )


def check_modules() -> None:
    logger.info("Checking 'modules' package...")
    module = Exp()
    assert objects_are_equal(module(torch.zeros(2, 3)), torch.ones(2, 3))


def check_utils() -> None:
    logger.info("Checking 'utils' package...")
    assert has_parameters(torch.nn.Linear(4, 6))


def main() -> None:
    check_version()
    check_packages()

    check_functional()
    check_modules()
    check_utils()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
