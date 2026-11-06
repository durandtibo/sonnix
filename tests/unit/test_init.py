from __future__ import annotations

from importlib.metadata import version

import sonnix

#################################
#     Tests for __version__     #
#################################


def test_version_matches_installed_metadata() -> None:
    assert sonnix.__version__ == version("sonnix")
