from __future__ import annotations

import subprocess
import sys
import textwrap

#################################
#     Tests for __version__     #
#################################

SCRIPT = textwrap.dedent(
    """
    import importlib.metadata as metadata

    def missing(name):
        raise metadata.PackageNotFoundError(name)

    metadata.version = missing
    import sonnix

    print(sonnix.__version__)
    """
)


def test_version_without_installed_metadata() -> None:
    result = subprocess.run(  # noqa: S603
        [sys.executable, "-c", SCRIPT], capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == "0.0.0+unknown"
