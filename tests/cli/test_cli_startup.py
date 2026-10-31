"""The CLI must start without importing the domain packages or their heavy dependencies.

Command modules import domain code inside the command bodies. A top-level import
in any of them loads numpy/pandas (and more) on every ``bioamla`` invocation,
including ``--help``. These run in a subprocess so modules already imported by
other tests don't mask a regression.
"""

import subprocess
import sys

import pytest

# Modules that must not be loaded just by building the CLI.
HEAVY_MODULES = (
    "numpy",
    "pandas",
    "scipy",
    "soundfile",
    "pydub",
    "bioamla.audio",
    "bioamla.datasets",
    "bioamla.detect",
    "bioamla.indices",
    "bioamla.viz",
)


def _loaded_after(code: str) -> set[str]:
    probe = f"{code}\nimport sys\nprint('\\n'.join(sys.modules))"
    out = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, check=True
    ).stdout
    return set(out.split())


@pytest.mark.parametrize(
    "code",
    [
        "import bioamla.cli.cli",
        # Build the whole command tree, as `bioamla --help` does.
        "from click.testing import CliRunner\n"
        "from bioamla.cli.cli import cli\n"
        "assert CliRunner().invoke(cli, ['--help']).exit_code == 0",
    ],
    ids=["import", "help"],
)
def test_cli_startup_skips_heavy_imports(code: str) -> None:
    loaded = _loaded_after(code)
    assert sorted(m for m in HEAVY_MODULES if m in loaded) == []


def test_annotation_format_choices_match_datasets() -> None:
    from bioamla.common.constants import ANNOTATION_FORMATS
    from bioamla.datasets._io import ANNOTATION_FORMATS as datasets_formats

    assert datasets_formats is ANNOTATION_FORMATS
