"""Tests for the bioamla console-script shim (bioamla._cli_entry)."""

import subprocess
import sys

import pytest

from bioamla import _cli_entry


def test_importing_the_shim_does_not_import_click():
    code = "import sys, bioamla._cli_entry; print('click' in sys.modules, 'rich' in sys.modules)"
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == "False False"


def test_runs_the_cli(monkeypatch):
    import importlib

    # bioamla.cli exports the Click group as ``cli``, which shadows the submodule.
    cli_module = importlib.import_module("bioamla.cli.cli")
    called = []
    monkeypatch.setattr(cli_module, "main", lambda: called.append(True))
    _cli_entry.main()
    assert called == [True]


@pytest.mark.parametrize("missing", ["click", "rich"])
def test_missing_cli_extra_prints_the_install_hint(monkeypatch, capsys, missing):
    def fail_import():
        raise ModuleNotFoundError(f"No module named '{missing}'", name=missing)

    _fake_cli_module(monkeypatch, fail_import)
    with pytest.raises(SystemExit) as excinfo:
        _cli_entry.main()
    assert excinfo.value.code == 1
    assert "pip install 'bioamla[cli]'" in capsys.readouterr().err


def test_other_missing_modules_are_not_masked(monkeypatch):
    def fail_import():
        raise ModuleNotFoundError("No module named 'torch'", name="torch")

    _fake_cli_module(monkeypatch, fail_import)
    with pytest.raises(ModuleNotFoundError, match="torch"):
        _cli_entry.main()


def test_console_script_points_at_the_shim():
    from importlib.metadata import entry_points

    (script,) = entry_points(group="console_scripts", name="bioamla")
    assert script.value == "bioamla._cli_entry:main"


def test_shim_runs_the_cli_end_to_end():
    result = subprocess.run(
        [sys.executable, "-m", "bioamla._cli_entry", "--help"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "Usage" in result.stdout


def _fake_cli_module(monkeypatch, on_import):
    """Make ``from bioamla.cli.cli import main`` run ``on_import`` (which raises)."""
    import importlib.abc

    class _Finder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname == "bioamla.cli.cli":
                on_import()
            return None

    monkeypatch.delitem(sys.modules, "bioamla.cli.cli", raising=False)
    monkeypatch.setattr(sys, "meta_path", [_Finder(), *sys.meta_path])
