"""Tests for bioamla._extras: the optional-dependency groups and their errors."""

from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path

import pytest

from bioamla import _extras
from bioamla._extras import (
    EXTRAS,
    MODULE_EXTRA,
    extra_hint,
    import_extra,
    install_hint,
    require_extra,
)
from bioamla.exceptions import DependencyError

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"


def _hide(monkeypatch, *names: str) -> None:
    """Make ``names`` look uninstalled to find_spec and import_module."""
    real_find_spec = importlib.util.find_spec
    real_import = importlib.import_module

    def find_spec(name, *args, **kwargs):
        if name.split(".")[0] in names:
            return None
        return real_find_spec(name, *args, **kwargs)

    def import_module(name, *args, **kwargs):
        if name.split(".")[0] in names:
            raise ModuleNotFoundError(f"No module named {name!r}", name=name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(_extras.importlib.util, "find_spec", find_spec)
    monkeypatch.setattr(_extras.importlib, "import_module", import_module)


def test_extras_match_pyproject():
    optional = tomllib.loads(PYPROJECT.read_text())["project"]["optional-dependencies"]
    assert set(EXTRAS) <= set(optional)
    assert optional["all"] == [f"bioamla[{','.join(EXTRAS)}]"]


def test_module_extra_prefers_smallest_extra():
    assert MODULE_EXTRA["librosa"] == "audio"
    assert MODULE_EXTRA["matplotlib"] == "viz"
    assert MODULE_EXTRA["torch"] == "ml"
    assert MODULE_EXTRA["sklearn"] == "cluster"


def test_install_hint():
    assert install_hint("ml") == "pip install 'bioamla[ml]'"


def test_require_extra_passes_when_installed(monkeypatch):
    _hide(monkeypatch)
    require_extra("audio", "test")  # the dev environment has every extra


def test_require_extra_names_missing_modules(monkeypatch):
    _hide(monkeypatch, "matplotlib")
    with pytest.raises(DependencyError) as exc:
        require_extra("viz", "bioamla.viz")
    message = str(exc.value)
    assert "bioamla.viz needs the [viz] extra" in message
    assert "missing: matplotlib" in message
    assert "librosa" not in message
    assert "pip install 'bioamla[viz]'" in message


def test_import_extra_returns_module():
    assert import_extra("json") is importlib.import_module("json")


def test_import_extra_raises_dependency_error(monkeypatch):
    _hide(monkeypatch, "librosa")
    with pytest.raises(DependencyError, match=r"bioamla\[audio\]"):
        import_extra("librosa")


def test_import_extra_uses_given_extra(monkeypatch):
    _hide(monkeypatch, "torch")
    with pytest.raises(DependencyError, match=r"bioamla\[ml\]"):
        import_extra("torch.nn", "ml")


def test_import_extra_reraises_unknown_module():
    with pytest.raises(ModuleNotFoundError):
        import_extra("bioamla_no_such_module")


@pytest.mark.parametrize(
    ("name", "extra", "expected"),
    [
        ("librosa", None, "audio"),
        ("torch.nn", None, "ml"),
        ("librosa", "viz", "viz"),  # the caller's extra wins
        ("numba", "audio", "audio"),  # even for a transitive dependency
        ("hdbscan", None, "cluster"),
    ],
)
def test_extra_hint(name, extra, expected):
    hint = extra_hint(ModuleNotFoundError(name=name), extra)
    assert hint == f"It comes with the [{expected}] extra: pip install 'bioamla[{expected}]'"


@pytest.mark.parametrize("name", [None, "", "nope", "bioamla.nope"])
def test_extra_hint_unknown(name):  # and no extra given
    assert extra_hint(ModuleNotFoundError(name=name)) is None


@pytest.mark.parametrize("package", ["viz", "indices", "detect", "ml", "cluster"])
def test_domain_package_guard(monkeypatch, package):
    _hide(monkeypatch, *EXTRAS[package])
    monkeypatch.delitem(sys.modules, f"bioamla.{package}", raising=False)
    with pytest.raises(DependencyError, match=rf"bioamla\[{package}\]"):
        importlib.import_module(f"bioamla.{package}")
