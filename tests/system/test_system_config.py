"""Tests for bioamla.system.config (error-wrapping thin layer)."""

from __future__ import annotations

import pytest

from bioamla.exceptions import ConfigError, InvalidInputError
from bioamla.system import config as syscfg
from bioamla.system.config import Config


def test_get_config_returns_config():
    cfg = syscfg.get_config()
    assert isinstance(cfg, Config)


def test_get_config_wraps_errors(monkeypatch):
    monkeypatch.setattr(syscfg, "_get_config", lambda: (_ for _ in ()).throw(RuntimeError("boom")))
    with pytest.raises(ConfigError, match="Failed to get config"):
        syscfg.get_config()


def test_find_config_file_none_when_missing(monkeypatch):
    monkeypatch.setattr(syscfg, "_find_config_file", lambda p: None)
    assert syscfg.find_config_file() is None


def test_find_config_file_returns_str(monkeypatch, tmp_path):
    f = tmp_path / "bioamla.toml"
    f.write_text("")
    monkeypatch.setattr(syscfg, "_find_config_file", lambda p: f)
    result = syscfg.find_config_file(str(f))
    assert result == str(f)
    assert isinstance(result, str)


def test_find_config_file_wraps_errors(monkeypatch):
    monkeypatch.setattr(
        syscfg, "_find_config_file", lambda p: (_ for _ in ()).throw(RuntimeError("x"))
    )
    with pytest.raises(ConfigError, match="Failed to find config file"):
        syscfg.find_config_file()


def test_get_config_locations_returns_strs():
    locs = syscfg.get_config_locations()
    assert isinstance(locs, list)
    assert all(isinstance(loc, str) for loc in locs)
    assert len(locs) > 0


def test_get_config_locations_wraps_errors(monkeypatch):
    monkeypatch.setattr(
        syscfg, "_get_config_locations", lambda: (_ for _ in ()).throw(RuntimeError("x"))
    )
    with pytest.raises(ConfigError, match="Failed to get config locations"):
        syscfg.get_config_locations()


def test_load_config_returns_config():
    cfg = syscfg.load_config()
    assert isinstance(cfg, Config)


def test_load_config_wraps_errors(monkeypatch):
    monkeypatch.setattr(
        syscfg, "_load_config", lambda p: (_ for _ in ()).throw(RuntimeError("x"))
    )
    with pytest.raises(ConfigError, match="Failed to load config"):
        syscfg.load_config()


def test_get_default_config_returns_config():
    cfg = syscfg.get_default_config()
    assert isinstance(cfg, Config)
    assert cfg.audio["sample_rate"] == 16000


def test_get_default_config_wraps_errors(monkeypatch):
    monkeypatch.setattr(
        syscfg, "_get_default_config", lambda: (_ for _ in ()).throw(RuntimeError("x"))
    )
    with pytest.raises(ConfigError, match="Failed to get default config"):
        syscfg.get_default_config()


def test_create_default_config_writes_file(tmp_path):
    dest = tmp_path / "bioamla.toml"
    result = syscfg.create_default_config(str(dest))
    assert result == str(dest)
    assert dest.exists()
    assert dest.read_text()


def test_create_default_config_exists_without_force(tmp_path):
    dest = tmp_path / "bioamla.toml"
    dest.write_text("existing")
    with pytest.raises(InvalidInputError, match="already exists"):
        syscfg.create_default_config(str(dest))


def test_create_default_config_force_overwrites(tmp_path):
    dest = tmp_path / "bioamla.toml"
    dest.write_text("existing")
    result = syscfg.create_default_config(str(dest), force=True)
    assert result == str(dest)
    # File was overwritten with default config content
    assert "existing" != dest.read_text()


def test_create_default_config_wraps_write_errors(monkeypatch, tmp_path):
    dest = tmp_path / "bioamla.toml"
    monkeypatch.setattr(
        syscfg,
        "_create_default_config_file",
        lambda p: (_ for _ in ()).throw(RuntimeError("disk full")),
    )
    with pytest.raises(ConfigError, match="Failed to create config file"):
        syscfg.create_default_config(str(dest))
