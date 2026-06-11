"""Tests for bioamla.common.config."""

import pytest

from bioamla.common import config as cfg
from bioamla.common.config import (
    DEFAULT_CONFIG,
    Config,
    create_default_config_file,
    find_config_file,
    get_config,
    get_config_locations,
    get_default_config,
    load_config,
    load_config_cascade,
    load_toml,
    reset_config,
    save_toml,
    set_config,
)
from bioamla.exceptions import ConfigError, NotFoundError


@pytest.fixture(autouse=True)
def isolate_config_locations(tmp_path, monkeypatch):
    """Never read the real user/system config. Point all locations at empty tmp paths."""
    fake = [
        tmp_path / "none_bioamla.toml",
        tmp_path / "none_user.toml",
        tmp_path / "none_etc.toml",
    ]
    monkeypatch.setattr(cfg, "CONFIG_LOCATIONS", fake)
    reset_config()
    yield
    reset_config()


class TestConfigDataclass:
    def test_get_with_default(self):
        c = Config(audio={"sample_rate": 8000})
        assert c.get("audio", "sample_rate") == 8000
        assert c.get("audio", "missing", "fallback") == "fallback"

    def test_get_none_section(self):
        c = Config(audio=None)  # type: ignore[arg-type]
        assert c.get("audio", "x", "d") == "d"

    def test_set(self):
        c = Config(audio={})
        c.set("audio", "sample_rate", 22050)
        assert c.audio["sample_rate"] == 22050

    def test_set_none_section_noop(self):
        c = Config(audio=None)  # type: ignore[arg-type]
        c.set("audio", "k", 1)  # should not raise
        assert c.audio is None

    def test_to_dict_and_from_dict_round_trip(self):
        c = Config.from_dict({"audio": {"sample_rate": 1}}, source="x.toml")
        d = c.to_dict()
        assert d["audio"]["sample_rate"] == 1
        assert "project" in d
        assert c._source == "x.toml"


class TestLoadSaveToml:
    def test_load_toml_round_trip(self, tmp_path):
        p = tmp_path / "c.toml"
        save_toml({"audio": {"sample_rate": 8000, "mono": False}}, p)
        data = load_toml(p)
        assert data["audio"]["sample_rate"] == 8000
        assert data["audio"]["mono"] is False

    def test_save_toml_types(self, tmp_path):
        p = tmp_path / "c.toml"
        save_toml(
            {
                "section": {
                    "s": "str",
                    "b": True,
                    "i": 5,
                    "f": 1.5,
                    "lst": ["a", 2],
                },
                "empty": {},
                "notdict": "ignored",
            },
            p,
        )
        data = load_toml(p)
        assert data["section"]["s"] == "str"
        assert data["section"]["b"] is True
        assert data["section"]["i"] == 5
        assert data["section"]["lst"] == ["a", 2]
        assert "empty" not in data

    def test_load_toml_missing(self, tmp_path):
        with pytest.raises(NotFoundError):
            load_toml(tmp_path / "nope.toml")

    def test_load_toml_malformed(self, tmp_path):
        p = tmp_path / "bad.toml"
        p.write_text("this is = = not valid toml [[[")
        with pytest.raises(ConfigError, match="Failed to parse"):
            load_toml(p)


class TestFindConfigFile:
    def test_explicit_path_found(self, tmp_path):
        p = tmp_path / "my.toml"
        p.write_text("")
        assert find_config_file(str(p)) == p

    def test_explicit_path_missing_returns_none(self, tmp_path):
        assert find_config_file(str(tmp_path / "nope.toml")) is None

    def test_standard_location_found(self, tmp_path, monkeypatch):
        loc = tmp_path / "found.toml"
        loc.write_text("")
        monkeypatch.setattr(cfg, "CONFIG_LOCATIONS", [loc])
        assert find_config_file() == loc

    def test_none_found(self):
        assert find_config_file() is None


class TestLoadConfig:
    def test_defaults_when_no_file(self):
        c = load_config()
        assert c.audio["sample_rate"] == DEFAULT_CONFIG["audio"]["sample_rate"]

    def test_explicit_overrides_defaults(self, tmp_path):
        p = tmp_path / "c.toml"
        save_toml({"audio": {"sample_rate": 99}}, p)
        c = load_config(str(p))
        assert c.audio["sample_rate"] == 99
        # untouched defaults preserved
        assert c.audio["mono"] == DEFAULT_CONFIG["audio"]["mono"]
        assert c._source == str(p)

    def test_load_config_bad_file_falls_back(self, tmp_path, caplog):
        p = tmp_path / "bad.toml"
        p.write_text("[[[ broken")
        c = load_config(str(p))
        # falls back to defaults, no raise
        assert c.audio["sample_rate"] == DEFAULT_CONFIG["audio"]["sample_rate"]


class TestCascade:
    def test_cascade_merges_user_over_defaults(self, tmp_path, monkeypatch):
        user = tmp_path / "user.toml"
        save_toml({"audio": {"sample_rate": 12345}}, user)
        monkeypatch.setattr(cfg, "CONFIG_LOCATIONS", [user])
        c = load_config_cascade()
        assert c.audio["sample_rate"] == 12345
        assert c._source == str(user)

    def test_cascade_priority_order(self, tmp_path, monkeypatch):
        # highest priority first in list
        high = tmp_path / "high.toml"
        low = tmp_path / "low.toml"
        save_toml({"audio": {"sample_rate": 100}}, high)
        save_toml({"audio": {"sample_rate": 200}}, low)
        monkeypatch.setattr(cfg, "CONFIG_LOCATIONS", [high, low])
        c = load_config_cascade()
        # high overrides low
        assert c.audio["sample_rate"] == 100

    def test_cascade_explicit_highest(self, tmp_path, monkeypatch):
        user = tmp_path / "user.toml"
        save_toml({"audio": {"sample_rate": 100}}, user)
        monkeypatch.setattr(cfg, "CONFIG_LOCATIONS", [user])
        explicit = tmp_path / "explicit.toml"
        save_toml({"audio": {"sample_rate": 777}}, explicit)
        c = load_config_cascade(str(explicit))
        assert c.audio["sample_rate"] == 777
        assert c._source == str(explicit)

    def test_cascade_explicit_missing(self, tmp_path):
        c = load_config_cascade(str(tmp_path / "nope.toml"))
        assert c._source == "defaults"

    def test_cascade_bad_location_skipped(self, tmp_path, monkeypatch):
        bad = tmp_path / "bad.toml"
        bad.write_text("[[[ broken")
        monkeypatch.setattr(cfg, "CONFIG_LOCATIONS", [bad])
        c = load_config_cascade()
        assert c.audio["sample_rate"] == DEFAULT_CONFIG["audio"]["sample_rate"]

    def test_cascade_bad_explicit_skipped(self, tmp_path):
        bad = tmp_path / "bad.toml"
        bad.write_text("[[[ broken")
        c = load_config_cascade(str(bad))
        assert c.audio["sample_rate"] == DEFAULT_CONFIG["audio"]["sample_rate"]


class TestGlobalConfig:
    def test_get_set_reset(self):
        c = get_config()
        assert isinstance(c, Config)
        custom = Config(audio={"sample_rate": 1})
        set_config(custom)
        assert get_config() is custom
        reset_config()
        assert get_config() is not custom

    def test_get_default_config(self):
        c = get_default_config()
        assert c.audio["sample_rate"] == DEFAULT_CONFIG["audio"]["sample_rate"]

    def test_get_config_locations_is_copy(self):
        locs = get_config_locations()
        assert isinstance(locs, list)
        locs.append("mutate")
        assert "mutate" not in get_config_locations()


class TestCreateDefaultConfigFile:
    def test_create_default(self, tmp_path):
        p = tmp_path / "bioamla.toml"
        result = create_default_config_file(str(p))
        assert result == str(p)
        data = load_toml(p)
        assert "audio" in data

    def test_create_default_default_path(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        result = create_default_config_file()
        assert result == "bioamla.toml"
        assert (tmp_path / "bioamla.toml").exists()


class TestDeprecatedKeyMigration:
    def test_migrate_old_key(self, tmp_path, monkeypatch):
        p = tmp_path / "c.toml"
        save_toml({"models": {"default_model": "OldModel"}}, p)
        monkeypatch.setattr(cfg, "CONFIG_LOCATIONS", [p])
        with pytest.warns(DeprecationWarning):
            c = load_config_cascade()
        assert c.models["default_ast_model"] == "OldModel"
        assert "default_model" not in c.models

    def test_migrate_both_keys_prefers_new(self, tmp_path):
        data = {"models": {"default_model": "Old", "default_ast_model": "New"}}
        with pytest.warns(DeprecationWarning):
            migrated = cfg._migrate_deprecated_keys(data)
        assert migrated["models"]["default_ast_model"] == "New"
        assert "default_model" not in migrated["models"]
