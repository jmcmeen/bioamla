"""Tests for bioamla.common.config (the load_toml helper)."""

import pytest

from bioamla.common.config import load_toml
from bioamla.exceptions import ConfigError, NotFoundError


class TestLoadToml:
    def test_load_toml_reads_sections(self, tmp_path):
        p = tmp_path / "c.toml"
        p.write_text("[audio]\nsample_rate = 8000\nmono = false\n")
        data = load_toml(p)
        assert data["audio"]["sample_rate"] == 8000
        assert data["audio"]["mono"] is False

    def test_load_toml_missing(self, tmp_path):
        with pytest.raises(NotFoundError):
            load_toml(tmp_path / "nope.toml")

    def test_load_toml_malformed(self, tmp_path):
        p = tmp_path / "bad.toml"
        p.write_text("this is = = not valid toml [[[")
        with pytest.raises(ConfigError, match="Failed to parse"):
            load_toml(p)
