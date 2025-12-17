"""
Unit tests for bioamla.config module.
"""

from pathlib import Path

import pytest

from bioamla.config import (
    Config,
    DEFAULT_CONFIG,
    create_default_config_file,
    find_config_file,
    get_default_config,
    load_config,
    load_toml,
    save_toml,
)


class TestConfig:
    """Tests for Config class."""

    def test_default_config(self):
        """Test getting default configuration."""
        config = get_default_config()

        assert isinstance(config, Config)
        assert config.audio.get("sample_rate") == 16000
        assert config.visualize.get("type") == "mel"

    def test_config_get(self):
        """Test Config.get method."""
        config = get_default_config()

        assert config.get("audio", "sample_rate") == 16000
        assert config.get("audio", "nonexistent", "default") == "default"

    def test_config_set(self):
        """Test Config.set method."""
        config = get_default_config()

        config.set("audio", "sample_rate", 44100)
        assert config.get("audio", "sample_rate") == 44100

    def test_config_to_dict(self):
        """Test Config.to_dict method."""
        config = get_default_config()
        d = config.to_dict()

        assert "audio" in d
        assert "visualize" in d
        assert "analysis" in d
        assert d["audio"]["sample_rate"] == 16000

    def test_config_from_dict(self):
        """Test Config.from_dict method."""
        data = {
            "audio": {"sample_rate": 22050},
            "visualize": {"type": "stft"},
        }
        config = Config.from_dict(data)

        assert config.audio["sample_rate"] == 22050
        assert config.visualize["type"] == "stft"


class TestTOMLOperations:
    """Tests for TOML load/save operations."""

    def test_save_toml(self, temp_dir):
        """Test saving TOML file."""
        config = {
            "audio": {"sample_rate": 16000, "mono": True},
            "visualize": {"type": "mel"},
        }

        filepath = temp_dir / "test.toml"
        result = save_toml(config, str(filepath))

        assert Path(result).exists()

    def test_load_toml(self, temp_dir):
        """Test loading TOML file."""
        # Create a simple TOML file
        filepath = temp_dir / "test.toml"
        content = """
[audio]
sample_rate = 22050
mono = true

[visualize]
type = "stft"
n_fft = 4096
"""
        filepath.write_text(content)

        config = load_toml(str(filepath))

        assert config["audio"]["sample_rate"] == 22050
        assert config["audio"]["mono"] is True
        assert config["visualize"]["type"] == "stft"
        assert config["visualize"]["n_fft"] == 4096

    def test_load_toml_missing_file(self):
        """Test loading non-existent TOML file."""
        with pytest.raises(FileNotFoundError):
            load_toml("/nonexistent/file.toml")

    def test_save_load_roundtrip(self, temp_dir):
        """Test that save then load preserves data."""
        original = {
            "audio": {"sample_rate": 16000, "mono": True},
            "visualize": {"type": "mel", "n_fft": 2048},
        }

        filepath = temp_dir / "roundtrip.toml"
        save_toml(original, str(filepath))
        loaded = load_toml(str(filepath))

        assert loaded["audio"]["sample_rate"] == original["audio"]["sample_rate"]
        assert loaded["audio"]["mono"] == original["audio"]["mono"]
        assert loaded["visualize"]["type"] == original["visualize"]["type"]


class TestConfigLoading:
    """Tests for config loading functions."""

    def test_load_config_defaults(self):
        """Test loading config returns defaults when no file exists."""
        config = load_config()

        assert config.audio.get("sample_rate") == DEFAULT_CONFIG["audio"]["sample_rate"]

    def test_load_config_from_file(self, temp_dir):
        """Test loading config from a specific file."""
        # Create config file
        filepath = temp_dir / "custom.toml"
        content = """
[audio]
sample_rate = 44100

[visualize]
type = "stft"
"""
        filepath.write_text(content)

        config = load_config(str(filepath))

        assert config.audio.get("sample_rate") == 44100
        assert config.visualize.get("type") == "stft"
        # Defaults should still be present
        assert config.visualize.get("n_fft") == DEFAULT_CONFIG["visualize"]["n_fft"]

    def test_find_config_file_explicit(self, temp_dir):
        """Test finding config file with explicit path."""
        filepath = temp_dir / "explicit.toml"
        filepath.write_text("[audio]\nsample_rate = 16000\n")

        result = find_config_file(str(filepath))
        assert result == filepath

    def test_find_config_file_not_found(self):
        """Test find_config_file returns None when not found."""
        result = find_config_file("/nonexistent/path.toml")
        assert result is None


class TestCreateDefaultConfig:
    """Tests for creating default config file."""

    def test_create_default_config_file(self, temp_dir):
        """Test creating default config file."""
        filepath = temp_dir / "bioamla.toml"
        result = create_default_config_file(str(filepath))

        assert Path(result).exists()

        # Verify it can be loaded
        config = load_toml(str(filepath))
        assert "audio" in config
        assert "visualize" in config


@pytest.fixture
def temp_dir(tmp_path):
    """Provide a temporary directory."""
    return tmp_path
