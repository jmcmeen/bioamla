"""Tests for the AST training-config generator."""

import pytest

from bioamla.exceptions import InvalidInputError
from bioamla.ml.train_config import write_train_config


def test_write_creates_file_with_three_sections(tmp_path):
    out = write_train_config(tmp_path / "ast_training.toml")
    assert out.exists()

    # The generated file must round-trip and expose exactly the sections that
    # `ast train --config` consumes.
    from bioamla.common.config import load_toml

    cfg = load_toml(out)
    assert set(cfg.keys()) == {"models", "training", "augmentation"}
    assert cfg["models"]["default_ast_model"]
    assert cfg["training"]["epochs"] == 1
    # Augmentation ships off-by-default.
    assert cfg["augmentation"]["add_noise"] is False
    assert cfg["augmentation"]["noise_probability"] == 0.5


def test_write_creates_parent_dirs(tmp_path):
    out = write_train_config(tmp_path / "nested" / "dir" / "train.toml")
    assert out.exists()


def test_existing_file_without_force_raises(tmp_path):
    target = tmp_path / "train.toml"
    target.write_text("old")
    with pytest.raises(InvalidInputError, match="already exists"):
        write_train_config(target)
    assert target.read_text() == "old"  # not overwritten


def test_force_overwrites(tmp_path):
    target = tmp_path / "train.toml"
    target.write_text("old")
    write_train_config(target, force=True)
    assert "[training]" in target.read_text()


def test_generated_keys_match_train_config_map(tmp_path):
    """Every [training]/[models]/[augmentation] key the CLI reads is documented."""
    from bioamla.cli.commands.models import _TRAIN_CONFIG_MAP
    from bioamla.common.config import load_toml

    cfg = load_toml(write_train_config(tmp_path / "t.toml"))
    for _flag, (section, key) in _TRAIN_CONFIG_MAP.items():
        assert key in cfg.get(section, {}), f"template missing [{section}].{key}"
