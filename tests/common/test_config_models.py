"""Validated config objects: construction, validate_config, JSON Schema export, aliases."""

import json
from dataclasses import asdict, field, replace
from pathlib import Path

import pytest

from bioamla.batch import BatchConfig
from bioamla.common.config import (
    CONFIG_CLASSES,
    config_dataclass,
    config_schema,
    resolve_config_class,
    validate_config,
)
from bioamla.exceptions import (
    ConfigError,
    ConfigValidationError,
    DependencyError,
    InvalidInputError,
)

SCHEMA_DIR = Path(__file__).resolve().parents[2] / "docs" / "schemas"


@config_dataclass
class _Inner:
    rate: float = 1.0


@config_dataclass
class _Outer:
    count: int = 1
    name: str | None = None
    inner: _Inner = field(default_factory=_Inner)


class TestConfigDataclass:
    def test_coerces_and_keeps_dataclass_behaviour(self) -> None:
        cfg = _Outer("3", "x")  # positional, and "3" coerces to 3
        assert cfg.count == 3
        assert asdict(cfg) == {"count": 3, "name": "x", "inner": {"rate": 1.0}}
        assert replace(cfg, count=5).count == 5

    def test_wrong_type_names_the_field(self) -> None:
        with pytest.raises(ConfigValidationError, match=r"count: .*integer.*\(got 'many'\)"):
            _Outer(count="many")

    def test_unknown_field_rejected(self) -> None:
        with pytest.raises(ConfigValidationError, match="bogus: unknown field"):
            _Outer(bogus=1)

    def test_validation_error_is_config_and_input_error(self) -> None:
        with pytest.raises(ConfigValidationError) as info:
            _Outer(count="many")
        assert isinstance(info.value, ConfigError)
        assert isinstance(info.value, InvalidInputError)

    def test_post_init_errors_propagate_unchanged(self) -> None:
        # BatchConfig's own check (neither input given) keeps raising InvalidInputError.
        with pytest.raises(InvalidInputError) as info:
            BatchConfig()
        assert not isinstance(info.value, ConfigValidationError)

    def test_batch_config_accepts_paths(self, tmp_path: Path) -> None:
        cfg = BatchConfig(input_dir=tmp_path, output_dir=tmp_path)
        assert cfg.input_dir == tmp_path


class TestValidateConfig:
    def test_builds_nested_config(self) -> None:
        cfg = validate_config(_Outer, {"count": 2, "inner": {"rate": "0.5"}})
        assert cfg.inner.rate == 0.5

    def test_errors_name_source_and_dotted_path(self) -> None:
        with pytest.raises(ConfigValidationError) as info:
            validate_config(_Outer, {"inner": {"rate": "fast", "extra": 1}}, source="a.toml")
        message = str(info.value)
        assert "config file a.toml" in message
        assert "inner.rate:" in message
        assert "inner.extra: unknown field" in message

    def test_private_fields_cannot_be_set_from_data(self) -> None:
        with pytest.raises(ConfigValidationError, match="_skip_validation: unknown field"):
            validate_config(BatchConfig, {"input_dir": "x", "_skip_validation": True})


class TestSchemas:
    def test_unknown_name(self) -> None:
        with pytest.raises(ConfigError, match="Unknown config 'nope'"):
            resolve_config_class("nope")

    @pytest.mark.parametrize("name", sorted(CONFIG_CLASSES))
    def test_schema_exports(self, name: str) -> None:
        try:
            schema = config_schema(name)
        except DependencyError:
            pytest.skip(f"{name} needs an extra that isn't installed")
        assert schema["type"] == "object"
        assert schema["additionalProperties"] is False
        assert not [k for k in schema["properties"] if k.startswith("_")]

    @pytest.mark.parametrize("name", sorted(CONFIG_CLASSES))
    def test_committed_schema_is_current(self, name: str) -> None:
        try:
            schema = config_schema(name)
        except DependencyError:
            pytest.skip(f"{name} needs an extra that isn't installed")
        committed = json.loads((SCHEMA_DIR / f"{name}.schema.json").read_text())
        assert committed == schema, (
            "docs/schemas is out of date; run: bioamla system schema --all -o docs/schemas"
        )

    def test_every_committed_schema_is_registered(self) -> None:
        names = {p.name.removesuffix(".schema.json") for p in SCHEMA_DIR.glob("*.schema.json")}
        assert names == set(CONFIG_CLASSES)


class TestRemovedAliases:
    """The old ``AugmentationConfig`` names were removed in 0.4.0."""

    @pytest.mark.parametrize(
        "module",
        [
            "bioamla.datasets",
            "bioamla.datasets.augmentation",
            "bioamla.ml",
            "bioamla.ml.preprocessing",
        ],
    )
    def test_old_name_is_gone(self, module: str) -> None:
        if module.startswith("bioamla.ml"):
            pytest.importorskip("torch")
        import importlib

        with pytest.raises(AttributeError):
            getattr(importlib.import_module(module), "AugmentationConfig")  # noqa: B009
