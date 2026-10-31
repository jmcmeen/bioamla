"""Settings files: locations, merging, TOML writing."""

import math

import pytest

from bioamla.common.config import load_toml
from bioamla.common.settings import (
    dump_toml,
    flatten,
    merge_settings,
    parse_value,
    read_settings_file,
    set_dotted,
    settings_layers,
    unset_dotted,
    user_config_path,
)
from bioamla.exceptions import ConfigError


def test_user_config_path_follows_xdg(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("os.name", "posix")
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    assert user_config_path() == tmp_path / "bioamla" / "config.toml"


def test_missing_file_is_empty(tmp_path) -> None:
    assert read_settings_file(tmp_path / "nope.toml") == {}


def test_bad_toml(tmp_path) -> None:
    (tmp_path / "c.toml").write_text("x = [")
    with pytest.raises(ConfigError):
        read_settings_file(tmp_path / "c.toml")


def test_merge_is_deep_and_later_wins() -> None:
    a = {"detect": {"energy": {"x": 1, "y": 2}}, "k": 1}
    b = {"detect": {"energy": {"y": 3}}, "k": {"t": 1}}
    assert merge_settings(a, b) == {"detect": {"energy": {"x": 1, "y": 3}}, "k": {"t": 1}}
    assert a == {"detect": {"energy": {"x": 1, "y": 2}}, "k": 1}  # inputs untouched


def test_layers_lowest_first(tmp_path) -> None:
    user_config_path().parent.mkdir(parents=True)
    user_config_path().write_text("a = 1\n")
    ws = tmp_path / "ws.toml"
    ws.write_text("a = 2\n")
    assert [data for _, data in settings_layers(ws)] == [{"a": 1}, {"a": 2}]


def test_flatten_set_unset() -> None:
    settings: dict = {}
    set_dotted(settings, "detect.energy.threshold_db", -25)
    set_dotted(settings, "detect.peaks.snr", 2.0)
    assert flatten(settings) == {"detect.energy.threshold_db": -25, "detect.peaks.snr": 2.0}
    assert unset_dotted(settings, "detect.energy.threshold_db")
    assert settings == {"detect": {"peaks": {"snr": 2.0}}}  # the empty table went too
    assert not unset_dotted(settings, "detect.energy.threshold_db")
    assert unset_dotted(settings, "detect.peaks.snr") and settings == {}


def test_set_through_a_value_fails() -> None:
    with pytest.raises(ConfigError, match="is a value"):
        set_dotted({"detect": 1}, "detect.energy.x", 1)


def test_dump_round_trips(tmp_path) -> None:
    settings = {
        "top": "level",
        "detect": {"energy": {"threshold_db": -25, "label": 'say "hi"', "flag": True}},
        "models": {"ast": {"predict": {"labels": ["a", "b"], "rate": 0.5}}},
        "odd key": {"x": 1},
    }
    path = tmp_path / "c.toml"
    path.write_text(dump_toml(settings))
    assert load_toml(path) == settings


def test_dump_special_floats(tmp_path) -> None:
    path = tmp_path / "c.toml"
    path.write_text(dump_toml({"a": math.inf, "b": -math.inf}))
    assert load_toml(path) == {"a": math.inf, "b": -math.inf}


def test_dump_rejects_other_types() -> None:
    with pytest.raises(ConfigError):
        dump_toml({"a": object()})


@pytest.mark.parametrize(
    ("text", "value"),
    [("-25", -25), ("0.5", 0.5), ("true", True), ('"x y"', "x y"), ("[1, 2]", [1, 2]),
     ("bioamla/scp-frogs", "bioamla/scp-frogs"), ("hello", "hello")],
)  # fmt: skip
def test_parse_value(text, value) -> None:
    assert parse_value(text) == value
