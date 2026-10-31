"""Option defaults from settings files and environment variables."""

import json
import logging

import pytest
from click.testing import CliRunner

from bioamla.cli.cli import cli
from bioamla.cli.settings_map import build_default_map, canonical_key, env_name, env_settings
from bioamla.common.settings import user_config_path


def threshold(*args: str, env: dict | None = None) -> float:
    """The --threshold-db `detect energy` would run with (from its --json params)."""
    result = CliRunner().invoke(cli, ["--json", "detect", "energy", *args], env=env)
    envelope = json.loads(result.stdout)
    assert envelope["ok"], envelope
    return envelope["result"]["params"]["threshold_db"]


@pytest.fixture
def audio(test_audio_path):
    pytest.importorskip("librosa")
    return test_audio_path


@pytest.fixture
def ws(tmp_path, monkeypatch):
    from bioamla.workspace import create_workspace

    ws = create_workspace(tmp_path / "ws")
    monkeypatch.chdir(ws.root)
    return ws


def write_user(text: str) -> None:
    user_config_path().parent.mkdir(parents=True, exist_ok=True)
    user_config_path().write_text(text)


class TestPrecedence:
    def test_each_level_overrides_the_one_before(self, audio, ws) -> None:
        assert threshold(audio) == -20.0  # the command's own default
        write_user("[detect.energy]\nthreshold_db = -30\n")
        assert threshold(audio) == -30.0  # user file
        ws.config_path.write_text("[detect.energy]\nthreshold-db = -10\n")
        assert threshold(audio) == -10.0  # workspace file (another spelling)
        env = {"BIOAMLA_DETECT_ENERGY_THRESHOLD_DB": "-15"}
        assert threshold(audio, env=env) == -15.0  # environment
        assert threshold(audio, "--threshold-db", "-5", env=env) == -5.0  # flag

    def test_outside_a_workspace_only_the_user_file(self, audio, tmp_path, monkeypatch) -> None:
        write_user("[detect.energy]\nthreshold_db = -30\n")
        monkeypatch.chdir(tmp_path)
        assert threshold(audio) == -30.0

    def test_unknown_keys_warn(self, audio, ws, caplog) -> None:
        ws.config_path.write_text("[detect.energy]\nbogus = 1\n[nocommand]\nx = 2\n")
        with caplog.at_level(logging.WARNING, logger="bioamla.cli"):
            assert threshold(audio) == -20.0
        assert "detect.energy.bogus" in caplog.text and "nocommand.x" in caplog.text

    def test_broken_file_is_an_error_envelope(self, audio, ws) -> None:
        ws.config_path.write_text("x = [")
        result = CliRunner().invoke(cli, ["--json", "detect", "energy", audio])
        envelope = json.loads(result.stdout)
        assert not envelope["ok"] and envelope["error"]["type"] == "ConfigError"
        assert str(ws.config_path) in envelope["error"]["message"]


class TestMapping:
    def test_nested_group_option(self) -> None:
        settings = {"models": {"ast": {"predict": {"model-path": "m"}}}}
        default_map, unknown = build_default_map(cli, settings)
        assert unknown == []
        (name,) = default_map["models"]["ast"]["predict"]
        assert default_map["models"]["ast"]["predict"][name] == "m"

    def test_renamed_parameter_by_long_name(self) -> None:
        default_map, _ = build_default_map(cli, {"audio": {"segment": {"duration_seconds": 4}}})
        assert default_map == {"audio": {"segment": {"duration": 4}}}

    def test_reserved_sections_are_not_commands(self) -> None:
        _, unknown = build_default_map(cli, {"remotes": {"lab": {"url": "http://x"}}})
        assert unknown == []

    def test_env_names_use_the_long_option_name(self) -> None:
        segment = cli.commands["audio"].commands["segment"]
        option = next(p for p in segment.params if p.name == "duration")
        assert env_name(["audio", "segment"], option) == "BIOAMLA_AUDIO_SEGMENT_DURATION_SECONDS"
        assert env_settings(cli, {"BIOAMLA_AUDIO_SEGMENT_DURATION_SECONDS": "2"}) == {
            "audio": {"segment": {"duration": "2"}}
        }
        assert env_settings(cli, {"PATH": "/bin"}) == {}

    def test_canonical_key(self) -> None:
        assert canonical_key(cli, "detect.energy.threshold-db") == "detect.energy.threshold_db"
        assert canonical_key(cli, "audio.segment.duration") == "audio.segment.duration_seconds"
        assert canonical_key(cli, "nope.x") == "nope.x"


class TestConfigCommands:
    def run(self, *args):
        return CliRunner().invoke(cli, ["workspace", "config", *args])

    def test_set_get_list_unset(self, ws) -> None:
        assert self.run("set", "detect.energy.threshold-db", "-12").exit_code == 0
        assert ws.config_path.read_text() == "[detect.energy]\nthreshold_db = -12.0\n"
        result = CliRunner().invoke(cli, ["--json", "workspace", "config", "get",
                                          "detect.energy.threshold_db"])  # fmt: skip
        got = json.loads(result.stdout)["result"]
        assert got["value"] == -12 and got["source"] == str(ws.config_path)

        assert self.run("set", "--user", "detect.energy.threshold_db", "-30").exit_code == 0
        listed = json.loads(CliRunner().invoke(cli, ["--json", "workspace", "config", "list"]
                                               ).stdout)["result"]["settings"]  # fmt: skip
        assert listed == [{"key": "detect.energy.threshold_db", "value": -12,
                           "source": str(ws.config_path)}]  # fmt: skip

        assert self.run("unset", "detect.energy.threshold_db").exit_code == 0
        assert "-30" in self.run("get", "detect.energy.threshold_db").output

    def test_set_replaces_other_spellings(self, ws) -> None:
        ws.config_path.write_text("[detect.energy]\nthreshold-db = -10\n")
        self.run("set", "detect.energy.threshold_db", "-12")
        assert ws.config_path.read_text() == "[detect.energy]\nthreshold_db = -12.0\n"

    def test_set_rejects_unknown_keys(self, ws) -> None:
        result = self.run("set", "detect.energy.bogus", "1")
        assert result.exit_code == 1 and "has no option 'bogus'" in result.output

    def test_set_without_a_workspace_needs_user(self, tmp_path, monkeypatch) -> None:
        monkeypatch.chdir(tmp_path)
        assert self.run("set", "detect.energy.threshold_db", "-1").exit_code == 1
        assert self.run("set", "--user", "detect.energy.threshold_db", "-1").exit_code == 0
        assert "threshold_db = -1" in user_config_path().read_text()

    def test_get_unset_key(self, ws) -> None:
        result = self.run("get", "detect.energy.threshold_db")
        assert result.exit_code == 1 and "not set" in result.output


class TestReviewCases:
    def test_root_options_ignore_env_and_settings(self, audio, ws) -> None:
        assert threshold(audio, env={"BIOAMLA_VERSION": "0.4.0", "BIOAMLA_JSON": "1"}) == -20.0
        write_user('version = true\nlog_level = "DEBUG"\n')
        assert threshold(audio) == -20.0  # no banner, no exit; just a warning

    def test_settings_cant_set_eager_options(self) -> None:
        _, unknown = build_default_map(cli, {"system": {"help": True}})
        assert unknown == ["system.help"]

    def test_multiple_option_from_env_and_file(self) -> None:
        extract = cli.commands["dataset"].commands["extract-clips"]
        include = next(p for p in extract.params if p.name == "include")
        from bioamla.cli.settings_map import _shape

        assert _shape(include, "frog") == ["frog"]
        assert _shape(include, ["frog", "toad"]) == ["frog", "toad"]
        got = env_settings(cli, {"BIOAMLA_DATASET_EXTRACT_CLIPS_INCLUDE": "frog toad"})
        assert got["dataset"]["extract-clips"]["include"] == ["frog", "toad"]

    def test_broken_file_doesnt_block_help(self, ws) -> None:
        write_user("x = [")
        result = CliRunner().invoke(cli, ["--help"])
        assert result.exit_code == 0 and "Usage" in result.output

    def test_set_converts_by_option_type(self, ws) -> None:
        run = TestConfigCommands().run
        assert run("set", "dataset.extract-clips.include", "frog").exit_code == 0
        assert run("set", "detect.energy.threshold_db", "-12").exit_code == 0
        result = run("set", "detect.energy.threshold_db", "loud")
        assert result.exit_code == 1 and "not valid for --threshold-db" in result.output
        from bioamla.common.config import load_toml

        assert load_toml(ws.config_path) == {
            "dataset": {"extract-clips": {"include": ["frog"]}},
            "detect": {"energy": {"threshold_db": -12.0}},
        }

    def test_set_reports_the_long_option(self, ws) -> None:
        result = CliRunner().invoke(cli, ["--json", "workspace", "config", "set",
                                          "dataset.extract-clips.include", "frog"])  # fmt: skip
        assert json.loads(result.stdout)["result"]["option"] == "--include"
