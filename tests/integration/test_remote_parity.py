"""The same command gives the same result locally and through a server (--remote)."""

import json
import shutil

import pytest
from click.testing import CliRunner

from bioamla.cli.cli import cli

pytestmark = pytest.mark.integration


@pytest.fixture
def remote(live_server, tmp_path, monkeypatch, test_audio_path):
    """A remote ``lab`` for the live server, with ``rec.wav`` (and a copy in ``more/``)
    in its workspace ``ws``; yields the workspace folder."""
    url, token, root = live_server
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    monkeypatch.setenv("BIOAMLA_LAB_TOKEN", token)
    monkeypatch.delenv("BIOAMLA_REMOTE", raising=False)
    assert CliRunner().invoke(cli, ["remote", "add", "lab", url]).exit_code == 0
    ws = root / "ws"
    shutil.copy(test_audio_path, ws / "rec.wav")
    (ws / "more").mkdir()
    shutil.copy(test_audio_path, ws / "more" / "a.wav")
    shutil.copy(test_audio_path, ws / "more" / "b.wav")
    return ws


def run_json(args):
    result = CliRunner().invoke(cli, ["--json", *args])
    body = json.loads(result.stdout)
    return result.exit_code, body


def relative(value, ws):
    """``value`` with paths into ``ws`` made relative to it: the server reports the
    absolute paths it resolved, a local run the paths as given."""
    if isinstance(value, dict):  # timings differ from run to run
        return {k: relative(v, ws) for k, v in value.items()
                if k not in ("duration_seconds", "start_time", "end_time")}  # fmt: skip
    if isinstance(value, list):
        return [relative(v, ws) for v in value]
    if isinstance(value, str):
        return value.replace(f"{ws}/", "")
    return value


# Each command runs in the workspace folder, once here and once on the server, with
# the same relative paths.
COMMANDS = [
    ["audio", "info", "rec.wav"],
    ["detect", "energy", "rec.wav", "--threshold-db", "-30"],
    ["indices", "compute", "rec.wav"],
    ["audio", "list", "more"],
    ["batch", "audio", "info", "--input-dir", "more", "--output-dir", "out"],
]


@pytest.mark.parametrize("args", COMMANDS, ids=[" ".join(c[:2]) for c in COMMANDS])
def test_same_json_locally_and_remotely(remote, monkeypatch, args):
    monkeypatch.chdir(remote)
    code, local = run_json(args)
    assert code == 0, local
    rcode, there = run_json(["--remote", "lab:ws", *args])
    assert rcode == 0, there
    assert there["command"] == local["command"]
    assert relative(there["result"], remote) == relative(local["result"], remote)


def test_same_human_output(remote, monkeypatch):
    monkeypatch.chdir(remote)
    local = CliRunner().invoke(cli, ["detect", "energy", "rec.wav"])
    there = CliRunner().invoke(cli, ["--remote", "lab:ws", "detect", "energy", "rec.wav"])
    assert there.exit_code == 0, there.output
    assert there.output == local.output


def test_remote_from_environment_and_workspace_config(remote, tmp_path, monkeypatch):
    from bioamla.workspace import create_workspace

    monkeypatch.setenv("BIOAMLA_REMOTE", "lab:ws")
    code, body = run_json(["audio", "info", "rec.wav"])  # rec.wav exists only there
    assert code == 0 and body["result"]["file"] == str(remote / "rec.wav")
    monkeypatch.delenv("BIOAMLA_REMOTE")

    local_ws = create_workspace(tmp_path / "laptop")
    local_ws.config_path.write_text('remote = "lab"\nworkspace = "ws"\n')
    monkeypatch.chdir(local_ws.root)
    code, body = run_json(["audio", "info", "rec.wav"])
    assert code == 0 and body["result"]["file"] == str(remote / "rec.wav")
    assert local_ws.history() == []  # recorded on the server, not here


def test_errors_match_a_local_run(remote, monkeypatch):
    monkeypatch.chdir(remote)
    code, local = run_json(["audio", "info", "missing.wav"])
    rcode, there = run_json(["--remote", "lab:ws", "audio", "info", "missing.wav"])
    assert (rcode, there["error"]["type"]) == (code, local["error"]["type"])


def test_local_only_commands_refuse_an_explicit_remote(remote, monkeypatch):
    for args in (["remote", "list"], ["audio", "play", "rec.wav"], ["serve", "--root", "x"]):
        result = CliRunner().invoke(cli, ["--remote", "lab:ws", *args])
        assert result.exit_code == 2, (args, result.output)
        assert "only on this machine" in result.output
    monkeypatch.setenv("BIOAMLA_REMOTE", "lab:ws")  # implicit: they just run here
    assert CliRunner().invoke(cli, ["remote", "list"]).exit_code == 0


def test_remote_needs_a_workspace_and_keeps_secrets_local(remote):
    result = CliRunner().invoke(cli, ["--remote", "lab", "audio", "info", "rec.wav"])
    assert result.exit_code == 2 and "Which workspace" in result.output
    result = CliRunner().invoke(cli, ["--remote", "lab:ws", "catalogs", "ebird", "nearby",
                                      "--lat", "1", "--lng", "1", "--api-key", "k"])  # fmt: skip
    assert result.exit_code == 2 and "aren't sent to a server" in result.output
