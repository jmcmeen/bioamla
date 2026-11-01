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


def untimed(value):
    """``value`` without its timings, which differ from run to run."""
    if isinstance(value, dict):
        return {k: untimed(v) for k, v in value.items()
                if k not in ("duration_seconds", "start_time", "end_time")}  # fmt: skip
    if isinstance(value, list):
        return [untimed(v) for v in value]
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
    # Paths come back relative to the remote workspace, as a local run reports them.
    assert untimed(there["result"]) == untimed(local["result"])


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
    assert code == 0 and body["result"]["file"] == "rec.wav"  # relative to the workspace
    monkeypatch.delenv("BIOAMLA_REMOTE")

    local_ws = create_workspace(tmp_path / "laptop")
    local_ws.config_path.write_text('remote = "lab"\nworkspace = "ws"\n')
    monkeypatch.chdir(local_ws.root)
    code, body = run_json(["audio", "info", "rec.wav"])
    assert code == 0 and body["result"]["file"] == "rec.wav"  # relative to the workspace
    assert local_ws.history() == []  # recorded on the server, not here


def test_errors_match_a_local_run(remote, monkeypatch):
    monkeypatch.chdir(remote)
    (remote / "bad.wav").write_text("not audio")
    for args in (["audio", "info", "missing.wav"], ["detect", "energy", "bad.wav"]):  # quick, job
        code, local = run_json(args)
        rcode, there = run_json(["--remote", "lab:ws", *args])
        assert (rcode, there["error"]["type"]) == (code, local["error"]["type"]), args

        def first_line(body):  # decoder details (addresses) differ from run to run
            return relative(body["error"]["message"], remote).splitlines()[0]

        assert first_line(there) == first_line(local), args


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


def test_local_commands_keep_their_checks_under_an_implicit_remote(remote):
    import click

    from bioamla.cli import remote_mode

    strict = click.Path.convert
    implicit = remote_mode.Target("lab", "ws", False, "$BIOAMLA_REMOTE")
    remote_mode.begin(implicit, "audio play")  # local-only: runs here, checks intact
    assert remote_mode.current() is None and click.Path.convert is strict
    remote_mode.begin(implicit, "audio info")  # goes to the server: its paths
    assert remote_mode.current() == implicit and click.Path.convert is not strict
    remote_mode.reset()
    assert click.Path.convert is strict


def test_a_remote_named_like_a_command(remote, live_server):
    url, _, _ = live_server
    assert CliRunner().invoke(cli, ["remote", "add", "serve", url, "--token-env",
                                    "BIOAMLA_LAB_TOKEN"]).exit_code == 0  # fmt: skip
    code, body = run_json(["--remote", "serve:ws", "audio", "info", "rec.wav"])
    assert code == 0, body


def test_keys_from_the_environment_stay_here(remote, monkeypatch):
    from bioamla.cli import remote_mode

    sent = {}

    class FakeClient:
        def run(self, ws, name, request, **kwargs):
            sent.update(request=request)
            return "ok"

    monkeypatch.setattr("bioamla.client.connect", lambda name: FakeClient())
    monkeypatch.setenv("EBIRD_API_KEY", "from-env")
    target = remote_mode.Target("lab", "ws", True, "--remote")
    request = {"species_code": "amerob", "lat": 1.0, "lng": 1.0}
    got = remote_mode.run(target, "catalogs.ebird.validate", request, on_progress=None,
                          on_notice=print, secrets={"EBIRD_API_KEY": "from-env"})  # fmt: skip
    assert got == "ok" and "from-env" not in json.dumps(sent)


def test_a_remote_run_does_not_hide_the_next_local_one(remote, monkeypatch, tmp_path):
    from bioamla.cli import remote_mode
    from bioamla.workspace import create_workspace

    assert run_json(["--remote", "lab:ws", "audio", "info", "rec.wav"])[0] == 0
    assert remote_mode.used()
    local = create_workspace(tmp_path / "here")
    monkeypatch.chdir(local.root)
    CliRunner().invoke(cli, ["audio", "info", "--no-such-flag"])  # fails while parsing
    assert not remote_mode.used()
    assert [r["command"] for r in local.history()] == ["audio info"]
