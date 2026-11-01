"""bioamla.client against a real server, and the `bioamla remote` commands."""

import json
import shutil

import pytest
from click.testing import CliRunner

from bioamla.cli.cli import cli
from bioamla.client import (
    Client,
    JobFailed,
    RemoteError,
    RemoteInvalidInput,
    RemoteNotFound,
    add_remote,
    get_remote,
    remotes,
    remove_remote,
)
from bioamla.exceptions import ConfigError, InvalidInputError, NotFoundError


@pytest.fixture
def user_config(tmp_path, monkeypatch):
    """A private user settings file."""
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    monkeypatch.delenv("APPDATA", raising=False)
    return tmp_path / "xdg" / "bioamla" / "config.toml"


def test_remotes_are_kept_in_user_settings_without_the_token(user_config, monkeypatch):
    added = add_remote("lab", "http://lab:8765/")
    assert (added.url, added.token_env) == ("http://lab:8765", "BIOAMLA_LAB_TOKEN")
    add_remote("gpu-box", "https://gpu:9000", token_env="GPU_TOKEN")
    assert [r.name for r in remotes()] == ["gpu-box", "lab"]
    text = user_config.read_text()
    assert "[remotes.lab]" in text and "token =" not in text
    with pytest.raises(ConfigError, match="BIOAMLA_LAB_TOKEN"):
        get_remote("lab").token()
    monkeypatch.setenv("BIOAMLA_LAB_TOKEN", "s3cret")
    assert get_remote("lab").token() == "s3cret"
    assert remove_remote("lab") is True and remove_remote("lab") is False
    with pytest.raises(NotFoundError):
        get_remote("lab")
    for name, url in (("bad name", "http://x"), ("ok", "ftp://x"), ("ok", "lab:8765")):
        with pytest.raises(InvalidInputError):
            add_remote(name, url)
    # An incomplete entry is left out of the list and can't hide the others.
    add_remote("good", "http://good:1")
    user_config.write_text(user_config.read_text() + '\n[remotes.broken]\nurl = "http://b"\n')
    assert [r.name for r in remotes()] == ["good", "gpu-box"]
    assert get_remote("good").url == "http://good:1"
    with pytest.raises(ConfigError, match="token_env"):
        get_remote("broken")


@pytest.mark.slow
def test_client_drives_a_server(live_server, test_audio_path, tmp_path):
    url, token, root = live_server
    client = Client(url, token)
    assert client.health() == {"status": "ok"}
    assert client.version()["api"] == "v1"

    made = client.create_workspace("site1", "Site 1")
    assert made["name"] == "Site 1"
    assert {w["workspace"] for w in client.workspaces()} == {"ws", "site1"}
    with pytest.raises(RemoteError) as exc:
        client.create_workspace("site1")
    assert exc.value.status == 409

    up = client.upload("site1", test_audio_path, "raw/rec.wav")
    assert up["path"] == "raw/rec.wav"
    listed = client.files("site1", recursive=True)["files"]
    assert "raw/rec.wav" in [f["path"] for f in listed]
    got = client.download("site1", "raw/rec.wav", tmp_path / "dl")  # a file named dl
    assert got == tmp_path / "dl" and got.read_bytes() == open(test_audio_path, "rb").read()
    with pytest.raises(InvalidInputError, match="exists"):
        client.download("site1", "raw/rec.wav", got)
    into = client.download("site1", "raw/rec.wav", f"{tmp_path}/new-folder/")
    assert into == tmp_path / "new-folder" / "rec.wav"

    info = client.run("site1", "audio.info", {"file": "raw/rec.wav"})  # quick
    assert info["info"]["sample_rate"] == 16000
    events = []
    detected = client.run("site1", "detect.energy", {"file": "raw/rec.wav"},
                          on_event=events.append)  # a job  # fmt: skip
    assert "count" in detected and events[-1]["status"] == "succeeded"
    jobs = client.jobs("site1")
    assert jobs[0]["operation"] == "detect.energy"
    assert client.job(jobs[0]["id"][:8])["status"] == "succeeded"  # a prefix will do
    with pytest.raises(RemoteNotFound):
        client.job("zzzz")

    client.set_setting("site1", "detect.energy.threshold_db", -30)
    assert client.settings("site1") == {"detect.energy.threshold_db": -30}
    assert client.unset_setting("site1", "detect.energy.threshold_db") is True

    with pytest.raises(RemoteNotFound):
        client.status("nope")
    with pytest.raises(RemoteInvalidInput):
        client.submit("site1", "audio.info", {"nope": 1})
    with pytest.raises(RemoteError) as exc:
        client.submit("site1", "audio.info", {"file": "/etc/hosts"})
    assert exc.value.status == 403 and exc.value.error_type == "AccessDenied"
    (root / "site1" / "bad.wav").write_text("not audio")
    with pytest.raises(JobFailed) as failed:
        client.run("site1", "detect.energy", {"file": "bad.wav"})
    assert failed.value.job["status"] == "failed"
    with pytest.raises(RemoteError, match="token"):
        Client(url, "wrong").version()


def test_an_unreachable_server_is_a_remote_error():
    with pytest.raises(RemoteError, match="Can't reach"):
        Client("http://127.0.0.1:9", "t", timeout=2).version()


@pytest.mark.slow
def test_remote_commands(live_server, user_config, monkeypatch, test_audio_path, tmp_path):
    url, token, root = live_server
    run = CliRunner().invoke
    assert run(cli, ["remote", "add", "lab", url]).exit_code == 0
    listed = json.loads(run(cli, ["--json", "remote", "list"]).stdout)["result"]["remotes"]
    assert listed == [{"name": "lab", "url": url, "token_env": "BIOAMLA_LAB_TOKEN",
                       "token_set": False}]  # fmt: skip
    monkeypatch.setenv("BIOAMLA_LAB_TOKEN", "  ")  # blank counts as unset
    assert json.loads(run(cli, ["--json", "remote", "list"]).stdout)["result"]["remotes"][0][
        "token_set"] is False  # fmt: skip
    missing = run(cli, ["remote", "ping", "lab"])
    assert missing.exit_code == 1 and "BIOAMLA_LAB_TOKEN" in missing.output
    monkeypatch.setenv("BIOAMLA_LAB_TOKEN", token)

    pinged = json.loads(run(cli, ["--json", "remote", "ping", "lab"]).stdout)["result"]
    assert pinged["api"] == "v1"
    made = run(cli, ["remote", "workspaces", "lab", "--create", "site1"])
    assert made.exit_code == 0, made.output
    assert "site1" in made.output

    rec = tmp_path / "rec.wav"
    shutil.copy(test_audio_path, rec)
    up = run(cli, ["--json", "remote", "upload", "lab", "site1", str(rec), "--to", "raw"])
    assert json.loads(up.stdout)["result"]["uploaded"][0]["path"] == "raw/rec.wav"
    again = run(cli, ["remote", "upload", "lab", "site1", str(rec), "--to", "raw"])
    assert again.exit_code == 1 and "overwrite" in again.output

    out = tmp_path / "out"
    down = run(cli, ["remote", "download", "lab", "site1", "raw/rec.wav", "-o", str(out)])
    assert down.exit_code == 0 and (out / "rec.wav").read_bytes() == rec.read_bytes()
    again = run(cli, ["remote", "download", "lab", "site1", "raw/rec.wav", "-o", str(out)])
    assert again.exit_code == 1 and "exists" in again.output
    both = run(cli, ["remote", "download", "lab", "site1", "raw/rec.wav", "x/rec.wav"])
    assert both.exit_code == 2 and "separate folders" in both.output
    partial = run(cli, ["remote", "upload", "lab", "site1", str(rec), str(rec)])
    assert partial.exit_code == 1 and "already uploaded: rec.wav" in partial.output

    job = Client(url, token).submit("site1", "detect.energy", {"file": "raw/rec.wav"})
    jobs = json.loads(run(cli, ["--json", "remote", "jobs", "lab"]).stdout)["result"]["jobs"]
    assert jobs[0]["id"] == job["id"]
    cancelled = run(cli, ["--json", "remote", "cancel", "lab", job["id"][:8]])
    assert cancelled.exit_code == 0, cancelled.output
    assert json.loads(cancelled.stdout)["result"]["id"] == job["id"]

    assert run(cli, ["remote", "remove", "lab"]).exit_code == 0
    assert run(cli, ["remote", "remove", "lab"]).exit_code == 1


def test_wait_resumes_a_dropped_event_stream(monkeypatch):
    client = Client("http://server", "t")
    streams = [
        [{"id": 1, "event": "status", "status": "running"}],
        RemoteError("Lost the connection"),
        [{"id": 2, "event": "progress", "done": 1, "total": 1},
         {"id": 3, "event": "status", "status": "succeeded"}],
    ]  # fmt: skip
    seen_after = []

    def events(job_id, after=0):
        seen_after.append(after)
        item = streams.pop(0)
        if isinstance(item, Exception):
            raise item
        yield from item

    states = iter(["running", "succeeded"])
    monkeypatch.setattr(client, "events", events)
    monkeypatch.setattr(client, "job", lambda job_id: {"id": job_id, "status": next(states)})
    monkeypatch.setattr("time.sleep", lambda s: None)
    progress = []
    job = client.wait("x" * 32, on_progress=lambda d, t: progress.append((d, t)))
    assert job["status"] == "succeeded"
    assert seen_after == [0, 1, 1] and progress == [(1, 1)]
