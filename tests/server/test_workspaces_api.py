"""Workspace and file routes."""

import os

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from bioamla.server import ServerConfig, create_app  # noqa: E402
from bioamla.workspace import create_workspace, open_workspace  # noqa: E402

AUTH = {"Authorization": "Bearer t"}


@pytest.fixture
def server(tmp_path):
    root = tmp_path / "root"
    create_workspace(root / "ws", name="Field season")
    archive = tmp_path / "archive"
    archive.mkdir()
    (archive / "a.wav").write_bytes(b"RIFF")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("no")
    app = create_app(ServerConfig(root=root, token="t", linkable=[archive]))
    return TestClient(app), root, archive, outside


def test_list_create_and_status(server):
    client, root, _, _ = server
    (root / "plain-folder").mkdir()
    listed = client.get("/v1/workspaces", headers=AUTH).json()["result"]
    assert [(w["workspace"], w["name"]) for w in listed] == [("ws", "Field season")]
    (root / "ws" / "up.wav.abc.part").write_bytes(b"")  # an unfinished upload: not listed

    made = client.post("/v1/workspaces", json={"name": "new", "title": "New"}, headers=AUTH)
    assert made.status_code == 201, made.text
    assert made.json()["result"]["name"] == "New"
    assert open_workspace(root / "new").name == "New"
    assert client.post("/v1/workspaces", json={"name": "new"}, headers=AUTH).status_code == 409
    # A link under the root (even a dangling one) is never followed to make a workspace.
    os.symlink(root / "elsewhere", root / "dangling")
    assert client.post("/v1/workspaces", json={"name": "dangling"}, headers=AUTH).status_code == 409
    assert not (root / "elsewhere").exists()
    for bad in ("../up", ".hidden", "a/b", ""):
        got = client.post("/v1/workspaces", json={"name": bad}, headers=AUTH)
        assert got.status_code == 400, bad

    status = client.get("/v1/workspaces/ws", headers=AUTH).json()["result"]
    assert status["workspace"] == "ws" and status["last_runs"] == []
    assert client.get("/v1/workspaces/nope", headers=AUTH).status_code == 404
    assert client.get("/v1/workspaces").status_code == 401


def test_config_set_get_and_unset(server):
    client, root, _, _ = server
    url = "/v1/workspaces/ws/config"
    got = client.put(f"{url}/detect.energy.threshold_db", json={"value": -25}, headers=AUTH)
    assert got.status_code == 200, got.text
    assert client.get(url, headers=AUTH).json()["result"]["settings"] == {
        "detect.energy.threshold_db": -25
    }
    assert open_workspace(root / "ws").settings() == {"detect.energy.threshold_db": -25}
    for key in ("detect.energy.nope", "nope.op.x", "plain"):
        assert client.put(f"{url}/{key}", json={"value": 1}, headers=AUTH).status_code == 400
    for value in ({"a": 1}, None, [None], [{"a": 1}]):
        got = client.put(f"{url}/detect.energy.low_freq", json={"value": value}, headers=AUTH)
        assert got.status_code in (400, 422), value  # null is a missing value: 422
    # One spelling per key: the other one is replaced, and unset removes any.
    space = open_workspace(root / "ws")
    space.set_setting("detect.energy.min-duration", 0.1)
    client.put(f"{url}/detect.energy.min_duration", json={"value": 0.2}, headers=AUTH)
    assert space.settings() == {"detect.energy.threshold_db": -25,
                                "detect.energy.min_duration": 0.2}  # fmt: skip
    space.set_setting("detect.energy.min-duration", 0.3)
    client.delete(f"{url}/detect.energy.min_duration", headers=AUTH)
    assert "detect.energy.min-duration" not in space.settings()
    removed = client.delete(f"{url}/detect.energy.threshold_db", headers=AUTH).json()["result"]
    assert removed["removed"] is True
    assert client.get(url, headers=AUTH).json()["result"]["settings"] == {}


def test_upload_list_and_download(server):
    client, root, _, _ = server
    url = "/v1/workspaces/ws/files"
    up = client.put(f"{url}/raw/site1/rec.wav", content=b"x" * 3000, headers=AUTH)
    assert up.status_code == 201, up.text
    assert up.json()["result"] == {"path": "raw/site1/rec.wav", "size": 3000}
    assert (root / "ws" / "raw" / "site1" / "rec.wav").read_bytes() == b"x" * 3000
    again = client.put(f"{url}/raw/site1/rec.wav", content=b"y", headers=AUTH)
    assert again.status_code == 409
    replaced = client.put(f"{url}/raw/site1/rec.wav?overwrite=true", content=b"y", headers=AUTH)
    assert replaced.status_code == 201
    assert not [p for p in (root / "ws" / "raw" / "site1").iterdir() if p.name.endswith(".part")]

    top = client.get(url, headers=AUTH).json()["result"]["files"]
    assert [(f["path"], f["type"]) for f in top] == [("raw", "dir")]  # no .bioamla
    deep = client.get(f"{url}?recursive=true", headers=AUTH).json()["result"]["files"]
    assert ("raw/site1/rec.wav", "file", 1) in [(f["path"], f["type"], f["size"]) for f in deep]
    assert all(not f["path"].endswith(".part") for f in deep)
    sub = client.get(f"{url}?path=raw/site1", headers=AUTH).json()["result"]["files"]
    assert [f["path"] for f in sub] == ["raw/site1/rec.wav"]

    down = client.get(f"{url}/raw/site1/rec.wav", headers=AUTH)
    assert down.status_code == 200 and down.content == b"y"
    assert client.get(f"{url}/raw/missing.wav", headers=AUTH).status_code == 404
    assert client.get(f"{url}?path=missing", headers=AUTH).status_code == 404


def test_paths_outside_the_roots_are_refused(server):
    client, root, archive, outside = server
    url = "/v1/workspaces/ws/files"
    assert client.get(f"{url}?path={outside}", headers=AUTH).status_code == 403
    assert client.get(f"{url}?path=../../outside", headers=AUTH).status_code == 403
    assert client.get(f"{url}/{outside}/secret.txt", headers=AUTH).status_code in (403, 404)
    assert client.get(f"{url}/..%2F..%2Foutside%2Fsecret.txt", headers=AUTH).status_code == 403
    got = client.put(f"{url}/..%2F..%2Foutside%2Fx.txt", content=b"x", headers=AUTH)
    assert got.status_code == 403 and not (outside / "x.txt").exists()
    # A link inside the workspace can't reach outside the allowed folders.
    os.symlink(outside, root / "ws" / "escape")
    assert client.get(f"{url}/escape/secret.txt", headers=AUTH).status_code == 403
    assert client.put(f"{url}/escape/new.txt", content=b"x", headers=AUTH).status_code == 403
    assert not (outside / "new.txt").exists()
    # No workspace's metadata can be read or overwritten, this one's or another's.
    create_workspace(root / "other", name="Other")
    for target in (".bioamla/workspace.toml", "..%2Fother%2F.bioamla%2Fartifacts.json",
                   f"{root}/other/.bioamla/history.jsonl"):  # fmt: skip
        got = client.put(f"{url}/{target}?overwrite=true", content=b"x", headers=AUTH)
        assert got.status_code == 403, target
    assert client.get(f"{url}/.bioamla/workspace.toml", headers=AUTH).status_code == 403
    assert open_workspace(root / "ws").name == "Field season"
    assert open_workspace(root / "other").name == "Other"
    # Uploads go only into the workspace in the URL, not another or a linkable folder.
    assert client.put(f"{url}/..%2Fother%2Fx.wav", content=b"x", headers=AUTH).status_code == 403
    assert client.put(f"{url}/{archive}/new.wav", content=b"x", headers=AUTH).status_code == 403
    assert not (root / "other" / "x.wav").exists() and not (archive / "new.wav").exists()
    # A file where a folder should be is a bad request, and leaves nothing behind.
    (root / "ws" / "a.wav").write_bytes(b"x")
    assert client.put(f"{url}/a.wav/b.wav", content=b"x", headers=AUTH).status_code == 400


def test_linkable_folders_can_be_listed_and_read(server):
    client, root, archive, _ = server
    os.symlink(archive, root / "ws" / "archive")
    url = "/v1/workspaces/ws/files"
    top = client.get(url, headers=AUTH).json()["result"]["files"]
    assert ("archive", "link") in [(f["path"], f["type"]) for f in top]
    listed = client.get(f"{url}?path=archive", headers=AUTH).json()["result"]["files"]
    assert [f["path"] for f in listed] == [str(archive / "a.wav")]
    assert client.get(f"{url}/archive/a.wav", headers=AUTH).content == b"RIFF"


def test_workspace_library_settings_and_files(tmp_path):
    ws = create_workspace(tmp_path / "ws")
    ws.set_setting("detect.energy.threshold_db", -30)
    ws.set_setting("models.ast.predict.model_path", "bioamla/scp-frogs")
    assert ws.settings()["detect.energy.threshold_db"] == -30
    assert ws.unset_setting("detect.energy.threshold_db") is True
    assert ws.unset_setting("detect.energy.threshold_db") is False
    (tmp_path / "ws" / "a.txt").write_text("hi")
    assert [f["path"] for f in ws.list_files()] == ["a.txt"]


def test_listing_hides_every_workspace_metadata_and_stops_at_a_limit(server):
    client, root, _, _ = server
    create_workspace(root / "ws" / "nested")
    for i in range(5):
        (root / "ws" / f"f{i}.txt").write_text("x")
    url = "/v1/workspaces/ws/files"
    deep = client.get(f"{url}?recursive=true", headers=AUTH).json()["result"]
    assert not [f for f in deep["files"] if ".bioamla" in f["path"]]
    assert deep["truncated"] is False
    space = open_workspace(root / "ws")
    assert len(space.list_files(limit=3)) == 3


def test_atomic_file_writer(tmp_path):
    from bioamla.common.files import AtomicFileWriter

    target = tmp_path / "sub" / "a.bin"
    with AtomicFileWriter(target) as w:
        w.write(b"ab")
        w.write(b"c")
        w.commit()
    assert target.read_bytes() == b"abc" and w.size == 3
    with pytest.raises(FileExistsError), AtomicFileWriter(target) as w:
        w.write(b"z")
        w.commit()
    with AtomicFileWriter(target, overwrite=True) as w:
        w.write(b"new")  # not committed: nothing changes
    assert target.read_bytes() == b"abc"
    assert sorted(p.name for p in target.parent.iterdir()) == ["a.bin"]


def _write_ramp(path, seconds=3.0, sr=16000):
    import numpy as np
    import soundfile as sf

    samples = (np.arange(int(seconds * sr)) / (seconds * sr)).astype("float32")  # 0 -> 1
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, samples, sr, subtype="FLOAT")
    return samples


def _decode(response):
    import io

    import soundfile as sf

    assert response.headers["content-type"].startswith("audio/wav"), response.text
    samples, sr = sf.read(io.BytesIO(response.content), dtype="float32")
    return samples, sr


def test_audio_window_streams_part_of_a_recording(server):
    import numpy as np

    client, root, _, _ = server
    samples = _write_ramp(root / "ws" / "raw" / "rec.wav")
    url = "/v1/workspaces/ws/audio/raw/rec.wav"

    got = client.get(f"{url}?start=1&duration=0.5", headers=AUTH)
    assert got.status_code == 200, got.text
    window, sr = _decode(got)
    assert sr == 16000 and len(window) == 8000
    assert np.allclose(window, samples[16000:24000])
    assert got.headers["x-bioamla-start"] == "1.0"
    assert got.headers["x-bioamla-duration"] == "0.5"
    assert got.headers["x-bioamla-sample-rate"] == "16000"
    assert got.headers["x-bioamla-channels"] == "1"

    # Cut short at the end of the recording, and resampled on request.
    short = client.get(f"{url}?start=2.9&duration=0.5", headers=AUTH)
    assert short.status_code == 200 and len(_decode(short)[0]) == 1600
    assert short.headers["x-bioamla-duration"] == "0.1"
    resampled = client.get(f"{url}?start=0&duration=1&sample_rate=8000", headers=AUTH)
    assert resampled.status_code == 200 and _decode(resampled)[1] == 8000
    assert len(_decode(resampled)[0]) == 8000

    # Deep into a long recording the start comes back exactly, not rounded.
    deep = client.get(f"{url}?start=2.123456789&duration=0.1", headers=AUTH)
    assert deep.headers["x-bioamla-start"] == "2.123456789"

    # Bad windows are 400s with the reason; a missing file is a 404.
    past = client.get(f"{url}?start=5&duration=1", headers=AUTH)
    assert past.status_code == 400
    assert past.json()["error"]["message"] == "start 5 s is past the end of raw/rec.wav (3 s long)"
    tiny = client.get(f"{url}?start=1&duration=0.00001", headers=AUTH)
    assert tiny.status_code == 400 and "rounds to no samples" in tiny.json()["error"]["message"]
    assert client.get(f"{url}?start=inf&duration=1", headers=AUTH).status_code in (400, 422)
    assert client.get(f"{url}?duration=1&sample_rate=2000000000", headers=AUTH).status_code in (
        400,
        422,
    )
    (root / "ws" / "raw" / "notes.csv").write_text("file,label\n")
    not_audio = client.get("/v1/workspaces/ws/audio/raw/notes.csv?duration=1", headers=AUTH)
    assert not_audio.status_code == 400, not_audio.text
    message = not_audio.json()["error"]["message"]
    assert (
        message.startswith("raw/notes.csv can't be decoded as audio") and str(root) not in message
    )
    assert client.get(f"{url}?start=0", headers=AUTH).status_code in (400, 422)  # no duration
    assert client.get(f"{url}?start=0&duration=0", headers=AUTH).status_code in (400, 422)
    assert (
        client.get("/v1/workspaces/ws/audio/raw/none.wav?duration=1", headers=AUTH).status_code
        == 404
    )
    assert client.get(f"{url}?duration=1").status_code == 401


def test_audio_window_length_is_capped_by_the_server(tmp_path):
    root = tmp_path / "root"
    create_workspace(root / "ws")
    _write_ramp(root / "ws" / "rec.wav")
    client = TestClient(create_app(ServerConfig(root=root, token="t", max_window=1.0)))
    ok = client.get("/v1/workspaces/ws/audio/rec.wav?duration=1", headers=AUTH)
    assert ok.status_code == 200
    too_long = client.get("/v1/workspaces/ws/audio/rec.wav?duration=1.5", headers=AUTH)
    assert too_long.status_code == 400 and "limit of 1 s" in too_long.json()["error"]["message"]


def test_audio_window_respects_the_path_policy(server):
    client, root, archive, outside = server
    _write_ramp(archive / "far.wav")
    _write_ramp(outside / "away.wav")
    linked = client.get(f"/v1/workspaces/ws/audio/{archive}/far.wav?duration=0.1", headers=AUTH)
    assert linked.status_code == 200
    refused = client.get(f"/v1/workspaces/ws/audio/{outside}/away.wav?duration=0.1", headers=AUTH)
    assert refused.status_code == 403


def test_the_window_cap_must_be_a_finite_positive_number(tmp_path):
    from bioamla.exceptions import InvalidInputError

    for bad in (float("inf"), float("nan"), 0.0, -1.0):
        with pytest.raises(InvalidInputError, match="max_window must be a positive number"):
            ServerConfig(root=tmp_path, token="t", max_window=bad)


def test_a_decoder_the_server_lacks_is_a_logged_500(server, monkeypatch, caplog):
    """A DependencyError is the server's problem: the client gets the reason, and so does
    the server log, where the operator can act on it."""
    import logging

    import bioamla.audio
    from bioamla.exceptions import DependencyError

    client, root, _, _ = server
    _write_ramp(root / "ws" / "rec.wav")

    def no_ffmpeg(*args, **kwargs):
        raise DependencyError("ffmpeg is required to read .mp3 files but was not found on PATH.")

    monkeypatch.setattr(bioamla.audio, "load_audio_data", no_ffmpeg)
    with caplog.at_level(logging.ERROR, logger="bioamla.server.app"):
        response = client.get("/v1/workspaces/ws/audio/rec.wav?duration=1", headers=AUTH)
    assert response.status_code == 500
    assert response.json()["error"] == {
        "type": "DependencyError",
        "message": "ffmpeg is required to read .mp3 files but was not found on PATH.",
    }
    assert "DependencyError handling GET /v1/workspaces/ws/audio/rec.wav" in caplog.text
    assert "ffmpeg is required" in caplog.text


@pytest.mark.skipif(os.name == "nt", reason="POSIX file modes")
def test_atomic_file_writer_gives_files_the_usual_mode(tmp_path):
    """A file the writer puts in place has the mode a plain open() would give it (0666 less
    the umask), whether it is new or replaces another; mkstemp's owner-only 0600 would make
    an overwritten download unreadable to anyone else."""
    import stat

    from bioamla.common.files import AtomicFileWriter

    target = tmp_path / "a.bin"
    before = os.umask(0o022)
    try:
        for overwrite in (False, True):
            with AtomicFileWriter(target, overwrite=overwrite) as w:
                w.write(b"x")
                w.commit()
            assert stat.S_IMODE(target.stat().st_mode) == 0o644, overwrite
    finally:
        os.umask(before)


def test_a_bad_decode_timeout_stops_the_server_at_start(tmp_path, monkeypatch):
    from bioamla.exceptions import ConfigError

    monkeypatch.setenv("BIOAMLA_DECODE_TIMEOUT", "soon")
    with pytest.raises(ConfigError, match="BIOAMLA_DECODE_TIMEOUT"):
        create_app(ServerConfig(root=tmp_path, token="t"))
