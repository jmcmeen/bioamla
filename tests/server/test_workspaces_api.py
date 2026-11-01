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
