"""Operation routes: one endpoint per remote-safe operation."""

import json

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from bioamla import operations  # noqa: E402
from bioamla.server import ServerConfig, create_app  # noqa: E402
from bioamla.workspace import create_workspace, open_workspace  # noqa: E402

AUTH = {"Authorization": "Bearer t"}


@pytest.fixture
def server(tmp_path):
    root = tmp_path / "root"
    create_workspace(root / "ws")
    outside = tmp_path / "outside"
    outside.mkdir()
    client = TestClient(create_app(ServerConfig(root=root, token="t")))
    return client, root / "ws", outside


def post(client, name, body, ws="ws", headers=AUTH):
    return client.post(f"/v1/workspaces/{ws}/operations/{name}", json=body, headers=headers)


def test_every_remote_operation_has_a_route_and_local_ones_none(server):
    client, _, _ = server
    listed = client.get("/v1/operations", headers=AUTH).json()["result"]
    names = {op["name"] for op in listed["operations"]}
    ops, _ = operations.all_operations()
    assert names == {op.name for op in ops if op.remote}
    for local in ("audio.play", "util.download", "system.deps-install", "catalogs.hf.cache-purge"):
        assert local not in names
        assert post(client, local, {}).status_code == 404
    energy = next(op for op in listed["operations"] if op["name"] == "detect.energy")
    assert energy["kind"] == "job" and "file" in energy["path_fields"]
    assert energy["request_schema"]["properties"]["file"]["format"] == "path"


def test_quick_operation_runs_in_the_workspace(server, test_audio_path):
    client, ws, _ = server
    (ws / "rec.wav").write_bytes(open(test_audio_path, "rb").read())
    got = post(client, "audio.info", {"file": "rec.wav"})
    assert got.status_code == 200, got.text
    body = got.json()
    assert body["ok"] and body["result"]["file"] == str(ws / "rec.wav")
    assert body["result"]["info"]["sample_rate"] == 16000
    record = open_workspace(ws).history()[-1]
    assert (record["operation"], record["ok"], record["via"]) == ("audio.info", True, "server")
    assert record["user"] == "owner"


def test_paths_must_stay_inside_the_allowed_folders(server):
    client, _, outside = server
    (outside / "x.wav").write_bytes(b"RIFF")
    for path in (str(outside / "x.wav"), "../../outside/x.wav"):
        got = post(client, "audio.info", {"file": path})
        assert got.status_code == 403, path
        assert got.json()["error"]["type"] == "AccessDenied"


def test_requests_are_validated(server):
    client, _, _ = server
    got = post(client, "audio.info", {"file": "x.wav", "bogus": 1})
    assert got.status_code == 422
    assert "bogus" in got.json()["error"]["message"]
    assert post(client, "audio.info", {}).status_code == 422  # file is required


def test_failures_are_recorded_and_reported(server):
    client, ws, _ = server
    got = post(client, "workspace.remove", {"artifact_id": "nope"})
    assert got.status_code == 404
    record = open_workspace(ws).history()[-1]
    assert record["ok"] is False and record["error"]["type"] == "NotFoundError"


def test_workspace_and_token_checks(server):
    client, _, _ = server
    assert post(client, "workspace.status", {}, ws="nope").status_code == 404
    assert post(client, "workspace.status", {}, ws=".hidden").status_code == 400
    assert post(client, "workspace.status", {}, headers={}).status_code == 401
    assert client.get("/v1/operations").status_code == 401


def test_spec_documents_each_request(server):
    client, _, _ = server
    spec = client.get("/v1/openapi.json", headers=AUTH).json()
    path = spec["paths"]["/v1/workspaces/{ws}/operations/audio.info"]["post"]
    ref = path["requestBody"]["content"]["application/json"]["schema"]["$ref"]
    schema = spec["components"]["schemas"][ref.rsplit("/", 1)[1]]
    assert "file" in schema["properties"]
    assert json.dumps(path["security"]) != "[]"


def test_workspace_operations_stay_in_the_run_workspace(server, tmp_path):
    client, ws, _ = server
    create_workspace(ws.parent / "other")
    create_workspace(tmp_path)  # a workspace around the server root
    for directory in ("../other", ".."):
        got = post(client, "workspace.list", {"directory": directory})
        assert got.status_code == 400, directory
        assert "not inside the workspace" in got.json()["error"]["message"]
    got = post(client, "workspace.list", {})
    assert got.json()["result"]["workspace"] == str(ws)


def test_unexpected_failures_are_recorded(server, mocker):
    client, ws, _ = server
    mocker.patch("bioamla.audio.operations.describe_audio_file", side_effect=RuntimeError("x"))
    (ws / "rec.wav").write_bytes(b"RIFF")
    client = TestClient(client.app, raise_server_exceptions=False)
    got = post(client, "audio.info", {"file": "rec.wav"})
    assert got.status_code == 500
    record = open_workspace(ws).history()[-1]
    assert (record["operation"], record["ok"]) == ("audio.info", False)
    assert record["error"]["type"] == "InternalError"


def test_reads_of_the_workspace_are_not_recorded(server):
    client, ws, _ = server
    for name in ("workspace.status", "workspace.history", "workspace.list"):
        assert post(client, name, {}).status_code == 200
    assert open_workspace(ws).history() == []
