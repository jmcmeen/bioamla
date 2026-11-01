"""Jobs: job-kind operations run in worker processes, with progress and cancel."""

import json
import shutil
import time

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from bioamla import operations  # noqa: E402
from bioamla.common.jsonable import to_jsonable  # noqa: E402
from bioamla.server import ServerConfig, create_app  # noqa: E402
from bioamla.server.jobs import is_ml  # noqa: E402
from bioamla.workspace import create_workspace, open_workspace  # noqa: E402

AUTH = {"Authorization": "Bearer t"}


@pytest.fixture
def server(tmp_path, test_audio_path):
    root = tmp_path / "root"
    ws = root / "ws"
    create_workspace(ws)
    shutil.copy(test_audio_path, ws / "rec.wav")
    with TestClient(create_app(ServerConfig(root=root, token="t", max_jobs=1))) as client:
        yield client, ws


def submit(client, name, body):
    got = client.post(f"/v1/workspaces/ws/operations/{name}", json=body, headers=AUTH)
    assert got.status_code == 202, got.text
    return got.json()["result"]


def wait(client, job_id, until=("succeeded", "failed", "cancelled"), timeout=60):
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = client.get(f"/v1/jobs/{job_id}", headers=AUTH).json()["result"]
        if job["status"] in until:
            return job
        time.sleep(0.05)
    raise AssertionError(f"job {job_id} still {job['status']}")


def events(client, job_id, last=0):
    headers = {**AUTH, "Last-Event-ID": str(last)} if last else AUTH
    found = []
    with client.stream("GET", f"/v1/jobs/{job_id}/events", headers=headers) as response:
        assert response.headers["content-type"].startswith("text/event-stream")
        for line in response.iter_lines():
            if line.startswith("data: "):
                found.append(json.loads(line[len("data: ") :]))
    return found


@pytest.mark.slow
def test_a_job_gives_the_same_result_as_a_local_run(server):
    client, ws = server
    job = submit(client, "detect.energy", {"file": "rec.wav"})
    assert job["status"] == "queued" and job["operation"] == "detect.energy"
    done = wait(client, job["id"])
    assert done["status"] == "succeeded", done["error"]
    local = operations.run("detect.energy", {"file": str(ws / "rec.wav")})
    assert done["result"] == json.loads(json.dumps(to_jsonable(local.to_dict())))

    record = open_workspace(ws).history()[-1]
    assert (record["operation"], record["ok"], record["job"]) == ("detect.energy", True, job["id"])
    listed = client.get("/v1/jobs?workspace=ws", headers=AUTH).json()["result"]
    assert [j["id"] for j in listed] == [job["id"]]


@pytest.mark.slow
def test_progress_streams_as_server_sent_events(server):
    client, ws = server
    for i in range(3):
        shutil.copy(ws / "rec.wav", ws / f"rec{i}.wav")
    job = submit(client, "batch.audio.info", {"input_dir": ".", "output_dir": "out"})
    seen = events(client, job["id"])
    kinds = [e["event"] for e in seen]
    assert kinds[0] == "status" and seen[0]["status"] == "queued"
    assert "progress" in kinds
    assert seen[-1]["event"] == "status" and seen[-1]["status"] == "succeeded"
    assert [e["id"] for e in seen] == sorted(e["id"] for e in seen)
    last = next(e for e in seen if e["event"] == "progress")
    assert last["total"] == 4
    # Resuming after an event only sends what came after it.
    assert events(client, job["id"], last=seen[-2]["id"]) == seen[-1:]
    assert wait(client, job["id"])["progress"]["done"] == 4


@pytest.mark.slow
def test_cancel_stops_a_running_job_and_drops_a_queued_one(server):
    client, ws = server
    client.app.state.jobs.cancel_grace = 0.2
    running = submit(client, "batch.audio.info", {"input_dir": "."})
    queued = submit(client, "detect.energy", {"file": "rec.wav"})  # max_jobs=1: waits
    wait(client, running["id"], until=("running",))
    got = client.post(f"/v1/jobs/{queued['id']}/cancel", headers=AUTH).json()["result"]
    assert got["status"] == "cancelled"
    client.post(f"/v1/jobs/{running['id']}/cancel", headers=AUTH)
    stopped = wait(client, running["id"])
    assert stopped["status"] == "cancelled"
    assert stopped["error"]["type"] == "OperationCancelled"
    record = open_workspace(ws).history()[-1]
    assert record["job"] == running["id"] and record["ok"] is False
    assert all(r.get("job") != queued["id"] for r in open_workspace(ws).history())


@pytest.mark.slow
def test_a_failing_job_reports_its_error(server):
    client, ws = server
    (ws / "bad.wav").write_text("not audio")
    job = submit(client, "detect.energy", {"file": "bad.wav"})
    done = wait(client, job["id"])
    assert done["status"] == "failed" and done["error"]["type"] != "WorkerError"
    assert open_workspace(ws).history()[-1]["ok"] is False


def test_bad_requests_are_refused_before_queueing(server, tmp_path):
    client, _ = server
    outside = tmp_path / "outside.wav"
    outside.write_bytes(b"")
    got = client.post(
        "/v1/workspaces/ws/operations/detect.energy", json={"file": str(outside)}, headers=AUTH
    )
    assert got.status_code == 403
    assert client.app.state.jobs.jobs == {}
    record = open_workspace(client.app.state.policy.root / "ws").history()[-1]
    assert record["operation"] == "detect.energy" and record["ok"] is False
    assert record["error"]["type"] == "AccessDenied"


def test_unknown_jobs_and_tokens(server):
    client, _ = server
    assert client.get("/v1/jobs/nope", headers=AUTH).status_code == 404
    assert client.post("/v1/jobs/nope/cancel", headers=AUTH).status_code == 404
    assert client.get("/v1/jobs").status_code == 401


def test_only_model_operations_count_as_ml_jobs():
    assert is_ml("models.ast.train") and is_ml("batch.models.predict")
    assert not is_ml("detect.energy") and not is_ml("batch.detect.energy")


def test_one_ml_job_at_a_time(tmp_path):
    from bioamla.server.jobs import JobRunner

    runner = JobRunner(tmp_path, max_jobs=4)
    started = []
    runner._start = lambda job: (started.append(job.operation), setattr(job, "status", "running"))

    for name in ("models.ast.train", "models.ast.predict", "detect.energy"):
        job = runner.jobs.setdefault(name, _job(name, tmp_path))
        job.created = len(runner.jobs)
    runner._start_queued()
    assert started == ["models.ast.train", "detect.energy"]


def _job(name, ws_dir):
    from bioamla.server.jobs import Job

    return Job(id=name, operation=name, workspace="ws", user="owner", request={}, ws_dir=ws_dir)


def test_a_bad_message_fails_that_job_and_the_runner_goes_on(tmp_path):
    from bioamla.server.jobs import JobRunner

    runner = JobRunner(tmp_path)
    job = _job("detect.energy", tmp_path)
    job.status = "running"

    class Conn:
        def __init__(self):
            self.sent = [("bogus",)]

        def poll(self):
            return bool(self.sent)

        def recv(self):
            return self.sent.pop()

        def close(self):
            pass

    class Proc:
        pid = None

        def is_alive(self):
            return True

        def kill(self):
            pass

    job._conn, job._process = Conn(), Proc()
    runner.jobs[job.id] = job
    assert runner._tick() == [job]
    assert job.status == "failed" and job.error["type"] == "WorkerError"
    assert job.events[-1]["status"] == "failed"


def test_old_finished_jobs_are_forgotten(tmp_path):
    from bioamla.server.jobs import JobRunner

    runner = JobRunner(tmp_path, keep_jobs=2)
    for i in range(4):
        job = _job(f"j{i}", tmp_path)
        job.status, job.finished = "succeeded", float(i)
        runner.jobs[job.id] = job
    queued = runner.jobs.setdefault("q", _job("q", tmp_path))
    runner._prune()
    assert set(runner.jobs) == {"j2", "j3", queued.id}
