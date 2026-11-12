"""JsonlTracker: a training run as JSON Lines (needs no backend)."""

import json

import pytest

from bioamla.exceptions import InvalidInputError, TrainingCancelled, TrainingError
from bioamla.ml.trackers import JsonlTracker
from bioamla.protocols import ExperimentTracker


def _lines(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_satisfies_the_protocol(tmp_path) -> None:
    assert isinstance(JsonlTracker(tmp_path / "m.jsonl"), ExperimentTracker)


@pytest.mark.parametrize("status", ["finished", "cancelled", "failed"])
def test_writes_one_line_per_call_ending_with_the_status(tmp_path, status) -> None:
    path = tmp_path / "run" / "metrics.jsonl"  # its folder is made
    tracker = JsonlTracker(path)
    tracker.start_run("run-1", {"lr": 1e-4, "aug": {"gain": True}})
    tracker.log_metrics(10, {"loss": 0.5})
    # Flushed as written: what a run measured is there before it ends.
    assert _lines(path)[-1] == {"event": "metrics", "step": 10, "metrics": {"loss": 0.5}}
    tracker.log_metrics(20, {"eval_accuracy": 0.9, "eval_loss": 0.3})
    tracker.log_artifact("/m/bioamla_run.json")
    tracker.end_run(status)
    assert _lines(path) == [
        {"event": "start", "name": "run-1", "params": {"lr": 1e-4, "aug": {"gain": True}}},
        {"event": "metrics", "step": 10, "metrics": {"loss": 0.5}},
        {"event": "metrics", "step": 20, "metrics": {"eval_accuracy": 0.9, "eval_loss": 0.3}},
        {"event": "artifact", "path": "/m/bioamla_run.json"},
        {"event": "end", "status": status},
    ]
    tracker.end_run(status)  # a second end writes nothing
    assert len(_lines(path)) == 5


def test_refuses_an_existing_file(tmp_path) -> None:
    path = tmp_path / "metrics.jsonl"
    path.write_text("")
    with pytest.raises(InvalidInputError, match="already exists"):
        JsonlTracker(path)


def test_the_file_is_created_up_front(tmp_path) -> None:
    path = tmp_path / "metrics.jsonl"
    JsonlTracker(path).close()
    assert path.exists()
    blocked = tmp_path / "file"
    blocked.write_text("")
    with pytest.raises(InvalidInputError, match="Can't create"):
        JsonlTracker(blocked / "metrics.jsonl")  # its "folder" is a file


@pytest.mark.parametrize(
    ("error", "status"),
    [(TrainingCancelled("stop"), "cancelled"), (TrainingError("bad"), "failed")],
)
def test_a_run_ending_before_training_starts_records_its_status(tmp_path, error, status) -> None:
    path = tmp_path / "metrics.jsonl"
    with pytest.raises(type(error)), JsonlTracker(path):
        raise error  # as train_ast does in preprocessing, before start_run
    assert _lines(path) == [{"event": "end", "status": status}]


def test_the_context_manager_leaves_an_ended_run_alone(tmp_path) -> None:
    path = tmp_path / "metrics.jsonl"
    with pytest.raises(TrainingError), JsonlTracker(path) as tracker:
        tracker.start_run("r", {})
        tracker.end_run("failed")
        raise TrainingError("bad")
    assert [x["event"] for x in _lines(path)] == ["start", "end"]


def test_a_reused_tracker_appends_each_run(tmp_path) -> None:
    path = tmp_path / "metrics.jsonl"
    tracker = JsonlTracker(path)
    for name in ("a", "b"):
        tracker.start_run(name, {})
        tracker.end_run("finished")
    assert [(x["event"], x.get("name")) for x in _lines(path)] == [
        ("start", "a"), ("end", None), ("start", "b"), ("end", None)
    ]  # fmt: skip


def test_values_are_strict_json(tmp_path) -> None:
    np = pytest.importorskip("numpy")
    path = tmp_path / "metrics.jsonl"
    tracker = JsonlTracker(path)
    tracker.start_run("r", {"dir": tmp_path, "labels": ("a", "b")})
    tracker.log_metrics(1, {"loss": float("nan"), "acc": np.float32(0.5), "n": np.int64(3),
                            "inf": float("inf")})  # fmt: skip
    tracker.end_run("finished")
    text = path.read_text()
    assert "NaN" not in text and "Infinity" not in text
    lines = _lines(path)
    assert lines[0]["params"] == {"dir": str(tmp_path), "labels": ["a", "b"]}
    assert lines[1]["metrics"] == {"loss": None, "acc": 0.5, "n": 3, "inf": None}
