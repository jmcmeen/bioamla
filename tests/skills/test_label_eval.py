"""End-to-end eval of the labeling skill on a synthetic workspace with known events.

The fixture is a folder of recordings over a noise floor whose events are known
exactly: strong ones that should become annotations, a faint one that should go to
review, a recording with nothing in it, and an event a person already labeled. The
skill runs all five steps with fixed thresholds, as an agent would, and passes when:

- every strong event is written as an agent annotation, and nothing else is;
- the faint event and the noise-only recording, and only they, are in the review queue,
  held back for their SNR (the energy detector's threshold is relative to a recording's
  loudest frame, so a recording of nothing but noise is one long event, with no
  quieter frames to measure its SNR against);
- the person's row is unchanged and no agent row overlaps it.

The model is the tiny local AST, so the labels it suggests are arbitrary; the model
confidence threshold is 0 and both its labels are in the vocabulary. The eval checks
what the skill decides from its measurements, not the model's accuracy.
"""

from __future__ import annotations

import csv
import json

import numpy as np
import pytest

from tests.skills.conftest import SR, tone, write_wav

pytestmark = pytest.mark.integration

NOISE = 0.002
#: file -> [(start, end, freq, amplitude, expected)], expected: annotate | review | person
EVENTS = {
    "rec/a.wav": [(0.5, 1.0, 2000, 0.5, "annotate"), (2.5, 3.0, 2000, 0.02, "review")],
    "rec/b.wav": [(1.5, 2.2, 3000, 0.4, "annotate")],
    "rec/c.wav": [],
    "rec/d.wav": [(1.0, 1.4, 2500, 0.5, "person")],
}
DURATION = 4.0
NOISE_ONLY = ("rec/c.wav", 0.0, DURATION)
PERSON_ROW = {
    "file": "rec/d.wav", "start_time": "0.95", "end_time": "1.45", "low_freq": "2000",
    "high_freq": "3000", "label": "wren", "labeled_by": "jm",
}  # fmt: skip
THRESHOLDS = ["--min-confidence", "0", "--min-snr", "30"]


def write_known(ws) -> None:
    """The recordings and the person's annotation into the workspace folder ``ws``."""
    rng = np.random.default_rng(1)
    for name, events in EVENTS.items():
        samples = rng.normal(0, NOISE, int(DURATION * SR)).astype(np.float32)
        for start, end, freq, amp, _ in events:
            i = int(start * SR)
            samples[i : i + int((end - start) * SR)] += tone(end - start, freq, amp)
        write_wav(ws / name, samples)
    with (ws / "annotations.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(PERSON_ROW))
        writer.writeheader()
        writer.writerow(PERSON_ROW)


@pytest.fixture
def known_workspace(workspace):
    write_known(workspace)
    return workspace


def expected(kind: str) -> list[tuple[str, float, float]]:
    found = [(f, s, e) for f, evs in EVENTS.items() for s, e, *_, k in evs if k == kind]
    return found + [NOISE_ONLY] if kind == "review" else found


def matches(found: tuple[str, float, float], want: tuple[str, float, float]) -> bool:
    return found[0] == want[0] and abs(found[1] - want[1]) < 0.1 and abs(found[2] - want[2]) < 0.1


def run_all(step, model) -> tuple[str, dict, dict]:
    """Run the five steps as an agent would; returns the run folder and the annotate
    and review summaries."""
    found = step("candidates", "rec", "--model", model, "--labels", "a,b",
                 "--threshold-db", "-40")  # fmt: skip
    assert found["files"] == 4
    run_dir = found["run_dir"]
    step("evidence", run_dir)
    step("crops", run_dir)
    return run_dir, step("annotate", run_dir, *THRESHOLDS), step("review", run_dir)


def check_outcome(ws, run_dir: str, annotated: dict, queued: dict) -> None:
    """The decisions, as the files in workspace folder ``ws`` record them."""
    # Every candidate is a known event (or the noise-only recording), and none is missed.
    candidates = json.loads((ws / run_dir / "candidates.json").read_text())["candidates"]
    spans = [(c["file"], c["start_time"], c["end_time"]) for c in candidates]
    known = [(f, s, e) for f, evs in EVENTS.items() for s, e, *_ in evs] + [NOISE_ONLY]
    assert all(any(matches(c, k) for k in known) for c in spans), spans
    assert len(spans) == len(known)

    with (ws / "annotations.csv").open(newline="") as f:
        rows = list(csv.DictReader(f))
    agent = [(r["file"], float(r["start_time"]), float(r["end_time"]))
             for r in rows if r["labeled_by"] == "agent"]  # fmt: skip
    assert len(agent) == len(expected("annotate"))
    assert all(any(matches(a, want) for a in agent) for want in expected("annotate"))
    (person,) = [r for r in rows if r["labeled_by"] != "agent"]
    assert {k: person[k] for k in PERSON_ROW} == PERSON_ROW
    assert annotated["skipped_overlapping_person"] == 1

    items = json.loads((ws / queued["queue"]).read_text())["items"]
    assert [(i["file"], i["start_time"], i["end_time"]) for i in items] == [
        next(s for s in spans if matches(s, want)) for want in expected("review")
    ]
    assert queued["reasons"] == {"snr": 2}
    assert items[1]["reasons"] == ["SNR could not be measured"]
    assert all((ws / i["image"]).exists() for i in items)


def test_label_skill_end_to_end(run_skill, known_workspace, tiny_ast) -> None:
    ws = known_workspace

    def step(script: str, *args) -> dict:
        code, out = run_skill("label", script, *args, cwd=ws)
        assert code == 0, (script, out)
        return out

    run_dir, annotated, queued = run_all(step, tiny_ast)
    assert annotated["remote"] is None
    check_outcome(ws, run_dir, annotated, queued)


def test_label_skill_on_a_server_workspace(
    run_skill, workspace, tiny_ast, live_server, tmp_path, monkeypatch
) -> None:
    """The same run from a laptop against a server workspace (``--remote``): the
    recordings and the person's annotation are on the server, and everything the skill
    writes lands there too, with the same decisions. The laptop's own annotations file
    is left alone."""
    import shutil

    from bioamla.client import add_remote

    url, token, root = live_server
    server_ws = root / "ws"
    write_known(server_ws)
    shutil.copytree(tiny_ast, server_ws / "model")
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    monkeypatch.setenv("BIOAMLA_LAB_TOKEN", token)
    monkeypatch.delenv("BIOAMLA_REMOTE", raising=False)
    # Load model weights on one thread in the server's job workers: transformers'
    # threaded loading can deadlock in safetensors (bioamla issue #291).
    monkeypatch.setenv("HF_DEACTIVATE_ASYNC_LOAD", "1")
    add_remote("lab", url)
    laptop = workspace  # a local workspace with an unrelated annotations file
    (laptop / "annotations.csv").write_text("file,label\nmine.wav,owl\n")

    def step(script: str, *args) -> dict:
        code, out = run_skill("label", script, *args, "--remote", "lab:ws", cwd=laptop)
        assert code == 0, (script, out)
        return out

    run_dir, annotated, queued = run_all(step, "model")
    assert annotated["remote"]["remote"] == "lab"
    check_outcome(server_ws, run_dir, annotated, queued)
    for name in ("candidates.json", "evidence.json", "crops.json", "decisions.json"):
        assert (server_ws / run_dir / name).is_file(), name
    assert (laptop / "annotations.csv").read_text() == "file,label\nmine.wav,owl\n"
    assert not (laptop / "rec").exists()  # the recordings stayed on the server

    # A second run merges into the server's file: the person's row is still kept, and
    # this run's agent rows replace the first run's.
    _, again, _ = run_all(step, "model")
    assert again["person_rows_kept"] == 1
    check_outcome(server_ws, run_dir, annotated, queued)
