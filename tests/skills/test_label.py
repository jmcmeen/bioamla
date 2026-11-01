"""Evals for the first-pass labeling skill (``skills/label``), with a tiny local AST."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from tests.skills.conftest import SR, skill_script, tone, write_wav

pytestmark = pytest.mark.integration


def burst(total: float, start: float, length: float, freq: float, amp: float) -> np.ndarray:
    """Silence with one tone burst."""
    samples = np.zeros(int(total * SR), dtype=np.float32)
    i = int(start * SR)
    samples[i : i + int(length * SR)] = tone(length, freq, amp)
    return samples


@pytest.fixture
def recordings(workspace):
    write_wav(workspace / "rec" / "one.wav", burst(3.0, 1.0, 0.5, 2000, 0.5))
    write_wav(workspace / "rec" / "quiet.wav", np.zeros(SR * 2, dtype=np.float32))
    return workspace


def test_candidates_pair_each_event_with_a_prediction(run_skill, recordings, tiny_ast) -> None:
    code, out = run_skill(
        "label", "candidates", "rec", "--model", tiny_ast, "--labels", "a",
        cwd=recordings,
    )  # fmt: skip
    assert code == 0, out
    assert out["files"] == 2 and out["candidates"] == 1
    data = json.loads((recordings / out["run_dir"] / "candidates.json").read_text())
    (c,) = data["candidates"]
    assert c["id"] == "c0001" and c["file"].endswith("one.wav")
    assert c["start_time"] == pytest.approx(1.0, abs=0.1)
    assert c["end_time"] == pytest.approx(1.5, abs=0.1)
    assert c["label"] in {"a", "b"} and 0 <= c["model_confidence"] <= 1
    assert c["in_vocabulary"] == (c["label"] == "a")
    assert out["outside_vocabulary"] == int(not c["in_vocabulary"])


def test_candidates_report_a_bioamla_error(run_skill, workspace) -> None:
    code, out = run_skill(
        "label", "candidates", "missing", "--model", "m", "--labels", "a", cwd=workspace
    )
    assert code == 1 and "not found" in out["error"]


@pytest.fixture
def loud_and_faint(workspace):
    """One recording over a noise floor: a loud burst at 1 s, a faint one at 3 s."""
    rng = np.random.default_rng(0)
    samples = burst(5.0, 1.0, 0.5, 2000, 0.5) + burst(5.0, 3.0, 0.5, 2000, 0.02)
    samples += rng.normal(0, 0.002, samples.size).astype(np.float32)
    write_wav(workspace / "rec" / "two.wav", samples)
    return workspace


def candidates_run(run_skill, workspace, model, *extra) -> str:
    code, out = run_skill(
        "label", "candidates", "rec", "--model", model, "--labels", "a", *extra, cwd=workspace
    )
    assert code == 0, out
    return out["run_dir"]


def test_evidence_measures_snr_per_candidate(run_skill, loud_and_faint, tiny_ast) -> None:
    run_dir = candidates_run(run_skill, loud_and_faint, tiny_ast, "--threshold-db", "-40")
    code, out = run_skill("label", "evidence", run_dir, cwd=loud_and_faint)
    assert code == 0, out
    assert out["candidates"] == 2 and out["unmeasured_snr"] == 0
    evidence = json.loads((loud_and_faint / run_dir / "evidence.json").read_text())
    loud, faint = evidence["candidates"]
    assert loud["snr_db"] > faint["snr_db"] > 0
    assert loud["snr_db"] > 30
    # tiny_ast has two labels: the margin is top-1 minus top-2, i.e. 2 * confidence - 1.
    assert loud["margin"] == pytest.approx(2 * loud["model_confidence"] - 1, abs=1e-6)
    assert loud["duration"] == pytest.approx(0.5, abs=0.1)
    assert {"detector_score", "model_confidence", "low_freq", "high_freq"} <= set(loud)


def test_evidence_without_candidates_is_an_error(run_skill, workspace) -> None:
    code, out = run_skill("label", "evidence", "runs/label/none", cwd=workspace)
    assert code == 1 and "No candidates" in out["error"]


def test_crops_cut_a_clip_and_image_per_candidate(run_skill, loud_and_faint, tiny_ast) -> None:
    run_dir = candidates_run(run_skill, loud_and_faint, tiny_ast, "--threshold-db", "-40")
    code, out = run_skill("label", "crops", run_dir, "--pad", "0.25", cwd=loud_and_faint)
    assert code == 0, out
    assert out["crops"] == 2
    crops = json.loads((loud_and_faint / run_dir / "crops.json").read_text())["crops"]
    for c in crops:
        assert (loud_and_faint / c["image"]).read_bytes()[:4] == b"\x89PNG"
        clip = sf.info(loud_and_faint / c["clip"])
        assert clip.duration == pytest.approx(c["box"]["end"] + 0.25, abs=0.02)
        assert c["box"]["start"] == pytest.approx(0.25, abs=0.01)
        assert c["image"].startswith(run_dir)
    assert crops[1]["clip_start"] == pytest.approx(2.75, abs=0.1)
    # The event's box is drawn: the image differs from the plain spectrogram of its clip.
    plain = loud_and_faint / "plain.png"
    subprocess.run([str(Path(sys.executable).parent / "bioamla"), "audio", "visualize",
                    crops[0]["clip"], "--type", "stft", "--output", str(plain)],
                   cwd=loud_and_faint, check=True, capture_output=True)  # fmt: skip
    assert plain.read_bytes() != (loud_and_faint / crops[0]["image"]).read_bytes()


def test_box_option_formats_the_box_for_audio_visualize() -> None:
    import importlib.util

    spec = importlib.util.spec_from_file_location("crops", skill_script("label", "crops"))
    crops = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(skill_script("label", "crops").parent))
    spec.loader.exec_module(crops)
    box = {"start": 0.25, "end": 0.75, "low_freq": 500.0, "high_freq": 5000.0}
    assert crops.box_option(box, "frog") == "0.25,0.75,500,5000,frog"
    assert crops.box_option({**box, "low_freq": None}) == "0.25,0.75"
    assert crops.box_option(box, "a,b") == "0.25,0.75,500,5000"


def read_rows(path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


PERSON_ROWS = (
    "file,start_time,end_time,low_freq,high_freq,label,reviewer,labeled_by\n"
    'rec/two.wav,2.9,3.6,500,5000,"frog, faint",jm,\n'
    "rec/other.wav,0.5,1.0,500,5000,bird,jm,jm\n"
    "rec/elsewhere.wav,0.1,0.2,500,5000,a,,agent\n"
)


def evidence_run(run_skill, workspace, model) -> str:
    run_dir = candidates_run(
        run_skill, workspace, model, "--threshold-db", "-40", "--labels", "a,b"
    )
    code, out = run_skill("label", "evidence", run_dir, cwd=workspace)
    assert code == 0, out
    return run_dir


def test_annotate_never_touches_rows_a_person_made(run_skill, loud_and_faint, tiny_ast) -> None:
    ws = loud_and_faint
    (ws / "annotations.csv").write_text(PERSON_ROWS)
    person = [r for r in read_rows(ws / "annotations.csv") if r["labeled_by"] != "agent"]
    for _ in range(2):
        run_dir = evidence_run(run_skill, ws, tiny_ast)
        code, out = run_skill(
            "label", "annotate", run_dir, "--min-confidence", "0", "--min-snr", "0", cwd=ws
        )
        assert code == 0, out
        assert out["accepted"] == 2 and out["written"] == 1
        assert out["skipped_overlapping_person"] == 1 and out["person_rows_kept"] == 2
        rows = read_rows(ws / "annotations.csv")
        # Same values in every original column (new columns are added empty).
        assert [{k: r[k] for k in person[0]} for r in rows if r["labeled_by"] != "agent"] == person
        assert all(r[k] == "" for r in rows[:2] for k in r.keys() - person[0].keys())
        agent = [r for r in rows if r["labeled_by"] == "agent"]
        # The other recording's agent row stays; this run's replaces the last run's.
        assert [r["file"] for r in agent] == ["rec/elsewhere.wav", "rec/two.wav"]
    (row,) = [r for r in agent if r["file"] == "rec/two.wav"]
    assert float(row["start_time"]) == pytest.approx(1.0, abs=0.1)
    assert row["method"].startswith("energy+ast:") and row["label"] in {"a", "b"}
    assert 0 < float(row["confidence"]) <= 1
    decisions = json.loads((ws / run_dir / "decisions.json").read_text())["decisions"]
    assert decisions[1]["reasons"] == ["overlaps a row a person labeled"]


def test_annotate_holds_back_candidates_below_threshold(
    run_skill, loud_and_faint, tiny_ast
) -> None:
    run_dir = evidence_run(run_skill, loud_and_faint, tiny_ast)
    code, out = run_skill(
        "label", "annotate", run_dir, "--min-confidence", "0", "--min-snr", "30",
        cwd=loud_and_faint,
    )  # fmt: skip
    assert code == 0, out
    assert out["accepted"] == 1 and out["not_accepted"] == 1
    decisions = json.loads((loud_and_faint / run_dir / "decisions.json").read_text())
    faint = decisions["decisions"][1]
    assert not faint["accepted"] and faint["reasons"][0].startswith("SNR 1")
    assert len(read_rows(loud_and_faint / "annotations.csv")) == 1


def test_review_queues_what_was_held_back(run_skill, loud_and_faint, tiny_ast) -> None:
    ws = loud_and_faint
    run_dir = evidence_run(run_skill, ws, tiny_ast)
    assert run_skill("label", "crops", run_dir, cwd=ws)[0] == 0
    code, out = run_skill(
        "label", "annotate", run_dir, "--min-confidence", "0", "--min-snr", "30", cwd=ws
    )
    assert code == 0 and out["not_accepted"] == 1
    code, out = run_skill("label", "review", run_dir, cwd=ws)
    assert code == 0, out
    assert out["queued"] == 1 and out["with_crops"] == 1 and out["reasons"] == {"snr": 1}
    (item,) = json.loads((ws / out["queue"]).read_text())["items"]
    assert item["id"] == "c0002" and item["start_time"] == pytest.approx(3.0, abs=0.1)
    assert item["evidence"]["snr_db"] < 30 and item["suggested_label"] in {"a", "b"}
    assert (ws / item["image"]).exists() and (ws / item["clip"]).exists()


def test_review_needs_the_decisions(run_skill, loud_and_faint, tiny_ast) -> None:
    run_dir = evidence_run(run_skill, loud_and_faint, tiny_ast)
    code, out = run_skill("label", "review", run_dir, cwd=loud_and_faint)
    assert code == 1 and "decisions.json" in out["error"]


def test_an_event_no_segment_covers_gets_no_label(monkeypatch) -> None:
    import argparse
    import importlib.util

    from tests.skills.conftest import skill_script

    spec = importlib.util.spec_from_file_location("candidates", skill_script("label", "candidates"))
    candidates = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(candidates)
    answers = {
        "detect": {"detections": [{"start_time": 5.0, "end_time": 5.5, "frequency_low": 500,
                                   "frequency_high": 5000, "confidence": 0.9}]},
        "models": {"predictions": [{"start_time": 0.0, "end_time": 3.0, "label": "a",
                                    "confidence": 0.99, "margin": 0.98}]},
    }  # fmt: skip
    monkeypatch.setattr(candidates, "run", lambda *args, **kw: answers[args[0]])
    args = argparse.Namespace(low_freq=500, high_freq=5000, threshold_db=-20, min_duration=0.05,
                              model="m", segment_seconds=3, labels=["a"])  # fmt: skip
    (row,) = candidates.file_candidates("rec.wav", args, ".")
    assert row["label"] is None and row["model_confidence"] is None and row["margin"] is None
