"""Evals for the first-pass labeling skill (``skills/label``), with a tiny local AST."""

from __future__ import annotations

import json

import numpy as np
import pytest
import soundfile as sf

from tests.skills.conftest import SR, tone, write_wav

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
    assert loud["margin"] is None
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
