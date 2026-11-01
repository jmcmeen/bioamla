"""Evals for the first-pass labeling skill (``skills/label``), with a tiny local AST."""

from __future__ import annotations

import json

import numpy as np
import pytest

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
