"""Eval for the inventory example skill: known files, known verdicts."""

import json

import pytest

from tests.skills.conftest import SKILLS, tone, write_wav

pytestmark = pytest.mark.integration


def test_inventory_flags_short_and_unreadable_files(run_skill, workspace):
    write_wav(workspace / "rec" / "long.wav", tone(2.0))
    write_wav(workspace / "rec" / "short.wav", tone(0.3))
    (workspace / "rec" / "broken.wav").write_bytes(b"RIFF\x00\x00\x00\x00WAVEnope")

    code, summary = run_skill("inventory", "inventory", "rec", "--min-duration", "1", cwd=workspace)

    assert code == 0, summary
    assert summary["files"] == 2
    reasons = {r["path"]: r["reason"] for r in summary["review"]}
    assert set(reasons) == {"rec/short.wav", "rec/broken.wav"}
    assert reasons["rec/short.wav"] == "shorter than 1.0s"
    assert reasons["rec/broken.wav"].startswith("unreadable:")
    saved = json.loads((workspace / summary["run_dir"] / "inventory.json").read_text())
    assert {f["path"] for f in saved["files"]} == {"rec/long.wav", "rec/short.wav"}
    assert summary["run_dir"].startswith("runs/inventory/")


def test_a_bioamla_error_is_reported_as_json(run_skill, workspace):
    code, summary = run_skill("inventory", "inventory", "missing-folder", cwd=workspace)
    assert code == 1
    assert "error" in summary


def test_skills_share_one_helper():
    """Every skill carries the same _bioamla.py (a skill folder is installed alone)."""
    copies = {p.read_text() for p in SKILLS.glob("*/scripts/_bioamla.py")}
    assert len(copies) == 1


def test_every_skill_has_a_skill_md_with_name_and_description():
    for skill in [p for p in SKILLS.iterdir() if p.is_dir()]:
        text = (skill / "SKILL.md").read_text()
        assert text.startswith("---\nname: ")
        assert "\ndescription: " in text.split("---")[1]
