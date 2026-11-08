"""Eval of the label-QA skill (``skills/label-qa``) on an annotation set with planted problems.

The fixture is two recordings of known length and an annotations file in which every
kind of problem the check step knows appears exactly once or twice, next to rows that
are fine, plus a person's reference labels for a subset of the recordings with a known
agreement. The skill passes when:

- the check step reports exactly the planted problems, by row and rule, and nothing else;
- the agreement step scores exactly the known hits, label disagreements, misses and
  false alarms;
- the gate fails that set for both reasons (exit 2), and passes a clean set (exit 0).
"""

from __future__ import annotations

import json

import pytest

from tests.skills.conftest import SR, tone, write_wav

pytestmark = pytest.mark.integration

HEADER = "file,start_time,end_time,low_freq,high_freq,label,labeled_by"
#: (row, rule) of every planted problem, as the check step must report them
PLANTED = {
    (0, "file"),  # row 9's recording doesn't exist (reported against the file, row 0)
    (2, "duplicate"),  # row 2 repeats row 1
    (4, "bounds"),  # row 4 ends after the 4 s recording
    (5, "bounds"),  # row 5's band is above Nyquist (a person's row: still checked)
    (6, "overlap"),  # row 6 (agent) overlaps row 5 (a person)
    (6, "vocabulary"),  # row 6's label isn't in the vocabulary
    (7, "value"),  # row 7's start_time isn't a number
}
ANNOTATIONS = [
    "rec/a.wav,0.5,1.0,2000,3000,wren,agent",  # 1 fine; matches the reference
    "rec/a.wav,0.5,1.0,2000,3000,wren,agent",  # 2 duplicate of 1
    "rec/a.wav,2.0,2.5,2000,3000,frog,agent",  # 3 fine; the reference says wren
    "rec/a.wav,3.8,4.5,2000,3000,wren,agent",  # 4 out of bounds
    "rec/a.wav,1.2,1.6,1000,9000,wren,jm",  # 5 a person's row, band above Nyquist
    "rec/a.wav,1.1,1.5,2000,3000,owl,agent",  # 6 overlaps 5, label not in vocabulary
    "rec/b.wav,x,1.0,2000,3000,wren,agent",  # 7 bad number
    "rec/b.wav,1.0,1.5,2000,3000,wren,agent",  # 8 fine; nothing in the reference there
    "rec/missing.wav,0,1,2000,3000,wren,agent",  # 9 recording doesn't exist
]
REFERENCE = [
    "rec/a.wav,0.5,1.0,2000,3000,wren,jm",  # hit (row 1)
    "rec/a.wav,2.0,2.5,2000,3000,wren,jm",  # matched to row 3, which says frog
    "rec/a.wav,3.0,3.4,2000,3000,frog,jm",  # miss
    "rec/b.wav,2.0,2.4,2000,3000,wren,jm",  # miss
]
CLEAN = [ANNOTATIONS[0], ANNOTATIONS[2], "rec/b.wav,1.0,1.5,2000,3000,frog,agent"]


def write_set(ws, rows: list[str], name: str = "annotations.csv") -> None:
    (ws / name).write_text("\n".join([HEADER, *rows]) + "\n")


def write_recordings(ws) -> None:
    write_wav(ws / "rec" / "a.wav", tone(4.0, 1000, 0.1))
    write_wav(ws / "rec" / "b.wav", tone(3.0, 1000, 0.1))


@pytest.fixture
def planted(workspace):
    write_recordings(workspace)
    write_set(workspace, ANNOTATIONS)
    write_set(workspace, REFERENCE, "reference.csv")
    return workspace


def findings(ws, run_dir: str) -> set[tuple[int, str]]:
    report = json.loads((ws / run_dir / "checks.json").read_text())
    return {(f["row"], f["rule"]) for f in report["findings"]}


def test_label_qa_finds_every_planted_problem_and_fails_the_set(run_skill, planted) -> None:
    ws = planted
    code, checked = run_skill(
        "label-qa", "check", "annotations.csv", "--labels", "wren,frog", cwd=ws
    )
    assert code == 0, checked
    assert checked["rows"] == 9 and checked["agent_rows"] == 8 and checked["person_rows"] == 1
    assert findings(ws, checked["run_dir"]) == PLANTED
    assert checked["findings"] == len(PLANTED)  # one finding per planted problem, no more
    assert checked["by_rule"] == {
        "file": 1, "duplicate": 1, "bounds": 2, "overlap": 1, "vocabulary": 1, "value": 1,
    }  # fmt: skip
    report = json.loads((ws / checked["run_dir"] / "checks.json").read_text())
    assert report["recordings"]["rec/a.wav"] == {"duration": 4.0, "sample_rate": SR}
    assert report["recordings"]["rec/missing.wav"] is None

    code, scored = run_skill("label-qa", "agreement", checked["run_dir"],
                             "--reference", "reference.csv", cwd=ws)  # fmt: skip
    assert code == 0, scored
    # Agent rows on the two reference recordings: rows 1, 2, 3, 4, 6 on a.wav and 7, 8 on
    # b.wav. Two match reference rows (1 and 3); the rest are false alarms, including the
    # rows the check step flagged.
    assert scored["reference_rows"] == 4 and scored["agent_rows"] == 7
    assert scored["matched"] == 2 and scored["label_disagreements"] == 1
    assert scored["misses"] == 2 and scored["false_alarms"] == 5
    report = json.loads((ws / checked["run_dir"] / "agreement.json").read_text())
    assert report["confusion"] == {"wren": {"wren": 1, "frog": 1}}
    assert report["per_label"]["wren"]["tp"] == 1 and report["per_label"]["wren"]["fn"] == 2
    assert report["per_label"]["frog"] == {
        "tp": 0,
        "fp": 1,
        "fn": 1,
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
    }
    assert report["per_label"]["owl"] == {
        "tp": 0, "fp": 1, "fn": 0, "precision": 0.0, "recall": None, "f1": 0.0,
    }  # fmt: skip
    assert scored["f1"] == pytest.approx(2 * 1 / (2 * 1 + 6 + 3))
    assert [m["label"] for m in report["misses"]] == ["frog", "wren"]
    (disagreement,) = report["disagreements"]
    assert disagreement["person"]["label"] == "wren" and disagreement["agent"]["label"] == "frog"
    assert disagreement["file"] == "rec/a.wav" and disagreement["iou"] == 1.0

    code, verdict = run_skill("label-qa", "gate", checked["run_dir"], "--min-f1", "0.5", cwd=ws)
    assert code == 2 and verdict["passed"] is False
    assert len(verdict["reasons"]) == 2
    assert verdict["reasons"][0].startswith("7 findings (at most 0 allowed)")
    assert verdict["reasons"][1].startswith("f1 0.182 is below 0.5")
    assert (ws / checked["run_dir"] / "gate.json").is_file()


def test_a_clean_set_passes_and_the_gate_can_insist_on_agreement(run_skill, workspace) -> None:
    ws = workspace
    write_recordings(ws)
    write_set(ws, CLEAN)
    write_set(ws, [REFERENCE[0], REFERENCE[1].replace("wren,jm", "frog,jm")], "reference.csv")
    code, checked = run_skill(
        "label-qa", "check", "annotations.csv", "--labels", "wren,frog", cwd=ws
    )
    assert code == 0 and checked["findings"] == 0, checked

    code, verdict = run_skill("label-qa", "gate", checked["run_dir"], cwd=ws)
    assert code == 0 and verdict["passed"] is True and verdict["agreement"] is None
    code, verdict = run_skill("label-qa", "gate", checked["run_dir"], "--require-agreement", cwd=ws)
    assert code == 2 and verdict["reasons"] == ["the agreement step didn't run"]

    code, scored = run_skill("label-qa", "agreement", checked["run_dir"],
                             "--reference", "reference.csv", cwd=ws)  # fmt: skip
    assert code == 0 and scored["f1"] == 1.0 and scored["false_alarms"] == 0
    code, verdict = run_skill("label-qa", "gate", checked["run_dir"], "--require-agreement",
                              "--min-precision", "0.9", cwd=ws)  # fmt: skip
    assert code == 0 and verdict["agreement"]["precision"] == 1.0


def test_check_without_the_required_columns_reports_only_that(run_skill, workspace) -> None:
    (workspace / "annotations.csv").write_text("file,label\nrec/a.wav,wren\n")
    code, checked = run_skill("label-qa", "check", "annotations.csv", cwd=workspace)
    assert code == 0 and checked["by_rule"] == {"columns": 1}
    (f,) = json.loads((workspace / checked["run_dir"] / "checks.json").read_text())["findings"]
    assert f["message"] == "missing columns: start_time, end_time"


def test_the_vocabulary_is_only_checked_when_given(run_skill, planted) -> None:
    code, checked = run_skill("label-qa", "check", "annotations.csv", cwd=planted)
    assert code == 0 and "vocabulary" not in checked["by_rule"]


def test_errors_are_json(run_skill, planted) -> None:
    code, out = run_skill("label-qa", "check", "nope.csv", cwd=planted)
    assert code == 1 and "nope.csv" in out["error"]
    code, checked = run_skill("label-qa", "check", "annotations.csv", cwd=planted)
    write_set(planted, ["rec/a.wav,0.5,1.0,2000,3000,wren,agent"], "agents-only.csv")
    code, out = run_skill("label-qa", "agreement", checked["run_dir"],
                          "--reference", "agents-only.csv", cwd=planted)  # fmt: skip
    assert code == 1 and "no usable rows a person labeled" in out["error"]
    code, out = run_skill("label-qa", "gate", "runs/label-qa/none", cwd=planted)
    assert code == 1 and "Run check.py first" in out["error"]


def test_label_qa_on_a_server_workspace(
    run_skill, workspace, live_server, tmp_path, monkeypatch
) -> None:
    """The same run from a laptop against a server workspace (``--remote``): the
    annotations, the reference and the recordings are on the server, every output lands
    there, and the laptop's own files are untouched."""
    from bioamla.client import add_remote

    url, token, root = live_server
    server_ws = root / "ws"
    write_recordings(server_ws)
    write_set(server_ws, ANNOTATIONS)
    write_set(server_ws, REFERENCE, "reference.csv")
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    monkeypatch.setenv("BIOAMLA_LAB_TOKEN", token)
    monkeypatch.delenv("BIOAMLA_REMOTE", raising=False)
    add_remote("lab", url)
    laptop = workspace
    write_set(laptop, CLEAN)  # an unrelated local file of the same name

    def step(script: str, *args) -> tuple[int, dict]:
        return run_skill("label-qa", script, *args, "--remote", "lab:ws", cwd=laptop)

    code, checked = step("check", "annotations.csv", "--labels", "wren,frog")
    assert code == 0 and checked["remote"]["remote"] == "lab", checked
    assert findings(server_ws, checked["run_dir"]) == PLANTED
    code, scored = step("agreement", checked["run_dir"], "--reference", "reference.csv")
    assert code == 0 and scored["matched"] == 2, scored
    # The server's nine-row file, not the laptop's three clean rows: seven agent rows on
    # the reference recordings and five false alarms.
    assert scored["agent_rows"] == 7 and scored["false_alarms"] == 5
    code, verdict = step("gate", checked["run_dir"])
    assert code == 2 and len(verdict["reasons"]) == 2
    for name in ("checks.json", "agreement.json", "gate.json"):
        assert (server_ws / checked["run_dir"] / name).is_file(), name
    assert (laptop / "annotations.csv").read_text() == "\n".join([HEADER, *CLEAN]) + "\n"
    assert not (laptop / "rec").exists()


def test_unusable_rows_are_value_findings_and_nothing_else(run_skill, workspace) -> None:
    """NaN times and a repeated bad row are value findings, and never reach the duplicate
    or overlap rules, which need numbers. A short row with a span but no label is a
    person's row with an empty label: a vocabulary finding, and the agent row over the
    same span overlaps it."""
    write_recordings(workspace)
    write_set(workspace, [
        "rec/a.wav,nan,nan,2000,3000,wren,agent",
        "rec/a.wav,0.5,1.0",  # short row: no band, no label, no labeled_by
        "rec/b.wav,x,1.0,2000,3000,wren,agent",
        "rec/b.wav,x,1.0,2000,3000,wren,agent",
        "rec/a.wav,0.5,1.0,2000,3000,wren,agent",  # fine, and overlaps the short row in time
    ])  # fmt: skip
    code, checked = run_skill("label-qa", "check", "annotations.csv", cwd=workspace)
    assert code == 0, checked
    assert checked["by_rule"] == {"value": 3, "vocabulary": 1, "overlap": 1}, checked["by_rule"]
    got = findings(workspace, checked["run_dir"])
    assert got == {(1, "value"), (2, "vocabulary"), (3, "value"), (4, "value"), (5, "overlap")}


def test_an_empty_vocabulary_allows_no_label(run_skill, workspace) -> None:
    write_recordings(workspace)
    write_set(workspace, CLEAN)
    code, checked = run_skill("label-qa", "check", "annotations.csv", "--labels", "", cwd=workspace)
    assert code == 0 and checked["by_rule"] == {"vocabulary": 3}


def test_agreement_ignores_short_reference_rows_and_never_matches_disjoint_spans(
    run_skill, workspace
) -> None:
    write_recordings(workspace)
    write_set(workspace, ["rec/a.wav,3.0,3.5,2000,3000,wren,agent"])
    write_set(
        workspace, ["rec/a.wav,0.5,1.0,2000,3000,wren,jm", "rec/a.wav,2.0,2.5"], "reference.csv"
    )
    code, checked = run_skill("label-qa", "check", "annotations.csv", cwd=workspace)
    code, scored = run_skill("label-qa", "agreement", checked["run_dir"], "--reference",
                             "reference.csv", "--min-iou", "0", cwd=workspace)  # fmt: skip
    assert code == 0, scored
    assert scored["reference_rows"] == 1 and scored["reference_rows_skipped"] == 1
    assert scored["matched"] == 0 and scored["misses"] == 1 and scored["false_alarms"] == 1
    assert scored["precision"] == 0.0 and scored["recall"] == 0.0 and scored["f1"] == 0.0


def test_the_gate_reports_a_metric_it_cannot_score(run_skill, workspace) -> None:
    """A reference with rows but no agent rows on its recordings: precision has nothing to
    score, and the gate says so instead of failing in its own code."""
    write_recordings(workspace)
    write_set(workspace, ["rec/a.wav,0.5,1.0,2000,3000,wren,agent"])
    write_set(workspace, ["rec/b.wav,0.5,1.0,2000,3000,wren,jm"], "reference.csv")
    code, checked = run_skill("label-qa", "check", "annotations.csv", cwd=workspace)
    code, scored = run_skill("label-qa", "agreement", checked["run_dir"], "--reference",
                             "reference.csv", cwd=workspace)  # fmt: skip
    assert code == 0 and scored["agent_rows"] == 0 and scored["precision"] is None
    code, verdict = run_skill("label-qa", "gate", checked["run_dir"], "--min-precision", "0.5",
                              cwd=workspace)  # fmt: skip
    assert code == 2
    assert verdict["reasons"] == [
        "f1 0.000 is below 0.8",
        "precision could not be scored (no agent rows on the reference recordings)",
    ]


def test_a_failed_check_leaves_no_run_folder(run_skill, workspace) -> None:
    code, out = run_skill("label-qa", "check", "nope.csv", cwd=workspace)
    assert code == 1 and not (workspace / "runs").exists()


def test_a_one_sided_band_schema_is_a_columns_finding(run_skill, workspace) -> None:
    write_recordings(workspace)
    (workspace / "annotations.csv").write_text(
        "file,start_time,end_time,low_freq,label,labeled_by\nrec/a.wav,0.5,1.0,2000,wren,agent\n"
    )
    code, checked = run_skill("label-qa", "check", "annotations.csv", cwd=workspace)
    assert code == 0 and checked["by_rule"] == {"columns": 1}, checked
    (f,) = json.loads((workspace / checked["run_dir"] / "checks.json").read_text())["findings"]
    assert f["message"] == "low_freq without high_freq: bands come as a pair"


def test_agreement_scores_the_file_the_check_step_read(run_skill, planted) -> None:
    """The annotations file changing between the steps doesn't change what the gate judges."""
    code, checked = run_skill("label-qa", "check", "annotations.csv", cwd=planted)
    write_set(planted, CLEAN)  # the live file changes after the check
    code, scored = run_skill("label-qa", "agreement", checked["run_dir"], "--reference",
                             "reference.csv", cwd=planted)  # fmt: skip
    assert code == 0 and scored["agent_rows"] == 7 and scored["false_alarms"] == 5


def test_a_failing_server_stops_the_check_instead_of_counting_as_findings(planted, tmp_path):
    """A bioamla error that isn't about the recording (a server that can't be reached, a
    refused token) is the step's error, not a file finding the gate could wave through."""
    import os
    import subprocess
    import sys

    from tests.skills.conftest import skill_script

    fake = tmp_path / "bioamla"
    fake.write_text(
        "#!/bin/sh\n"
        'echo \'{"ok": false, "error": {"type": "RemoteError", '
        '"message": "Cannot reach http://lab:8765"}}\'\n'
    )
    fake.chmod(0o755)
    proc = subprocess.run(
        [sys.executable, str(skill_script("label-qa", "check")), "annotations.csv"],
        cwd=planted, env={**os.environ, "BIOAMLA_BIN": str(fake)},
        capture_output=True, text=True, check=False,
    )  # fmt: skip
    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert json.loads(proc.stdout)["error"].startswith("RemoteError: Cannot reach")
    assert not (planted / "runs").exists()  # the failed run left nothing behind


def test_a_bioamla_that_cannot_run_is_a_json_error(planted, tmp_path):
    """A BIOAMLA_BIN that isn't there (a removed virtual environment, say) is reported the
    way every other failure is: JSON on stdout, exit 1, not a traceback."""
    import os
    import subprocess
    import sys

    from tests.skills.conftest import skill_script

    gone = tmp_path / "old-venv" / "bin" / "bioamla"
    proc = subprocess.run(
        [sys.executable, str(skill_script("label-qa", "check")), "annotations.csv"],
        cwd=planted, env={**os.environ, "BIOAMLA_BIN": str(gone)},
        capture_output=True, text=True, check=False,
    )  # fmt: skip
    assert proc.returncode == 1, proc.stdout + proc.stderr
    error = json.loads(proc.stdout)["error"]
    assert f"can't run {gone}" in error and "BIOAMLA_BIN" in error
    assert "Traceback" not in proc.stderr


def test_thresholds_outside_zero_to_one_are_usage_errors(planted) -> None:
    import subprocess
    import sys

    from tests.skills.conftest import skill_script

    for script, flag, bad in (
        ("gate", "--min-f1", "1.5"),
        ("gate", "--min-precision", "nan"),
        ("gate", "--max-findings", "-1"),
        ("agreement", "--min-iou", "2"),
    ):
        args = [sys.executable, str(skill_script("label-qa", script)), "runs/label-qa/x", flag, bad]
        if script == "agreement":
            args += ["--reference", "reference.csv"]
        proc = subprocess.run(args, cwd=planted, capture_output=True, text=True, check=False)
        assert proc.returncode == 2 and "error: argument" in proc.stderr, (
            script,
            flag,
            proc.stderr,
        )


def test_a_file_cell_that_looks_like_an_option_is_just_a_missing_file(run_skill, workspace):
    """Data never reaches the CLI as an option: a file called --help is looked up as a file."""
    write_recordings(workspace)
    write_set(workspace, ["--help,0.5,1.0,2000,3000,wren,agent"])
    code, checked = run_skill("label-qa", "check", "annotations.csv", cwd=workspace)
    assert code == 0 and checked["by_rule"] == {"file": 1}, checked
    (f,) = json.loads((workspace / checked["run_dir"] / "checks.json").read_text())["findings"]
    assert f["file"] == "--help" and "not found" in f["message"]


def test_the_reference_cannot_be_the_checked_file_and_a_rerun_drops_the_old_verdict(
    run_skill, planted
) -> None:
    code, checked = run_skill("label-qa", "check", "annotations.csv", cwd=planted)
    code, out = run_skill("label-qa", "agreement", checked["run_dir"], "--reference",
                          "annotations.csv", cwd=planted)  # fmt: skip
    assert code == 1 and "can't be the checked file itself" in out["error"]
    code, _ = run_skill("label-qa", "agreement", checked["run_dir"], "--reference",
                        "reference.csv", cwd=planted)  # fmt: skip
    code, verdict = run_skill("label-qa", "gate", checked["run_dir"], cwd=planted)
    assert code == 2 and (planted / checked["run_dir"] / "gate.json").exists()
    # The reference copy lives apart from the checked snapshot, and rerunning the
    # agreement step removes the verdict that rested on the old report.
    assert (planted / checked["run_dir"] / "reference" / "reference.csv").exists()
    assert (planted / checked["run_dir"] / "workspace" / "annotations.csv").exists()
    code, _ = run_skill("label-qa", "agreement", checked["run_dir"], "--reference",
                        "reference.csv", cwd=planted)  # fmt: skip
    assert not (planted / checked["run_dir"] / "gate.json").exists()


def test_held_out_recordings_with_no_events_count_their_agent_rows_as_false_alarms(
    run_skill, workspace
) -> None:
    write_recordings(workspace)
    write_set(workspace, ["rec/a.wav,0.5,1.0,2000,3000,wren,agent",
                          "rec/b.wav,1.0,1.5,2000,3000,wren,agent"])  # fmt: skip
    write_set(workspace, ["rec/a.wav,0.5,1.0,2000,3000,wren,jm"], "reference.csv")
    code, checked = run_skill("label-qa", "check", "annotations.csv", cwd=workspace)
    code, scored = run_skill("label-qa", "agreement", checked["run_dir"], "--reference",
                             "reference.csv", cwd=workspace)  # fmt: skip
    assert scored["files"] == 1 and scored["false_alarms"] == 0  # b.wav isn't covered
    code, scored = run_skill("label-qa", "agreement", checked["run_dir"], "--reference",
                             "reference.csv", "--recordings", "rec/b.wav", cwd=workspace)  # fmt: skip
    assert scored["files"] == 2 and scored["false_alarms"] == 1 and scored["precision"] == 0.5
