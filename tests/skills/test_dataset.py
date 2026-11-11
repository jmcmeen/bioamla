"""Eval for the dataset skill: check, gate and push a dataset with known problems."""

from __future__ import annotations

import csv
import importlib.util
import json
import os
import subprocess
import sys
import wave
from pathlib import Path

import pytest

from tests.skills.conftest import SKILLS, skill_python, skill_script

FIELDS = ["file_name", "label", "split", "license", "attribution", "site", "source_file"]
BY = "CC BY 4.0"


def row(name, label, split, license=BY, attribution="A. Recordist", site="", source=None):
    return {
        "file_name": name,
        "label": label,
        "split": split,
        "license": license,
        "attribution": attribution,
        "site": site,
        "source_file": source or name,
    }


def pcm_wav(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(b"\x00\x01" * 1600)
    return path


def guano_wav(path: Path, position=(36.523456, -82.218765)) -> Path:
    from riffy import GuanoMetadata, WAVParser

    pcm_wav(path)
    parser = WAVParser(path)
    guano = GuanoMetadata(version="1.0")
    guano.loc_position = position
    guano.write_to_parser(parser)
    parser.write_wav(path, overwrite=True)
    return path


def write_dataset(root: Path, rows: list[dict], fields=FIELDS, name="frogs") -> Path:
    folder = root / "datasets" / name
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / "metadata.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    for r in rows:
        if r.get("file_name", "").endswith(".wav") and not (folder / r["file_name"]).exists():
            pcm_wav(folder / r["file_name"])
    return folder


def clean_rows() -> list[dict]:
    return [
        row("train/frog/a.wav", "frog", "train", site="s1"),
        row("train/bird/b.wav", "bird", "train", site="s1"),
        row("test/frog/c.wav", "frog", "test", site="s2"),
        row("test/bird/d.wav", "bird", "test", site="s2"),
    ]


def rules(summary_or_checks) -> dict:
    return summary_or_checks["by_rule"]


def check(run_skill, ws, *args):
    return run_skill("dataset", "check", "datasets/frogs", *args, cwd=ws)


def checks_json(ws, summary) -> dict:
    return json.loads((ws / summary["run_dir"] / "checks.json").read_text())


def test_a_clean_dataset_passes_check_and_gate(run_skill, workspace):
    write_dataset(workspace, clean_rows())
    code, summary = check(run_skill, workspace, "--license", "CC-BY-4.0", "--group-by", "site")
    assert code == 0, summary
    assert summary["findings"] == 0, checks_json(workspace, summary)["findings"]
    assert summary["splits"] == {"test": 2, "train": 2}
    report = checks_json(workspace, summary)
    assert report["counts"] == {"test": {"bird": 1, "frog": 1}, "train": {"bird": 1, "frog": 1}}
    assert len(report["metadata_sha256"]) == 64

    code, verdict = run_skill("dataset", "gate", summary["run_dir"], cwd=workspace)
    assert code == 0, verdict
    assert verdict["passed"] is True
    assert verdict["coordinates"] == "round"
    assert (workspace / summary["run_dir"] / "gate.json").exists()


@pytest.mark.parametrize(
    ("license", "target", "message"),
    [
        ("CC BY-ND 4.0", "CC-BY-4.0", "no derivatives"),
        ("cc-by-nc", "CC-BY-4.0", "non-commercial"),
        ("https://creativecommons.org/licenses/by-sa/4.0/", "CC-BY-4.0", "same license"),
        ("CC BY-SA 4.0", "CC-BY-NC-SA-4.0", "same license"),
        ("CC BY-SA 3.0", "CC-BY-SA-4.0", "same license version"),
        ("CC BY 4.0", "CC0", "requires attribution"),
        ("All rights reserved", "CC-BY-4.0", "not a license"),
        ("", "CC-BY-4.0", "no license"),
    ],
)
def test_licenses_the_dataset_cant_carry(run_skill, workspace, license, target, message):
    rows = [{**r, "license": "CC0"} for r in clean_rows()]  # fine under any license
    rows[2]["license"] = license
    write_dataset(workspace, rows)
    code, summary = check(run_skill, workspace, "--license", target)
    assert code == 0, summary
    findings = checks_json(workspace, summary)["findings"]
    licensed = [f for f in findings if f["rule"] == "license"]
    assert [f["row"] for f in licensed] == [4], findings
    assert message in licensed[0]["message"]
    assert licensed[0]["file"] == "test/frog/c.wav"


@pytest.mark.parametrize(
    ("license", "target"),
    [
        ("CC0 1.0", "CC-BY-4.0"),
        ("cc-by-nc", "CC-BY-NC-SA-4.0"),  # NC kept; SA added is fine for a BY-NC source
        ("//creativecommons.org/licenses/by-nc-sa/4.0/", "CC BY-NC-SA 4.0"),
        ("Public Domain Mark", "CC-BY-NC-4.0"),
    ],
)
def test_licenses_the_dataset_can_carry(run_skill, workspace, license, target):
    rows = clean_rows()
    rows[2]["license"] = license
    write_dataset(workspace, rows)
    _, summary = check(run_skill, workspace, "--license", target)
    assert "license" not in rules(summary), checks_json(workspace, summary)["findings"]


def test_a_dataset_partitioned_by_bioamla_checks_clean(run_skill, workspace):
    """The metadata `dataset partition` writes, grouped by site: no leakage."""
    from tests.skills.conftest import bioamla_bin

    rows = [
        row(f"{label}{i}.wav", label, "", site=f"s{i % 4}")
        for i, label in enumerate(["frog", "bird"] * 6)
    ]
    folder = write_dataset(workspace, rows)
    subprocess.run(
        [bioamla_bin(), "--json", "dataset", "partition", str(folder), "--mode", "column",
         "--group-by", "site", "--train", "0.5", "--val", "0", "--test", "0.5", "--seed", "1"],
        check=True, capture_output=True,
    )  # fmt: skip
    code, summary = check(run_skill, workspace, "--license", "CC-BY-4.0", "--group-by", "site")
    assert code == 0, summary
    assert summary["findings"] == 0, checks_json(workspace, summary)["findings"]
    assert set(summary["splits"]) == {"train", "test"}


def test_attribution_is_needed_for_a_by_license(run_skill, workspace):
    rows = clean_rows()
    rows[1]["attribution"] = ""
    rows[3]["attribution"] = ""
    rows[3]["license"] = "CC0"  # no credit needed
    write_dataset(workspace, rows)
    _, summary = check(run_skill, workspace, "--license", "CC-BY-4.0")
    findings = checks_json(workspace, summary)["findings"]
    assert [(f["rule"], f["row"]) for f in findings] == [("attribution", 3)]


def test_attr_lic_is_read_before_license(run_skill, workspace):
    """The order `dataset license` reads them in, so the checked license is the credited one."""
    fields = [*FIELDS, "attr_lic"]
    rows = [{**r, "attr_lic": ""} for r in clean_rows()]
    rows[0]["attr_lic"] = "CC BY-ND 4.0"  # license still says CC BY 4.0
    write_dataset(workspace, rows, fields=fields)
    _, summary = check(run_skill, workspace, "--license", "CC-BY-4.0")
    [found] = checks_json(workspace, summary)["findings"]
    assert (found["rule"], found["row"]) == ("license", 2)


def test_an_observer_alone_isnt_credit(run_skill, workspace):
    fields = [*FIELDS, "observer"]
    rows = [{**r, "observer": "J. Doe"} for r in clean_rows()]
    rows[0]["attribution"] = ""
    write_dataset(workspace, rows, fields=fields)
    _, summary = check(run_skill, workspace, "--license", "CC-BY-4.0")
    assert rules(summary) == {"attribution": 1}


def test_a_row_without_a_split(run_skill, workspace):
    rows = clean_rows()
    rows[3]["split"] = ""
    write_dataset(workspace, rows)
    _, summary = check(run_skill, workspace, "--license", "CC-BY-4.0")
    assert rules(summary) == {"split": 1}
    assert summary["splits"] == {"test": 1, "train": 2}


def test_a_site_in_two_splits_leaks(run_skill, workspace):
    rows = clean_rows()
    rows[2]["site"] = "s1"
    write_dataset(workspace, rows)
    _, summary = check(run_skill, workspace, "--license", "CC-BY-4.0", "--group-by", "site")
    [leak] = checks_json(workspace, summary)["findings"]
    assert (leak["rule"], leak["value"], leak["splits"], leak["rows"]) == (
        "leakage",
        "s1",
        ["test", "train"],
        3,
    )


def test_source_file_is_the_default_grouping(run_skill, workspace):
    rows = clean_rows()
    rows[2]["source_file"] = "train/frog/a.wav"  # a clip of a training recording
    write_dataset(workspace, rows)
    _, summary = check(run_skill, workspace, "--license", "CC-BY-4.0")
    assert rules(summary) == {"leakage": 1}


def test_missing_columns(run_skill, workspace):
    fields = ["file_name", "label", "license", "attribution"]
    write_dataset(workspace, clean_rows(), fields=fields)
    _, summary = check(run_skill, workspace, "--license", "CC-BY-4.0", "--group-by", "site")
    messages = [f["message"] for f in checks_json(workspace, summary)["findings"]]
    assert messages == ["no split column", "no site column"]


def test_locations_in_the_metadata_need_redaction(run_skill, workspace):
    fields = [*FIELDS, "latitude", "longitude"]
    rows = [{**r, "latitude": "36.52", "longitude": "-82.21"} for r in clean_rows()]
    write_dataset(workspace, rows, fields=fields)
    _, kept = check(run_skill, workspace, "--license", "CC-BY-4.0", "--coordinates", "keep")
    assert rules(kept) == {"location": 1}
    assert kept["locations"]["metadata_columns"] == ["latitude", "longitude"]
    _, rounded = check(run_skill, workspace, "--license", "CC-BY-4.0")
    assert rules(rounded) == {}


def test_a_position_in_a_wav_file_needs_redaction(run_skill, workspace):
    folder = write_dataset(workspace, clean_rows())
    guano_wav(folder / "test" / "frog" / "c.wav")
    _, summary = check(run_skill, workspace, "--license", "CC-BY-4.0", "--coordinates", "keep")
    assert rules(summary) == {"location": 1}
    assert summary["locations"]["with_position"] == "datasets/frogs/test/frog/c.wav"
    _, stripped = check(run_skill, workspace, "--license", "CC-BY-4.0", "--coordinates", "strip")
    assert rules(stripped) == {}


def test_files_the_push_cant_redact(run_skill, workspace):
    folder = write_dataset(workspace, clean_rows())
    (folder / "detections.jsonl").write_text('{"latitude": 36.523456, "longitude": -82.2}\n')
    (folder / "notes.txt").write_text("recorded by the creek\n")  # no coordinates
    _, rounded = check(run_skill, workspace, "--license", "CC-BY-4.0")
    [found] = checks_json(workspace, rounded)["findings"]
    assert (found["rule"], found["files"]) == ("location", ["detections.jsonl"])
    _, kept = check(run_skill, workspace, "--license", "CC-BY-4.0", "--coordinates", "keep")
    assert rules(kept) == {}  # keep publishes as is anyway; nothing in the metadata or WAVs


def test_an_unreadable_wav_is_a_finding_not_an_error(run_skill, workspace):
    folder = write_dataset(workspace, clean_rows())
    (folder / "test" / "frog" / "c.wav").write_bytes(b"not a wav")
    code, summary = check(run_skill, workspace, "--license", "CC-BY-4.0", "--coordinates", "keep")
    assert code == 0, summary
    assert rules(summary) == {"file": 1}


@pytest.mark.parametrize(
    ("args", "reason"),
    [
        (["--min-per-class", "2"], "fewer than 2 rows"),
        (["--require-splits", "train,val,test"], "the val split has no rows"),
        (["--max-imbalance", "1.5"], "2.0 times"),
    ],
)
def test_the_gate_applies_balance_thresholds(run_skill, workspace, args, reason):
    rows = [*clean_rows(), row("train/frog/e.wav", "frog", "train", site="s1"),
            row("train/frog/f.wav", "frog", "train", site="s1")]  # fmt: skip
    write_dataset(workspace, rows)
    _, summary = check(run_skill, workspace, "--license", "CC-BY-4.0")
    code, verdict = run_skill("dataset", "gate", summary["run_dir"], *args, cwd=workspace)
    assert code == 2, verdict
    assert any(reason in r for r in verdict["reasons"]), verdict["reasons"]


def test_the_gate_fails_on_findings_unless_a_rule_is_accepted(run_skill, workspace):
    fields = [*FIELDS, "latitude", "longitude"]
    rows = [{**r, "latitude": "36.52", "longitude": "-82.21"} for r in clean_rows()]
    write_dataset(workspace, rows, fields=fields)
    _, summary = check(run_skill, workspace, "--license", "CC-BY-4.0", "--coordinates", "keep")
    code, verdict = run_skill("dataset", "gate", summary["run_dir"], cwd=workspace)
    assert code == 2
    assert verdict["reasons"] == ["1 findings (at most 0 allowed): location 1"]
    code, verdict = run_skill(
        "dataset", "gate", summary["run_dir"], "--accept", "location", cwd=workspace
    )
    assert code == 0, verdict
    assert verdict["accepted_findings"] == 1
    assert verdict["coordinates"] == "keep"


@pytest.mark.parametrize(
    "args",
    [
        ["datasets/frogs"],  # no --license
        ["datasets/frogs", "--license", "GPL-3.0"],
        ["datasets/frogs", "--license", "CC-BY-ND-4.0"],  # a dataset of clips can't be ND
        ["datasets/frogs", "--license", "CC-BY-4.0", "--coordinate-decimals", "9"],
    ],
)
def test_bad_arguments_exit_1_not_2(run_skill, workspace, args):
    write_dataset(workspace, clean_rows())
    code, summary = run_skill("dataset", "check", *args, cwd=workspace)
    assert code == 1
    assert "error" in summary


def test_a_missing_dataset_is_an_error(run_skill, workspace):
    code, summary = run_skill(
        "dataset", "check", "datasets/none", "--license", "CC-BY-4.0", cwd=workspace
    )
    assert code == 1
    assert "not in the workspace" in summary["error"]
    assert not (workspace / "runs").exists()


# --- push: a stand-in bioamla records the push instead of reaching the Hub ---

FAKE_BIOAMLA = """\
import json, sys
args = sys.argv[1:]
if args[:4] == ["--json", "catalogs", "hf", "push-dataset"]:
    with open({log!r}, "w") as f:
        json.dump(args, f)
    print(json.dumps({{"bioamla": "x", "command": "catalogs hf push-dataset", "ok": True,
                      "result": {{"url": "https://huggingface.co/datasets/" + args[-1],
                                 "warnings": []}}}}))
else:
    print(json.dumps({{"ok": False, "error": {{"type": "Error", "message": "unexpected"}}}}))
"""


def run_push(workspace: Path, *args: str) -> tuple[int, dict, Path]:
    log = workspace.parent / "push-args.json"
    fake = workspace.parent / "fake_bioamla.py"
    fake.write_text(FAKE_BIOAMLA.format(log=str(log)))
    fake.chmod(0o755)
    wrapper = workspace.parent / "bioamla"
    wrapper.write_text(f'#!/bin/sh\nexec {sys.executable} {fake} "$@"\n')
    wrapper.chmod(0o755)
    env = {**os.environ, "BIOAMLA_BIN": str(wrapper)}
    env.pop("BIOAMLA_REMOTE", None)
    proc = subprocess.run(
        [skill_python(), str(skill_script("dataset", "push")), *args],
        cwd=workspace, env=env, capture_output=True, text=True, check=False,
    )  # fmt: skip
    return proc.returncode, json.loads(proc.stdout), log


def gated(run_skill, workspace, *check_args) -> str:
    write_dataset(workspace, clean_rows())
    _, summary = check(run_skill, workspace, "--license", "CC-BY-4.0", *check_args)
    run_skill("dataset", "gate", summary["run_dir"], cwd=workspace)
    return summary["run_dir"]


@pytest.mark.skipif(os.name == "nt", reason="the stand-in bioamla is a shell script")
def test_push_uses_the_checked_coordinate_mode_and_is_private(run_skill, workspace):
    run = gated(run_skill, workspace, "--coordinates", "strip")
    code, report, log = run_push(workspace, run, "me/frogs")
    assert code == 0, report
    assert report["pushed"] is True
    args = json.loads(log.read_text())
    assert args[args.index("--coordinates") + 1] == "strip"
    assert "--private" in args and "--public" not in args
    assert args[args.index("--unredacted") + 1] == "refuse"  # no location sign-off
    assert args[-3:] == ["--", "datasets/frogs", "me/frogs"]
    assert (workspace / run / "push.json").exists()


@pytest.mark.skipif(os.name == "nt", reason="the stand-in bioamla is a shell script")
def test_a_signed_off_location_finding_publishes_unredactable_files(run_skill, workspace):
    folder = write_dataset(workspace, clean_rows())
    (folder / "detections.jsonl").write_text('{"latitude": 36.523456}\n')
    _, summary = check(run_skill, workspace, "--license", "CC-BY-4.0")
    run_skill("dataset", "gate", summary["run_dir"], "--accept", "location", cwd=workspace)
    code, report, log = run_push(workspace, summary["run_dir"], "me/frogs")
    assert code == 0, report
    args = json.loads(log.read_text())
    assert args[args.index("--unredacted") + 1] == "publish"


@pytest.mark.skipif(os.name == "nt", reason="the stand-in bioamla is a shell script")
def test_push_refuses_a_failed_gate(run_skill, workspace):
    rows = clean_rows()
    rows[0]["license"] = "CC BY-ND 4.0"
    write_dataset(workspace, rows)
    _, summary = check(run_skill, workspace, "--license", "CC-BY-4.0")
    run_skill("dataset", "gate", summary["run_dir"], cwd=workspace)
    code, report, log = run_push(workspace, summary["run_dir"], "me/frogs")
    assert code == 2
    assert report["reason"] == "the gate didn't pass"
    assert not log.exists()


@pytest.mark.skipif(os.name == "nt", reason="the stand-in bioamla is a shell script")
def test_push_refuses_a_dataset_changed_since_the_check(run_skill, workspace):
    run = gated(run_skill, workspace)
    with (workspace / "datasets" / "frogs" / "metadata.csv").open("a") as f:
        f.write("test/frog/z.wav,frog,test,CC BY-ND 4.0,,s2,z.wav\n")
    code, report, log = run_push(workspace, run, "me/frogs")
    assert code == 2
    assert "changed since the check" in report["reason"]
    assert not log.exists()


def test_push_refuses_without_a_gate(run_skill, workspace):
    write_dataset(workspace, clean_rows())
    _, summary = check(run_skill, workspace, "--license", "CC-BY-4.0")
    code, report = run_skill("dataset", "push", summary["run_dir"], "me/frogs", cwd=workspace)
    assert code == 2
    assert "hasn't run" in report["reason"]


def _load(script: str):
    spec = importlib.util.spec_from_file_location(
        f"dataset_{script}", SKILLS / "dataset" / "scripts" / f"{script}.py"
    )
    module = importlib.util.module_from_spec(spec)
    saved = list(sys.path)
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path[:] = saved
    return module


@pytest.mark.parametrize(
    ("text", "name"),
    [
        ("CC BY-NC-SA 4.0", "CC BY-NC-SA 4.0"),
        ("cc-by-nc-nd", "CC BY-NC-ND"),
        ("http://creativecommons.org/licenses/by/3.0/", "CC BY 3.0"),
        ("CC0", "public domain"),
        ("CC_BY_NC", "CC BY-NC"),
        ("GPL-3.0", None),
        ("cc-sa", None),
    ],
)
def test_license_names_and_urls_are_read(text, name):
    parsed = _load("check").parse_license(text)
    assert (parsed.name if parsed else None) == name
