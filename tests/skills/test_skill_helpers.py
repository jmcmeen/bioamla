"""The skill helpers in ``_bioamla.py``: where commands run, and moving files."""

import importlib.util
import json
from pathlib import Path

import pytest

from tests.skills.conftest import skill_script


def load(monkeypatch, name="_bioamla"):
    folder = skill_script("label", "crops").parent
    monkeypatch.syspath_prepend(str(folder))
    spec = importlib.util.spec_from_file_location(f"label_{name}", folder / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_skill_helpers_are_identical():
    copies = {
        p.read_bytes() for p in skill_script("label", "x").parents[2].glob("*/scripts/_bioamla.py")
    }
    assert len(copies) == 1


def test_remote_comes_from_the_environment_then_the_workspace(monkeypatch, tmp_path):
    helpers = load(monkeypatch)
    monkeypatch.delenv("BIOAMLA_REMOTE", raising=False)
    assert helpers.remote(tmp_path) is None
    ws = tmp_path / "ws"
    (ws / ".bioamla").mkdir(parents=True)
    (ws / ".bioamla" / "workspace.toml").write_text('format_version = 1\nid = "x"\n')
    (ws / ".bioamla" / "config.toml").write_text('remote = "lab"\nworkspace = "site"\n')
    (ws / "sub").mkdir()
    assert helpers.remote(ws / "sub")["workspace"] == "site"  # found walking up
    monkeypatch.setenv("BIOAMLA_REMOTE", "hive:field")
    assert helpers.remote(ws) == {"remote": "hive", "workspace": "field",
                                  "source": "$BIOAMLA_REMOTE"}  # fmt: skip
    monkeypatch.setenv("BIOAMLA_REMOTE", "hive")  # no workspace: a clear error
    with pytest.raises(helpers.BioamlaCommandError, match="NAME:WORKSPACE"):
        helpers.remote(ws)


def test_paths_must_stay_in_the_workspace(monkeypatch, tmp_path):
    helpers = load(monkeypatch)
    monkeypatch.delenv("BIOAMLA_REMOTE", raising=False)
    for bad in ("/data/annotations.csv", "../annotations.csv", "a/../../b.csv", ""):
        with pytest.raises(ValueError):
            helpers.fetch(bad, "copy", tmp_path)


def test_a_path_that_looks_like_an_option_is_refused(monkeypatch, tmp_path):
    helpers = load(monkeypatch)
    for bad in ("-o", "--help", "-x.wav"):
        with pytest.raises(ValueError, match="not starting with -"):
            helpers.fetch(bad, "into", tmp_path)


def test_local_fetch_never_keeps_a_stale_copy(monkeypatch, tmp_path):
    helpers = load(monkeypatch)
    monkeypatch.delenv("BIOAMLA_REMOTE", raising=False)
    (tmp_path / "a.csv").write_text("x\n")
    copy = helpers.fetch("a.csv", "run/workspace", tmp_path)
    assert copy.read_text() == "x\n"
    (tmp_path / "a.csv").unlink()
    assert not helpers.fetch("a.csv", "run/workspace", tmp_path).exists()
    # publish puts a file in place relative to the root, wherever the script runs
    src = tmp_path / "run" / "new.csv"
    src.write_text("y\n")
    monkeypatch.chdir(tmp_path.parent)
    helpers.publish("run/new.csv", "out/new.csv", tmp_path)
    assert (tmp_path / "out" / "new.csv").read_text() == "y\n"


def test_annotate_refuses_a_file_that_changed_meanwhile(monkeypatch, tmp_path):
    annotate = load(monkeypatch, "annotate")
    monkeypatch.delenv("BIOAMLA_REMOTE", raising=False)
    run = tmp_path / "runs" / "r"
    run.mkdir(parents=True)
    (run / "candidates.json").write_text(json.dumps(
        {"files": ["a.wav"], "model": "m", "candidates": []}))  # fmt: skip
    (run / "evidence.json").write_text(json.dumps({"candidates": []}))
    (tmp_path / "annotations.csv").write_text("file,labeled_by\na.wav,jm\n")
    real_fetch = annotate.fetch

    def fetch(path, into, root):  # someone edits the file between the two reads
        if Path(into).name == "check":
            (tmp_path / "annotations.csv").write_text("file,labeled_by\na.wav,jm\nb.wav,jm\n")
        return real_fetch(path, into, root)

    monkeypatch.setattr(annotate, "fetch", fetch)
    code = annotate.main(["runs/r", "--root", str(tmp_path)])
    assert code == 1
    assert "b.wav" in (tmp_path / "annotations.csv").read_text()  # their edit survives


def test_a_hung_bioamla_command_is_an_error_not_a_wait(monkeypatch):
    import subprocess

    helpers = load(monkeypatch)
    monkeypatch.setenv("BIOAMLA_TIMEOUT", "0.5")

    def hang(argv, **kwargs):
        assert kwargs["timeout"] == 0.5
        raise subprocess.TimeoutExpired(argv, kwargs["timeout"])

    monkeypatch.setattr(helpers.subprocess, "run", hang)
    with pytest.raises(helpers.BioamlaCommandError, match="gave no answer in 0.5 s") as caught:
        helpers.run("audio", "info", "x.wav")
    assert caught.value.kind == "Timeout"


def test_run_folders_are_created_without_a_check_then_create_race(monkeypatch, tmp_path):
    helpers = load(monkeypatch)
    first = helpers.run_dir("qa", tmp_path)
    second = helpers.run_dir("qa", tmp_path)
    assert first != second and first.is_dir() and second.is_dir()
    assert second.name.startswith(first.name.split("-")[0])


def test_json_is_written_whole_or_not_at_all(monkeypatch, tmp_path):
    helpers = load(monkeypatch)
    out = helpers.write_json(tmp_path / "a" / "b.json", {"x": 1})
    assert out.read_text() == '{\n  "x": 1\n}\n'
    assert not (tmp_path / "a" / "b.json.tmp").exists()
