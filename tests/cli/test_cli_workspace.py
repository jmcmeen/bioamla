"""`bioamla workspace` commands."""

import json

import pytest
from click.testing import CliRunner

from bioamla.cli.cli import cli


def run(*args: str):
    return CliRunner().invoke(cli, list(args))


def as_json(*args: str) -> dict:
    result = run("--json", *args)
    envelope = json.loads(result.stdout)
    assert envelope["ok"], envelope
    return envelope["result"]


@pytest.fixture
def ws(tmp_path):
    root = tmp_path / "survey"
    as_json("workspace", "init", str(root), "--name", "Survey")
    return root


def test_init_and_status(ws) -> None:
    status = as_json("workspace", "-C", str(ws), "status")
    assert status["name"] == "Survey" and status["artifacts"] == {}


def test_init_twice_fails(ws) -> None:
    result = run("workspace", "init", str(ws))
    assert result.exit_code == 1 and "already a workspace" in result.output


def test_status_outside_a_workspace(tmp_path) -> None:
    result = run("workspace", "-C", str(tmp_path), "status")
    assert result.exit_code == 1
    assert "bioamla workspace init" in result.output


def test_found_from_a_subfolder(ws, monkeypatch) -> None:
    (ws / "a" / "b").mkdir(parents=True)
    monkeypatch.chdir(ws / "a" / "b")
    assert as_json("workspace", "status")["name"] == "Survey"


def test_add_import_list_remove(ws, tmp_path) -> None:
    (ws / "recs").mkdir()
    (ws / "recs" / "a.wav").write_bytes(b"RIFF")
    outside = tmp_path / "b.wav"
    outside.write_bytes(b"RIFF")

    added = as_json("workspace", "-C", str(ws), "add", str(ws / "recs"), str(outside))
    assert [a["path"] for a in added["artifacts"]] == ["recs", str(outside)]
    imported = as_json("workspace", "-C", str(ws), "import", str(outside), "--kind", "dataset")
    (copy,) = imported["artifacts"]
    assert copy["mode"] == "imported" and copy["kind"] == "dataset" and copy["exists"]

    listed = as_json("workspace", "-C", str(ws), "list")["artifacts"]
    assert len(listed) == 3
    assert as_json("workspace", "-C", str(ws), "list", "--kind", "dataset")["artifacts"] == [copy]

    removed = as_json("workspace", "-C", str(ws), "remove", copy["id"], "--delete-import")
    assert removed["removed"]["id"] == copy["id"]
    assert len(as_json("workspace", "-C", str(ws), "list")["artifacts"]) == 2


def test_missing_link_is_reported(ws) -> None:
    (ws / "a.wav").write_bytes(b"RIFF")
    as_json("workspace", "-C", str(ws), "add", str(ws / "a.wav"))
    (ws / "a.wav").unlink()
    result = run("workspace", "-C", str(ws), "list")
    assert result.exit_code == 0 and "MISSING" in result.output


def test_add_missing_path(ws, tmp_path) -> None:
    result = run("workspace", "-C", str(ws), "add", str(tmp_path / "nope.wav"))
    assert result.exit_code == 1 and "does not exist" in result.output


def test_remove_unknown(ws) -> None:
    result = run("workspace", "-C", str(ws), "remove", "nope")
    assert result.exit_code == 1 and "No artifact" in result.output


def test_relative_paths_follow_dash_c(ws, tmp_path, monkeypatch) -> None:
    (ws / "recs").mkdir()
    monkeypatch.chdir(tmp_path)  # not inside the workspace
    (linked,) = as_json("workspace", "-C", str(ws), "add", "recs")["artifacts"]
    assert linked["path"] == "recs"


def test_dash_c_must_exist(ws) -> None:
    result = run("workspace", "-C", str(ws / "typo"), "status")
    assert result.exit_code == 2 and "does not exist" in result.output


def test_one_missing_path_adds_nothing(ws) -> None:
    (ws / "a.wav").write_bytes(b"RIFF")
    result = run("workspace", "-C", str(ws), "add", str(ws / "a.wav"), str(ws / "missing.wav"))
    assert result.exit_code == 1
    assert as_json("workspace", "-C", str(ws), "list")["artifacts"] == []


def test_name_needs_a_single_path(ws) -> None:
    (ws / "a.wav").write_bytes(b"RIFF")
    (ws / "b.wav").write_bytes(b"RIFF")
    result = run("workspace", "-C", str(ws), "add", "a.wav", "b.wav", "--name", "Site A")
    assert result.exit_code == 2 and "single PATH" in result.output


def test_delete_import_on_a_linked_artifact_says_so(ws) -> None:
    (ws / "a.wav").write_bytes(b"RIFF")
    (linked,) = as_json("workspace", "-C", str(ws), "add", "a.wav")["artifacts"]
    result = run("workspace", "-C", str(ws), "remove", linked["id"], "--delete-import")
    assert result.exit_code == 0 and "left where it is" in result.output
    assert (ws / "a.wav").exists()
