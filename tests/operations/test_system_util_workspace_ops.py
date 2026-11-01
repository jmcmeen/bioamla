"""system.*, util.* and workspace.* operations, without the CLI and against it."""

import json
import zipfile
from pathlib import Path

import pytest
from click.testing import CliRunner

from bioamla import operations
from bioamla.cli.cli import cli
from bioamla.exceptions import NotFoundError, OptionConflictError


def _cli(*args: str) -> dict:
    result = CliRunner().invoke(cli, ["--json", *args])
    assert result.exit_code == 0, result.output
    return json.loads(result.stdout)["result"]


def test_workspace_lifecycle_through_the_registry(tmp_path, test_audio_path):
    root = tmp_path / "survey"
    info = operations.run("workspace.init", {"directory": str(root), "name": "survey"})
    assert info["name"] == "survey"
    added = operations.run("workspace.add", {"directory": str(root), "paths": [test_audio_path]})
    artifact = added["artifacts"][0]
    assert artifact["exists"] and artifact["mode"] == "linked"
    listed = operations.run("workspace.list", {"directory": str(root)})
    assert [a["id"] for a in listed["artifacts"]] == [artifact["id"]]
    status = operations.run("workspace.status", {"directory": str(root)})
    assert status["artifacts"] == {artifact["kind"]: 1} and status["last_runs"] == []
    removed = operations.run(
        "workspace.remove", {"directory": str(root), "artifact_id": artifact["id"]}
    )
    assert removed["deleted_copy"] is False
    assert operations.run("workspace.history", {"directory": str(root)})["runs"] == []


def test_workspace_needs_one(tmp_path):
    with pytest.raises(NotFoundError, match="Not inside a bioamla workspace"):
        operations.run("workspace.status", {"directory": str(tmp_path)})
    with pytest.raises(OptionConflictError):
        operations.run(
            "workspace.add", {"directory": str(tmp_path), "paths": ["a", "b"], "name": "x"}
        )


def test_workspace_cli_matches_the_registry(tmp_path, test_audio_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _cli("workspace", "init", "--name", "w")
    got = _cli("workspace", "add", test_audio_path)
    want = operations.run("workspace.list", {"directory": str(tmp_path)})
    assert [a["id"] for a in got["artifacts"]] == [a["id"] for a in want["artifacts"]]
    assert len(_cli("workspace", "history")["runs"]) == 2  # init and add were recorded


def test_zip_and_unzip(tmp_path):
    src = tmp_path / "src"
    (src / "a").mkdir(parents=True)
    (src / "a" / "x.txt").write_text("hi")
    operations.run("util.zip", {"source": str(src), "output": str(tmp_path / "s.zip")})
    assert zipfile.ZipFile(tmp_path / "s.zip").namelist() == ["a/x.txt"]
    got = _cli("util", "unzip", str(tmp_path / "s.zip"), str(tmp_path / "out"))
    assert (Path(got["output"]) / "a" / "x.txt").read_text() == "hi"


def test_unzip_refuses_a_bad_archive(tmp_path):
    from bioamla.exceptions import BioamlaError

    (tmp_path / "bad.zip").write_text("not a zip")
    with pytest.raises(BioamlaError, match="Extraction failed"):
        operations.run(
            "util.unzip", {"file": str(tmp_path / "bad.zip"), "output_dir": str(tmp_path)}
        )


def test_schemas(tmp_path):
    listing = operations.run("system.schema", {})
    assert "batch" in listing["configs"]
    one = operations.run("system.schema", {"name": "batch"})
    assert one["schema"]["title"] == "BatchConfig"
    exported = operations.run("system.export-schemas", {"output_dir": str(tmp_path)})
    assert str(tmp_path / "batch.schema.json") in exported["written"]
    assert _cli("system", "schema", "batch")["schema"] == one["schema"]


def test_local_only_operations():
    for name in ("system.deps-install", "util.download"):
        assert operations.get(name).remote is False, name
    for name in ("system.deps", "util.unzip", "workspace.add"):
        assert operations.get(name).remote is True, name


def test_version_and_devices():
    assert operations.run("system.version", {}).bioamla_version
    assert operations.run("system.devices", {}).devices


def test_failed_run_with_dash_c_is_recorded_in_that_workspace(tmp_path, monkeypatch):
    ws = tmp_path / "ws"
    _cli("workspace", "init", str(ws))
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    run = CliRunner().invoke(cli, ["workspace", "-C", str(ws), "remove", "nope"])
    assert run.exit_code != 0
    failed = _cli("workspace", "-C", str(ws), "history", "--failed")["runs"]
    assert [r["argv"][-1] for r in failed] == ["nope"]


def test_file_errors_are_bioamla_errors(tmp_path):
    import os

    from bioamla.exceptions import BioamlaError, ConfigError

    old = tmp_path / "old.txt"
    old.write_text("x")
    os.utime(old, (0, 0))  # 1970: ZIP can't store it
    with pytest.raises(BioamlaError, match="ZIP creation failed"):
        operations.run("util.zip", {"source": str(old), "output": str(tmp_path / "o.zip")})
    with pytest.raises(NotFoundError):
        operations.run("util.unzip", {"file": str(tmp_path / "missing.zip")})
    with pytest.raises(ConfigError, match="Failed to write"):
        operations.run("system.schema", {"name": "batch", "output": str(tmp_path / "no" / "x")})
    from bioamla.common.files import download_to_dir

    with pytest.raises(BioamlaError, match="Download failed"):
        download_to_dir("example.com/no-scheme.wav", tmp_path)
