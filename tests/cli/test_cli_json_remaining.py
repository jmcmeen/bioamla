"""`bioamla --json` results for the audio transforms, system, util, catalogs and models commands."""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from bioamla.cli.cli import cli


def _run(*args: str):
    result = CliRunner().invoke(cli, ["--json", *args])
    return result, json.loads(result.stdout)


def _ok(*args: str) -> dict:
    result, envelope = _run(*args)
    assert result.exit_code == 0, result.output
    assert envelope["ok"] is True
    return envelope["result"]


@pytest.mark.parametrize(
    ("command", "extra"),
    [
        ("trim", ["--start", "0.1", "--end", "0.4"]),
        ("normalize", []),
        ("resample", ["-r", "8000"]),
        ("gain", ["--gain-db", "3"]),
    ],
)
def test_audio_transforms(test_audio_path, tmp_path, command, extra) -> None:
    out = tmp_path / f"{command}.wav"
    result = _ok("audio", command, test_audio_path, str(out), *extra)
    assert result["file"] == test_audio_path and result["output"] == str(
        out
    )  # like other per-file results
    assert out.exists() and "output" not in result["params"]


def test_audio_segment_reports_the_count(test_audio_path, tmp_path) -> None:
    result = _ok("audio", "segment", test_audio_path, str(tmp_path / "seg"), "-d", "0.5")
    assert result["segments"] == len(list((tmp_path / "seg").glob("*.wav"))) > 0


def test_audio_play_refuses_json(test_audio_path) -> None:
    result, envelope = _run("audio", "play", test_audio_path)
    assert result.exit_code == 1 and "no --json output" in envelope["error"]["message"]


def test_system_commands() -> None:
    assert _ok("system", "version")["bioamla_version"]
    assert "devices" in _ok("system", "devices")
    assert "dependencies" in _ok("system", "deps")
    assert "batch" in _ok("system", "schema")["configs"]
    assert _ok("system", "schema", "batch")["schema"]["title"] == "BatchConfig"


def test_system_deps_install_needs_yes() -> None:
    result, envelope = _run("system", "deps", "--install")
    assert result.exit_code == 2 and envelope["error"]["type"] == "UsageError"


def test_util_zip_and_unzip(tmp_path) -> None:
    src = tmp_path / "src"
    src.mkdir()
    (src / "a.txt").write_text("x")
    archive = tmp_path / "a.zip"
    assert _ok("util", "zip", str(src), str(archive))["output"] == str(archive)
    unzipped = _ok("util", "unzip", str(archive), str(tmp_path / "out"))
    assert Path(unzipped["output"]).exists()


def test_models_init_config(tmp_path) -> None:
    out = tmp_path / "train.toml"
    assert _ok("models", "ast", "init-config", "-o", str(out))["output"] == str(out)
    result, envelope = _run("models", "ast", "init-config", "-o", str(out))
    assert result.exit_code == 1 and "--force" in envelope["error"]["message"]


def test_catalog_search_and_lookup() -> None:
    recording = SimpleNamespace(to_dict=lambda: {"id": "1", "scientific_name": "Rana"})
    with patch(
        "bioamla.catalogs.xeno_canto.search", return_value=SimpleNamespace(recordings=[recording])
    ):
        result = _ok("catalogs", "xc", "search", "--species", "Rana")
    assert result == {"count": 1, "recordings": [{"id": "1", "scientific_name": "Rana"}]}

    with patch("bioamla.catalogs.inat.search", return_value=SimpleNamespace(observations=[])):
        result = _ok("catalogs", "inat", "search", "--species", "Rana")
    assert result == {"count": 0, "observations": [], "output": None}


def test_hf_cache_purge_needs_yes() -> None:
    with patch("bioamla.catalogs.huggingface.scan_cache", return_value=[]):
        result, envelope = _run("catalogs", "hf", "cache", "--purge")
        assert result.exit_code == 2 and envelope["error"]["type"] == "UsageError"
        assert _ok("catalogs", "hf", "cache") == {"repos": [], "total_bytes": 0, "purged": None}


def test_deps_install_does_nothing_when_all_installed() -> None:
    report = SimpleNamespace(all_installed=True, install_command="brew install x", os_type="macos")
    with (
        patch("bioamla.system.dependency.check_all", return_value=report),
        patch("bioamla.system.dependency.install") as install,
    ):
        result = _ok("system", "deps", "--install", "--yes")
    install.assert_not_called()
    assert result["installed"] is None


def test_deps_install_without_an_install_command_fails() -> None:
    from bioamla.exceptions import ProcessingError

    report = SimpleNamespace(all_installed=False, install_command=None, os_type="windows")
    with (
        patch("bioamla.system.dependency.check_all", return_value=report),
        patch("bioamla.system.dependency.install", side_effect=ProcessingError("install manually")),
    ):
        result, envelope = _run("system", "deps", "--install", "--yes")
    assert result.exit_code == 1 and envelope["ok"] is False


def test_inat_search_rows_project_filter_and_empty_output(tmp_path) -> None:
    obs = {"id": 7, "taxon": {"name": "Rana", "preferred_common_name": "frog"}, "sounds": [{}]}
    with patch(
        "bioamla.catalogs.inat.search", return_value=SimpleNamespace(observations=[obs])
    ) as m:
        result = _ok("catalogs", "inat", "search", "--project-id", "frogs-tn")
    assert m.call_args.kwargs["project_id"] == "frogs-tn"
    assert (
        result["observations"][0]["scientific_name"] == "Rana"
    )  # the curated row, not the API dict
    assert "taxon" not in result["observations"][0]

    with patch("bioamla.catalogs.inat.search", return_value=SimpleNamespace(observations=[])):
        result = _ok(
            "catalogs", "inat", "search", "--species", "Rana", "-o", str(tmp_path / "o.csv")
        )
    assert result == {"count": 0, "observations": [], "output": None}


def test_hf_cache_purge_failures_are_a_failure() -> None:
    from bioamla.catalogs._models import CachedRepo, PurgeResult

    repo = CachedRepo(repo_id="a/b", repo_type="model", size_bytes=10)
    purged = PurgeResult(deleted=0, freed_bytes=0, failures=["a/b: permission denied"])
    with (
        patch("bioamla.catalogs.huggingface.scan_cache", return_value=[repo]),
        patch("bioamla.catalogs.huggingface.purge_cache", return_value=purged),
    ):
        result, envelope = _run("catalogs", "hf", "cache", "--purge", "--yes")
    assert result.exit_code == 1
    assert envelope["error"]["details"]["purged"]["failures"] == ["a/b: permission denied"]
