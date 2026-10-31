"""`bioamla --json` results for the batch commands."""

import json
import shutil
from pathlib import Path

import pytest
from click.testing import CliRunner

from bioamla.cli.cli import cli


def _ok(*args: str) -> dict:
    result = CliRunner().invoke(cli, ["--json", *args])
    assert result.exit_code == 0, result.output
    envelope = json.loads(result.stdout)
    assert envelope["ok"] is True
    return envelope["result"]


@pytest.fixture
def audio_dir(test_audio_path, tmp_path) -> Path:
    directory = tmp_path / "in"
    directory.mkdir()
    for name in ("a.wav", "b.wav"):
        shutil.copy(test_audio_path, directory / name)
    return directory


def test_directory_mode_reports_the_batch_result(audio_dir, tmp_path) -> None:
    out = tmp_path / "out"
    result = _ok("batch", "audio", "info", "--input-dir", str(audio_dir), "--output-dir", str(out))
    assert (result["total_files"], result["successful"], result["failed"]) == (2, 2, 0)
    assert result["results_file"] == str(out / "audio_info.csv")
    assert "metadata" not in result  # per-file rows stay in the results file
    assert result["errors"] == [] and result["duration_seconds"] >= 0


def test_csv_mode_reports_failures_and_the_updated_csv(audio_dir, tmp_path) -> None:
    meta = audio_dir / "meta.csv"
    meta.write_text("file_name,label\na.wav,x\nmissing.wav,y\n")
    out = tmp_path / "out"
    result = _ok("batch", "audio", "info", "--input-file", str(meta), "--output-dir", str(out))
    assert (result["total_files"], result["failed"]) == (2, 1)
    assert "missing.wav" in result["errors"][0]
    assert Path(result["metadata_csv"]).exists()
    assert result["output_dir"] == str(out)  # CSV mode reports the directory given


def test_detect_batch(audio_dir, tmp_path) -> None:
    pytest.importorskip("librosa")
    out = tmp_path / "out"
    result = _ok(
        "batch", "detect", "energy", "--input-dir", str(audio_dir), "--output-dir", str(out)
    )
    assert result["successful"] == 2
    assert result["results_file"] == str(out / "detections_energy.json")


def test_visualize_uses_the_same_keys(audio_dir, tmp_path) -> None:
    pytest.importorskip("matplotlib")
    pytest.importorskip("librosa")
    result = _ok(
        "batch",
        "audio",
        "visualize",
        "--input-dir",
        str(audio_dir),
        "--output-dir",
        str(tmp_path / "v"),
    )
    assert (result["total_files"], result["successful"], result["failed"]) == (2, 2, 0)


def test_detect_without_output_dir_reports_where_it_wrote(audio_dir) -> None:
    pytest.importorskip("librosa")
    result = _ok("batch", "detect", "energy", "--input-dir", str(audio_dir))
    assert result["output_dir"] == str(audio_dir)
    assert Path(result["results_file"]).exists()


def test_results_file_is_null_when_nothing_was_written(tmp_path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    result = _ok(
        "batch", "audio", "metadata", "--input-dir", str(empty), "--output-dir", str(tmp_path / "o")
    )
    assert result["results_file"] is None


def test_validate_with_invalid_files_is_a_failure(audio_dir, tmp_path) -> None:
    (audio_dir / "broken.wav").write_bytes(b"RIFF....WAVEjunk")
    result = CliRunner().invoke(
        cli, ["--json", "batch", "audio", "validate", "--input-dir", str(audio_dir)]
    )
    assert result.exit_code == 1
    envelope = json.loads(result.stdout)
    assert envelope["ok"] is False  # not ok:true followed by exit 1
    assert envelope["error"]["details"]["failed"] == 1
    assert "broken.wav" in envelope["error"]["details"]["errors"][0]
