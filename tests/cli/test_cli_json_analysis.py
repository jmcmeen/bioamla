"""`bioamla --json` results for the detect, indices and audio info commands."""

import json

import pytest
from click.testing import CliRunner

from bioamla.cli.cli import cli


def _json(*args: str) -> dict:
    result = CliRunner().invoke(cli, ["--json", *args])
    assert result.exit_code == 0, result.output
    envelope = json.loads(result.stdout)
    assert envelope["ok"] is True
    return envelope["result"]


@pytest.mark.parametrize("command", ["energy", "ribbit", "peaks", "accelerating"])
def test_detect(test_audio_path, command) -> None:
    pytest.importorskip("librosa")
    result = _json("detect", command, test_audio_path)
    assert result["file"] == test_audio_path
    assert result["count"] == len(result["detections"])  # same key for every detector
    assert result["output"] is None
    assert "output_format" not in result["params"]


def test_detect_peaks_items_match_the_csv_columns(test_audio_path, tmp_path) -> None:
    pytest.importorskip("librosa")
    out = tmp_path / "p.csv"
    result = _json("detect", "peaks", test_audio_path, "-o", str(out))
    header = out.read_text().splitlines()[0].split(",")
    for item in result["detections"]:
        assert list(item) == header


def test_detect_params_use_option_names_and_report_output(test_audio_path, tmp_path) -> None:
    pytest.importorskip("librosa")
    out = tmp_path / "d.csv"
    result = _json("detect", "energy", test_audio_path, "--threshold-db", "-30", "-o", str(out))
    assert result["params"]["threshold_db"] == -30.0  # the option name, not the parameter name
    assert result["output"] == str(out) and out.exists()


@pytest.mark.parametrize(
    ("index", "_"), [("aci", 0), ("adi", 0), ("aei", 0), ("bio", 0), ("ndsi", 0)]
)
def test_single_index(test_audio_path, index, _) -> None:
    pytest.importorskip("librosa")
    result = _json("indices", index, test_audio_path)
    assert result["index"] == index
    assert "output" not in result  # these commands have no -o
    assert isinstance(result["value"], (int, float)) or result["value"] is None


def test_indices_compute_entropy_and_temporal(test_audio_path) -> None:
    pytest.importorskip("librosa")
    assert {"aci", "adi", "ndsi"} <= set(_json("indices", "compute", test_audio_path)["indices"])
    assert set(_json("indices", "entropy", test_audio_path)["indices"]) == {
        "h_spectral",
        "h_temporal",
    }
    segments = _json("indices", "temporal", test_audio_path, "--window-seconds", "0.5")["segments"]
    assert segments and {"segment", "start_time", "aci"} <= set(segments[0])


def test_audio_info_and_list(test_audio_path) -> None:
    from pathlib import Path

    info = _json("audio", "info", test_audio_path, "--metadata")
    assert info["file"] == test_audio_path
    assert info["info"]["sample_rate"] > 0
    assert info["recording_metadata"] == {}  # a WAV with no recorder chunks
    assert "recording_metadata" not in _json("audio", "info", test_audio_path)

    listing = _json("audio", "list", str(Path(test_audio_path).parent))
    assert listing["count"] == len(listing["files"]) >= 1


def test_human_output_unchanged_without_json(test_audio_path) -> None:
    pytest.importorskip("librosa")
    result = CliRunner().invoke(cli, ["indices", "aci", test_audio_path])
    assert result.exit_code == 0
    assert not result.stdout.lstrip().startswith("{")


def test_audio_info_metadata_null_for_non_wav(tmp_path, mocker) -> None:
    from bioamla.audio.info import AudioInfo
    from bioamla.exceptions import InvalidInputError

    flac = tmp_path / "a.flac"
    flac.write_bytes(b"")
    mocker.patch(
        "bioamla.audio.get_audio_info",
        return_value=AudioInfo(
            duration=1.0, sample_rate=16000, channels=1, samples=16000, format="FLAC"
        ),
    )
    mocker.patch("bioamla.audio.get_recording_metadata", side_effect=InvalidInputError("not a WAV"))
    assert _json("audio", "info", str(flac), "--metadata")["recording_metadata"] is None
