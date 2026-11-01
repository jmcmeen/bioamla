"""CLI tests for `bioamla batch` commands.

Two input modes are exercised per command family:

- Directory mode (`--input-dir`): the heavy domain `batch_*` helper is mocked
  (patched where the command body imports it from) and we assert the report
  output / branching.
- CSV-metadata mode (`--input-file`): the real `bioamla.batch` engine
  (`load_csv`/`run_csv_batch`/`write_csv`) runs over a small CSV pointing at a
  real audio file, with only the per-file domain work mocked.

No network/model downloads occur; ML/audio backends are mocked.
"""

import csv
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
from click.testing import CliRunner

from bioamla.batch import BatchResult
from bioamla.cli.cli import cli
from bioamla.exceptions import ProcessingError


@pytest.fixture
def runner():
    return CliRunner()


def _result(n=2, ok=2, failed=0, errors=None):
    return BatchResult(total_files=n, successful=ok, failed=failed, errors=errors or [])


@pytest.fixture
def csv_with_audio(tmp_path, test_audio_path):
    """A metadata CSV (with file_name column) referencing a real audio file."""
    csv_path = tmp_path / "meta.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["file_name", "label"])
        w.writeheader()
        w.writerow({"file_name": test_audio_path, "label": "frog"})
    return str(csv_path)


# --------------------------------------------------------------------------- #
# help / registration
# --------------------------------------------------------------------------- #


def test_batch_group_help(runner):
    result = runner.invoke(cli, ["batch", "--help"])
    assert result.exit_code == 0
    for grp in ("audio", "detect", "index", "models", "cluster"):
        assert grp in result.output


@pytest.mark.parametrize(
    ("group", "cmd"),
    [
        ("audio", "info"),
        ("audio", "validate"),
        ("audio", "metadata"),
        ("audio", "convert"),
        ("audio", "resample"),
        ("audio", "normalize"),
        ("audio", "trim"),
        ("audio", "filter"),
        ("audio", "denoise"),
        ("audio", "segment"),
        ("audio", "visualize"),
        ("detect", "energy"),
        ("detect", "ribbit"),
        ("detect", "peaks"),
        ("detect", "accelerating"),
        ("models", "predict"),
        ("models", "embed"),
    ],
)
def test_batch_subcommand_help(runner, group, cmd):
    result = runner.invoke(cli, ["batch", group, cmd, "--help"])
    assert result.exit_code == 0


def test_batch_index_help(runner):
    result = runner.invoke(cli, ["batch", "index", "--help"])
    assert result.exit_code == 0


def test_batch_cluster_help(runner):
    result = runner.invoke(cli, ["batch", "cluster", "--help"])
    assert result.exit_code == 0


# --------------------------------------------------------------------------- #
# input option validation (shared decorator)
# --------------------------------------------------------------------------- #


def test_requires_input(runner):
    result = runner.invoke(cli, ["batch", "audio", "info"])
    assert result.exit_code != 0


def test_mutually_exclusive_inputs(runner, test_audio_dir, csv_with_audio):
    result = runner.invoke(
        cli,
        ["batch", "audio", "info", "--input-dir", test_audio_dir, "--input-file", csv_with_audio],
    )
    assert result.exit_code != 0


# --------------------------------------------------------------------------- #
# audio info — directory + CSV
# --------------------------------------------------------------------------- #


def _audio_info_obj():
    info = MagicMock()
    info.duration = 1.0
    info.sample_rate = 16000
    info.channels = 1
    info.samples = 16000
    info.format = "WAV"
    return info


def test_audio_info_dir(runner, test_audio_dir, tmp_path, mocker):
    mocker.patch(
        "bioamla.audio.discovery.list_audio_files",
        return_value=[Path(test_audio_dir) / "audio_0.wav"],
    )
    mocker.patch("bioamla.audio.info.get_audio_info", return_value=_audio_info_obj())
    out = tmp_path / "out"
    result = runner.invoke(
        cli,
        ["batch", "audio", "info", "--input-dir", test_audio_dir, "--output-dir", str(out)],
    )
    assert result.exit_code == 0
    assert "Processed" in result.output
    assert (out / "audio_info.csv").exists()


def test_audio_info_csv(runner, csv_with_audio, mocker):
    mocker.patch("bioamla.audio.info.get_audio_info", return_value=_audio_info_obj())
    result = runner.invoke(cli, ["batch", "audio", "info", "--input-file", csv_with_audio])
    assert result.exit_code == 0
    assert "Updated metadata CSV written to" in result.output
    # The CSV should have been augmented with the analysis columns.
    rows = list(csv.DictReader(open(csv_with_audio)))
    assert "duration" in rows[0]


def test_audio_info_error(runner, test_audio_dir, mocker):
    mocker.patch(
        "bioamla.audio.batch.list_audio_files",
        side_effect=ProcessingError("listfail"),
    )
    result = runner.invoke(cli, ["batch", "audio", "info", "--input-dir", test_audio_dir])
    assert result.exit_code != 0


# --------------------------------------------------------------------------- #
# audio validate / metadata (real riffy, no mocks)
# --------------------------------------------------------------------------- #


def _audiomoth_wav(path: Path) -> None:
    """Write a PCM WAV carrying an AudioMoth INFO comment."""
    import wave

    from riffy import InfoMetadata, WAVParser

    src = path.with_name("src_" + path.name)
    with wave.open(str(src), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(8000)
        w.writeframes(b"\x00" * 800)
    parser = WAVParser(src)
    info = InfoMetadata()
    info.set("IART", "AudioMoth 248D9B045EC9EE79")
    info.set(
        "ICMT",
        "Recorded at 19:30:00 12/11/2021 (UTC) by AudioMoth 248D9B045EC9EE79 "
        "at medium gain while battery was 4.1V and temperature was 14.0C.",
    )
    info.write_to_parser(parser)
    parser.write_wav(path)
    src.unlink()


def test_audio_validate_dir_all_valid(runner, test_audio_dir, tmp_path):
    out = tmp_path / "out"
    result = runner.invoke(
        cli,
        ["batch", "audio", "validate", "--input-dir", test_audio_dir, "--output-dir", str(out)],
    )
    assert result.exit_code == 0, result.output
    assert "3 successful, 0 failed" in result.output
    rows = list(csv.DictReader(open(out / "validation_report.csv")))
    assert [r["valid"] for r in rows] == ["true", "true", "true"]
    assert list(rows[0]) == ["file_name", "valid", "error"]


def test_audio_validate_dir_invalid_and_skipped(runner, test_audio_dir, tmp_path, caplog):
    (Path(test_audio_dir) / "broken.wav").write_bytes(b"RIFF\x00\x00\x00\x00WAVEnope")
    (Path(test_audio_dir) / "clip.flac").write_bytes(b"fLaC")
    out = tmp_path / "out"
    result = runner.invoke(
        cli,
        ["batch", "audio", "validate", "--input-dir", test_audio_dir, "--output-dir", str(out)],
    )
    assert result.exit_code == 1, result.output
    assert "1 failed" in result.output
    # The failure itself is reported as a warning from the batch engine.
    assert any("broken.wav" in r.getMessage() for r in caplog.records if r.levelname == "WARNING")
    by_name = {
        Path(r["file_name"]).name: r for r in csv.DictReader(open(out / "validation_report.csv"))
    }
    assert by_name["broken.wav"]["valid"] == "false"
    assert by_name["broken.wav"]["error"]
    assert by_name["clip.flac"]["valid"] == "skipped"
    assert by_name["audio_0.wav"]["valid"] == "true"
    assert by_name["audio_0.wav"]["error"] == ""


def test_audio_validate_dir_no_output(runner, test_audio_dir):
    result = runner.invoke(cli, ["batch", "audio", "validate", "--input-dir", test_audio_dir])
    assert result.exit_code == 0, result.output
    assert "Results saved" not in result.output


def test_audio_validate_csv(runner, csv_with_audio):
    result = runner.invoke(cli, ["batch", "audio", "validate", "--input-file", csv_with_audio])
    assert result.exit_code == 0, result.output
    rows = list(csv.DictReader(open(csv_with_audio)))
    assert rows[0]["valid"] == "true"
    assert rows[0]["label"] == "frog"


def test_audio_validate_csv_invalid_exits_1(runner, tmp_path):
    bad = tmp_path / "bad.wav"
    bad.write_bytes(b"not riff")
    csv_path = tmp_path / "meta.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["file_name"])
        w.writeheader()
        w.writerow({"file_name": str(bad)})
    result = runner.invoke(cli, ["batch", "audio", "validate", "--input-file", str(csv_path)])
    assert result.exit_code == 1, result.output
    rows = list(csv.DictReader(open(csv_path)))
    assert rows[0]["valid"] == "false"


def test_audio_metadata_dir(runner, test_audio_dir, tmp_path):
    _audiomoth_wav(Path(test_audio_dir) / "moth.wav")
    (Path(test_audio_dir) / "clip.flac").write_bytes(b"fLaC")
    out = tmp_path / "out"
    result = runner.invoke(
        cli,
        ["batch", "audio", "metadata", "--input-dir", test_audio_dir, "--output-dir", str(out)],
    )
    assert result.exit_code == 0, result.output
    assert "4 successful" in result.output
    rows = {
        Path(r["file_name"]).name: r for r in csv.DictReader(open(out / "recorder_metadata.csv"))
    }
    assert "clip.flac" not in rows
    assert rows["moth.wav"]["recorded_at"] == "2021-11-12T19:30:00+00:00"
    assert rows["moth.wav"]["device_id"] == "248D9B045EC9EE79"
    assert rows["moth.wav"]["metadata_sources"] == "info+audiomoth"
    assert rows["audio_0.wav"]["recorded_at"] == ""
    assert rows["audio_0.wav"]["metadata_sources"] == ""
    header = next(csv.reader(open(out / "recorder_metadata.csv")))
    assert header[:4] == ["file_name", "recorded_at", "latitude", "longitude"]


def test_audio_metadata_csv(runner, tmp_path):
    moth = tmp_path / "moth.wav"
    _audiomoth_wav(moth)
    other = tmp_path / "clip.flac"
    other.write_bytes(b"fLaC")
    csv_path = tmp_path / "meta.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["file_name", "label"])
        w.writeheader()
        w.writerow({"file_name": str(moth), "label": "frog"})
        w.writerow({"file_name": str(other), "label": "bird"})
    result = runner.invoke(cli, ["batch", "audio", "metadata", "--input-file", str(csv_path)])
    assert result.exit_code == 0, result.output
    rows = list(csv.DictReader(open(csv_path)))
    assert rows[0]["device_model"] == "AudioMoth"
    assert rows[0]["label"] == "frog"
    assert rows[1]["device_model"] == ""
    assert rows[1]["label"] == "bird"
    header = next(csv.reader(open(csv_path)))
    assert header[:5] == ["file_name", "label", "recorded_at", "latitude", "longitude"]
    assert header[-1] == "metadata_sources"


def test_audio_metadata_error(runner, test_audio_dir, mocker):
    mocker.patch(
        "bioamla.audio.batch.list_audio_files",
        side_effect=ProcessingError("listfail"),
    )
    result = runner.invoke(cli, ["batch", "audio", "metadata", "--input-dir", test_audio_dir])
    assert result.exit_code != 0
    assert "listfail" in result.output


# --------------------------------------------------------------------------- #
# audio convert / resample / normalize / trim / filter / denoise — directory
# --------------------------------------------------------------------------- #


def test_audio_convert_dir(runner, test_audio_dir, tmp_path, mocker):
    m = mocker.patch("bioamla.audio.batch.batch_convert_files", return_value=_result())
    out = tmp_path / "out"
    result = runner.invoke(
        cli,
        [
            "batch",
            "audio",
            "convert",
            "--input-dir",
            test_audio_dir,
            "--output-dir",
            str(out),
            "-f",
            "flac",
        ],
    )
    assert result.exit_code == 0
    assert "Processed" in result.output
    m.assert_called_once()


def test_audio_convert_dir_requires_output(runner, test_audio_dir, mocker):
    mocker.patch("bioamla.audio.batch.batch_convert_files", return_value=_result())
    result = runner.invoke(cli, ["batch", "audio", "convert", "--input-dir", test_audio_dir])
    assert result.exit_code != 0


def test_audio_resample_dir(runner, test_audio_dir, tmp_path, mocker):
    m = mocker.patch("bioamla.audio.batch.batch_transform_files", return_value=_result())
    out = tmp_path / "out"
    result = runner.invoke(
        cli,
        [
            "batch",
            "audio",
            "resample",
            "--input-dir",
            test_audio_dir,
            "--output-dir",
            str(out),
            "-r",
            "8000",
        ],
    )
    assert result.exit_code == 0
    m.assert_called_once()


def test_audio_normalize_dir_peak(runner, test_audio_dir, tmp_path, mocker):
    m = mocker.patch("bioamla.audio.batch.batch_transform_files", return_value=_result())
    out = tmp_path / "out"
    result = runner.invoke(
        cli,
        [
            "batch",
            "audio",
            "normalize",
            "--input-dir",
            test_audio_dir,
            "--output-dir",
            str(out),
            "--peak",
        ],
    )
    assert result.exit_code == 0
    m.assert_called_once()


def test_audio_normalize_dir_rms(runner, test_audio_dir, tmp_path, mocker):
    mocker.patch("bioamla.audio.batch.batch_transform_files", return_value=_result())
    out = tmp_path / "out"
    result = runner.invoke(
        cli,
        ["batch", "audio", "normalize", "--input-dir", test_audio_dir, "--output-dir", str(out)],
    )
    assert result.exit_code == 0


def test_audio_trim_dir(runner, test_audio_dir, tmp_path, mocker):
    mocker.patch("bioamla.audio.batch.batch_transform_files", return_value=_result())
    out = tmp_path / "out"
    result = runner.invoke(
        cli,
        [
            "batch",
            "audio",
            "trim",
            "--input-dir",
            test_audio_dir,
            "--output-dir",
            str(out),
            "-s",
            "0.0",
            "-e",
            "0.5",
        ],
    )
    assert result.exit_code == 0


def test_audio_trim_dir_silence(runner, test_audio_dir, tmp_path, mocker):
    mocker.patch("bioamla.audio.batch.batch_transform_files", return_value=_result())
    out = tmp_path / "out"
    result = runner.invoke(
        cli,
        [
            "batch",
            "audio",
            "trim",
            "--input-dir",
            test_audio_dir,
            "--output-dir",
            str(out),
            "--trim-silence",
        ],
    )
    assert result.exit_code == 0


def test_audio_filter_dir_lowpass(runner, test_audio_dir, tmp_path, mocker):
    mocker.patch("bioamla.audio.batch.batch_transform_files", return_value=_result())
    out = tmp_path / "out"
    result = runner.invoke(
        cli,
        [
            "batch",
            "audio",
            "filter",
            "--input-dir",
            test_audio_dir,
            "--output-dir",
            str(out),
            "--lowpass",
            "4000",
        ],
    )
    assert result.exit_code == 0


def test_audio_filter_dir_bandpass(runner, test_audio_dir, tmp_path, mocker):
    mocker.patch("bioamla.audio.batch.batch_transform_files", return_value=_result())
    out = tmp_path / "out"
    result = runner.invoke(
        cli,
        [
            "batch",
            "audio",
            "filter",
            "--input-dir",
            test_audio_dir,
            "--output-dir",
            str(out),
            "--bandpass-low",
            "500",
            "--bandpass-high",
            "4000",
        ],
    )
    assert result.exit_code == 0


def test_audio_filter_no_option(runner, test_audio_dir):
    result = runner.invoke(cli, ["batch", "audio", "filter", "--input-dir", test_audio_dir])
    assert result.exit_code != 0


def test_audio_filter_bandpass_incomplete(runner, test_audio_dir):
    result = runner.invoke(
        cli,
        ["batch", "audio", "filter", "--input-dir", test_audio_dir, "--bandpass-low", "500"],
    )
    assert result.exit_code != 0


def test_audio_denoise_dir(runner, test_audio_dir, tmp_path, mocker):
    mocker.patch("bioamla.audio.batch.batch_transform_files", return_value=_result())
    out = tmp_path / "out"
    result = runner.invoke(
        cli,
        ["batch", "audio", "denoise", "--input-dir", test_audio_dir, "--output-dir", str(out)],
    )
    assert result.exit_code == 0


# --------------------------------------------------------------------------- #
# audio segment / visualize — directory
# --------------------------------------------------------------------------- #


def test_audio_segment_dir(runner, test_audio_dir, tmp_path, mocker):
    mocker.patch(
        "bioamla.audio.discovery.list_audio_files",
        return_value=[Path(test_audio_dir) / "audio_0.wav"],
    )
    seg = mocker.patch("bioamla.audio.batch.segment_audio_file", return_value=[])
    out = tmp_path / "segs"
    result = runner.invoke(
        cli,
        [
            "batch",
            "audio",
            "segment",
            "--input-dir",
            test_audio_dir,
            "--output-dir",
            str(out),
            "-d",
            "1.0",
        ],
    )
    assert result.exit_code == 0
    seg.assert_called()


def test_audio_segment_bad_duration(runner, test_audio_dir, tmp_path):
    out = tmp_path / "segs"
    result = runner.invoke(
        cli,
        [
            "batch",
            "audio",
            "segment",
            "--input-dir",
            test_audio_dir,
            "--output-dir",
            str(out),
            "-d",
            "0",
        ],
    )
    assert result.exit_code != 0


def test_audio_visualize_dir(runner, test_audio_dir, tmp_path, mocker):
    mocker.patch(
        "bioamla.viz.batch_generate_spectrograms",
        return_value={"files_processed": 3, "files_failed": 0},
    )
    out = tmp_path / "viz"
    result = runner.invoke(
        cli,
        ["batch", "audio", "visualize", "--input-dir", test_audio_dir, "--output-dir", str(out)],
    )
    assert result.exit_code == 0
    assert "Processed 3 files" in result.output


def test_audio_visualize_requires_output(runner, test_audio_dir):
    result = runner.invoke(cli, ["batch", "audio", "visualize", "--input-dir", test_audio_dir])
    assert result.exit_code != 0


# --------------------------------------------------------------------------- #
# audio CSV-mode transforms (real engine, mocked per-file domain work)
# --------------------------------------------------------------------------- #


def test_audio_resample_csv(runner, csv_with_audio, mocker):
    mocker.patch(
        "bioamla.audio.io.process_file",
        side_effect=lambda inp, outp, proc, sr: outp,
    )
    result = runner.invoke(
        cli, ["batch", "audio", "resample", "--input-file", csv_with_audio, "-r", "8000"]
    )
    assert result.exit_code == 0
    assert "Updated metadata CSV written to" in result.output


def test_audio_convert_csv(runner, csv_with_audio, mocker):
    mocker.patch(
        "bioamla.audio.convert.convert_audio_file",
        side_effect=lambda inp, outp, **kw: outp,
    )
    result = runner.invoke(
        cli, ["batch", "audio", "convert", "--input-file", csv_with_audio, "-f", "flac"]
    )
    assert result.exit_code == 0


# --------------------------------------------------------------------------- #
# detect — directory + CSV + error
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("method", ["energy", "ribbit", "peaks", "accelerating"])
def test_detect_dir(runner, test_audio_dir, tmp_path, mocker, method):
    m = mocker.patch("bioamla.detect.batch.batch_detect_dir", return_value=_result())
    out = tmp_path / "out"
    result = runner.invoke(
        cli,
        ["batch", "detect", method, "--input-dir", test_audio_dir, "--output-dir", str(out)],
    )
    assert result.exit_code == 0
    assert "Processed" in result.output
    m.assert_called_once()
    assert m.call_args.kwargs["method"] == method


def test_detect_dir_default_output(runner, test_audio_dir, mocker):
    # When no --output-dir, it falls back to the input dir.
    m = mocker.patch("bioamla.detect.batch.batch_detect_dir", return_value=_result())
    result = runner.invoke(cli, ["batch", "detect", "energy", "--input-dir", test_audio_dir])
    assert result.exit_code == 0
    assert m.call_args.args[1] == test_audio_dir


def test_detect_dir_error(runner, test_audio_dir, mocker):
    mocker.patch("bioamla.detect.batch.batch_detect_dir", side_effect=ProcessingError("detfail"))
    result = runner.invoke(cli, ["batch", "detect", "energy", "--input-dir", test_audio_dir])
    assert result.exit_code != 0


def test_detect_csv(runner, csv_with_audio, mocker):
    detector = MagicMock()
    detector.detect_from_file.return_value = []
    mocker.patch("bioamla.detect.batch._build_detector", return_value=detector)
    result = runner.invoke(cli, ["batch", "detect", "energy", "--input-file", csv_with_audio])
    assert result.exit_code == 0
    detector.detect_from_file.assert_called()


# --------------------------------------------------------------------------- #
# indices calculate — directory + CSV
# --------------------------------------------------------------------------- #


def test_indices_calculate_dir(runner, test_audio_dir, tmp_path, mocker):
    mocker.patch(
        "bioamla.indices.batch.list_audio_files",
        return_value=["a.wav", "b.wav"],
    )

    def fake_compute(path, **kwargs):
        if path == "b.wav":
            raise ProcessingError("boom")
        return mocker.Mock(to_dict=lambda: {"aci": 1.0})

    mocker.patch("bioamla.indices.compute.compute_indices_from_file", side_effect=fake_compute)
    out = tmp_path / "out"
    result = runner.invoke(
        cli,
        ["batch", "index", "--input-dir", test_audio_dir, "--output-dir", str(out)],
    )
    assert result.exit_code == 0
    assert (out / "indices.csv").exists()
    assert "1 successful, 1 failed" in result.output


def test_indices_calculate_csv(runner, csv_with_audio, mocker):
    fake = MagicMock()
    fake.to_dict.return_value = {"aci": 1.0, "adi": 0.5, "filepath": "x", "filename": "x"}
    mocker.patch("bioamla.indices.compute_indices_from_file", return_value=fake)
    result = runner.invoke(cli, ["batch", "index", "--input-file", csv_with_audio])
    assert result.exit_code == 0
    rows = list(csv.DictReader(open(csv_with_audio)))
    assert "aci" in rows[0]


def test_indices_calculate_error(runner, test_audio_dir, mocker):
    mocker.patch(
        "bioamla.indices.batch.list_audio_files",
        side_effect=ProcessingError("x"),
    )
    result = runner.invoke(cli, ["batch", "index", "--input-dir", test_audio_dir])
    assert result.exit_code != 0


# --------------------------------------------------------------------------- #
# models predict / embed
# --------------------------------------------------------------------------- #


def test_models_predict_dir(runner, test_audio_dir, tmp_path, mocker):
    res = _result()
    res.metadata = {
        "predictions": [{"filepath": "a.wav", "predicted_label": "frog", "confidence": 0.9}]
    }
    mocker.patch("bioamla.ml.batch.batch_predict_files", return_value=res)
    out = tmp_path / "out"
    result = runner.invoke(
        cli,
        [
            "batch",
            "models",
            "predict",
            "--input-dir",
            test_audio_dir,
            "--output-dir",
            str(out),
            "-m",
            "some/model",
        ],
    )
    assert result.exit_code == 0
    assert (out / "predictions.json").exists()


def test_models_predict_requires_model(runner, test_audio_dir):
    result = runner.invoke(cli, ["batch", "models", "predict", "--input-dir", test_audio_dir])
    assert result.exit_code != 0


def test_models_predict_segments_dir(runner, test_audio_dir, tmp_path, mocker):
    res = _result()
    res.metadata = {
        "segments": [
            {
                "filepath": "a.wav",
                "start_time": 0.0,
                "end_time": 3.0,
                "predicted_label": "frog",
                "confidence": 0.8,
            }
        ]
    }
    mocker.patch("bioamla.ml.batch.batch_predict_segments", return_value=res)
    out = tmp_path / "out"
    result = runner.invoke(
        cli,
        [
            "batch",
            "models",
            "predict",
            "--input-dir",
            test_audio_dir,
            "--output-dir",
            str(out),
            "-m",
            "some/model",
            "--segment-seconds",
            "3",
        ],
    )
    assert result.exit_code == 0
    assert (out / "predictions.csv").exists()


def test_models_predict_error(runner, test_audio_dir, mocker):
    mocker.patch("bioamla.ml.batch.batch_predict_files", side_effect=ProcessingError("mfail"))
    result = runner.invoke(
        cli,
        ["batch", "models", "predict", "--input-dir", test_audio_dir, "-m", "some/model"],
    )
    assert result.exit_code != 0


def test_models_embed_dir(runner, test_audio_dir, tmp_path, mocker):
    mocker.patch("bioamla.ml.batch.batch_embed_files", return_value=_result())
    out = tmp_path / "out"
    result = runner.invoke(
        cli,
        [
            "batch",
            "models",
            "embed",
            "--input-dir",
            test_audio_dir,
            "--output-dir",
            str(out),
            "-m",
            "some/model",
        ],
    )
    assert result.exit_code == 0
    assert "Processed" in result.output


def test_models_embed_requires_output(runner, test_audio_dir):
    result = runner.invoke(
        cli,
        ["batch", "models", "embed", "--input-dir", test_audio_dir, "-m", "some/model"],
    )
    assert result.exit_code != 0


def test_models_embed_csv(runner, csv_with_audio, tmp_path, mocker):
    extractor = MagicMock()
    emb = MagicMock()
    emb.embeddings = np.zeros(8, dtype=np.float32)
    extractor.extract.return_value = emb
    mocker.patch("bioamla.ml.embedding.EmbeddingExtractor", return_value=extractor)
    mocker.patch("bioamla.ml.embedding.EmbeddingConfig", MagicMock())
    out = tmp_path / "emb"
    result = runner.invoke(
        cli,
        [
            "batch",
            "models",
            "embed",
            "--input-file",
            csv_with_audio,
            "--output-dir",
            str(out),
            "-m",
            "some/model",
        ],
    )
    assert result.exit_code == 0
    extractor.extract.assert_called()


# --------------------------------------------------------------------------- #
# cluster
# --------------------------------------------------------------------------- #


def _cluster_result(files=()):
    """What a successful clustering run returns: assignments per file."""
    result = _result()
    result.metadata = {
        "output_file": "out/cluster_assignments.json",
        "file_assignments": [{"filepath": str(f), "cluster_labels": [1]} for f in files],
    }
    return result


def test_cluster_dir(runner, test_audio_dir, tmp_path, mocker):
    m = mocker.patch("bioamla.cluster.batch.cluster_batch_files", return_value=_cluster_result())
    out = tmp_path / "out"
    result = runner.invoke(
        cli,
        [
            "batch",
            "cluster",
            "--input-dir",
            test_audio_dir,
            "--output-dir",
            str(out),
            "--method",
            "kmeans",
            "--n-clusters",
            "3",
        ],
    )
    assert result.exit_code == 0
    m.assert_called_once()


def test_cluster_requires_output(runner, test_audio_dir):
    result = runner.invoke(cli, ["batch", "cluster", "--input-dir", test_audio_dir])
    assert result.exit_code != 0


def test_cluster_error(runner, test_audio_dir, tmp_path, mocker):
    mocker.patch("bioamla.cluster.batch.cluster_batch_files", side_effect=ProcessingError("cfail"))
    out = tmp_path / "out"
    result = runner.invoke(
        cli,
        ["batch", "cluster", "--input-dir", test_audio_dir, "--output-dir", str(out)],
    )
    assert result.exit_code != 0


def test_cluster_csv(runner, csv_with_audio, tmp_path, mocker):
    mocker.patch("bioamla.cluster.batch.cluster_embedding_files", return_value=_cluster_result())
    out = tmp_path / "out"
    result = runner.invoke(
        cli,
        [
            "batch",
            "cluster",
            "--input-file",
            csv_with_audio,
            "--output-dir",
            str(out),
        ],
    )
    assert result.exit_code == 0


def test_indices_selection_applies(runner, test_audio_dir, tmp_path):
    out = tmp_path / "out"
    result = runner.invoke(
        cli,
        [
            "batch",
            "index",
            "--input-dir",
            test_audio_dir,
            "--output-dir",
            str(out),
            "--indices",
            "aci,ndsi,h_spectral",
        ],
    )
    assert result.exit_code == 0, result.output
    header = (out / "indices.csv").read_text().splitlines()[0].split(",")
    assert header == [
        "filepath",
        "aci",
        "ndsi",
        "anthrophony",
        "biophony",
        "sample_rate",
        "duration",
        "h_spectral",
        "success",
    ]


def test_indices_unknown_name_rejected(runner, test_audio_dir):
    result = runner.invoke(
        cli, ["batch", "index", "--input-dir", test_audio_dir, "--indices", "aci,bogus"]
    )
    assert result.exit_code != 0
    assert "Unknown indices: bogus" in result.output


def test_indices_default_columns_unchanged(runner, test_audio_dir, tmp_path):
    out = tmp_path / "out"
    result = runner.invoke(
        cli, ["batch", "index", "--input-dir", test_audio_dir, "--output-dir", str(out)]
    )
    assert result.exit_code == 0, result.output
    header = (out / "indices.csv").read_text().splitlines()[0].split(",")
    assert header == [
        "filepath",
        "aci",
        "adi",
        "aei",
        "bio",
        "ndsi",
        "anthrophony",
        "biophony",
        "sample_rate",
        "duration",
        "success",
    ]


def test_indices_empty_selection_rejected(runner, test_audio_dir):
    result = runner.invoke(cli, ["batch", "index", "--input-dir", test_audio_dir, "--indices", ","])
    assert result.exit_code != 0
    assert "at least one index" in result.output
