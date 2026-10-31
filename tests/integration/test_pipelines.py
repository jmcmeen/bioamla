"""Pipelines: one command's output fed to the next, through ``bioamla --json``.

Each step reads the paths and values the previous step's JSON envelope reported,
the way a script or an agent chains commands. Unit tests cover each command on
its own; these cover the hand-offs between them.
"""

import csv
import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from click.testing import CliRunner

from bioamla.cli.cli import cli

pytestmark = pytest.mark.integration

SR = 22050


def bioamla(*args: str) -> dict:
    """Run ``bioamla --json <args>``; return the result, failing on an error envelope."""
    run = CliRunner().invoke(cli, ["--json", *args])
    envelope = json.loads(run.stdout)
    assert envelope["ok"], envelope.get("error")
    return envelope["result"]


def bioamla_error(*args: str) -> dict:
    """Run ``bioamla --json <args>`` expecting failure; return the error."""
    run = CliRunner().invoke(cli, ["--json", *args])
    envelope = json.loads(run.stdout)
    assert not envelope["ok"]
    return envelope["error"]


def read_csv(path: str | Path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


@pytest.fixture
def recording(tmp_path: Path) -> Path:
    """Eight seconds: seven chirped calls, alternating low and high, over a faint hum."""
    t = np.arange(SR * 8) / SR
    x = 0.003 * np.sin(2 * np.pi * 60 * t)
    for i, start in enumerate(np.arange(0.5, 7.5, 1.0)):
        m = (t >= start) & (t < start + 0.4)
        tt = t[m] - start
        f0 = 2000 if i % 2 else 4000
        x[m] += 0.5 * np.sin(2 * np.pi * (f0 * tt + 3000 * tt**2)) * np.hanning(m.sum())
    path = tmp_path / "calls.wav"
    sf.write(path, x, SR)
    return path


@pytest.fixture(autouse=True)
def _needs_audio_extra() -> None:
    pytest.importorskip("librosa")


class TestDetectionsToDataset:
    @pytest.mark.parametrize("fmt", ["csv", "json"])
    def test_detections_become_clips_with_their_band(self, recording, tmp_path, fmt) -> None:
        detections = bioamla(
            "detect", "energy", str(recording), "-o", str(tmp_path / f"det.{fmt}"),
            "--format", fmt,
        )  # fmt: skip
        assert detections["count"] == 7

        summary = bioamla("annotation", "summary", detections["output"])
        assert summary["total_annotations"] == detections["count"]

        clips = bioamla(
            "dataset", "extract-clips", str(recording), str(tmp_path / "clips"),
            "--annotations", detections["output"],
        )  # fmt: skip
        assert clips["clips_written"] == detections["count"]
        rows = read_csv(clips["metadata_file"])
        assert {r["low_freq"] for r in rows} == {"500.0"}  # the detector's band survives
        assert {r["high_freq"] for r in rows} == {"5000.0"}
        for row in rows:
            assert (Path(clips["output_dir"]) / row["file_name"]).is_file()

        stats = bioamla("dataset", "stats", clips["metadata_file"])  # the CSV path itself
        assert stats["total_files"] == clips["clips_written"]

    def test_partition_reports_splits_it_cannot_fill(self, recording, tmp_path) -> None:
        det = bioamla("detect", "energy", str(recording), "-o", str(tmp_path / "d.csv"))
        clips = bioamla(
            "dataset", "extract-clips", str(recording), str(tmp_path / "clips"),
            "--annotations", det["output"],
        )  # fmt: skip

        grouped = bioamla("dataset", "partition", clips["output_dir"], "--mode", "column")
        assert grouped["groups"] == 1  # every clip comes from the one recording
        assert grouped["splits"] == {"train": 7}
        assert grouped["empty_splits"] == ["val", "test"]

        by_row = bioamla(
            "dataset", "partition", clips["output_dir"], "--mode", "column", "--group-by", "",
            "--train", "0.6", "--val", "0.2", "--test", "0.2",
        )  # fmt: skip
        assert by_row["empty_splits"] == []
        stats = bioamla("dataset", "stats", clips["output_dir"])
        assert sum(stats["splits"].values()) == 7
        assert set(stats["splits"]) == {"train", "val", "test"}


class TestBatchOverOneCsv:
    def test_columns_accumulate_across_batch_commands(self, recording, tmp_path) -> None:
        seg = bioamla("audio", "segment", str(recording), str(tmp_path / "seg"), "-d", "2")
        files = sorted(Path(seg["output"]).glob("*.wav"))
        assert len(files) == seg["segments"] == 4
        meta = Path(seg["output"]) / "meta.csv"
        meta.write_text("file_name,site\n" + "".join(f"{f.name},A\n" for f in files))

        bioamla("batch", "audio", "features", "--input-file", str(meta), "--trajectory")
        bioamla("batch", "detect", "energy", "--input-file", str(meta))
        bioamla("batch", "index", "--input-file", str(meta))
        bioamla("batch", "detect", "energy", "--input-file", str(meta))  # a second run

        rows = read_csv(meta)
        header = list(rows[0])
        assert len(header) == len(set(header))  # nothing duplicated by the rerun
        assert {"site", "features_file", "n_frames", "num_detections", "aci", "ndsi"} <= set(header)
        for row in rows:
            assert row["site"] == "A"
            table = Path(seg["output"]) / row["features_file"]
            assert "pc1" in table.read_text().splitlines()[0]
            assert int(row["n_frames"]) > 0


class TestTrainingConfig:
    def test_init_config_round_trip_and_validation(self, recording, tmp_path) -> None:
        pytest.importorskip("torch")
        from bioamla.common.config import load_toml, validate_config
        from bioamla.ml.train_config import AstTrainConfigFile

        config = bioamla("models", "ast", "init-config", "-o", str(tmp_path / "train.toml"))
        path = Path(config["output"])
        data = load_toml(path)
        schema = bioamla("system", "schema", "ast-train-file")["schema"]
        assert set(data) <= set(schema["properties"])  # every section is in the schema
        validate_config(AstTrainConfigFile, data, source=str(path))

        path.write_text(path.read_text().replace("epochs = 1 ", 'epochs = "many" ', 1))
        error = bioamla_error(
            "models", "ast", "train", "--train-dataset", str(tmp_path), "--config", str(path)
        )
        assert error["type"] == "ConfigValidationError"
        assert "training.epochs" in error["message"]


@pytest.mark.usefixtures("requires_ffmpeg_cli")
class TestFeaturesAndVideo:
    def test_trajectory_agrees_between_table_and_video(self, recording, tmp_path) -> None:
        pytest.importorskip("matplotlib")
        features = bioamla("audio", "features", str(recording), "--trajectory")
        video = bioamla(
            "audio", "video", str(recording), "-t", "trajectory", "-o", str(tmp_path / "t.mp4"),
            "--size", "160x120", "--fps", "5",
        )  # fmt: skip
        assert video["duration"] == pytest.approx(features["duration"], abs=0.01)
        assert video["active_frames"] == sum(f["pc1"] is not None for f in features["frames"])
        np.testing.assert_allclose(
            video["explained_variance_ratio"], features["explained_variance_ratio"]
        )
