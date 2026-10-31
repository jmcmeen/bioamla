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


class TestRavenConfidenceRoundTrip:
    @pytest.mark.parametrize("source_format", ["csv", "raven"])
    def test_confidence_custom_fields_and_band_survive(self, tmp_path, source_format) -> None:
        if source_format == "csv":
            source = tmp_path / "in.csv"
            source.write_text(
                "start_time,end_time,low_freq,high_freq,label,channel,confidence,Confidence,site\n"
                "0.123456,1.234567,8343.75,12345.679,bird,2,0,stale,north\n"
                "2,3,0,100.125,frog,1,0.812345678901,,south\n"
                "4,5,100,200,call,1,,high,east\n"
                "6,7,,,noise,1,,,west\n"
            )
            middle, target = tmp_path / "sel.txt", tmp_path / "back.csv"
        else:
            source = tmp_path / "in.txt"
            source.write_text(
                "Selection\tChannel\tBegin Time (s)\tEnd Time (s)\tLow Freq (Hz)\t"
                "High Freq (Hz)\tAnnotation\tConfidence\tsite\n"
                "1\t2\t0.123456\t1.234567\t8343.750\t12345.679\tbird\t0\tnorth\n"
                "2\t1\t2\t3\t0.000\t100.125\tfrog\t0.812345678901\tsouth\n"
                "3\t1\t4\t5\t100.000\t200.000\tcall\thigh\teast\n"
                "4\t1\t6\t7\t\t\tnoise\t\twest\n"
            )
            middle, target = tmp_path / "ann.csv", tmp_path / "back.txt"

        first = bioamla("annotation", "convert", str(source), str(middle))
        second = bioamla("annotation", "convert", first["output"], str(target))
        assert first["count"] == second["count"] == 4

        csv_path = second["output"] if source_format == "csv" else first["output"]
        rows = read_csv(csv_path)
        assert [row["confidence"] for row in rows] == ["0.0", "0.812345678901", "", ""]
        assert [row["site"] for row in rows] == ["north", "south", "east", "west"]
        assert float(rows[0]["low_freq"]) == 8343.75
        assert float(rows[0]["high_freq"]) == 12345.679
        assert float(rows[0]["start_time"]) == 0.123456
        assert float(rows[0]["end_time"]) == 1.234567
        assert rows[0]["channel"] == "2"
        assert rows[2]["Confidence"] == "high"

        raven_path = first["output"] if source_format == "csv" else second["output"]
        with open(raven_path, newline="") as stream:
            reader = csv.DictReader(stream, delimiter="\t")
            selections = list(reader)
        assert reader.fieldnames.count("Confidence") == 1
        assert [row["Confidence"] for row in selections] == ["0.0", "0.812345678901", "high", ""]
        assert selections[0]["Low Freq (Hz)"] == "8343.750"
        assert selections[0]["High Freq (Hz)"] == "12345.679"
        assert selections[3]["Low Freq (Hz)"] == selections[3]["High Freq (Hz)"] == ""


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


class TestArbimon:
    def test_detections_to_arbimon_and_back_into_clips(self, recording, tmp_path) -> None:
        """Two recordings' detections in one Arbimon file; clips come from the right one."""
        other = tmp_path / "other.wav"
        audio, sr = sf.read(recording)
        sf.write(other, audio[: len(audio) // 2], sr)  # the first four calls only

        rows = []
        for wav in (recording, other):
            det = bioamla("detect", "energy", str(wav), "-o", str(tmp_path / f"{wav.stem}.csv"))
            out = tmp_path / f"{wav.stem}.arbimon.csv"
            conv = bioamla(
                "annotation", "convert", det["output"], str(out), "--to", "arbimon",
                "--recording", wav.name,
            )  # fmt: skip
            assert conv["count"] == det["count"]
            rows += read_csv(out)
        combined = tmp_path / "rois.csv"
        with open(combined, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

        summary = bioamla("annotation", "summary", str(combined))  # detected as arbimon
        assert summary["total_annotations"] == len(rows) == 7 + 4

        for wav, expected in ((recording, 7), (other, 4)):
            clips = bioamla(
                "dataset", "extract-clips", str(wav), str(tmp_path / f"clips-{wav.stem}"),
                "--annotations", str(combined),
            )  # fmt: skip
            assert clips["clips_written"] == expected  # only this recording's rows
            assert {r["low_freq"] for r in read_csv(clips["metadata_file"])} == {"500.0"}

        back = bioamla("annotation", "convert", str(combined), str(tmp_path / "sel.txt"))
        assert back["from_format"] == "arbimon" and back["to_format"] == "raven"


@pytest.fixture(scope="session")
def tiny_ast(tmp_path_factory) -> Path:
    """A randomly initialised AST, about 130 KB, built in a temp dir: no download and
    nothing committed. Its embeddings mean nothing; the hand-offs are what's tested."""
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    torch.manual_seed(0)
    config = transformers.ASTConfig(
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=64,
        num_mel_bins=128,
        max_length=128,
        num_labels=2,
        id2label={0: "a", 1: "b"},
        label2id={"a": 0, "b": 1},
    )
    path = tmp_path_factory.mktemp("tiny-ast")
    transformers.ASTForAudioClassification(config).save_pretrained(path)
    transformers.ASTFeatureExtractor(num_mel_bins=128, max_length=128).save_pretrained(path)
    return path


class TestEmbedToCluster:
    @pytest.fixture
    def tones(self, tmp_path) -> Path:
        folder = tmp_path / "in"
        (folder / "b").mkdir(parents=True)
        t = np.arange(16000 * 2) / 16000
        for name, freq in (
            ("a1.wav", 1000),
            ("a2.wav", 1100),
            ("b/a1.wav", 4000),
            ("b2.wav", 4100),
        ):
            sf.write(folder / name, 0.5 * np.sin(2 * np.pi * freq * t), 16000)
        return folder

    def test_directory_mode(self, tiny_ast, tones, tmp_path) -> None:
        pytest.importorskip("sklearn")
        emb = bioamla(
            "batch", "models", "embed", "--input-dir", str(tones),
            "--output-dir", str(tmp_path / "emb"), "-m", str(tiny_ast),
        )  # fmt: skip
        assert emb["successful"] == 4
        written = sorted(p.relative_to(emb["output_dir"]).as_posix()
                         for p in Path(emb["output_dir"]).rglob("*.npy"))  # fmt: skip
        assert written == ["a1_embeddings.npy", "a2_embeddings.npy", "b/a1_embeddings.npy",
                           "b2_embeddings.npy"]  # fmt: skip
        clusters = bioamla(
            "batch", "cluster", "--input-dir", emb["output_dir"],
            "--output-dir", str(tmp_path / "cl"), "--method", "kmeans", "--n-clusters", "2",
        )  # fmt: skip
        assignments = json.loads(Path(clusters["results_file"]).read_text())
        assert assignments["n_samples"] == 4
        assert sum(assignments["cluster_sizes"].values()) == 4

    def test_csv_mode_across_output_dirs(self, tiny_ast, tones, tmp_path) -> None:
        """embed -> cluster -> audio info, each writing the CSV to a new folder."""
        pytest.importorskip("sklearn")
        meta = tones / "meta.csv"
        meta.write_text("file_name,site\na1.wav,A\na2.wav,A\nb/a1.wav,B\nb2.wav,B\n")

        bioamla(
            "batch", "models", "embed", "--input-file", str(meta),
            "--output-dir", str(tmp_path / "emb"), "-m", str(tiny_ast),
        )  # fmt: skip
        emb_csv = tmp_path / "emb" / "meta.csv"
        rows = read_csv(emb_csv)
        assert len({r["embedding_file"] for r in rows}) == 4  # a1.wav twice, two files
        for row in rows:  # every path resolves from the CSV's new folder
            assert (emb_csv.parent / row["file_name"]).is_file()
            assert (emb_csv.parent / row["embedding_file"]).is_file()

        bioamla(
            "batch", "cluster", "--input-file", str(emb_csv), "--output-dir",
            str(tmp_path / "cl"), "--method", "kmeans", "--n-clusters", "2",
        )  # fmt: skip
        cl_csv = tmp_path / "cl" / "meta.csv"
        rows = read_csv(cl_csv)
        assert {r["cluster"] for r in rows} <= {"0", "1"} and all(r["cluster"] for r in rows)

        bioamla("batch", "audio", "info", "--input-file", str(cl_csv), "--output-dir",
                str(tmp_path / "info"))  # fmt: skip
        info_csv = tmp_path / "info" / "meta.csv"
        for row in read_csv(info_csv):
            assert row["site"] in ("A", "B") and row["cluster"] and row["sample_rate"] == "16000"
            assert (info_csv.parent / row["file_name"]).is_file()
            assert (info_csv.parent / row["embedding_file"]).is_file()

    def test_failed_rows_neither_keep_stale_embeddings_nor_reach_cluster(
        self, tiny_ast, tones, tmp_path
    ) -> None:
        pytest.importorskip("sklearn")
        meta = tones / "meta.csv"
        meta.write_text("file_name\na1.wav\na2.wav\nb2.wav\n")
        out = tmp_path / "emb"
        bioamla("batch", "models", "embed", "--input-file", str(meta), "--output-dir", str(out),
                "-m", str(tiny_ast))  # fmt: skip
        assert all(r["embedding_file"] for r in read_csv(out / "meta.csv"))

        (tones / "a2.wav").write_bytes(b"not audio")  # this file now fails
        bioamla("batch", "models", "embed", "--input-file", str(out / "meta.csv"),
                "--output-dir", str(tmp_path / "emb2"), "-m", str(tiny_ast))  # fmt: skip
        rows = {Path(r["file_name"]).name: r for r in read_csv(tmp_path / "emb2" / "meta.csv")}
        assert rows["a2.wav"]["embedding_file"] == ""  # not the first run's file
        assert rows["a1.wav"]["embedding_file"] and rows["b2.wav"]["embedding_file"]

        result = bioamla(
            "batch", "cluster", "--input-file", str(tmp_path / "emb2" / "meta.csv"),
            "--output-dir", str(tmp_path / "cl"), "--method", "kmeans", "--n-clusters", "2",
        )  # fmt: skip
        assert result["total_files"] == 2 and result["failed"] == 0  # the .wav isn't tried
        rows = {Path(r["file_name"]).name: r for r in read_csv(tmp_path / "cl" / "meta.csv")}
        assert rows["a2.wav"]["cluster"] == ""


class TestWorkspace:
    def test_init_add_and_detect_inside(self, recording, tmp_path, monkeypatch) -> None:
        root = tmp_path / "survey"
        created = bioamla("workspace", "init", str(root))
        recs = root / "recordings"
        recs.mkdir()
        (recs / recording.name).write_bytes(recording.read_bytes())

        monkeypatch.chdir(root / "recordings")  # commands find the workspace from below
        (linked,) = bioamla("workspace", "add", ".")["artifacts"]
        assert linked["path"] == "recordings" and linked["kind"] == "folder"

        detections = bioamla("detect", "energy", recording.name, "-o", "../det.csv")
        assert detections["count"] == 7
        status = bioamla("workspace", "status")
        assert status["id"] == created["id"] and status["artifacts"] == {"folder": 1}

    def test_settings_written_by_config_set_reach_commands(self, recording, tmp_path, monkeypatch):
        root = tmp_path / "survey"
        bioamla("workspace", "init", str(root))
        monkeypatch.chdir(root)
        default = bioamla("detect", "energy", str(recording))
        bioamla("workspace", "config", "set", "detect.energy.threshold_db", "-1")
        tuned = bioamla("detect", "energy", str(recording))
        assert tuned["params"]["threshold_db"] == -1.0
        assert tuned["count"] < default["count"]
        bioamla("workspace", "config", "unset", "detect.energy.threshold_db")
        assert bioamla("detect", "energy", str(recording))["params"]["threshold_db"] == -20.0
