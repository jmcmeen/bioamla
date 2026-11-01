"""Pipelines: one command's output fed to the next, through ``bioamla --json``.

Each step reads the paths and values the previous step's JSON envelope reported,
the way a script or an agent chains commands. Unit tests cover each command on
its own; these cover the hand-offs between them.
"""

import csv
import json
import os
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from click.testing import CliRunner

from bioamla.cli.cli import cli

pytestmark = pytest.mark.integration


def _envelope(args: tuple[str, ...]) -> dict:
    run = CliRunner().invoke(cli, ["--json", *args])
    try:
        return json.loads(run.stdout)
    except json.JSONDecodeError:
        raise AssertionError(
            f"bioamla {' '.join(args)} printed no envelope (exit {run.exit_code}): {run.output!r}"
        ) from run.exception


def bioamla(*args: str) -> dict:
    """Run ``bioamla --json <args>``; return the result, failing on an error envelope."""
    envelope = _envelope(args)
    assert envelope["ok"], envelope.get("error")
    return envelope["result"]


def bioamla_error(*args: str) -> dict:
    """Run ``bioamla --json <args>`` expecting failure; return the error."""
    envelope = _envelope(args)
    assert not envelope["ok"]
    return envelope["error"]


def read_csv(path: str | Path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


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

    def test_detections_outlined_on_a_spectrogram(self, recording, tmp_path) -> None:
        pytest.importorskip("matplotlib")
        from PIL import Image

        from bioamla.viz import annotation_boxes

        detections = bioamla("detect", "energy", str(recording), "-o", str(tmp_path / "d.csv"))
        boxes = annotation_boxes(detections["output"])
        assert len(boxes) == detections["count"] == 7
        assert {(b.low_freq, b.high_freq) for b in boxes} == {(500.0, 5000.0)}
        plain = bioamla("audio", "visualize", str(recording), "-o", str(tmp_path / "plain.png"))
        boxed = bioamla(
            "audio", "visualize", str(recording), "-o", str(tmp_path / "boxed.png"),
            "--annotations", detections["output"],
        )  # fmt: skip
        a = np.asarray(Image.open(plain["output"]))
        b = np.asarray(Image.open(boxed["output"]))
        assert a.shape == b.shape and (a != b).any()

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

    def test_runs_are_recorded_with_their_outputs(self, recording, tmp_path, monkeypatch):
        root = tmp_path / "survey"
        bioamla("workspace", "init", str(root))
        monkeypatch.chdir(root)
        bioamla("detect", "energy", str(recording), "-o", "det.csv")
        bioamla_error("detect", "energy", "missing.wav")
        runs = bioamla("workspace", "history")["runs"]
        detect_ok, detect_failed = runs[-2:]
        assert detect_ok["ok"] and detect_ok["outputs"] == ["det.csv"]
        assert (root / detect_ok["outputs"][0]).is_file()
        assert not detect_failed["ok"] and detect_failed["error"]["type"] == "BadParameter"
        assert [r["command"] for r in runs][0] == "workspace init"


class TestProgressEvents:
    def test_batch_command_streams_progress(self, recording, tmp_path) -> None:
        import os
        import subprocess
        import sys

        folder = tmp_path / "in"
        folder.mkdir()
        for i in range(3):
            (folder / f"r{i}.wav").write_bytes(recording.read_bytes())
        env = {**os.environ, "BIOAMLA_PROGRESS": "jsonl"}
        proc = subprocess.run(
            [sys.executable, "-m", "bioamla._cli_entry", "--json", "batch", "audio", "features",
             "--input-dir", str(folder), "--output-dir", str(tmp_path / "out")],
            capture_output=True, text=True, env=env, check=True,
        )  # fmt: skip
        assert json.loads(proc.stdout)["ok"]  # stdout is still exactly the envelope
        got = [json.loads(line) for line in proc.stderr.splitlines()
               if line.startswith('{"event"')]  # fmt: skip
        assert got[0]["event"] == "start" and got[-1]["event"] == "end"
        assert got[-1]["done"] == 3 and got[-1]["ok"] is True
        progress = [e["done"] for e in got if e["event"] == "progress"]
        assert progress == [1, 2, 3]  # every file of a small batch


class TestCatalog:
    def test_catalog_setting_keys_configure_commands(self, recording, tmp_path, monkeypatch):
        for name in [n for n in os.environ if n.startswith("BIOAMLA_DETECT_ENERGY_")]:
            monkeypatch.delenv(name)  # the environment would win over the workspace setting
        catalog = bioamla("system", "commands", "detect", "energy")
        (energy,) = [c for c in catalog["commands"] if c["path"] == "detect energy"]
        threshold = next(o for o in energy["options"] if o["name"] == "threshold_db")

        root = tmp_path / "survey"
        bioamla("workspace", "init", str(root))
        monkeypatch.chdir(root)
        bioamla("workspace", "config", "set", threshold["setting"], "-3")
        assert bioamla("detect", "energy", str(recording))["params"]["threshold_db"] == -3.0


def test_batch_audio_transforms_chain_in_directory_mode(recording: Path, tmp_path: Path) -> None:
    """resample -> normalize -> trim -> filter -> denoise, each reading the folder the
    previous step wrote, on real files (no mocks)."""
    src = tmp_path / "in"
    src.mkdir()
    (src / "calls.wav").write_bytes(recording.read_bytes())
    steps = [
        ("resample", ["-r", "16000"]),
        ("normalize", ["-d", "-20"]),
        ("trim", ["-s", "1", "-e", "6"]),
        ("filter", ["--highpass", "500"]),
        ("denoise", ["--strength", "0.5"]),
    ]
    folder = src
    for command, options in steps:
        out = tmp_path / command
        result = bioamla(
            "batch", "audio", command, "--input-dir", str(folder), "--output-dir", str(out),
            *options,
        )  # fmt: skip
        assert (result["successful"], result["failed"]) == (1, 0), command
        assert result["output_files"] == [str(out / "calls.wav")]
        folder = out
    data, sr = sf.read(folder / "calls.wav")
    assert sr == 16000
    assert len(data) == 5 * 16000  # trimmed to 1-6 s; denoising keeps the length


class TestModelBatchLibraryParity:
    """`batch models predict|embed` reproduced with library calls alone (no CLI)."""

    @pytest.fixture
    def folder(self, tmp_path) -> Path:
        folder = tmp_path / "in"
        folder.mkdir()
        t = np.arange(16000 * 2) / 16000
        for name, freq in (("a.wav", 1000), ("b.wav", 3000)):
            sf.write(folder / name, 0.5 * np.sin(2 * np.pi * freq * t), 16000)
        return folder

    def test_predict_whole_files_and_segments(self, tiny_ast, folder, tmp_path) -> None:
        from bioamla.batch import BatchConfig
        from bioamla.ml import batch_predict

        for extra, name in (
            ([], "predictions.json"),
            (["--segment-seconds", "1"], "predictions.csv"),
        ):
            cli_out, lib_out = tmp_path / f"cli{len(extra)}", tmp_path / f"lib{len(extra)}"
            got = bioamla(
                "batch", "models", "predict", "--input-dir", str(folder),
                "--output-dir", str(cli_out), "-m", str(tiny_ast), *extra,
            )  # fmt: skip
            want = batch_predict(
                BatchConfig(input_dir=str(folder), output_dir=str(lib_out)),
                model_path=str(tiny_ast),
                segment_duration=1 if extra else 0,
            )
            assert got["results_file"] == str(cli_out / name)
            assert want.metadata["output_file"] == str(lib_out / name)
            cli_text = (cli_out / name).read_text().replace(str(cli_out), "")
            assert cli_text == (lib_out / name).read_text().replace(str(lib_out), "")

    def test_embed_csv_mode(self, tiny_ast, folder, tmp_path) -> None:
        from bioamla.batch import BatchConfig
        from bioamla.ml import batch_embed

        manifest = folder / "files.csv"
        manifest.write_text("file_name\na.wav\nb.wav\n")
        got = bioamla(
            "batch", "models", "embed", "--input-file", str(manifest),
            "--output-dir", str(tmp_path / "cli"), "-m", str(tiny_ast),
        )  # fmt: skip
        want = batch_embed(
            BatchConfig(input_file=str(manifest), output_dir=str(tmp_path / "lib")),
            model_path=str(tiny_ast),
        )
        cli_rows = read_csv(got["metadata_csv"])
        lib_rows = read_csv(want.metadata["metadata_csv"])
        assert [r["embedding_file"] for r in cli_rows] == [r["embedding_file"] for r in lib_rows]
        for row in cli_rows:
            a = np.load(tmp_path / "cli" / row["embedding_file"])
            b = np.load(tmp_path / "lib" / row["embedding_file"])
            np.testing.assert_allclose(a, b, rtol=1e-6)

    def test_predict_without_output_dir_saves_beside_the_input(
        self, tiny_ast, folder, tmp_path
    ) -> None:
        manifest = folder / "files.csv"
        manifest.write_text("file_name\nb.wav\na.wav\n")
        got = bioamla(
            "batch", "models", "predict", "--input-file", str(manifest), "-m", str(tiny_ast),
            "--max-workers", "2",
        )  # fmt: skip
        assert got["results_file"] == str(folder / "predictions.json")
        names = [
            Path(p["filepath"]).name for p in json.loads((folder / "predictions.json").read_text())
        ]
        assert names == ["b.wav", "a.wav"]  # CSV order, whatever order the workers finished


class TestModelCommandsLibraryParity:
    """`models ast predict|embed|info|evaluate` reproduced with library calls alone."""

    @pytest.fixture
    def clips(self, tmp_path) -> Path:
        folder = tmp_path / "clips"
        folder.mkdir()
        t = np.arange(16000 * 2) / 16000
        for name, freq in (("a.wav", 1000), ("b.wav", 3000)):
            sf.write(folder / name, 0.5 * np.sin(2 * np.pi * freq * t), 16000)
        (tmp_path / "truth.csv").write_text("file_name,label\na.wav,a\nb.wav,b\n")
        return folder

    def test_predict_embed_info_evaluate(self, tiny_ast, clips, tmp_path) -> None:
        from bioamla.ml import classify_file, embed_file, evaluate_model, get_model_info

        got = bioamla(
            "models", "ast", "predict", str(clips / "a.wav"), "--model-path", str(tiny_ast),
            "--segment-seconds", "1", "-o", str(tmp_path / "c.csv"),
        )  # fmt: skip
        want = classify_file(
            clips / "a.wav", model_path=str(tiny_ast), segment_duration=1,
            output=tmp_path / "l.csv",
        )  # fmt: skip
        assert got["count"] == want.count == 2
        assert (tmp_path / "c.csv").read_text() == (tmp_path / "l.csv").read_text()

        ranked = bioamla(
            "models", "ast", "predict", str(clips / "a.wav"), "--model-path", str(tiny_ast),
            "--segment-seconds", "1", "--top-k", "2", "-o", str(tmp_path / "k.csv"),
        )  # fmt: skip
        want = classify_file(clips / "a.wav", model_path=str(tiny_ast), segment_duration=1,
                             top_k=2)  # fmt: skip
        assert ranked["predictions"] == json.loads(json.dumps(want.to_dict()["predictions"]))
        for p in ranked["predictions"]:
            assert [t["label"] for t in p["top"]][0] == p["label"]
            assert p["margin"] == pytest.approx(
                p["top"][0]["confidence"] - p["top"][1]["confidence"]
            )
        rows = read_csv(tmp_path / "k.csv")
        assert set(rows[0]) >= {"margin", "top"} and rows[0]["top"].count(";") == 1

        got = bioamla("models", "ast", "embed", str(clips / "b.wav"), "--model-path",
                      str(tiny_ast), "-o", str(tmp_path / "c.npy"))  # fmt: skip
        want = embed_file(clips / "b.wav", tmp_path / "l.npy", model_path=str(tiny_ast))
        assert got["shape"] == want.shape
        np.testing.assert_allclose(np.load(tmp_path / "c.npy"), np.load(tmp_path / "l.npy"))

        assert (
            bioamla("models", "ast", "info", str(tiny_ast))["labels"]
            == get_model_info(str(tiny_ast))["labels"]
        )

        got = bioamla("models", "ast", "evaluate", str(clips), "--model-path", str(tiny_ast),
                      "-g", str(tmp_path / "truth.csv"))  # fmt: skip
        want = evaluate_model(clips, model_path=str(tiny_ast), ground_truth=tmp_path / "truth.csv")
        assert got["metrics"] == json.loads(json.dumps(want.to_dict()["metrics"]))

    def test_train_operation_reports_progress_and_stops_when_cancelled(
        self, tiny_ast, clips, tmp_path
    ) -> None:
        from bioamla import operations

        try:  # training data is decoded by torchcodec, which needs FFmpeg 4-8 (#256)
            import torchcodec.decoders  # noqa: F401
        except Exception as e:  # noqa: BLE001
            pytest.skip(f"torchcodec can't load here: {str(e).splitlines()[0]}")

        data = tmp_path / "by_class"
        for label, name in (("a", "a.wav"), ("b", "b.wav")):
            (data / label).mkdir(parents=True)
            for i in range(4):
                (data / label / f"{i}{name}").write_bytes((clips / name).read_bytes())
        steps = []
        ctx = operations.OperationContext(
            on_progress=lambda done, total: steps.append(done), cancelled=lambda: len(steps) >= 1
        )
        from bioamla.exceptions import TrainingCancelled

        with pytest.raises(TrainingCancelled):
            operations.run(
                "models.ast.train",
                {
                    "train_dataset": str(data),
                    "training_dir": str(tmp_path / "run"),
                    "options": {
                        "base_model": str(tiny_ast), "num_train_epochs": 3,
                        "per_device_train_batch_size": 2, "report_to": "none",
                        "dataloader_num_workers": 0, "logging_steps": 1,
                    },
                },
                ctx,
            )  # fmt: skip
        assert steps, "training reported no progress before it stopped"
