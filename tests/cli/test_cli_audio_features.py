"""`bioamla audio features` and `bioamla batch audio features`."""

import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import soundfile as sf
from click.testing import CliRunner

from bioamla.audio import DESCRIPTORS
from bioamla.cli.cli import cli


@pytest.fixture
def chirps(tmp_path: Path) -> Path:
    """Two seconds of chirps with gaps, as a WAV."""
    sr = 22050
    t = np.arange(2 * sr) / sr
    x = 0.005 * np.sin(2 * np.pi * 50 * t)
    for start in (0.2, 0.9, 1.5):
        m = (t >= start) & (t < start + 0.3)
        tt = t[m] - start
        x[m] += 0.5 * np.sin(2 * np.pi * (2000 * tt + 5000 * tt**2)) * np.hanning(m.sum())
    path = tmp_path / "chirps.wav"
    sf.write(path, x, sr)
    return path


def _run(*args: str):
    return CliRunner().invoke(cli, list(args))


def _json(*args: str) -> dict:
    result = _run("--json", *args)
    assert result.exit_code == 0, result.output
    envelope = json.loads(result.stdout)
    assert envelope["ok"] is True
    return envelope["result"]


class TestAudioFeatures:
    def test_summary_without_output(self, chirps) -> None:
        result = _run("audio", "features", str(chirps))
        assert result.exit_code == 0, result.output
        assert "centroid_hz" in result.output
        assert "--output" in result.output

    @pytest.mark.parametrize("suffix", [".csv", ".parquet"])
    def test_writes_table(self, chirps, tmp_path, suffix) -> None:
        out = tmp_path / f"t{suffix}"
        result = _run("audio", "features", str(chirps), "-o", str(out), "--trajectory")
        assert result.exit_code == 0, result.output
        table = pd.read_csv(out) if suffix == ".csv" else pd.read_parquet(out)
        assert list(table.columns[:11]) == ["time", *DESCRIPTORS]
        assert list(table.columns[-3:]) == ["pc1", "pc2", "pc3"]

    def test_json_with_output_is_a_summary(self, chirps, tmp_path) -> None:
        out = tmp_path / "t.csv"
        result = _json("audio", "features", str(chirps), "-o", str(out), "--trajectory")
        assert result["output"] == str(out)
        assert result["n_frames"] > 100
        assert len(result["explained_variance_ratio"]) == 3
        assert result["config"]["n_fft"] == 1024
        assert result["params"]["trajectory"] is True
        assert "frames" not in result

    def test_json_without_output_has_frames(self, chirps) -> None:
        result = _json("audio", "features", str(chirps), "--trajectory")
        assert len(result["frames"]) == result["n_frames"]
        pcs = [row["pc1"] for row in result["frames"]]
        assert None in pcs  # gated frames
        assert any(isinstance(v, float) for v in pcs)
        assert result["explained_variance_ratio"] is not None

    def test_options_reach_the_analysis(self, chirps) -> None:
        result = _json(
            "audio", "features", str(chirps), "--n-fft", "512", "--n-mfcc", "8", "--no-denoise"
        )
        assert result["config"]["n_fft"] == 512
        assert result["config"]["denoise"] is False
        assert "mfcc_7" in result["frames"][0] and "mfcc_8" not in result["frames"][0]

    @pytest.mark.parametrize(
        ("args", "message"),
        [
            (["-o", "x.txt"], ".csv or .parquet"),
            (["--hop", "0"], "hop must be at least 1"),
            (["--f-max", "50000"], "Nyquist"),
        ],
    )
    def test_errors(self, chirps, args, message) -> None:
        result = _run("audio", "features", str(chirps), *args)
        assert result.exit_code == 1
        assert message in result.output

    def test_json_error(self, tmp_path) -> None:
        short = tmp_path / "short.wav"
        sf.write(short, np.zeros(1000), 22050)
        result = _run("--json", "audio", "features", str(short))
        envelope = json.loads(result.stdout)
        assert envelope["ok"] is False
        assert envelope["error"]["type"] == "InvalidInputError"


class TestBatchAudioFeatures:
    @pytest.fixture
    def tree(self, chirps, tmp_path) -> Path:
        root = tmp_path / "in"
        (root / "sub").mkdir(parents=True)
        (root / "a.wav").write_bytes(chirps.read_bytes())
        (root / "sub" / "b.wav").write_bytes(chirps.read_bytes())
        sf.write(root / "sub" / "short.wav", np.zeros(1000), 22050)
        return root

    def test_directory_mode(self, tree, tmp_path) -> None:
        out = tmp_path / "out"
        result = _json(
            "batch", "audio", "features", "--input-dir", str(tree), "--output-dir", str(out),
            "--trajectory",
        )  # fmt: skip
        assert result["successful"] == 2 and result["failed"] == 1
        assert (out / "a.features.csv").exists()
        assert (out / "sub" / "b.features.csv").exists()
        assert "pc1" in pd.read_csv(out / "a.features.csv").columns
        index = pd.read_csv(out / "frame_features.csv")
        assert len(index) == 2 and set(index.columns) >= {"features_file", "n_frames"}

    def test_directory_mode_needs_output_dir(self, tree) -> None:
        result = _run("batch", "audio", "features", "--input-dir", str(tree))
        assert result.exit_code == 1
        assert "--output-dir is required" in result.output

    def test_csv_mode_merges_columns(self, tree, tmp_path) -> None:
        meta = tree / "meta.csv"
        meta.write_text("file_name,label\na.wav,x\nsub/b.wav,y\n")
        out = tmp_path / "out"
        result = _run(
            "batch", "audio", "features", "--input-file", str(meta), "--output-dir", str(out),
            "-f", "parquet",
        )  # fmt: skip
        assert result.exit_code == 0, result.output
        rows = list(csv.DictReader((out / "meta.csv").open()))
        assert [r["label"] for r in rows] == ["x", "y"]
        assert rows[1]["features_file"] == str(Path("sub") / "b.features.parquet")  # from out/
        assert int(rows[0]["n_frames"]) > 100
        assert (out / "a.features.parquet").exists()

    def test_csv_mode_in_place_is_relative(self, tree) -> None:
        meta = tree / "meta.csv"
        meta.write_text("file_name\nsub/b.wav\n")
        result = _run("batch", "audio", "features", "--input-file", str(meta))
        assert result.exit_code == 0, result.output
        rows = list(csv.DictReader(meta.open()))
        assert rows[0]["features_file"] == str(Path("sub") / "b.features.csv")
        assert (tree / "sub" / "b.features.csv").exists()

    def test_csv_mode_same_stem_does_not_collide(self, tree, chirps, tmp_path) -> None:
        import shutil

        sf.write(tree / "a.flac", sf.read(chirps)[0], 22050)
        other = tmp_path / "elsewhere"
        other.mkdir()
        shutil.copy(chirps, other / "a.wav")
        meta = tree / "meta.csv"
        meta.write_text(f"file_name\na.wav\na.flac\n{other / 'a.wav'}\n")
        out = tmp_path / "out"
        result = _run(
            "batch", "audio", "features", "--input-file", str(meta), "--output-dir", str(out)
        )
        assert result.exit_code == 0, result.output
        tables = [r["features_file"] for r in csv.DictReader((out / "meta.csv").open())]
        assert len(set(tables)) == 3
        assert all((out / t).exists() for t in tables)  # relative to the written CSV
