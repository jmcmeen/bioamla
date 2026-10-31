"""CSV-mode batch detect and batch cluster keep their results (no silent loss)."""

import csv
import json
import shutil
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from click.testing import CliRunner

from bioamla.cli.cli import cli


def _rows(path: Path) -> list[dict]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def test_csv_mode_detect_keeps_detections(test_audio_path, tmp_path) -> None:
    pytest.importorskip("librosa")
    src = tmp_path / "in"
    src.mkdir()
    for name in ("a.wav", "b.wav"):
        shutil.copy(test_audio_path, src / name)
    meta = src / "meta.csv"
    meta.write_text("file_name,label\na.wav,x\nb.wav,y\n")
    out = tmp_path / "out"

    result = CliRunner().invoke(
        cli,
        [
            "--json",
            "batch",
            "detect",
            "energy",
            "--input-file",
            str(meta),
            "--output-dir",
            str(out),
            "--threshold-db",
            "-60",
        ],
    )
    assert result.exit_code == 0, result.output
    report = json.loads(result.stdout)["result"]

    rows = _rows(Path(report["metadata_csv"]))
    assert [r["label"] for r in rows] == ["x", "y"]  # original columns kept
    assert all(r["num_detections"].isdigit() for r in rows)

    detections = json.loads(Path(report["results_file"]).read_text())
    assert [Path(d["filepath"]).name for d in detections] == ["a.wav", "b.wav"]
    assert [d["num_detections"] for d in detections] == [int(r["num_detections"]) for r in rows]


@pytest.fixture
def embeddings_csv(tmp_path) -> Path:
    rng = np.random.default_rng(0)
    src = tmp_path / "emb"
    src.mkdir()
    lines = ["file_name,label"]
    for i in range(12):
        center = 0.0 if i < 6 else 8.0
        np.save(src / f"e{i}.npy", rng.normal(center, 0.5, (1, 4)))
        lines.append(f"e{i}.npy,l{i}")
    meta = src / "meta.csv"
    meta.write_text("\n".join(lines) + "\n")
    return meta


def test_csv_mode_cluster_adds_cluster_columns(embeddings_csv, tmp_path) -> None:
    pytest.importorskip("sklearn")
    out = tmp_path / "out"
    result = CliRunner().invoke(
        cli,
        [
            "--json",
            "batch",
            "cluster",
            "--input-file",
            str(embeddings_csv),
            "--output-dir",
            str(out),
            "--method",
            "kmeans",
            "--n-clusters",
            "2",
        ],
    )
    assert result.exit_code == 0, result.output
    report = json.loads(result.stdout)["result"]
    rows = _rows(Path(report["metadata_csv"]))
    clusters = [r["cluster"] for r in rows]
    assert clusters[:6] == [clusters[0]] * 6 and clusters[6:] == [clusters[6]] * 6
    assert clusters[0] != clusters[6]  # the two blobs land in different clusters
    assert rows[0]["cluster_labels"] == rows[0]["cluster"]


def test_failed_clustering_is_reported_as_a_failure(embeddings_csv, tmp_path) -> None:
    with patch("bioamla.cluster.batch.cluster_embeddings", side_effect=ValueError("bad input")):
        result = CliRunner().invoke(
            cli,
            [
                "--json",
                "batch",
                "cluster",
                "--input-file",
                str(embeddings_csv),
                "--output-dir",
                str(tmp_path / "out"),
            ],
        )
    assert result.exit_code == 1
    envelope = json.loads(result.stdout)
    assert envelope["error"]["type"] == "ClusteringError"
    assert "bad input" in envelope["error"]["message"]


def test_file_cluster_ignores_noise_and_breaks_ties_by_label() -> None:
    from bioamla.cluster.batch import file_cluster

    assert file_cluster([-1, -1, 3]) == 3  # a real assignment beats noise
    assert file_cluster([5, 2]) == 2 and file_cluster([2, 5]) == 2  # not row order
    assert file_cluster([-1, -1]) == -1


def test_cluster_embedding_files_dedupes_and_raises_when_nothing_loads(tmp_path) -> None:
    pytest.importorskip("sklearn")
    from bioamla.cluster import cluster_embedding_files
    from bioamla.exceptions import ClusteringError

    rng = np.random.default_rng(1)
    paths = []
    for i in range(6):
        p = tmp_path / f"e{i}.npy"
        np.save(p, rng.normal(0, 1, (1, 3)))
        paths.append(p)
    result = cluster_embedding_files(
        [*paths, paths[0]], tmp_path / "o", method="kmeans", n_clusters=2
    )
    assert result.total_files == 6  # the duplicate is clustered once
    assert len(result.metadata["file_assignments"]) == 6

    with pytest.raises(ClusteringError, match="No embeddings could be loaded"):
        cluster_embedding_files([tmp_path / "missing.npy"], tmp_path / "o2")
    assert cluster_embedding_files([], tmp_path / "o3").total_files == 0  # nothing to do


def test_rerun_with_missing_files_blanks_columns_and_reports_no_stale_file(
    test_audio_path, tmp_path
) -> None:
    pytest.importorskip("librosa")
    src = tmp_path / "in"
    src.mkdir()
    shutil.copy(test_audio_path, src / "a.wav")
    meta = src / "meta.csv"
    meta.write_text("file_name\na.wav\n")
    out = tmp_path / "out"
    args = [
        "--json",
        "batch",
        "detect",
        "energy",
        "--input-file",
        str(meta),
        "--output-dir",
        str(out),
    ]
    assert CliRunner().invoke(cli, args).exit_code == 0

    (src / "a.wav").unlink()
    (out / "meta.csv").replace(meta)  # rerun on the CSV that already has num_detections
    report = json.loads(CliRunner().invoke(cli, args).stdout)["result"]
    assert report["failed"] == 1
    assert report["results_file"] is None  # not last run's detections_energy.json
    assert _rows(Path(report["metadata_csv"]))[0]["num_detections"] == ""


def test_csv_mode_cluster_says_where_the_csv_went(embeddings_csv, tmp_path) -> None:
    pytest.importorskip("sklearn")
    result = CliRunner().invoke(
        cli,
        [
            "batch",
            "cluster",
            "--input-file",
            str(embeddings_csv),
            "--output-dir",
            str(tmp_path / "o"),
            "--method",
            "kmeans",
            "--n-clusters",
            "2",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Updated metadata CSV written to" in result.output
