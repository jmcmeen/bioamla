"""CLI tests for `bioamla cluster` commands."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from click.testing import CliRunner

from bioamla.cli.cli import cli
from bioamla.exceptions import ClusteringError


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture
def embeddings_file(tmp_path):
    arr = np.random.RandomState(0).rand(20, 8).astype(np.float32)
    path = tmp_path / "emb.npy"
    np.save(path, arr)
    return path


@pytest.fixture
def labels_file(tmp_path):
    labels = np.array([i % 3 for i in range(20)])
    path = tmp_path / "labels.npy"
    np.save(path, labels)
    return path


def test_cluster_group_help(runner: CliRunner) -> None:
    result = runner.invoke(cli, ["cluster", "--help"])
    assert result.exit_code == 0
    for sub in ["reduce", "cluster", "analyze", "novelty"]:
        assert sub in result.output


# --- reduce --------------------------------------------------------------


def test_cluster_reduce(runner: CliRunner, embeddings_file, tmp_path) -> None:
    out = tmp_path / "reduced.npy"
    reduced = np.zeros((20, 2), dtype=np.float32)
    with patch("bioamla.cluster.reduce_dimensions", return_value=reduced) as m:
        result = runner.invoke(
            cli, ["cluster", "reduce", str(embeddings_file), "-o", str(out), "-m", "pca"]
        )
    assert result.exit_code == 0, result.output
    assert out.exists()
    m.assert_called_once()


def test_cluster_reduce_quiet(runner: CliRunner, embeddings_file, tmp_path) -> None:
    out = tmp_path / "reduced.npy"
    with patch("bioamla.cluster.reduce_dimensions", return_value=np.zeros((20, 2))):
        result = runner.invoke(
            cli, ["cluster", "reduce", str(embeddings_file), "-o", str(out), "-q"]
        )
    assert result.exit_code == 0
    assert result.output.strip() == ""


def test_cluster_reduce_error(runner: CliRunner, embeddings_file, tmp_path) -> None:
    out = tmp_path / "reduced.npy"
    with patch("bioamla.cluster.reduce_dimensions", side_effect=ClusteringError("bad")):
        result = runner.invoke(
            cli, ["cluster", "reduce", str(embeddings_file), "-o", str(out)]
        )
    assert result.exit_code != 0


# --- cluster -------------------------------------------------------------


def test_cluster_cluster(runner: CliRunner, embeddings_file, tmp_path) -> None:
    out = tmp_path / "clabels.npy"
    fake_clusterer = MagicMock()
    fake_clusterer.fit_predict.return_value = np.array([0] * 10 + [1] * 10)
    fake_clusterer.n_clusters_ = 2
    with patch("bioamla.cluster.AudioClusterer", return_value=fake_clusterer), patch(
        "bioamla.cluster.ClusteringConfig", MagicMock()
    ):
        result = runner.invoke(
            cli, ["cluster", "cluster", str(embeddings_file), "-o", str(out), "-m", "hdbscan"]
        )
    assert result.exit_code == 0, result.output
    assert "Found 2 clusters" in result.output
    assert out.exists()


def test_cluster_cluster_error(runner: CliRunner, embeddings_file, tmp_path) -> None:
    out = tmp_path / "clabels.npy"
    with patch("bioamla.cluster.ClusteringConfig", MagicMock()), patch(
        "bioamla.cluster.AudioClusterer", side_effect=ClusteringError("fail")
    ):
        result = runner.invoke(
            cli, ["cluster", "cluster", str(embeddings_file), "-o", str(out)]
        )
    assert result.exit_code != 0


# --- analyze -------------------------------------------------------------


def _analysis():
    return SimpleNamespace(
        n_clusters=3,
        n_samples=20,
        n_noise=2,
        silhouette_score=0.5,
        calinski_harabasz_score=100.0,
        cluster_stats={"0": {"size": 6}},
    )


def test_cluster_analyze(runner: CliRunner, embeddings_file, labels_file) -> None:
    with patch("bioamla.cluster.analyze_clusters_summary", return_value=_analysis()):
        result = runner.invoke(
            cli, ["cluster", "analyze", str(embeddings_file), str(labels_file)]
        )
    assert result.exit_code == 0, result.output
    assert "Cluster Analysis" in result.output
    assert "Silhouette" in result.output


def test_cluster_analyze_output(runner: CliRunner, embeddings_file, labels_file, tmp_path) -> None:
    out = tmp_path / "analysis.json"
    with patch("bioamla.cluster.analyze_clusters_summary", return_value=_analysis()):
        result = runner.invoke(
            cli,
            ["cluster", "analyze", str(embeddings_file), str(labels_file), "-o", str(out)],
        )
    assert result.exit_code == 0
    assert out.exists()
    assert "n_clusters" in out.read_text()


def test_cluster_analyze_error(runner: CliRunner, embeddings_file, labels_file) -> None:
    with patch(
        "bioamla.cluster.analyze_clusters_summary", side_effect=ClusteringError("x")
    ):
        result = runner.invoke(
            cli, ["cluster", "analyze", str(embeddings_file), str(labels_file)]
        )
    assert result.exit_code != 0


# --- novelty -------------------------------------------------------------


def test_cluster_novelty(runner: CliRunner, embeddings_file, tmp_path) -> None:
    out = tmp_path / "novelty.npy"
    summary = SimpleNamespace(n_novel=3, novel_percentage=15.0)
    is_novel = np.array([True] * 3 + [False] * 17)
    scores = np.random.RandomState(1).rand(20)
    with patch(
        "bioamla.cluster.detect_novelty", return_value=(summary, is_novel, scores)
    ):
        result = runner.invoke(
            cli, ["cluster", "novelty", str(embeddings_file), "-o", str(out), "-m", "distance"]
        )
    assert result.exit_code == 0, result.output
    assert "novel samples" in result.output
    assert out.exists()


def test_cluster_novelty_with_labels(
    runner: CliRunner, embeddings_file, labels_file, tmp_path
) -> None:
    out = tmp_path / "novelty.npy"
    summary = SimpleNamespace(n_novel=0, novel_percentage=0.0)
    is_novel = np.array([False] * 20)
    scores = np.zeros(20)
    with patch(
        "bioamla.cluster.detect_novelty", return_value=(summary, is_novel, scores)
    ):
        result = runner.invoke(
            cli,
            ["cluster", "novelty", str(embeddings_file), "-o", str(out),
             "--labels", str(labels_file)],
        )
    assert result.exit_code == 0


def test_cluster_novelty_error(runner: CliRunner, embeddings_file, tmp_path) -> None:
    out = tmp_path / "novelty.npy"
    with patch("bioamla.cluster.detect_novelty", side_effect=ClusteringError("x")):
        result = runner.invoke(
            cli, ["cluster", "novelty", str(embeddings_file), "-o", str(out)]
        )
    assert result.exit_code != 0
