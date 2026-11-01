"""cluster.* operations, and parity with the CLI on real embeddings."""

import json
from pathlib import Path

import numpy as np
import pytest
from click.testing import CliRunner

from bioamla import operations
from bioamla.cli.cli import cli

pytest.importorskip("sklearn")

from bioamla import cluster as lib  # noqa: E402


def _cli(*args: str) -> dict:
    result = CliRunner().invoke(cli, ["--json", "cluster", *args])
    assert result.exit_code == 0, result.output
    return json.loads(result.stdout)["result"]


@pytest.fixture
def embeddings(tmp_path) -> Path:
    rng = np.random.default_rng(0)
    data = np.vstack([rng.normal(0, 0.1, (20, 6)), rng.normal(4, 0.1, (20, 6))])
    path = tmp_path / "emb.npy"
    np.save(path, data.astype(np.float32))
    return path


def test_reduce(embeddings, tmp_path):
    got = _cli("reduce", str(embeddings), "-o", str(tmp_path / "c.npy"), "-n", "3")
    want = lib.reduce_embeddings_file(embeddings, tmp_path / "l.npy", n_components=3)
    assert got["shape"] == want.shape == [40, 3]
    np.testing.assert_allclose(np.load(tmp_path / "c.npy"), np.load(tmp_path / "l.npy"))


def test_fit_analyze_novelty(embeddings, tmp_path):
    got = _cli("fit", str(embeddings), "-o", str(tmp_path / "c.npy"), "-m", "kmeans", "-k", "2")
    want = lib.cluster_embeddings_file(
        embeddings, tmp_path / "l.npy", method="kmeans", n_clusters=2
    )
    assert got["n_clusters"] == want.n_clusters == 2
    assert got["cluster_sizes"] == want.cluster_sizes

    got = _cli("analyze", str(embeddings), str(tmp_path / "l.npy"), "-o", str(tmp_path / "a.json"))
    want = lib.analyze_clusters_file(embeddings, tmp_path / "l.npy")
    assert got["silhouette_score"] == pytest.approx(want.analysis.silhouette_score)
    assert json.loads((tmp_path / "a.json").read_text())["n_clusters"] == 2

    got = _cli("novelty", str(embeddings), "-o", str(tmp_path / "nc.npy"))
    want = lib.novelty_file(embeddings, tmp_path / "nl.npy")
    assert got["n_novel"] == want.n_novel
    np.testing.assert_array_equal(np.load(tmp_path / "nc.npy"), np.load(tmp_path / "nl.npy"))


def test_wrong_shapes_are_input_errors(tmp_path):
    from bioamla.exceptions import InvalidInputError

    np.save(tmp_path / "one_d.npy", np.zeros(5))
    with pytest.raises(InvalidInputError, match="2-D"):
        lib.cluster_embeddings_file(tmp_path / "one_d.npy", tmp_path / "x.npy")


def test_operations_check_paths(embeddings, tmp_path):
    seen = []

    def check(path):
        seen.append(Path(path).name)
        return path

    operations.run(
        "cluster.fit",
        {
            "embeddings_file": str(embeddings),
            "output": str(tmp_path / "o.npy"),
            "method": "kmeans",
            "n_clusters": 2,
        },
        operations.OperationContext(path_check=check),
    )
    assert seen == ["emb.npy", "o.npy"]
    assert operations.get("cluster.novelty").kind == "job"


def test_outputs_report_the_npy_file_written(embeddings, tmp_path):
    """np.save adds .npy to a name without it; results name the file actually written."""
    got = lib.reduce_embeddings_file(embeddings, tmp_path / "reduced")
    assert got.output == str(tmp_path / "reduced.npy") and Path(got.output).exists()
    fit = lib.cluster_embeddings_file(
        embeddings, tmp_path / "labels", method="kmeans", n_clusters=2
    )
    assert fit.output.endswith("labels.npy") and Path(fit.output).exists()
