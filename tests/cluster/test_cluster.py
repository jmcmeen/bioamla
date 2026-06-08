"""Tests for the cluster domain (flattened, exception-based API)."""

import builtins
import importlib
import json

import numpy as np
import pytest

from bioamla.cluster import (
    AudioClusterer,
    ClusterAnalysis,
    ClusteringConfig,
    ClusteringSummary,
    IncrementalReducer,
    NoveltyDetectionSummary,
    NoveltyDetector,
    ReductionConfig,
    analyze_clusters_summary,
    cluster_batch_files,
    cluster_embeddings,
    detect_novelty,
    discover_novel_sounds,
    export_clusters,
    export_clusters_to_csv,
    load_embeddings_batch,
    reduce_dimensions,
)
from bioamla.exceptions import (
    BioamlaError,
    ClusteringError,
    DependencyError,
    InvalidInputError,
)

# sklearn / umap / hdbscan may or may not be present; gate accordingly.
sklearn = pytest.importorskip("sklearn", reason="scikit-learn not installed")


@pytest.fixture
def blob_embeddings() -> np.ndarray:
    """Two well-separated Gaussian blobs in 8 dimensions."""
    rng = np.random.default_rng(0)
    a = rng.normal(loc=0.0, scale=0.1, size=(30, 8))
    b = rng.normal(loc=5.0, scale=0.1, size=(30, 8))
    return np.vstack([a, b]).astype(np.float64)


class TestReduceDimensions:
    def test_pca_reduction(self, blob_embeddings: np.ndarray) -> None:
        reduced = reduce_dimensions(blob_embeddings, method="pca", n_components=2)
        assert reduced.shape == (60, 2)

    def test_unknown_method_raises(self, blob_embeddings: np.ndarray) -> None:
        with pytest.raises(InvalidInputError, match="Unknown reduction method"):
            reduce_dimensions(blob_embeddings, method="bogus")

    def test_reduction_config_defaults(self) -> None:
        cfg = ReductionConfig()
        assert cfg.method == "umap"
        assert cfg.n_components == 2


class TestIncrementalReducer:
    def test_fit_transform_pca(self, blob_embeddings: np.ndarray) -> None:
        reducer = IncrementalReducer(method="pca", n_components=2)
        reduced = reducer.fit_transform(blob_embeddings)
        assert reduced.shape == (60, 2)
        assert reducer.fitted

    def test_transform_before_fit_raises(self, blob_embeddings: np.ndarray) -> None:
        reducer = IncrementalReducer(method="pca", n_components=2)
        with pytest.raises(ClusteringError, match="must be fitted"):
            reducer.transform(blob_embeddings)

    def test_unsupported_method_raises(self, blob_embeddings: np.ndarray) -> None:
        reducer = IncrementalReducer(method="tsne", n_components=2)
        with pytest.raises(InvalidInputError, match="Unsupported method"):
            reducer.fit(blob_embeddings)


class TestAudioClusterer:
    def test_kmeans_clustering(self, blob_embeddings: np.ndarray) -> None:
        config = ClusteringConfig(method="kmeans", n_clusters=2)
        clusterer = AudioClusterer(config=config)
        labels = clusterer.fit_predict(blob_embeddings)
        assert len(labels) == 60
        assert clusterer.n_clusters_ == 2

    def test_unknown_method_raises(self, blob_embeddings: np.ndarray) -> None:
        clusterer = AudioClusterer(method="bogus")
        with pytest.raises(InvalidInputError, match="Unknown clustering method"):
            clusterer.fit(blob_embeddings)

    def test_predict_before_fit_raises(self, blob_embeddings: np.ndarray) -> None:
        clusterer = AudioClusterer(method="kmeans")
        with pytest.raises(ClusteringError, match="must be fitted"):
            clusterer.predict(blob_embeddings)


class TestClusterEmbeddings:
    def test_returns_summary(self, blob_embeddings: np.ndarray) -> None:
        summary = cluster_embeddings(blob_embeddings, method="kmeans", n_clusters=2)
        assert isinstance(summary, ClusteringSummary)
        assert summary.n_clusters == 2
        assert summary.n_samples == 60
        assert len(summary.labels) == 60
        assert summary.silhouette_score > 0.5  # well-separated blobs

    def test_summary_to_dict(self, blob_embeddings: np.ndarray) -> None:
        summary = cluster_embeddings(blob_embeddings, method="kmeans", n_clusters=2)
        d = summary.to_dict()
        assert d["n_clusters"] == 2
        assert "labels" in d


class TestAnalyzeClusters:
    def test_returns_analysis(self, blob_embeddings: np.ndarray) -> None:
        clusterer = AudioClusterer(method="kmeans", config=ClusteringConfig(n_clusters=2))
        labels = clusterer.fit_predict(blob_embeddings)
        analysis = analyze_clusters_summary(blob_embeddings, labels)
        assert isinstance(analysis, ClusterAnalysis)
        assert analysis.n_clusters == 2
        assert analysis.n_samples == 60
        assert analysis.silhouette_score > 0.5


class TestNoveltyDetection:
    def test_distance_method(self, blob_embeddings: np.ndarray) -> None:
        detector = NoveltyDetector(method="distance", threshold=1.5)
        detector.fit(blob_embeddings)
        results = detector.predict(blob_embeddings)
        assert len(results) == 60

    def test_detect_novelty_helper(self, blob_embeddings: np.ndarray) -> None:
        summary, is_novel, scores = detect_novelty(blob_embeddings, method="distance")
        assert isinstance(summary, NoveltyDetectionSummary)
        assert is_novel.shape == (60,)
        assert scores.shape == (60,)
        assert summary.n_samples == 60

    def test_detect_outlier_flagged(self) -> None:
        rng = np.random.default_rng(1)
        normal = rng.normal(0.0, 0.05, size=(40, 4))
        outlier = np.array([[100.0, 100.0, 100.0, 100.0]])
        embeddings = np.vstack([normal, outlier])
        _, is_novel, _ = detect_novelty(embeddings, method="distance", threshold=3.0)
        assert is_novel[-1]  # the far outlier is flagged novel

    def test_unknown_method_raises(self, blob_embeddings: np.ndarray) -> None:
        detector = NoveltyDetector(method="bogus")
        with pytest.raises(InvalidInputError, match="Unknown method"):
            detector.fit(blob_embeddings)

    def test_discover_novel_sounds(self, blob_embeddings: np.ndarray) -> None:
        is_novel = discover_novel_sounds(blob_embeddings, method="distance")
        assert is_novel.shape == (60,)
        assert is_novel.dtype == bool


class TestExport:
    def test_export_clusters_manifest(self, tmp_path) -> None:
        labels = np.array([0, 0, 1, -1])
        filepaths = ["a.wav", "b.wav", "c.wav", "d.wav"]
        out = export_clusters(labels, filepaths, str(tmp_path / "clusters"))
        manifest_path = tmp_path / "clusters" / "manifest.json"
        assert manifest_path.exists()
        manifest = json.loads(manifest_path.read_text())
        assert manifest["cluster_0"] == ["a.wav", "b.wav"]
        assert manifest["cluster_1"] == ["c.wav"]
        assert manifest["noise"] == ["d.wav"]
        assert out == str(tmp_path / "clusters")

    def test_export_to_csv(self, tmp_path) -> None:
        labels = np.array([0, 1, 0])
        filepaths = ["a.wav", "b.wav", "c.wav"]
        reduced = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
        out_path = tmp_path / "out.csv"
        result = export_clusters_to_csv(labels, filepaths, str(out_path), reduced)
        assert result == str(out_path)
        lines = out_path.read_text().strip().splitlines()
        assert lines[0] == "filepath,cluster,x,y"
        assert lines[1].startswith("a.wav,0,")
        assert len(lines) == 4


class TestBatchClustering:
    def test_cluster_batch_files(self, tmp_path) -> None:
        rng = np.random.default_rng(2)
        in_dir = tmp_path / "embeddings"
        in_dir.mkdir()
        # Two files, each with a separated blob.
        np.save(in_dir / "f0.npy", rng.normal(0.0, 0.1, size=(15, 6)))
        np.save(in_dir / "f1.npy", rng.normal(5.0, 0.1, size=(15, 6)))

        out_dir = tmp_path / "out"
        result = cluster_batch_files(
            in_dir, out_dir, method="kmeans", n_clusters=2
        )

        assignments_path = out_dir / "cluster_assignments.json"
        assert assignments_path.exists()
        data = json.loads(assignments_path.read_text())
        assert data["n_samples"] == 30
        assert len(data["file_assignments"]) == 2
        assert result.metadata["n_clusters"] == 2

    def test_load_embeddings_batch(self, tmp_path) -> None:
        np.save(tmp_path / "a.npy", np.ones((3, 4)))
        np.save(tmp_path / "b.npy", np.zeros((2, 4)))
        embeddings, filepaths = load_embeddings_batch(tmp_path)
        assert len(embeddings) == 2
        assert len(filepaths) == 2

    def test_unsupported_extension_raises(self, tmp_path) -> None:
        from bioamla.cluster.batch import load_embedding_file

        bad = tmp_path / "x.txt"
        bad.write_text("nope")
        with pytest.raises(InvalidInputError, match="Unsupported embedding file format"):
            load_embedding_file(bad)


class TestDependencyError:
    def test_missing_hdbscan_raises_dependency_error(
        self, blob_embeddings: np.ndarray, monkeypatch
    ) -> None:
        """If hdbscan is unavailable, AudioClusterer must raise DependencyError."""
        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "hdbscan" or name.startswith("hdbscan."):
                raise ImportError("simulated missing hdbscan")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        clusterer = AudioClusterer(method="hdbscan")
        with pytest.raises(DependencyError, match=r"bioamla\[cluster\]"):
            clusterer.fit(blob_embeddings)

    def test_missing_umap_raises_dependency_error(
        self, blob_embeddings: np.ndarray, monkeypatch
    ) -> None:
        """If umap is unavailable, UMAP reduction must raise DependencyError."""
        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "umap" or name.startswith("umap."):
                raise ImportError("simulated missing umap")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        with pytest.raises(DependencyError, match=r"bioamla\[cluster\]"):
            reduce_dimensions(blob_embeddings, method="umap")

    def test_dependency_error_is_bioamla_error(self) -> None:
        assert issubclass(DependencyError, BioamlaError)
        assert issubclass(ClusteringError, BioamlaError)


def test_module_imports_on_slim_install() -> None:
    """The cluster package must import without triggering heavy backends."""
    mod = importlib.import_module("bioamla.cluster")
    assert hasattr(mod, "AudioClusterer")
    assert hasattr(mod, "reduce_dimensions")
