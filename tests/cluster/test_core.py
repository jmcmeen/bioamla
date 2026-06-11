"""Coverage-focused tests for :mod:`bioamla.cluster.core`.

Targets gaps not exercised by ``test_cluster.py``: the dispatch helpers
(centers/stats/predict), similarity sorting, novelty internals, cluster analysis
metrics, and the lazy umap/hdbscan code paths (patched to stay fast).
"""

from __future__ import annotations

import csv
from unittest import mock

import numpy as np
import pytest

from bioamla.cluster.core import (
    AudioClusterer,
    ClusteringConfig,
    IncrementalReducer,
    NoveltyDetector,
    NoveltyResult,
    ReductionConfig,
    analyze_clusters,
    cluster_embeddings,
    compute_cluster_similarity,
    detect_novelty,
    discover_novel_sounds,
    export_clusters,
    export_clusters_to_csv,
    find_optimal_clusters,
    reduce_dimensions,
    sort_by_similarity,
    sort_clusters_by_similarity,
)
from bioamla.exceptions import ClusteringError, InvalidInputError

pytest.importorskip("sklearn", reason="scikit-learn not installed")


@pytest.fixture
def blob_embeddings() -> np.ndarray:
    """Two well-separated Gaussian blobs in 8 dimensions."""
    rng = np.random.default_rng(0)
    a = rng.normal(loc=0.0, scale=0.1, size=(30, 8))
    b = rng.normal(loc=5.0, scale=0.1, size=(30, 8))
    return np.vstack([a, b]).astype(np.float64)


# ---------------------------------------------------------------------------
# Dimensionality reduction (umap/tsne patched to stay fast)
# ---------------------------------------------------------------------------


class TestReduceDimensionsPatched:
    def test_umap_path_patched(self, blob_embeddings: np.ndarray) -> None:
        fake = mock.MagicMock()
        fake.UMAP.return_value.fit_transform.return_value = np.zeros((60, 2))
        with mock.patch.dict("sys.modules", {"umap": fake}):
            reduced = reduce_dimensions(blob_embeddings, method="umap", n_components=2)
        assert reduced.shape == (60, 2)
        fake.UMAP.assert_called_once()

    def test_tsne_path(self, blob_embeddings: np.ndarray) -> None:
        reduced = reduce_dimensions(blob_embeddings, method="tsne", n_components=2)
        assert reduced.shape == (60, 2)

    def test_config_object_used_when_no_method(self, blob_embeddings: np.ndarray) -> None:
        cfg = ReductionConfig(method="pca", n_components=3)
        reduced = reduce_dimensions(blob_embeddings, config=cfg)
        assert reduced.shape == (60, 3)


class TestIncrementalReducerUmap:
    def test_umap_fit_transform_patched(self, blob_embeddings: np.ndarray) -> None:
        reducer_inst = mock.MagicMock()
        reducer_inst.transform.return_value = np.zeros((60, 2))
        fake = mock.MagicMock()
        fake.UMAP.return_value = reducer_inst
        with mock.patch.dict("sys.modules", {"umap": fake}):
            reducer = IncrementalReducer(method="umap", n_components=2)
            out = reducer.fit_transform(blob_embeddings)
        assert out.shape == (60, 2)
        assert reducer.fitted


# ---------------------------------------------------------------------------
# AudioClusterer helpers
# ---------------------------------------------------------------------------


class TestAudioClustererHelpers:
    def test_dbscan_method(self, blob_embeddings: np.ndarray) -> None:
        config = ClusteringConfig(method="dbscan", eps=1.0, min_samples=3)
        clusterer = AudioClusterer(config=config)
        labels = clusterer.fit_predict(blob_embeddings)
        assert len(labels) == 60

    def test_agglomerative_method(self, blob_embeddings: np.ndarray) -> None:
        config = ClusteringConfig(method="agglomerative", n_clusters=2)
        clusterer = AudioClusterer(config=config)
        labels = clusterer.fit_predict(blob_embeddings)
        assert len(labels) == 60

    def test_predict_kmeans_has_predict(self, blob_embeddings: np.ndarray) -> None:
        clusterer = AudioClusterer(config=ClusteringConfig(method="kmeans", n_clusters=2))
        clusterer.fit(blob_embeddings)
        preds = clusterer.predict(blob_embeddings[:5])
        assert len(preds) == 5

    def test_predict_hdbscan_approximate(self, blob_embeddings: np.ndarray) -> None:
        pytest.importorskip("hdbscan")
        clusterer = AudioClusterer(config=ClusteringConfig(method="hdbscan", min_cluster_size=5))
        clusterer.fit(blob_embeddings)
        preds = clusterer.predict(blob_embeddings[:5])
        assert len(preds) == 5

    def test_predict_fallback_to_nearest(self, blob_embeddings: np.ndarray) -> None:
        # Agglomerative has no predict and isn't hdbscan -> _predict_nearest.
        config = ClusteringConfig(method="agglomerative", n_clusters=2)
        clusterer = AudioClusterer(config=config)
        clusterer.fit(blob_embeddings)
        clusterer.get_cluster_centers(blob_embeddings)  # sets _cluster_centers
        preds = clusterer.predict(blob_embeddings[:5])
        assert len(preds) == 5

    def test_predict_nearest_without_centers_raises(self, blob_embeddings: np.ndarray) -> None:
        config = ClusteringConfig(method="agglomerative", n_clusters=2)
        clusterer = AudioClusterer(config=config)
        clusterer.fit(blob_embeddings)
        with pytest.raises(ClusteringError, match="No cluster centers"):
            clusterer._predict_nearest(blob_embeddings[:5])

    def test_get_cluster_centers(self, blob_embeddings: np.ndarray) -> None:
        clusterer = AudioClusterer(config=ClusteringConfig(method="kmeans", n_clusters=2))
        clusterer.fit(blob_embeddings)
        centers = clusterer.get_cluster_centers(blob_embeddings)
        assert centers.shape == (2, 8)

    def test_get_cluster_centers_before_fit_raises(self, blob_embeddings: np.ndarray) -> None:
        clusterer = AudioClusterer(config=ClusteringConfig(method="kmeans", n_clusters=2))
        with pytest.raises(ClusteringError, match="must be fitted"):
            clusterer.get_cluster_centers(blob_embeddings)

    def test_get_cluster_stats(self, blob_embeddings: np.ndarray) -> None:
        clusterer = AudioClusterer(config=ClusteringConfig(method="kmeans", n_clusters=2))
        clusterer.fit(blob_embeddings)
        stats = clusterer.get_cluster_stats(blob_embeddings)
        assert len(stats) == 2
        for entry in stats.values():
            assert "size" in entry and "mean" in entry and "is_noise" in entry

    def test_get_cluster_stats_before_fit_raises(self, blob_embeddings: np.ndarray) -> None:
        clusterer = AudioClusterer(config=ClusteringConfig(method="kmeans", n_clusters=2))
        with pytest.raises(ClusteringError, match="must be fitted"):
            clusterer.get_cluster_stats(blob_embeddings)


# ---------------------------------------------------------------------------
# find_optimal_clusters
# ---------------------------------------------------------------------------


class TestFindOptimalClusters:
    def test_silhouette(self, blob_embeddings: np.ndarray) -> None:
        k = find_optimal_clusters(blob_embeddings, method="silhouette", k_range=(2, 4))
        assert k == 2

    def test_elbow(self, blob_embeddings: np.ndarray) -> None:
        k = find_optimal_clusters(blob_embeddings, method="elbow", k_range=(2, 5))
        assert 2 <= k <= 5

    def test_unknown_method_falls_back_to_argmax(self, blob_embeddings: np.ndarray) -> None:
        # An unrecognized method computes no scores -> argmax of empty raises.
        with pytest.raises(ValueError):
            find_optimal_clusters(blob_embeddings, method="bogus", k_range=(2, 3))


# ---------------------------------------------------------------------------
# Similarity & sorting
# ---------------------------------------------------------------------------


class TestClusterSimilarity:
    def test_cosine(self, blob_embeddings: np.ndarray) -> None:
        labels = np.array([0] * 30 + [1] * 30)
        sim = compute_cluster_similarity(blob_embeddings, labels, metric="cosine")
        assert sim.shape == (2, 2)

    def test_euclidean(self, blob_embeddings: np.ndarray) -> None:
        labels = np.array([0] * 30 + [1] * 30)
        sim = compute_cluster_similarity(blob_embeddings, labels, metric="euclidean")
        assert sim.shape == (2, 2)

    def test_unknown_metric_raises(self, blob_embeddings: np.ndarray) -> None:
        labels = np.array([0] * 30 + [1] * 30)
        with pytest.raises(InvalidInputError, match="Unknown metric"):
            compute_cluster_similarity(blob_embeddings, labels, metric="bogus")

    def test_noise_label_dropped(self, blob_embeddings: np.ndarray) -> None:
        labels = np.array([-1] * 10 + [0] * 20 + [1] * 30)
        sim = compute_cluster_similarity(blob_embeddings, labels, metric="cosine")
        assert sim.shape == (2, 2)


class TestSortBySimilarity:
    def test_nearest_neighbor_default_start(self) -> None:
        emb = np.array([[0.0, 0.0], [0.1, 0.0], [5.0, 5.0]])
        order = sort_by_similarity(emb, method="nearest_neighbor")
        assert order[0] == 0
        assert set(order.tolist()) == {0, 1, 2}

    def test_nearest_neighbor_with_reference(self) -> None:
        emb = np.array([[0.0, 0.0], [0.1, 0.0], [5.0, 5.0]])
        order = sort_by_similarity(emb, reference=np.array([5.0, 5.0]))
        assert order[0] == 2

    def test_spectral(self) -> None:
        rng = np.random.default_rng(3)
        emb = rng.normal(size=(8, 3))
        order = sort_by_similarity(emb, method="spectral")
        assert sorted(order.tolist()) == list(range(8))

    def test_unknown_method_raises(self) -> None:
        with pytest.raises(InvalidInputError, match="Unknown method"):
            sort_by_similarity(np.zeros((3, 2)), method="bogus")


class TestSortClustersBySimilarity:
    def test_basic(self, blob_embeddings: np.ndarray) -> None:
        labels = np.array([0] * 30 + [1] * 30)
        order = sort_clusters_by_similarity(blob_embeddings, labels)
        assert sorted(order) == [0, 1]

    def test_single_cluster_short_circuits(self, blob_embeddings: np.ndarray) -> None:
        labels = np.zeros(60, dtype=int)
        order = sort_clusters_by_similarity(blob_embeddings, labels)
        assert order == [0]

    def test_explicit_reference_label(self, blob_embeddings: np.ndarray) -> None:
        labels = np.array([0] * 30 + [1] * 30)
        order = sort_clusters_by_similarity(blob_embeddings, labels, reference_label=1)
        assert order[0] == 1

    def test_noise_dropped(self, blob_embeddings: np.ndarray) -> None:
        labels = np.array([-1] * 30 + [1] * 30)
        order = sort_clusters_by_similarity(blob_embeddings, labels)
        assert order == [1]


# ---------------------------------------------------------------------------
# Novelty detector internals
# ---------------------------------------------------------------------------


class TestNoveltyDetectorInternals:
    def test_distance_with_labels(self, blob_embeddings: np.ndarray) -> None:
        labels = np.array([0] * 30 + [1] * 30)
        detector = NoveltyDetector(method="distance")
        detector.fit(blob_embeddings, labels)
        assert detector.cluster_centers.shape == (2, 8)
        results = detector.predict(blob_embeddings)
        assert len(results) == 60
        assert all(isinstance(r, NoveltyResult) for r in results)

    def test_isolation_forest(self, blob_embeddings: np.ndarray) -> None:
        detector = NoveltyDetector(method="isolation_forest", contamination=0.1)
        detector.fit(blob_embeddings)
        results = detector.predict(blob_embeddings)
        assert len(results) == 60

    def test_lof(self, blob_embeddings: np.ndarray) -> None:
        detector = NoveltyDetector(method="lof", contamination=0.1)
        detector.fit(blob_embeddings)
        results = detector.predict(blob_embeddings)
        assert len(results) == 60

    def test_get_novel_samples_limited(self) -> None:
        rng = np.random.default_rng(4)
        normal = rng.normal(0.0, 0.05, size=(40, 4))
        outliers = np.array([[50.0] * 4, [60.0] * 4, [70.0] * 4])
        emb = np.vstack([normal, outliers])
        detector = NoveltyDetector(method="distance", threshold=3.0)
        detector.fit(normal)
        novel = detector.get_novel_samples(emb, n_samples=2)
        assert len(novel) == 2

    def test_get_novel_samples_all(self) -> None:
        rng = np.random.default_rng(5)
        normal = rng.normal(0.0, 0.05, size=(20, 4))
        detector = NoveltyDetector(method="distance", threshold=3.0)
        detector.fit(normal)
        novel = detector.get_novel_samples(normal)
        assert isinstance(novel, list)


class TestDiscoverAndDetectNovelty:
    def test_discover_with_known_labels(self, blob_embeddings: np.ndarray) -> None:
        known_labels = np.array([0] * 30 + [1] * 30)
        is_novel = discover_novel_sounds(blob_embeddings, known_labels=known_labels)
        assert is_novel.shape == (60,)

    def test_discover_return_scores(self, blob_embeddings: np.ndarray) -> None:
        is_novel, scores = discover_novel_sounds(
            blob_embeddings, method="distance", return_scores=True
        )
        assert is_novel.shape == (60,)
        assert scores.shape == (60,)

    def test_detect_novelty_with_known_embeddings(self, blob_embeddings: np.ndarray) -> None:
        known = blob_embeddings[:30]
        summary, is_novel, scores = detect_novelty(
            blob_embeddings, known_embeddings=known, method="distance"
        )
        assert summary.n_samples == 60
        assert is_novel.shape == (60,)
        assert scores.shape == (60,)

    def test_detect_novelty_isolation_forest(self, blob_embeddings: np.ndarray) -> None:
        summary, is_novel, scores = detect_novelty(
            blob_embeddings, method="isolation_forest", contamination=0.1
        )
        assert summary.method == "isolation_forest"
        assert summary.n_novel + summary.n_known == 60


# ---------------------------------------------------------------------------
# analyze_clusters & cluster_embeddings error path
# ---------------------------------------------------------------------------


class TestAnalyzeClusters:
    def test_with_metadata_and_noise(self, blob_embeddings: np.ndarray) -> None:
        labels = np.array([-1] * 5 + [0] * 25 + [1] * 30)
        metadata = [{"file": f"f{i}.wav"} for i in range(60)]
        analysis = analyze_clusters(blob_embeddings, labels, metadata=metadata)
        assert analysis["n_clusters"] == 2
        assert analysis["n_noise"] == 5
        assert analysis["silhouette_score"] > 0.0
        # metadata_sample is attached per non-noise cluster.
        assert "metadata_sample" in analysis["cluster_stats"][0]

    def test_single_cluster_zero_metrics(self, blob_embeddings: np.ndarray) -> None:
        labels = np.zeros(60, dtype=int)
        analysis = analyze_clusters(blob_embeddings, labels)
        assert analysis["silhouette_score"] == 0.0
        assert analysis["calinski_harabasz_score"] == 0.0


class TestClusterEmbeddingsErrors:
    def test_invalid_method_reraises_invalidinput(self, blob_embeddings: np.ndarray) -> None:
        with pytest.raises(InvalidInputError):
            cluster_embeddings(blob_embeddings, method="bogus")

    def test_generic_failure_wrapped_in_clustering_error(self) -> None:
        # Too-few-samples for kmeans n_clusters triggers sklearn ValueError ->
        # wrapped as ClusteringError.
        tiny = np.zeros((1, 4))
        with pytest.raises(ClusteringError):
            cluster_embeddings(tiny, method="kmeans", n_clusters=10)


# ---------------------------------------------------------------------------
# Export helpers (copy_files branch + CSV without coordinates)
# ---------------------------------------------------------------------------


class TestExportCoverage:
    def test_export_clusters_copy_files(self, tmp_path) -> None:
        src = tmp_path / "src"
        src.mkdir()
        f0 = src / "a.wav"
        f0.write_bytes(b"RIFF")
        f1 = src / "b.wav"
        f1.write_bytes(b"RIFF")
        labels = np.array([0, 1])
        out = export_clusters(labels, [str(f0), str(f1)], str(tmp_path / "out"), copy_files=True)
        assert (tmp_path / "out" / "cluster_0" / "a.wav").exists()
        assert (tmp_path / "out" / "cluster_1" / "b.wav").exists()
        assert out == str(tmp_path / "out")

    def test_export_csv_without_coordinates(self, tmp_path) -> None:
        labels = np.array([0, 1, 0])
        out_path = tmp_path / "out.csv"
        export_clusters_to_csv(labels, ["a.wav", "b.wav", "c.wav"], str(out_path))
        rows = list(csv.DictReader(out_path.read_text().splitlines()))
        assert list(rows[0].keys()) == ["filepath", "cluster"]
        assert rows[0]["cluster"] == "0"
