"""
Unit tests for bioamla.clustering module.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


class TestReductionConfig:
    """Tests for ReductionConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        from bioamla.clustering import ReductionConfig

        config = ReductionConfig()
        assert config.method == "umap"
        assert config.n_components == 2
        assert config.n_neighbors == 15
        assert config.perplexity == 30.0

    def test_custom_values(self):
        """Test custom configuration values."""
        from bioamla.clustering import ReductionConfig

        config = ReductionConfig(
            method="tsne",
            n_components=3,
            perplexity=50.0,
        )
        assert config.method == "tsne"
        assert config.n_components == 3
        assert config.perplexity == 50.0


class TestReduceDimensions:
    """Tests for reduce_dimensions function."""

    def test_reduce_pca(self):
        """Test PCA reduction."""
        from bioamla.clustering import reduce_dimensions

        embeddings = np.random.randn(100, 256)
        reduced = reduce_dimensions(embeddings, method="pca", n_components=2)

        assert reduced.shape == (100, 2)

    def test_reduce_pca_3d(self):
        """Test PCA reduction to 3D."""
        from bioamla.clustering import reduce_dimensions

        embeddings = np.random.randn(50, 128)
        reduced = reduce_dimensions(embeddings, method="pca", n_components=3)

        assert reduced.shape == (50, 3)

    def test_reduce_tsne(self):
        """Test t-SNE reduction."""
        from bioamla.clustering import reduce_dimensions

        embeddings = np.random.randn(50, 64)
        reduced = reduce_dimensions(
            embeddings,
            method="tsne",
            n_components=2,
            n_iter=250  # Fewer iterations for speed
        )

        assert reduced.shape == (50, 2)

    def test_reduce_with_config(self):
        """Test reduction with config object."""
        from bioamla.clustering import ReductionConfig, reduce_dimensions

        config = ReductionConfig(method="pca", n_components=2)
        embeddings = np.random.randn(100, 256)
        reduced = reduce_dimensions(embeddings, config=config)

        assert reduced.shape == (100, 2)

    def test_reduce_invalid_method(self):
        """Test error on invalid method."""
        from bioamla.clustering import reduce_dimensions

        embeddings = np.random.randn(100, 256)
        with pytest.raises(ValueError, match="Unknown reduction method"):
            reduce_dimensions(embeddings, method="invalid")


class TestIncrementalReducer:
    """Tests for IncrementalReducer."""

    def test_fit_transform_pca(self):
        """Test fit_transform with PCA."""
        from bioamla.clustering import IncrementalReducer

        reducer = IncrementalReducer(method="pca", n_components=2)
        embeddings = np.random.randn(100, 256)

        reduced = reducer.fit_transform(embeddings)

        assert reduced.shape == (100, 2)
        assert reducer.fitted

    def test_transform_after_fit(self):
        """Test transform on new data after fitting."""
        from bioamla.clustering import IncrementalReducer

        reducer = IncrementalReducer(method="pca", n_components=2)
        train_embeddings = np.random.randn(100, 256)
        test_embeddings = np.random.randn(20, 256)

        reducer.fit(train_embeddings)
        reduced = reducer.transform(test_embeddings)

        assert reduced.shape == (20, 2)

    def test_transform_before_fit_raises(self):
        """Test error when transforming before fitting."""
        from bioamla.clustering import IncrementalReducer

        reducer = IncrementalReducer(method="pca", n_components=2)
        embeddings = np.random.randn(20, 256)

        with pytest.raises(RuntimeError, match="must be fitted"):
            reducer.transform(embeddings)

    def test_invalid_method_for_incremental(self):
        """Test error on unsupported method for incremental."""
        from bioamla.clustering import IncrementalReducer

        reducer = IncrementalReducer(method="tsne", n_components=2)
        embeddings = np.random.randn(100, 256)

        with pytest.raises(ValueError, match="Unsupported method"):
            reducer.fit(embeddings)


class TestClusteringConfig:
    """Tests for ClusteringConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        from bioamla.clustering import ClusteringConfig

        config = ClusteringConfig()
        assert config.method == "hdbscan"
        assert config.min_cluster_size == 5
        assert config.n_clusters == 10

    def test_custom_values(self):
        """Test custom configuration values."""
        from bioamla.clustering import ClusteringConfig

        config = ClusteringConfig(
            method="kmeans",
            n_clusters=20,
        )
        assert config.method == "kmeans"
        assert config.n_clusters == 20


class TestAudioClusterer:
    """Tests for AudioClusterer."""

    def test_kmeans_clustering(self):
        """Test k-means clustering."""
        from bioamla.clustering import AudioClusterer

        clusterer = AudioClusterer(method="kmeans", n_clusters=3)
        embeddings = np.random.randn(100, 64)

        labels = clusterer.fit_predict(embeddings)

        assert labels.shape == (100,)
        assert clusterer.n_clusters_ == 3
        assert set(labels) == {0, 1, 2}

    def test_kmeans_predict(self):
        """Test k-means predict on new data."""
        from bioamla.clustering import AudioClusterer

        clusterer = AudioClusterer(method="kmeans", n_clusters=3)
        train_embeddings = np.random.randn(100, 64)
        test_embeddings = np.random.randn(20, 64)

        clusterer.fit(train_embeddings)
        labels = clusterer.predict(test_embeddings)

        assert labels.shape == (20,)

    def test_dbscan_clustering(self):
        """Test DBSCAN clustering."""
        from bioamla.clustering import AudioClusterer

        clusterer = AudioClusterer(method="dbscan", eps=0.5, min_samples=3)
        # Create clustered data
        cluster1 = np.random.randn(30, 64) + np.array([5] * 64)
        cluster2 = np.random.randn(30, 64) - np.array([5] * 64)
        embeddings = np.vstack([cluster1, cluster2])

        labels = clusterer.fit_predict(embeddings)

        assert labels.shape == (60,)
        # Should find at least some clusters
        assert clusterer.n_clusters_ >= 1

    def test_agglomerative_clustering(self):
        """Test agglomerative clustering."""
        from bioamla.clustering import AudioClusterer

        clusterer = AudioClusterer(method="agglomerative", n_clusters=4)
        embeddings = np.random.randn(100, 64)

        labels = clusterer.fit_predict(embeddings)

        assert labels.shape == (100,)
        assert clusterer.n_clusters_ == 4

    def test_get_cluster_centers(self):
        """Test computing cluster centers."""
        from bioamla.clustering import AudioClusterer

        clusterer = AudioClusterer(method="kmeans", n_clusters=3)
        embeddings = np.random.randn(100, 64)

        clusterer.fit(embeddings)
        centers = clusterer.get_cluster_centers(embeddings)

        assert centers.shape == (3, 64)

    def test_get_cluster_stats(self):
        """Test getting cluster statistics."""
        from bioamla.clustering import AudioClusterer

        clusterer = AudioClusterer(method="kmeans", n_clusters=3)
        embeddings = np.random.randn(100, 64)

        clusterer.fit(embeddings)
        stats = clusterer.get_cluster_stats(embeddings)

        assert len(stats) == 3
        for label in [0, 1, 2]:
            assert "size" in stats[label]
            assert "mean" in stats[label]
            assert "std" in stats[label]

    def test_invalid_method(self):
        """Test error on invalid method."""
        from bioamla.clustering import AudioClusterer

        clusterer = AudioClusterer(method="invalid")
        embeddings = np.random.randn(100, 64)

        with pytest.raises(ValueError, match="Unknown clustering method"):
            clusterer.fit(embeddings)


class TestFindOptimalClusters:
    """Tests for find_optimal_clusters function."""

    def test_silhouette_method(self):
        """Test silhouette method."""
        from bioamla.clustering import find_optimal_clusters

        # Create well-separated clusters
        cluster1 = np.random.randn(50, 32) + np.array([10] * 32)
        cluster2 = np.random.randn(50, 32) - np.array([10] * 32)
        cluster3 = np.random.randn(50, 32) + np.array([-10] * 16 + [10] * 16)
        embeddings = np.vstack([cluster1, cluster2, cluster3])

        k = find_optimal_clusters(embeddings, method="silhouette", k_range=(2, 6))

        assert 2 <= k <= 6

    def test_elbow_method(self):
        """Test elbow method."""
        from bioamla.clustering import find_optimal_clusters

        embeddings = np.random.randn(100, 32)
        k = find_optimal_clusters(embeddings, method="elbow", k_range=(2, 8))

        assert 2 <= k <= 8


class TestClusterSimilarity:
    """Tests for cluster similarity functions."""

    def test_compute_cluster_similarity_cosine(self):
        """Test cosine similarity computation."""
        from bioamla.clustering import compute_cluster_similarity

        embeddings = np.random.randn(100, 64)
        labels = np.array([0] * 50 + [1] * 50)

        similarity = compute_cluster_similarity(embeddings, labels, metric="cosine")

        assert similarity.shape == (2, 2)
        assert np.allclose(np.diag(similarity), 1.0, atol=1e-5)

    def test_compute_cluster_similarity_euclidean(self):
        """Test euclidean similarity computation."""
        from bioamla.clustering import compute_cluster_similarity

        embeddings = np.random.randn(100, 64)
        labels = np.array([0] * 50 + [1] * 50)

        similarity = compute_cluster_similarity(embeddings, labels, metric="euclidean")

        assert similarity.shape == (2, 2)
        # Diagonal should be highest (max self-similarity)
        assert similarity[0, 0] >= similarity[0, 1]

    def test_compute_cluster_similarity_with_noise(self):
        """Test similarity with noise points."""
        from bioamla.clustering import compute_cluster_similarity

        embeddings = np.random.randn(110, 64)
        labels = np.array([0] * 50 + [1] * 50 + [-1] * 10)  # -1 is noise

        similarity = compute_cluster_similarity(embeddings, labels, metric="cosine")

        # Should only have 2 clusters (noise excluded)
        assert similarity.shape == (2, 2)


class TestSortBySimilarity:
    """Tests for sort_by_similarity function."""

    def test_nearest_neighbor_sorting(self):
        """Test nearest neighbor sorting."""
        from bioamla.clustering import sort_by_similarity

        embeddings = np.array([
            [0, 0],
            [1, 1],
            [0.1, 0.1],
            [2, 2],
        ])

        sorted_indices = sort_by_similarity(embeddings, method="nearest_neighbor")

        assert len(sorted_indices) == 4
        assert set(sorted_indices) == {0, 1, 2, 3}

    def test_nearest_neighbor_with_reference(self):
        """Test sorting with reference point."""
        from bioamla.clustering import sort_by_similarity

        embeddings = np.array([
            [0, 0],
            [10, 10],
            [1, 1],
        ])
        reference = np.array([10, 10])

        sorted_indices = sort_by_similarity(
            embeddings, reference=reference, method="nearest_neighbor"
        )

        # First index should be closest to reference (index 1)
        assert sorted_indices[0] == 1

    def test_spectral_sorting(self):
        """Test spectral sorting."""
        from bioamla.clustering import sort_by_similarity

        embeddings = np.random.randn(50, 32)
        sorted_indices = sort_by_similarity(embeddings, method="spectral")

        assert len(sorted_indices) == 50
        assert set(sorted_indices) == set(range(50))


class TestSortClustersBySimilarity:
    """Tests for sort_clusters_by_similarity function."""

    def test_basic_sorting(self):
        """Test basic cluster sorting."""
        from bioamla.clustering import sort_clusters_by_similarity

        embeddings = np.random.randn(100, 64)
        labels = np.array([0] * 25 + [1] * 25 + [2] * 25 + [3] * 25)

        sorted_labels = sort_clusters_by_similarity(embeddings, labels)

        assert len(sorted_labels) == 4
        assert set(sorted_labels) == {0, 1, 2, 3}

    def test_sorting_with_reference(self):
        """Test sorting with reference cluster."""
        from bioamla.clustering import sort_clusters_by_similarity

        embeddings = np.random.randn(100, 64)
        labels = np.array([0] * 25 + [1] * 25 + [2] * 25 + [3] * 25)

        sorted_labels = sort_clusters_by_similarity(
            embeddings, labels, reference_label=2
        )

        assert sorted_labels[0] == 2

    def test_sorting_excludes_noise(self):
        """Test that noise cluster is excluded."""
        from bioamla.clustering import sort_clusters_by_similarity

        embeddings = np.random.randn(110, 64)
        labels = np.array([0] * 50 + [1] * 50 + [-1] * 10)

        sorted_labels = sort_clusters_by_similarity(embeddings, labels)

        assert -1 not in sorted_labels
        assert len(sorted_labels) == 2


class TestNoveltyDetector:
    """Tests for NoveltyDetector."""

    def test_distance_based_detection(self):
        """Test distance-based novelty detection."""
        from bioamla.clustering import NoveltyDetector

        # Create known clusters
        known_embeddings = np.random.randn(100, 64)
        labels = np.array([0] * 50 + [1] * 50)

        detector = NoveltyDetector(method="distance")
        detector.fit(known_embeddings, labels)

        # Test on similar data (not novel)
        similar = np.random.randn(10, 64)
        results = detector.predict(similar)

        assert len(results) == 10
        assert all(hasattr(r, "novelty_score") for r in results)
        assert all(hasattr(r, "is_novel") for r in results)

    def test_distance_based_without_labels(self):
        """Test distance-based detection without labels."""
        from bioamla.clustering import NoveltyDetector

        embeddings = np.random.randn(100, 64)

        detector = NoveltyDetector(method="distance")
        detector.fit(embeddings)  # No labels

        test_embeddings = np.random.randn(10, 64)
        results = detector.predict(test_embeddings)

        assert len(results) == 10

    def test_isolation_forest_detection(self):
        """Test isolation forest novelty detection."""
        from bioamla.clustering import NoveltyDetector

        embeddings = np.random.randn(100, 64)

        detector = NoveltyDetector(method="isolation_forest", contamination=0.1)
        detector.fit(embeddings)

        test_embeddings = np.random.randn(10, 64)
        results = detector.predict(test_embeddings)

        assert len(results) == 10

    def test_lof_detection(self):
        """Test local outlier factor detection."""
        from bioamla.clustering import NoveltyDetector

        embeddings = np.random.randn(100, 64)

        detector = NoveltyDetector(method="lof", contamination=0.1)
        detector.fit(embeddings)

        test_embeddings = np.random.randn(10, 64)
        results = detector.predict(test_embeddings)

        assert len(results) == 10

    def test_get_novel_samples(self):
        """Test getting novel sample indices."""
        from bioamla.clustering import NoveltyDetector

        # Known data clustered together
        known_embeddings = np.zeros((100, 64)) + np.random.randn(100, 64) * 0.1

        detector = NoveltyDetector(method="distance", threshold=0.5)
        detector.fit(known_embeddings)

        # Mix of similar and novel (outlier) data
        similar = np.zeros((5, 64)) + np.random.randn(5, 64) * 0.1
        novel = np.ones((5, 64)) * 100  # Far away
        test_embeddings = np.vstack([similar, novel])

        novel_indices = detector.get_novel_samples(test_embeddings)

        # Novel samples should be detected
        assert len(novel_indices) > 0


class TestDiscoverNovelSounds:
    """Tests for discover_novel_sounds function."""

    def test_basic_discovery(self):
        """Test basic novel sound discovery."""
        from bioamla.clustering import discover_novel_sounds

        embeddings = np.random.randn(100, 64)
        is_novel = discover_novel_sounds(embeddings)

        assert is_novel.shape == (100,)
        assert is_novel.dtype == bool

    def test_discovery_with_known_labels(self):
        """Test discovery with known labels."""
        from bioamla.clustering import discover_novel_sounds

        embeddings = np.random.randn(100, 64)
        labels = np.array([0] * 50 + [-1] * 50)  # 50 known, 50 unknown

        is_novel = discover_novel_sounds(embeddings, known_labels=labels)

        assert is_novel.shape == (100,)

    def test_discovery_with_scores(self):
        """Test discovery returning scores."""
        from bioamla.clustering import discover_novel_sounds

        embeddings = np.random.randn(100, 64)
        is_novel, scores = discover_novel_sounds(embeddings, return_scores=True)

        assert is_novel.shape == (100,)
        assert scores.shape == (100,)


class TestAnalyzeClusters:
    """Tests for analyze_clusters function."""

    def test_basic_analysis(self):
        """Test basic cluster analysis."""
        from bioamla.clustering import analyze_clusters

        embeddings = np.random.randn(100, 64)
        labels = np.array([0] * 50 + [1] * 50)

        analysis = analyze_clusters(embeddings, labels)

        assert analysis["n_clusters"] == 2
        assert analysis["n_samples"] == 100
        assert "silhouette_score" in analysis
        assert "calinski_harabasz_score" in analysis
        assert "cluster_stats" in analysis

    def test_analysis_with_noise(self):
        """Test analysis with noise points."""
        from bioamla.clustering import analyze_clusters

        embeddings = np.random.randn(110, 64)
        labels = np.array([0] * 50 + [1] * 50 + [-1] * 10)

        analysis = analyze_clusters(embeddings, labels)

        assert analysis["n_clusters"] == 2
        assert analysis["n_noise"] == 10

    def test_analysis_with_metadata(self):
        """Test analysis with metadata."""
        from bioamla.clustering import analyze_clusters

        embeddings = np.random.randn(10, 64)
        labels = np.array([0] * 5 + [1] * 5)
        metadata = [{"file": f"file_{i}.wav"} for i in range(10)]

        analysis = analyze_clusters(embeddings, labels, metadata=metadata)

        assert "metadata_sample" in analysis["cluster_stats"][0]


class TestExportClusters:
    """Tests for export_clusters function."""

    def test_export_manifest_only(self, tmp_path):
        """Test exporting manifest without copying files."""
        from bioamla.clustering import export_clusters

        labels = np.array([0, 0, 1, 1, -1])
        filepaths = ["a.wav", "b.wav", "c.wav", "d.wav", "e.wav"]

        output_dir = export_clusters(
            labels, filepaths, str(tmp_path), copy_files=False
        )

        assert Path(output_dir).exists()
        manifest_path = Path(output_dir) / "manifest.json"
        assert manifest_path.exists()

        with open(manifest_path) as f:
            manifest = json.load(f)

        assert "cluster_0" in manifest
        assert "cluster_1" in manifest
        assert "noise" in manifest
        assert len(manifest["cluster_0"]) == 2
        assert len(manifest["cluster_1"]) == 2
        assert len(manifest["noise"]) == 1


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def temp_dir(tmp_path):
    """Provide a temporary directory."""
    return tmp_path
