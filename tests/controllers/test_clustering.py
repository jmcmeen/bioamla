# tests/controllers/test_clustering.py
"""
Tests for ClusteringController.
"""

from unittest.mock import MagicMock, Mock

import numpy as np
import pytest

from bioamla.controllers.clustering import (
    ClusteringController,
    ClusteringSummary,
    NoveltyDetectionSummary,
)


class TestClusteringController:
    """Tests for ClusteringController."""

    @pytest.fixture
    def controller(self):
        return ClusteringController()

    @pytest.fixture
    def sample_embeddings(self):
        """Create sample embeddings for testing."""
        return np.random.randn(50, 768).astype(np.float32)

    @pytest.fixture
    def mock_clusterer(self, mocker):
        """Mock the AudioClusterer."""
        mocker.patch("bioamla.core.analysis.clustering.ClusteringConfig")
        mock_clusterer_class = mocker.patch("bioamla.core.analysis.clustering.AudioClusterer")

        instance = Mock()
        # Return labels with 3 clusters and some noise
        instance.fit_predict.return_value = np.array([0] * 15 + [1] * 15 + [2] * 15 + [-1] * 5)
        mock_clusterer_class.return_value = instance

        return instance

    def test_cluster_success(self, controller, sample_embeddings, mock_clusterer, mocker):
        """Test that clustering succeeds with valid embeddings."""
        # Mock silhouette score
        mock_silhouette = mocker.patch("sklearn.metrics.silhouette_score")
        mock_silhouette.return_value = 0.45

        # Mock run tracking
        mocker.patch.object(controller, "_start_run", return_value="test_run")
        mocker.patch.object(controller, "_complete_run")

        result = controller.cluster(sample_embeddings, method="hdbscan")

        assert result.success is True
        assert result.data is not None
        assert isinstance(result.data, ClusteringSummary)
        assert result.data.n_clusters == 3
        assert result.data.n_noise == 5
        assert result.data.method == "hdbscan"

    def test_cluster_kmeans_success(self, controller, sample_embeddings, mock_clusterer, mocker):
        """Test that k-means clustering succeeds."""
        mock_silhouette = mocker.patch("sklearn.metrics.silhouette_score")
        mock_silhouette.return_value = 0.5

        mocker.patch.object(controller, "_start_run", return_value="test_run")
        mocker.patch.object(controller, "_complete_run")

        result = controller.cluster(sample_embeddings, method="kmeans", n_clusters=5)

        assert result.success is True
        assert result.data is not None


class TestDimensionalityReduction:
    """Tests for dimensionality reduction methods."""

    @pytest.fixture
    def controller(self):
        return ClusteringController()

    @pytest.fixture
    def sample_embeddings(self):
        """Create sample embeddings for testing."""
        return np.random.randn(20, 768).astype(np.float32)

    def test_reduce_dimensions_success(self, controller, sample_embeddings, mocker):
        """Test that dimensionality reduction succeeds."""
        mock_reduce = mocker.patch("bioamla.core.analysis.clustering.reduce_dimensions")
        mock_reduce.return_value = np.random.randn(20, 2).astype(np.float32)

        result = controller.reduce_dimensions(sample_embeddings, method="pca", n_components=2)

        assert result.success is True
        assert result.data is not None
        assert result.data["reduced_shape"] == (20, 2)
        assert result.data["method"] == "pca"

    def test_reduce_for_visualization_returns_coordinates(
        self, controller, sample_embeddings, mocker
    ):
        """Test that visualization reduction returns x,y coordinates."""
        mock_reduce = mocker.patch("bioamla.core.analysis.clustering.reduce_dimensions")
        mock_reduce.return_value = np.random.randn(20, 2).astype(np.float32)

        result = controller.reduce_for_visualization(sample_embeddings)

        assert result.success is True
        assert "x" in result.data
        assert "y" in result.data
        assert len(result.data["x"]) == 20


class TestNoveltyDetection:
    """Tests for novelty detection methods."""

    @pytest.fixture
    def controller(self):
        return ClusteringController()

    @pytest.fixture
    def sample_embeddings(self):
        """Create sample embeddings for testing."""
        return np.random.randn(30, 768).astype(np.float32)

    def test_detect_novelty_success(self, controller, sample_embeddings, mocker):
        """Test that novelty detection succeeds."""
        mock_detector = mocker.patch("bioamla.core.analysis.clustering.NoveltyDetector")

        # Create mock novelty results
        mock_result = MagicMock()
        mock_result.sample_idx = 0
        mock_result.is_novel = True
        mock_result.novelty_score = 0.9

        mock_known_result = MagicMock()
        mock_known_result.sample_idx = 1
        mock_known_result.is_novel = False
        mock_known_result.novelty_score = 0.2

        instance = Mock()
        # 5 novel samples out of 30
        novel_results = [mock_result] * 5 + [mock_known_result] * 25
        for i, r in enumerate(novel_results):
            r.sample_idx = i
        instance.predict.return_value = novel_results
        mock_detector.return_value = instance

        result = controller.detect_novelty(sample_embeddings, method="distance")

        assert result.success is True
        assert result.data is not None
        assert isinstance(result.data, NoveltyDetectionSummary)
        assert result.data.n_novel == 5
        assert result.data.n_known == 25


class TestFindOptimalK:
    """Tests for find_optimal_k method."""

    @pytest.fixture
    def controller(self):
        return ClusteringController()

    @pytest.fixture
    def sample_embeddings(self):
        """Create sample embeddings for testing."""
        return np.random.randn(50, 768).astype(np.float32)

    def test_find_optimal_k_success(self, controller, sample_embeddings, mocker):
        """Test that find_optimal_k succeeds."""
        mock_find = mocker.patch("bioamla.core.analysis.clustering.find_optimal_clusters")
        mock_find.return_value = 5

        result = controller.find_optimal_k(sample_embeddings, k_range=(2, 10))

        assert result.success is True
        assert result.data["optimal_k"] == 5
        assert result.data["k_range"] == (2, 10)

    def test_find_optimal_k_with_silhouette(self, controller, sample_embeddings, mocker):
        """Test that find_optimal_k works with silhouette method."""
        mock_find = mocker.patch("bioamla.core.analysis.clustering.find_optimal_clusters")
        mock_find.return_value = 7

        result = controller.find_optimal_k(
            sample_embeddings, k_range=(2, 15), method="silhouette"
        )

        assert result.success is True
        assert result.data["method"] == "silhouette"


class TestAnalyzeClusters:
    """Tests for analyze_clusters method."""

    @pytest.fixture
    def controller(self):
        return ClusteringController()

    @pytest.fixture
    def sample_embeddings(self):
        """Create sample embeddings for testing."""
        return np.random.randn(50, 768).astype(np.float32)

    @pytest.fixture
    def sample_labels(self):
        """Create sample cluster labels."""
        return np.array([0] * 20 + [1] * 20 + [2] * 10)

    def test_analyze_clusters_success(
        self, controller, sample_embeddings, sample_labels, mocker
    ):
        """Test that cluster analysis succeeds."""
        mock_analyze = mocker.patch("bioamla.core.analysis.clustering.analyze_clusters")
        mock_analyze.return_value = {
            "n_clusters": 3,
            "n_samples": 50,
            "n_noise": 0,
            "silhouette_score": 0.45,
            "calinski_harabasz_score": 120.5,
            "cluster_stats": {
                0: {"size": 20, "centroid_dist": 0.5},
                1: {"size": 20, "centroid_dist": 0.6},
                2: {"size": 10, "centroid_dist": 0.4},
            },
        }

        result = controller.analyze_clusters(sample_embeddings, sample_labels)

        assert result.success is True
        assert result.data.n_clusters == 3
        assert result.data.silhouette_score == 0.45

    def test_analyze_clusters_with_filepaths(
        self, controller, sample_embeddings, sample_labels, mocker
    ):
        """Test that cluster analysis works with filepaths."""
        mock_analyze = mocker.patch("bioamla.core.analysis.clustering.analyze_clusters")
        mock_analyze.return_value = {
            "n_clusters": 3,
            "n_samples": 50,
            "n_noise": 0,
            "silhouette_score": 0.45,
            "calinski_harabasz_score": 120.5,
            "cluster_stats": {},
        }

        filepaths = [f"/path/to/file_{i}.wav" for i in range(50)]
        result = controller.analyze_clusters(sample_embeddings, sample_labels, filepaths)

        assert result.success is True
        mock_analyze.assert_called_once()


class TestExportClusters:
    """Tests for export methods."""

    @pytest.fixture
    def controller(self):
        return ClusteringController()

    @pytest.fixture
    def sample_labels(self):
        """Create sample cluster labels."""
        return np.array([0, 0, 1, 1, 2, 2, -1])

    @pytest.fixture
    def sample_filepaths(self):
        """Create sample file paths."""
        return [f"/path/to/file_{i}.wav" for i in range(7)]

    def test_export_clusters_success(
        self, controller, sample_labels, sample_filepaths, tmp_path, mocker
    ):
        """Test that export_clusters succeeds."""
        mock_export = mocker.patch("bioamla.core.analysis.clustering.export_clusters")
        mock_export.return_value = str(tmp_path / "clusters")

        result = controller.export_clusters(
            sample_labels,
            sample_filepaths,
            str(tmp_path / "clusters"),
            copy_files=False,
        )

        assert result.success is True
        assert result.data["n_clusters"] == 3

    def test_export_to_csv_success(
        self, controller, sample_labels, sample_filepaths, tmp_path, mocker
    ):
        """Test that export_to_csv succeeds."""
        mocker.patch("bioamla.core.files.TextFile")

        output_csv = tmp_path / "clusters.csv"
        result = controller.export_to_csv(sample_labels, sample_filepaths, str(output_csv))

        assert result.success is True
        assert result.data["n_rows"] == 7

    def test_export_to_csv_with_coordinates(
        self, controller, sample_labels, sample_filepaths, tmp_path, mocker
    ):
        """Test that export_to_csv includes 2D coordinates when provided."""
        mocker.patch("bioamla.core.files.TextFile")

        reduced = np.random.randn(7, 2).astype(np.float32)
        output_csv = tmp_path / "clusters.csv"

        result = controller.export_to_csv(
            sample_labels, sample_filepaths, str(output_csv), reduced_embeddings=reduced
        )

        assert result.success is True


class TestClusterSimilarity:
    """Tests for cluster similarity methods."""

    @pytest.fixture
    def controller(self):
        return ClusteringController()

    @pytest.fixture
    def sample_embeddings(self):
        """Create sample embeddings for testing."""
        return np.random.randn(50, 768).astype(np.float32)

    @pytest.fixture
    def sample_labels(self):
        """Create sample cluster labels."""
        return np.array([0] * 20 + [1] * 20 + [2] * 10)

    def test_compute_similarity_success(
        self, controller, sample_embeddings, sample_labels, mocker
    ):
        """Test that compute_similarity succeeds."""
        mock_similarity = mocker.patch(
            "bioamla.core.analysis.clustering.compute_cluster_similarity"
        )
        mock_similarity.return_value = np.array(
            [[1.0, 0.5, 0.3], [0.5, 1.0, 0.4], [0.3, 0.4, 1.0]]
        )

        result = controller.compute_similarity(
            sample_embeddings, sample_labels, metric="cosine"
        )

        assert result.success is True
        assert result.data["n_clusters"] == 3
        assert result.data["metric"] == "cosine"

    def test_sort_clusters_by_similarity_success(
        self, controller, sample_embeddings, sample_labels, mocker
    ):
        """Test that sort_clusters_by_similarity succeeds."""
        mock_sort = mocker.patch(
            "bioamla.core.analysis.clustering.sort_clusters_by_similarity"
        )
        mock_sort.return_value = [0, 2, 1]  # Sorted order

        result = controller.sort_clusters_by_similarity(sample_embeddings, sample_labels)

        assert result.success is True
        assert result.data["sorted_labels"] == [0, 2, 1]


class TestGetMostNovel:
    """Tests for get_most_novel method."""

    @pytest.fixture
    def controller(self):
        return ClusteringController()

    @pytest.fixture
    def sample_embeddings(self):
        """Create sample embeddings for testing."""
        return np.random.randn(30, 768).astype(np.float32)

    def test_get_most_novel_success(self, controller, sample_embeddings, mocker):
        """Test that get_most_novel succeeds."""
        mock_detector = mocker.patch("bioamla.core.analysis.clustering.NoveltyDetector")

        # Create mock novelty results with varying scores
        mock_results = []
        for i in range(30):
            r = MagicMock()
            r.sample_idx = i
            r.novelty_score = i * 0.1  # Increasing scores
            r.is_novel = i >= 25
            mock_results.append(r)

        instance = Mock()
        instance.predict.return_value = mock_results
        mock_detector.return_value = instance

        result = controller.get_most_novel(sample_embeddings, n_samples=5)

        assert result.success is True
        assert len(result.data["indices"]) == 5
        assert len(result.data["scores"]) == 5


class TestClusteringSummary:
    """Tests for ClusteringSummary dataclass."""

    def test_clustering_summary_fields(self):
        """Test that ClusteringSummary has all expected fields."""
        summary = ClusteringSummary(
            n_clusters=5,
            n_samples=100,
            n_noise=10,
            noise_percentage=10.0,
            silhouette_score=0.45,
            method="hdbscan",
            labels=[0, 1, 2, 3, 4],
            cluster_sizes={0: 20, 1: 20, 2: 20, 3: 20, 4: 10, -1: 10},
        )

        assert summary.n_clusters == 5
        assert summary.n_samples == 100
        assert summary.n_noise == 10
        assert summary.silhouette_score == 0.45


class TestNoveltyDetectionSummary:
    """Tests for NoveltyDetectionSummary dataclass."""

    def test_novelty_detection_summary_fields(self):
        """Test that NoveltyDetectionSummary has all expected fields."""
        summary = NoveltyDetectionSummary(
            n_samples=100,
            n_novel=15,
            n_known=85,
            novel_percentage=15.0,
            method="distance",
            threshold=0.5,
            novel_indices=[1, 5, 10, 15, 20],
        )

        assert summary.n_samples == 100
        assert summary.n_novel == 15
        assert summary.novel_percentage == 15.0
