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
