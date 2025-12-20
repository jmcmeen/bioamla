# tests/controllers/conftest.py
"""
Controller-specific pytest fixtures.
"""

from unittest.mock import Mock

import numpy as np
import pytest


@pytest.fixture
def mock_soundfile(mocker):
    """Patch soundfile.read and soundfile.write."""
    mock_sf = mocker.patch("soundfile.read")
    mock_sf.return_value = (np.zeros(16000, dtype=np.float32), 16000)

    mock_write = mocker.patch("soundfile.write")
    return mock_sf, mock_write


@pytest.fixture
def mock_ast_model(mocker):
    """Mock AST inference model."""
    mock = mocker.patch("bioamla.core.ml.inference.ASTInference")
    instance = Mock()
    instance.predict.return_value = [
        {
            "label": "speech",
            "confidence": 0.95,
            "start_time": 0.0,
            "end_time": 1.0,
            "top_k_labels": ["speech", "music", "noise"],
            "top_k_scores": [0.95, 0.03, 0.02],
        }
    ]
    instance.get_embeddings.return_value = np.random.randn(1, 768).astype(np.float32)
    mock.return_value = instance
    return instance


@pytest.fixture
def mock_ribbit_detector(mocker):
    """Mock RIBBIT detector."""
    mock = mocker.patch("bioamla.core.detection.ribbit.RibbitDetector")

    # Create mock result
    mock_result = Mock()
    mock_result.filepath = "/test/audio.wav"
    mock_result.profile_name = "test_preset"
    mock_result.num_detections = 2
    mock_result.total_detection_time = 0.5
    mock_result.detection_percentage = 50.0
    mock_result.duration = 1.0
    mock_result.processing_time = 0.1
    mock_result.error = None
    mock_result.detections = []

    instance = Mock()
    instance.detect.return_value = mock_result
    mock.return_value = instance
    mock.from_preset.return_value = instance

    return instance


@pytest.fixture
def mock_clusterer(mocker):
    """Mock AudioClusterer."""
    mock = mocker.patch("bioamla.core.analysis.clustering.AudioClusterer")
    instance = Mock()
    instance.fit_predict.return_value = np.array([0, 0, 1, 1, -1])
    mock.return_value = instance
    return instance


@pytest.fixture
def mock_embedding_extractor(mocker):
    """Mock EmbeddingExtractor."""
    mock = mocker.patch("bioamla.core.ml.embeddings.EmbeddingExtractor")
    instance = Mock()
    instance.extract.return_value = np.random.randn(768).astype(np.float32)
    instance.extract_batch.return_value = np.random.randn(5, 768).astype(np.float32)
    mock.return_value = instance
    return instance


@pytest.fixture
def sample_embeddings():
    """Return sample embeddings for clustering tests."""
    return np.random.randn(10, 768).astype(np.float32)


@pytest.fixture
def sample_labels():
    """Return sample cluster labels."""
    return np.array([0, 0, 0, 1, 1, 1, 2, 2, -1, -1])
