# tests/controllers/test_indices.py
"""
Tests for IndicesController.
"""

from unittest.mock import Mock

import numpy as np
import pytest

from bioamla.controllers.audio_file import AudioData
from bioamla.controllers.indices import (
    BatchIndicesResult,
    IndicesController,
    IndicesResult,
)


class TestIndicesController:
    """Tests for IndicesController."""

    @pytest.fixture
    def controller(self):
        return IndicesController()

    @pytest.fixture
    def audio_data(self):
        """Create test audio data."""
        sr = 16000
        t = np.linspace(0, 1, sr, dtype=np.float32)
        # Generate test signal with multiple frequency components
        samples = (
            0.3 * np.sin(2 * np.pi * 440 * t)
            + 0.2 * np.sin(2 * np.pi * 880 * t)
            + 0.1 * np.sin(2 * np.pi * 1320 * t)
        ).astype(np.float32)
        return AudioData(samples=samples, sample_rate=sr)

    def test_calculate_success(self, controller, audio_data, mocker):
        """Test that index calculation succeeds."""
        # Mock the core index functions
        mock_compute = mocker.patch("bioamla.controllers.indices.compute_all_indices")
        mock_indices = Mock()
        mock_indices.to_dict.return_value = {
            "aci": 1.5,
            "adi": 2.0,
            "aei": 0.6,
            "bio": 10.0,
            "ndsi": 0.3,
        }
        mock_compute.return_value = mock_indices

        mocker.patch("bioamla.controllers.indices.spectral_entropy", return_value=0.85)
        mocker.patch("bioamla.controllers.indices.temporal_entropy", return_value=0.75)

        result = controller.calculate(audio_data)

        assert result.success is True
        assert result.data is not None
        assert isinstance(result.data, IndicesResult)
        assert result.data.h_spectral == 0.85
        assert result.data.h_temporal == 0.75

    def test_calculate_single_index_success(self, controller, audio_data, mocker):
        """Test that single index calculation succeeds."""
        mock_compute = mocker.patch("bioamla.controllers.indices.compute_aci")
        mock_compute.return_value = 1.5

        result = controller.calculate_single_index(audio_data, "aci")

        assert result.success is True
        assert result.data == 1.5

    def test_calculate_single_index_unknown_fails(self, controller, audio_data):
        """Test that unknown index name fails."""
        result = controller.calculate_single_index(audio_data, "unknown_index")

        assert result.success is False
        assert "Unknown index" in result.error


class TestBatchIndices:
    """Tests for batch index operations."""

    @pytest.fixture
    def controller(self):
        return IndicesController()

    def test_calculate_batch_success(self, controller, tmp_dir_with_audio_files, mocker):
        """Test that batch calculation succeeds."""
        # Mock storage for run tracking
        mocker.patch.object(controller, "_start_run", return_value="test_run")
        mocker.patch.object(controller, "_complete_run")

        # Mock the core index functions
        mock_compute = mocker.patch("bioamla.controllers.indices.compute_all_indices")
        mock_indices = Mock()
        mock_indices.to_dict.return_value = {
            "aci": 1.5,
            "adi": 2.0,
            "aei": 0.6,
            "bio": 10.0,
            "ndsi": 0.3,
        }
        mock_compute.return_value = mock_indices

        mocker.patch("bioamla.controllers.indices.spectral_entropy", return_value=0.85)
        mocker.patch("bioamla.controllers.indices.temporal_entropy", return_value=0.75)

        result = controller.calculate_batch(tmp_dir_with_audio_files)

        assert result.success is True
        assert result.data is not None
        assert isinstance(result.data, BatchIndicesResult)
        assert result.data.total == 3
        assert result.data.successful == 3

    def test_calculate_batch_empty_dir_fails(self, controller, tmp_path, mocker):
        """Test that batch calculation fails on empty directory."""
        mocker.patch.object(controller, "_start_run", return_value="test_run")
        mocker.patch.object(controller, "_fail_run")

        result = controller.calculate_batch(str(tmp_path))

        assert result.success is False
        assert "No audio files found" in result.error


class TestAvailableIndices:
    """Tests for index information methods."""

    @pytest.fixture
    def controller(self):
        return IndicesController()

    def test_get_available_indices_returns_list(self, controller):
        """Test that available indices returns a list."""
        indices = controller.get_available_indices()

        assert isinstance(indices, list)
        assert "aci" in indices
        assert "ndsi" in indices

    def test_describe_index_returns_description(self, controller):
        """Test that describe_index returns a description."""
        description = controller.describe_index("aci")

        assert description is not None
        assert "Acoustic Complexity Index" in description
