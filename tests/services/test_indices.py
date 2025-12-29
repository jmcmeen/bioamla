"""Tests for IndicesService - core paths for pre-migration verification."""

import numpy as np
import pytest

from bioamla.models.audio import AudioData
from bioamla.repository.local import LocalFileRepository
from bioamla.services.indices import IndicesService


class TestIndicesServiceCompute:
    """Tests for acoustic indices computation."""

    def test_compute_all_indices(
        self, mock_repository, sample_audio_data: AudioData
    ) -> None:
        """Test computing all acoustic indices without error."""
        service = IndicesService(mock_repository)

        result = service.calculate(sample_audio_data)

        assert result.success, f"Indices calculation failed: {result.error}"
        assert result.data is not None

        # Check that all indices are present
        indices_dict = result.data.to_dict()
        assert "aci" in indices_dict
        assert "adi" in indices_dict
        assert "aei" in indices_dict
        assert "bio" in indices_dict
        assert "ndsi" in indices_dict

    def test_compute_indices_with_entropy(
        self, mock_repository, sample_audio_data: AudioData
    ) -> None:
        """Test computing indices with entropy measures."""
        service = IndicesService(mock_repository)

        result = service.calculate(sample_audio_data, include_entropy=True)

        assert result.success
        assert result.data.h_spectral is not None
        assert result.data.h_temporal is not None

    def test_compute_indices_returns_valid_values(
        self, mock_repository, sample_audio_data: AudioData
    ) -> None:
        """Test that computed indices are valid numeric values."""
        service = IndicesService(mock_repository)

        result = service.calculate(sample_audio_data)

        assert result.success
        indices_dict = result.data.to_dict()

        for key, value in indices_dict.items():
            if isinstance(value, (int, float)):
                assert not np.isnan(value), f"{key} is NaN"
                assert not np.isinf(value), f"{key} is infinite"


class TestIndicesServiceSingleIndex:
    """Tests for single index computation."""

    @pytest.mark.parametrize(
        "index_name",
        ["aci", "adi", "aei", "bio", "ndsi", "h_spectral", "h_temporal"],
    )
    def test_calculate_single_index(
        self, mock_repository, sample_audio_data: AudioData, index_name: str
    ) -> None:
        """Test computing each individual index."""
        service = IndicesService(mock_repository)

        result = service.calculate_single_index(sample_audio_data, index_name)

        assert result.success, f"{index_name} calculation failed: {result.error}"
        assert isinstance(result.data, float)
        assert not np.isnan(result.data)

    def test_calculate_single_index_invalid_name(
        self, mock_repository, sample_audio_data: AudioData
    ) -> None:
        """Test error handling for invalid index name."""
        service = IndicesService(mock_repository)

        result = service.calculate_single_index(sample_audio_data, "invalid_index")

        assert not result.success
        assert "Unknown index" in result.error


class TestIndicesServiceTemporal:
    """Tests for temporal indices computation."""

    def test_calculate_temporal(
        self, mock_repository, sample_audio_3s: AudioData
    ) -> None:
        """Test temporal indices computation over sliding windows."""
        service = IndicesService(mock_repository)

        result = service.calculate_temporal(
            sample_audio_3s,
            window_duration=1.0,
            hop_duration=1.0,
        )

        assert result.success, f"Temporal calculation failed: {result.error}"
        assert result.data is not None
        # 3 second audio with 1 second windows = 3 windows
        assert result.data.num_windows == 3


class TestIndicesServiceHelpers:
    """Tests for helper methods."""

    def test_get_available_indices(self, mock_repository) -> None:
        """Test getting list of available indices."""
        service = IndicesService(mock_repository)

        indices = service.get_available_indices()

        assert isinstance(indices, list)
        assert len(indices) > 0
        assert "aci" in indices
        assert "ndsi" in indices

    def test_describe_index(self, mock_repository) -> None:
        """Test getting index descriptions."""
        service = IndicesService(mock_repository)

        description = service.describe_index("aci")

        assert description is not None
        assert isinstance(description, str)
        assert len(description) > 0

    def test_describe_index_invalid(self, mock_repository) -> None:
        """Test description for invalid index returns None."""
        service = IndicesService(mock_repository)

        description = service.describe_index("invalid")

        assert description is None
