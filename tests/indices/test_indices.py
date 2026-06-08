"""Tests for the indices domain (flattened, exception-based API)."""

import numpy as np
import pytest

from bioamla.audio import AudioData
from bioamla.exceptions import InvalidInputError
from bioamla.indices import (
    AVAILABLE_INDICES,
    AcousticIndices,
    compute_all_indices,
    compute_index,
    describe_index,
    temporal_indices,
)


class TestComputeAllIndices:
    def test_compute_all_indices(self, sample_audio_data: AudioData) -> None:
        result = compute_all_indices(sample_audio_data.samples, sample_audio_data.sample_rate)
        assert isinstance(result, AcousticIndices)
        d = result.to_dict()
        for key in ("aci", "adi", "aei", "bio", "ndsi"):
            assert key in d

    def test_compute_indices_with_entropy(self, sample_audio_data: AudioData) -> None:
        result = compute_all_indices(
            sample_audio_data.samples, sample_audio_data.sample_rate, include_entropy=True
        )
        assert result.h_spectral is not None
        assert result.h_temporal is not None

    def test_compute_indices_returns_valid_values(self, sample_audio_data: AudioData) -> None:
        result = compute_all_indices(sample_audio_data.samples, sample_audio_data.sample_rate)
        for key, value in result.to_dict().items():
            if isinstance(value, (int, float)):
                assert not np.isnan(value), f"{key} is NaN"
                assert not np.isinf(value), f"{key} is infinite"


class TestComputeSingleIndex:
    @pytest.mark.parametrize(
        "index_name",
        ["aci", "adi", "aei", "bio", "ndsi", "h_spectral", "h_temporal"],
    )
    def test_compute_index(self, sample_audio_data: AudioData, index_name: str) -> None:
        value = compute_index(sample_audio_data.samples, sample_audio_data.sample_rate, index_name)
        assert isinstance(value, float)
        assert not np.isnan(value)

    def test_compute_index_invalid_name(self, sample_audio_data: AudioData) -> None:
        with pytest.raises(InvalidInputError, match="Unknown index"):
            compute_index(sample_audio_data.samples, sample_audio_data.sample_rate, "invalid_index")


class TestTemporalIndices:
    def test_temporal_indices(self, sample_audio_3s: AudioData) -> None:
        windows = temporal_indices(
            sample_audio_3s.samples,
            sample_audio_3s.sample_rate,
            window_duration=1.0,
            hop_duration=1.0,
        )
        # 3-second audio with 1-second windows => 3 windows
        assert len(windows) == 3


class TestIndicesHelpers:
    def test_available_indices(self) -> None:
        assert "aci" in AVAILABLE_INDICES
        assert "ndsi" in AVAILABLE_INDICES

    def test_describe_index(self) -> None:
        description = describe_index("aci")
        assert isinstance(description, str)
        assert len(description) > 0

    def test_describe_index_invalid(self) -> None:
        assert describe_index("invalid") is None
