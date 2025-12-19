"""
Unit tests for bioamla.core.torchaudio module.
"""

import pytest
import torch

from bioamla.core.torchaudio import (
    load_audio_from_bytes,
    load_waveform_tensor,
    resample_waveform_tensor,
    split_waveform_tensor,
)


class TestLoadWaveformTensor:
    """Tests for load_waveform_tensor function."""

    def test_loads_wav_file(self, mock_audio_file):
        """Test loading a WAV file."""
        waveform, sample_rate = load_waveform_tensor(str(mock_audio_file))

        assert isinstance(waveform, torch.Tensor)
        assert isinstance(sample_rate, int)
        assert sample_rate == 16000
        assert waveform.dim() == 2  # (channels, samples)

    def test_waveform_shape(self, mock_audio_file):
        """Test that waveform has expected shape."""
        waveform, sample_rate = load_waveform_tensor(str(mock_audio_file))

        # 1 second at 16kHz = 16000 samples
        assert waveform.shape[0] == 1  # mono
        assert waveform.shape[1] == 16000


class TestResampleWaveformTensor:
    """Tests for resample_waveform_tensor function."""

    def test_resamples_correctly(self):
        """Test resampling a waveform tensor."""
        # Create a 1-second waveform at 16kHz
        original_sr = 16000
        target_sr = 8000
        waveform = torch.zeros(1, original_sr)

        resampled = resample_waveform_tensor(waveform, original_sr, target_sr)

        assert resampled.shape[1] == target_sr  # Half the samples

    def test_upsample(self):
        """Test upsampling a waveform."""
        original_sr = 8000
        target_sr = 16000
        waveform = torch.zeros(1, original_sr)

        resampled = resample_waveform_tensor(waveform, original_sr, target_sr)

        assert resampled.shape[1] == target_sr

    def test_same_rate_no_change(self):
        """Test that same sample rate returns similar tensor."""
        sr = 16000
        waveform = torch.randn(1, sr)

        resampled = resample_waveform_tensor(waveform, sr, sr)

        assert resampled.shape == waveform.shape


class TestSplitWaveformTensor:
    """Tests for split_waveform_tensor function."""

    def test_splits_into_segments(self):
        """Test splitting waveform into segments."""
        freq = 16000
        duration_seconds = 30
        clip_seconds = 10
        overlap_seconds = 0

        waveform = torch.zeros(1, freq * duration_seconds)

        segments = split_waveform_tensor(waveform, freq, clip_seconds, overlap_seconds)

        # 30 seconds / 10 seconds = 3 segments
        assert len(segments) == 3

    def test_segment_shape(self):
        """Test that segments have correct shape."""
        freq = 16000
        clip_seconds = 10
        waveform = torch.zeros(1, freq * 30)

        segments = split_waveform_tensor(waveform, freq, clip_seconds, 0)

        for segment, start, end in segments:
            assert segment.shape[1] == freq * clip_seconds
            assert end - start == freq * clip_seconds

    def test_overlap(self):
        """Test splitting with overlap."""
        freq = 16000
        duration_seconds = 20
        clip_seconds = 10
        overlap_seconds = 5

        waveform = torch.zeros(1, freq * duration_seconds)

        segments = split_waveform_tensor(waveform, freq, clip_seconds, overlap_seconds)

        # With 5s overlap, step is 5s, so we get more segments
        # Positions: 0-10, 5-15, 10-20 = 3 segments
        assert len(segments) == 3

    def test_segment_positions(self):
        """Test that segment positions are correct."""
        freq = 16000
        waveform = torch.zeros(1, freq * 30)

        segments = split_waveform_tensor(waveform, freq, 10, 0)

        expected_positions = [
            (0, 160000),
            (160000, 320000),
            (320000, 480000)
        ]

        for (_segment, start, end), (exp_start, exp_end) in zip(segments, expected_positions):
            assert start == exp_start
            assert end == exp_end

    def test_short_audio_returns_empty(self):
        """Test that audio shorter than clip length returns empty list."""
        freq = 16000
        clip_seconds = 10
        waveform = torch.zeros(1, freq * 5)  # Only 5 seconds

        segments = split_waveform_tensor(waveform, freq, clip_seconds, 0)

        assert len(segments) == 0


class TestLoadAudioFromBytes:
    """Tests for load_audio_from_bytes function."""

    def test_loads_from_bytes(self, mock_audio_file):
        """Test loading audio from bytes."""
        with open(mock_audio_file, "rb") as f:
            audio_bytes = f.read()

        audio_array, sample_rate = load_audio_from_bytes(audio_bytes)

        assert sample_rate == 16000  # Default target
        assert len(audio_array.shape) == 1  # Should be 1D

    def test_resamples_to_target(self, mock_audio_file):
        """Test that audio is resampled to target sample rate."""
        with open(mock_audio_file, "rb") as f:
            audio_bytes = f.read()

        # Request different sample rate
        audio_array, sample_rate = load_audio_from_bytes(audio_bytes, target_sr=8000)

        assert sample_rate == 8000

    def test_invalid_bytes_raises(self):
        """Test that invalid bytes raise ValueError."""
        invalid_bytes = b"not audio data"

        with pytest.raises(ValueError, match="Could not process audio bytes"):
            load_audio_from_bytes(invalid_bytes)
