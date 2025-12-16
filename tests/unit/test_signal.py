"""
Unit tests for bioamla.core.signal module.
"""

import struct
from pathlib import Path

import numpy as np
import pytest

from bioamla.signal import (
    AudioEvent,
    AudioSegment,
    bandpass_filter,
    batch_process,
    detect_onsets,
    highpass_filter,
    load_audio,
    lowpass_filter,
    normalize_loudness,
    peak_normalize,
    process_file,
    resample_audio,
    save_audio,
    segment_on_silence,
    spectral_denoise,
    split_audio_on_silence,
    trim_audio,
    trim_silence,
)


def _create_mock_wav(path: Path, duration: float = 1.0, freq: float = 440.0) -> None:
    """Create a WAV file with a sine wave for testing."""
    sample_rate = 16000
    num_samples = int(sample_rate * duration)
    bits_per_sample = 16
    num_channels = 1

    byte_rate = sample_rate * num_channels * bits_per_sample // 8
    block_align = num_channels * bits_per_sample // 8
    data_size = num_samples * block_align
    file_size = 36 + data_size

    with open(path, "wb") as f:
        f.write(b"RIFF")
        f.write(struct.pack("<I", file_size))
        f.write(b"WAVE")
        f.write(b"fmt ")
        f.write(struct.pack("<I", 16))
        f.write(struct.pack("<H", 1))
        f.write(struct.pack("<H", num_channels))
        f.write(struct.pack("<I", sample_rate))
        f.write(struct.pack("<I", byte_rate))
        f.write(struct.pack("<H", block_align))
        f.write(struct.pack("<H", bits_per_sample))
        f.write(b"data")
        f.write(struct.pack("<I", data_size))
        for i in range(num_samples):
            sample = int(16000 * np.sin(2 * np.pi * freq * i / sample_rate))
            f.write(struct.pack("<h", sample))


def _create_test_audio(duration: float = 1.0, freq: float = 440.0, sr: int = 16000) -> np.ndarray:
    """Create a test sine wave audio array."""
    t = np.arange(int(sr * duration)) / sr
    audio = np.sin(2 * np.pi * freq * t).astype(np.float32)
    return audio


class TestBandpassFilter:
    """Tests for bandpass_filter function."""

    def test_filters_audio(self):
        """Test that bandpass filter works."""
        audio = _create_test_audio(freq=1000)
        sr = 16000

        filtered = bandpass_filter(audio, sr, 500, 2000)

        assert isinstance(filtered, np.ndarray)
        assert len(filtered) == len(audio)

    def test_attenuates_out_of_band(self):
        """Test that frequencies outside band are attenuated."""
        sr = 16000
        # Create audio with 100Hz component (outside band)
        low_freq_audio = _create_test_audio(freq=100, sr=sr)

        filtered = bandpass_filter(low_freq_audio, sr, 500, 2000)

        # Energy should be reduced
        original_energy = np.sum(low_freq_audio ** 2)
        filtered_energy = np.sum(filtered ** 2)
        assert filtered_energy < original_energy * 0.5

    def test_invalid_range_raises(self):
        """Test that invalid frequency range raises error."""
        audio = _create_test_audio()

        with pytest.raises(ValueError, match="Low frequency"):
            bandpass_filter(audio, 16000, 5000, 1000)


class TestLowpassFilter:
    """Tests for lowpass_filter function."""

    def test_filters_audio(self):
        """Test that lowpass filter works."""
        audio = _create_test_audio(freq=440)
        sr = 16000

        filtered = lowpass_filter(audio, sr, 1000)

        assert isinstance(filtered, np.ndarray)
        assert len(filtered) == len(audio)


class TestHighpassFilter:
    """Tests for highpass_filter function."""

    def test_filters_audio(self):
        """Test that highpass filter works."""
        audio = _create_test_audio(freq=440)
        sr = 16000

        filtered = highpass_filter(audio, sr, 200)

        assert isinstance(filtered, np.ndarray)
        assert len(filtered) == len(audio)


class TestSpectralDenoise:
    """Tests for spectral_denoise function."""

    def test_denoises_audio(self):
        """Test that spectral denoise works."""
        audio = _create_test_audio()
        # Add noise
        noisy = audio + np.random.randn(len(audio)).astype(np.float32) * 0.1

        denoised = spectral_denoise(noisy, 16000)

        assert isinstance(denoised, np.ndarray)
        assert len(denoised) > 0

    def test_preserves_signal(self):
        """Test that signal is roughly preserved."""
        audio = _create_test_audio(freq=440)

        denoised = spectral_denoise(audio, 16000, noise_reduce_factor=0.5)

        # Should have similar energy
        assert np.sum(denoised ** 2) > 0


class TestSegmentOnSilence:
    """Tests for segment_on_silence function."""

    def test_returns_segments(self):
        """Test that segments are returned."""
        # Create audio with silence in middle
        audio = np.concatenate([
            _create_test_audio(duration=0.5),
            np.zeros(8000, dtype=np.float32),  # 0.5s silence
            _create_test_audio(duration=0.5),
        ])

        segments = segment_on_silence(audio, 16000, silence_threshold_db=-40)

        assert isinstance(segments, list)
        for seg in segments:
            assert isinstance(seg, AudioSegment)

    def test_segment_has_correct_attributes(self):
        """Test that segments have correct attributes."""
        audio = _create_test_audio(duration=1.0)

        segments = segment_on_silence(audio, 16000, silence_threshold_db=-60)

        if segments:
            seg = segments[0]
            assert hasattr(seg, 'start_time')
            assert hasattr(seg, 'end_time')
            assert hasattr(seg, 'start_sample')
            assert hasattr(seg, 'end_sample')


class TestSplitAudioOnSilence:
    """Tests for split_audio_on_silence function."""

    def test_returns_chunks(self):
        """Test that audio chunks are returned."""
        audio = _create_test_audio(duration=1.0)

        chunks = split_audio_on_silence(audio, 16000, silence_threshold_db=-60)

        assert isinstance(chunks, list)
        for chunk, start, end in chunks:
            assert isinstance(chunk, np.ndarray)
            assert isinstance(start, float)
            assert isinstance(end, float)


class TestDetectOnsets:
    """Tests for detect_onsets function."""

    def test_returns_events(self):
        """Test that events are returned."""
        # Create audio with clear onset
        audio = np.concatenate([
            np.zeros(4000, dtype=np.float32),
            _create_test_audio(duration=0.5),
        ])

        events = detect_onsets(audio, 16000)

        assert isinstance(events, list)
        for event in events:
            assert isinstance(event, AudioEvent)

    def test_event_has_correct_attributes(self):
        """Test that events have correct attributes."""
        audio = _create_test_audio()

        events = detect_onsets(audio, 16000)

        if events:
            event = events[0]
            assert hasattr(event, 'time')
            assert hasattr(event, 'strength')


class TestNormalizeLoudness:
    """Tests for normalize_loudness function."""

    def test_normalizes_audio(self):
        """Test that audio is normalized."""
        audio = _create_test_audio() * 0.1  # Quiet audio

        normalized = normalize_loudness(audio, 16000, target_db=-20)

        assert isinstance(normalized, np.ndarray)
        assert len(normalized) == len(audio)

    def test_increases_quiet_audio(self):
        """Test that quiet audio is made louder."""
        audio = _create_test_audio() * 0.01

        normalized = normalize_loudness(audio, 16000, target_db=-20)

        # Should be louder
        assert np.max(np.abs(normalized)) > np.max(np.abs(audio))

    def test_handles_silence(self):
        """Test that silence is handled gracefully."""
        audio = np.zeros(16000, dtype=np.float32)

        normalized = normalize_loudness(audio, 16000, target_db=-20)

        assert np.allclose(normalized, 0)


class TestPeakNormalize:
    """Tests for peak_normalize function."""

    def test_normalizes_to_peak(self):
        """Test that audio is normalized to target peak."""
        audio = _create_test_audio() * 0.5

        normalized = peak_normalize(audio, target_peak=0.95)

        assert np.max(np.abs(normalized)) == pytest.approx(0.95, abs=0.01)


class TestResampleAudio:
    """Tests for resample_audio function."""

    def test_resamples_audio(self):
        """Test that audio is resampled."""
        audio = _create_test_audio(sr=16000)

        resampled = resample_audio(audio, 16000, 8000)

        # Should have half the samples
        assert len(resampled) == len(audio) // 2

    def test_same_rate_returns_same(self):
        """Test that same rate returns same audio."""
        audio = _create_test_audio()

        resampled = resample_audio(audio, 16000, 16000)

        assert np.array_equal(audio, resampled)


class TestTrimAudio:
    """Tests for trim_audio function."""

    def test_trims_audio(self):
        """Test that audio is trimmed."""
        audio = _create_test_audio(duration=2.0)

        trimmed = trim_audio(audio, 16000, start_time=0.5, end_time=1.5)

        # Should have 1 second of audio
        assert len(trimmed) == 16000

    def test_trim_start_only(self):
        """Test trimming start only."""
        audio = _create_test_audio(duration=2.0)

        trimmed = trim_audio(audio, 16000, start_time=1.0)

        assert len(trimmed) == 16000

    def test_trim_end_only(self):
        """Test trimming end only."""
        audio = _create_test_audio(duration=2.0)

        trimmed = trim_audio(audio, 16000, end_time=1.0)

        assert len(trimmed) == 16000

    def test_invalid_range_raises(self):
        """Test that invalid range raises error."""
        audio = _create_test_audio()

        with pytest.raises(ValueError, match="Invalid trim range"):
            trim_audio(audio, 16000, start_time=1.0, end_time=0.5)


class TestTrimSilence:
    """Tests for trim_silence function."""

    def test_trims_silence(self):
        """Test that silence is trimmed."""
        # Create audio with silence at start and end
        audio = np.concatenate([
            np.zeros(4000, dtype=np.float32),
            _create_test_audio(duration=0.5),
            np.zeros(4000, dtype=np.float32),
        ])

        trimmed = trim_silence(audio, 16000, threshold_db=-40)

        # Should be shorter than original
        assert len(trimmed) < len(audio)


class TestLoadAndSaveAudio:
    """Tests for load_audio and save_audio functions."""

    def test_load_audio(self, temp_dir):
        """Test loading audio file."""
        audio_path = temp_dir / "test.wav"
        _create_mock_wav(audio_path)

        audio, sr = load_audio(str(audio_path))

        assert isinstance(audio, np.ndarray)
        assert sr == 16000
        assert len(audio) == 16000

    def test_save_audio(self, temp_dir):
        """Test saving audio file."""
        audio = _create_test_audio()
        output_path = temp_dir / "output.wav"

        result = save_audio(str(output_path), audio, 16000)

        assert output_path.exists()
        assert result == str(output_path)

    def test_save_creates_directory(self, temp_dir):
        """Test that save creates parent directory."""
        audio = _create_test_audio()
        output_path = temp_dir / "nested" / "dir" / "output.wav"

        save_audio(str(output_path), audio, 16000)

        assert output_path.exists()


class TestProcessFile:
    """Tests for process_file function."""

    def test_processes_file(self, temp_dir):
        """Test processing a single file."""
        input_path = temp_dir / "input.wav"
        output_path = temp_dir / "output.wav"
        _create_mock_wav(input_path)

        def processor(audio, sr):
            return audio * 0.5

        result = process_file(str(input_path), str(output_path), processor)

        assert output_path.exists()


class TestBatchProcess:
    """Tests for batch_process function."""

    def test_processes_directory(self, temp_dir):
        """Test batch processing a directory."""
        input_dir = temp_dir / "input"
        input_dir.mkdir()
        output_dir = temp_dir / "output"

        for i in range(3):
            _create_mock_wav(input_dir / f"audio_{i}.wav")

        def processor(audio, sr):
            return audio

        result = batch_process(
            str(input_dir),
            str(output_dir),
            processor,
            verbose=False,
        )

        assert result["files_processed"] == 3
        assert result["files_failed"] == 0

    def test_handles_empty_directory(self, temp_dir):
        """Test handling empty directory."""
        input_dir = temp_dir / "empty"
        input_dir.mkdir()
        output_dir = temp_dir / "output"

        def processor(audio, sr):
            return audio

        result = batch_process(
            str(input_dir),
            str(output_dir),
            processor,
            verbose=False,
        )

        assert result["files_processed"] == 0

    def test_raises_on_missing_directory(self, temp_dir):
        """Test that missing directory raises error."""
        def processor(audio, sr):
            return audio

        with pytest.raises(FileNotFoundError):
            batch_process(
                "/nonexistent/dir",
                str(temp_dir / "output"),
                processor,
            )
