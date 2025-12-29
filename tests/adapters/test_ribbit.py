"""Tests for RIBBIT adapter."""

import numpy as np
import pytest
import scipy.io.wavfile as wav

from bioamla.adapters.opensoundscape import (
    RIBBIT_PRESETS,
    RibbitDetection,
    get_ribbit_preset,
    list_ribbit_presets,
    ribbit_detect,
    ribbit_detect_preset,
    ribbit_detect_samples,
)


@pytest.fixture
def sample_audio_path(tmp_path) -> str:
    """Create a test audio file with a pulsing tone (simulated frog call)."""
    sample_rate = 16000
    duration = 3.0
    t = np.linspace(0, duration, int(sample_rate * duration), dtype=np.float32)

    # Create a pulsing tone at 1000Hz with 5Hz pulse rate
    carrier = np.sin(2 * np.pi * 1000 * t)
    pulse_rate = 5.0  # Hz
    modulator = 0.5 * (1 + np.sin(2 * np.pi * pulse_rate * t))
    samples = (carrier * modulator * 0.5 * 32767).astype(np.int16)

    audio_path = tmp_path / "pulsing_tone.wav"
    wav.write(str(audio_path), sample_rate, samples)

    return str(audio_path)


@pytest.fixture
def sample_audio_samples() -> tuple:
    """Create test audio samples with a pulsing tone."""
    sample_rate = 16000
    duration = 3.0
    t = np.linspace(0, duration, int(sample_rate * duration), dtype=np.float32)

    # Create a pulsing tone
    carrier = np.sin(2 * np.pi * 1000 * t)
    pulse_rate = 5.0
    modulator = 0.5 * (1 + np.sin(2 * np.pi * pulse_rate * t))
    samples = (carrier * modulator * 0.5).astype(np.float32)

    return samples, sample_rate


class TestRibbitDetection:
    """Tests for RibbitDetection dataclass."""

    def test_create_detection(self) -> None:
        """Test creating a RibbitDetection."""
        detection = RibbitDetection(
            start_time=1.0,
            end_time=2.5,
            score=0.85,
        )

        assert detection.start_time == 1.0
        assert detection.end_time == 2.5
        assert detection.score == 0.85

    def test_duration_property(self) -> None:
        """Test duration property."""
        detection = RibbitDetection(
            start_time=1.0,
            end_time=3.0,
            score=0.9,
        )

        assert detection.duration == 2.0

    def test_to_dict(self) -> None:
        """Test to_dict method."""
        detection = RibbitDetection(
            start_time=0.5,
            end_time=1.5,
            score=0.75,
        )

        d = detection.to_dict()

        assert d["start_time"] == 0.5
        assert d["end_time"] == 1.5
        assert d["score"] == 0.75
        assert d["duration"] == 1.0


class TestRibbitPresets:
    """Tests for RIBBIT preset functions."""

    def test_list_presets(self) -> None:
        """Test listing available presets."""
        presets = list_ribbit_presets()

        assert isinstance(presets, list)
        assert len(presets) > 0
        assert "american_bullfrog" in presets
        assert "spring_peeper" in presets
        assert "generic_mid_freq" in presets

    def test_get_preset(self) -> None:
        """Test getting a preset's parameters."""
        params = get_ribbit_preset("american_bullfrog")

        assert "signal_band" in params
        assert "pulse_rate_range" in params
        assert params["signal_band"] == (100, 400)

    def test_get_preset_invalid(self) -> None:
        """Test getting an invalid preset raises ValueError."""
        with pytest.raises(ValueError, match="Unknown preset"):
            get_ribbit_preset("nonexistent_frog")

    def test_presets_dict_has_required_keys(self) -> None:
        """Test all presets have required keys."""
        for name, params in RIBBIT_PRESETS.items():
            assert "signal_band" in params, f"{name} missing signal_band"
            assert "pulse_rate_range" in params, f"{name} missing pulse_rate_range"


class TestRibbitDetect:
    """Tests for ribbit_detect function."""

    def test_detect_file(self, sample_audio_path: str) -> None:
        """Test detection on audio file."""
        detections, metadata = ribbit_detect(
            audio_path=sample_audio_path,
            signal_band=(800, 1200),
            pulse_rate_range=(3.0, 8.0),
            score_threshold=0.1,  # Low threshold for test
        )

        assert isinstance(detections, list)
        assert isinstance(metadata, dict)
        assert "duration" in metadata
        assert "sample_rate" in metadata
        assert "num_detections" in metadata
        assert metadata["duration"] > 0

    def test_detect_returns_scores_and_times(self, sample_audio_path: str) -> None:
        """Test that metadata includes scores and times."""
        detections, metadata = ribbit_detect(
            audio_path=sample_audio_path,
            signal_band=(800, 1200),
            pulse_rate_range=(3.0, 8.0),
        )

        assert "scores" in metadata
        assert "times" in metadata
        assert isinstance(metadata["scores"], list)
        assert isinstance(metadata["times"], list)


class TestRibbitDetectSamples:
    """Tests for ribbit_detect_samples function."""

    def test_detect_samples(self, sample_audio_samples: tuple) -> None:
        """Test detection on audio samples."""
        samples, sample_rate = sample_audio_samples

        detections, metadata = ribbit_detect_samples(
            samples=samples,
            sample_rate=sample_rate,
            signal_band=(800, 1200),
            pulse_rate_range=(3.0, 8.0),
            score_threshold=0.1,
        )

        assert isinstance(detections, list)
        assert isinstance(metadata, dict)
        assert metadata["sample_rate"] == sample_rate

    def test_detect_stereo_samples(self, sample_audio_samples: tuple) -> None:
        """Test detection handles stereo samples."""
        samples, sample_rate = sample_audio_samples
        # Create stereo
        stereo_samples = np.stack([samples, samples], axis=-1)

        detections, metadata = ribbit_detect_samples(
            samples=stereo_samples,
            sample_rate=sample_rate,
            signal_band=(800, 1200),
            pulse_rate_range=(3.0, 8.0),
        )

        assert isinstance(detections, list)


class TestRibbitDetectPreset:
    """Tests for ribbit_detect_preset function."""

    def test_detect_with_preset(self, sample_audio_path: str) -> None:
        """Test detection using a preset."""
        detections, metadata = ribbit_detect_preset(
            audio_path=sample_audio_path,
            preset="generic_mid_freq",
        )

        assert isinstance(detections, list)
        assert isinstance(metadata, dict)

    def test_detect_preset_with_override(self, sample_audio_path: str) -> None:
        """Test detection with threshold override."""
        detections, metadata = ribbit_detect_preset(
            audio_path=sample_audio_path,
            preset="generic_mid_freq",
            score_threshold=0.9,  # Higher threshold
        )

        assert isinstance(detections, list)

    def test_detect_invalid_preset(self, sample_audio_path: str) -> None:
        """Test detection with invalid preset raises ValueError."""
        with pytest.raises(ValueError, match="Unknown preset"):
            ribbit_detect_preset(
                audio_path=sample_audio_path,
                preset="invalid_preset",
            )


class TestRibbitIntegration:
    """Integration tests for RIBBIT adapter."""

    def test_full_workflow(self, sample_audio_path: str) -> None:
        """Test complete RIBBIT workflow."""
        # List available presets
        presets = list_ribbit_presets()
        assert len(presets) > 0

        # Get preset parameters
        params = get_ribbit_preset("generic_mid_freq")
        assert "signal_band" in params

        # Run detection with preset
        detections, metadata = ribbit_detect_preset(
            audio_path=sample_audio_path,
            preset="generic_mid_freq",
        )

        assert isinstance(detections, list)
        assert metadata["duration"] > 0

        # Verify detection structure if any found
        for det in detections:
            assert isinstance(det, RibbitDetection)
            assert det.end_time > det.start_time
            assert 0 <= det.score <= 1
