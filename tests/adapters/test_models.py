"""Tests for model zoo adapters (BirdNET, Perch, HawkEars).

These tests verify the adapter interfaces work correctly.
Inference tests are marked as slow since they require model downloads.
"""

import numpy as np
import pytest
import scipy.io.wavfile as wav

from bioamla.adapters.opensoundscape import (
    PredictionResult,
)


class TestPredictionResult:
    """Tests for PredictionResult dataclass."""

    def test_create_prediction_result(self) -> None:
        """Test creating a PredictionResult."""
        result = PredictionResult(
            label="American Robin",
            confidence=0.95,
            start_time=0.0,
            end_time=3.0,
            filepath="test.wav",
        )

        assert result.label == "American Robin"
        assert result.confidence == 0.95
        assert result.start_time == 0.0
        assert result.end_time == 3.0
        assert result.filepath == "test.wav"

    def test_prediction_result_optional_filepath(self) -> None:
        """Test PredictionResult with optional filepath."""
        result = PredictionResult(
            label="Song Sparrow",
            confidence=0.8,
            start_time=1.0,
            end_time=4.0,
        )

        assert result.filepath is None


@pytest.mark.slow
class TestCheckModelAvailability:
    """Tests for check_model_availability function - slow due to model loading."""

    def test_returns_dict(self) -> None:
        """Test that check_model_availability returns a dict."""
        from bioamla.adapters.opensoundscape import check_model_availability
        avail = check_model_availability()

        assert isinstance(avail, dict)
        assert "BirdNET" in avail
        assert "Perch" in avail
        assert "HawkEars" in avail

    def test_values_are_bool(self) -> None:
        """Test that all values are booleans."""
        from bioamla.adapters.opensoundscape import check_model_availability
        avail = check_model_availability()

        for model, available in avail.items():
            assert isinstance(available, bool), f"{model} availability should be bool"


class TestBirdNETAdapterUnit:
    """Unit tests for BirdNETAdapter that don't require inference."""

    def test_import(self) -> None:
        """Test BirdNETAdapter can be imported."""
        from bioamla.adapters.opensoundscape import BirdNETAdapter
        assert BirdNETAdapter is not None

    def test_constants(self) -> None:
        """Test BirdNETAdapter constants."""
        from bioamla.adapters.opensoundscape import BirdNETAdapter
        assert BirdNETAdapter.SAMPLE_RATE == 48000
        assert BirdNETAdapter.SAMPLE_DURATION == 3.0


class TestHawkEarsAdapterUnit:
    """Unit tests for HawkEarsAdapter that don't require inference."""

    def test_import(self) -> None:
        """Test HawkEarsAdapter can be imported."""
        from bioamla.adapters.opensoundscape import HawkEarsAdapter
        assert HawkEarsAdapter is not None


class TestPerchAdapterUnit:
    """Unit tests for PerchAdapter that don't require inference."""

    def test_import(self) -> None:
        """Test PerchAdapter can be imported."""
        from bioamla.adapters.opensoundscape import PerchAdapter
        assert PerchAdapter is not None

    def test_constants(self) -> None:
        """Test PerchAdapter constants."""
        from bioamla.adapters.opensoundscape import PerchAdapter
        assert PerchAdapter.SAMPLE_RATE == 32000
        assert PerchAdapter.SAMPLE_DURATION == 5.0


@pytest.fixture
def sample_audio_48k(tmp_path) -> str:
    """Create a temporary test audio file at 48kHz for BirdNET."""
    sample_rate = 48000
    duration = 3.0
    t = np.linspace(0, duration, int(sample_rate * duration), dtype=np.float32)
    frequency = 1000
    samples = (0.5 * np.sin(2 * np.pi * frequency * t) * 32767).astype(np.int16)

    audio_path = tmp_path / "test_audio_48k.wav"
    wav.write(str(audio_path), sample_rate, samples)

    return str(audio_path)


@pytest.fixture
def sample_audio_16k(tmp_path) -> str:
    """Create a temporary test audio file at 16kHz for HawkEars."""
    sample_rate = 16000
    duration = 5.0
    t = np.linspace(0, duration, int(sample_rate * duration), dtype=np.float32)
    frequency = 440
    samples = (0.5 * np.sin(2 * np.pi * frequency * t) * 32767).astype(np.int16)

    audio_path = tmp_path / "test_audio_16k.wav"
    wav.write(str(audio_path), sample_rate, samples)

    return str(audio_path)


@pytest.mark.slow
class TestBirdNETAdapterInference:
    """Inference tests for BirdNETAdapter - require model download."""

    def test_init(self) -> None:
        """Test BirdNETAdapter initialization."""
        from bioamla.adapters.opensoundscape import BirdNETAdapter
        try:
            adapter = BirdNETAdapter()
            assert adapter is not None
            assert adapter.sample_rate == 48000
            assert adapter.sample_duration == 3.0
        except ImportError:
            pytest.skip("BirdNET dependencies not available")

    def test_config(self) -> None:
        """Test config method."""
        from bioamla.adapters.opensoundscape import BirdNETAdapter
        try:
            adapter = BirdNETAdapter()
            config = adapter.config()

            assert config["model"] == "BirdNET"
            assert config["sample_rate"] == 48000
            assert config["sample_duration"] == 3.0
        except ImportError:
            pytest.skip("BirdNET dependencies not available")

    def test_predict(self, sample_audio_48k: str) -> None:
        """Test prediction on audio file."""
        from bioamla.adapters.opensoundscape import BirdNETAdapter
        try:
            adapter = BirdNETAdapter()
            predictions = adapter.predict(sample_audio_48k)

            assert predictions is not None
            import pandas as pd
            assert isinstance(predictions, pd.DataFrame)
        except ImportError:
            pytest.skip("BirdNET dependencies not available")


@pytest.mark.slow
class TestHawkEarsAdapterInference:
    """Inference tests for HawkEarsAdapter - require model download."""

    def test_init(self) -> None:
        """Test HawkEarsAdapter initialization."""
        from bioamla.adapters.opensoundscape import HawkEarsAdapter
        try:
            adapter = HawkEarsAdapter()
            assert adapter is not None
            assert adapter.variant == "default"
        except ImportError:
            pytest.skip("HawkEars dependencies not available")

    def test_config(self) -> None:
        """Test config method."""
        from bioamla.adapters.opensoundscape import HawkEarsAdapter
        try:
            adapter = HawkEarsAdapter()
            config = adapter.config()

            assert "HawkEars" in config["model"]
            assert "sample_rate" in config
            assert "sample_duration" in config
        except ImportError:
            pytest.skip("HawkEars dependencies not available")
