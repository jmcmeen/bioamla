"""Performance benchmarks for audio operations.

Run with: pytest tests/benchmarks/ --benchmark-only
Save baseline: pytest tests/benchmarks/ --benchmark-save=baseline
Compare: pytest tests/benchmarks/ --benchmark-compare=baseline

These benchmarks establish baseline performance metrics before the
OpenSoundscape migration to ensure no performance regression.
"""

import numpy as np
import pytest

from bioamla.models.audio import AudioData
from bioamla.repository.local import LocalFileRepository
from bioamla.services.audio_transform import AudioTransformService


@pytest.fixture
def benchmark_audio_data() -> AudioData:
    """Create audio data for benchmarking (10 seconds at 44100 Hz)."""
    sample_rate = 44100
    duration = 10.0
    frequency = 440.0

    t = np.linspace(0, duration, int(sample_rate * duration), dtype=np.float32)
    samples = 0.5 * np.sin(2 * np.pi * frequency * t)

    return AudioData(
        samples=samples,
        sample_rate=sample_rate,
        channels=1,
    )


@pytest.fixture
def audio_transform_service() -> AudioTransformService:
    """Create service for benchmarking."""
    repository = LocalFileRepository()
    return AudioTransformService(repository)


@pytest.mark.benchmark
class TestResampleBenchmarks:
    """Benchmarks for resampling operations."""

    def test_benchmark_resample_44100_to_16000(
        self,
        benchmark,
        audio_transform_service: AudioTransformService,
        benchmark_audio_data: AudioData,
    ) -> None:
        """Benchmark downsampling from 44100 Hz to 16000 Hz."""
        result = benchmark(
            audio_transform_service.resample,
            benchmark_audio_data,
            target_sample_rate=16000,
        )
        assert result.success

    def test_benchmark_resample_16000_to_44100(
        self,
        benchmark,
        audio_transform_service: AudioTransformService,
    ) -> None:
        """Benchmark upsampling from 16000 Hz to 44100 Hz."""
        # Create 16kHz audio
        sample_rate = 16000
        duration = 10.0
        t = np.linspace(0, duration, int(sample_rate * duration), dtype=np.float32)
        samples = 0.5 * np.sin(2 * np.pi * 440.0 * t)
        audio = AudioData(samples=samples, sample_rate=sample_rate, channels=1)

        result = benchmark(
            audio_transform_service.resample,
            audio,
            target_sample_rate=44100,
        )
        assert result.success


@pytest.mark.benchmark
class TestFilterBenchmarks:
    """Benchmarks for filtering operations."""

    def test_benchmark_bandpass(
        self,
        benchmark,
        audio_transform_service: AudioTransformService,
        benchmark_audio_data: AudioData,
    ) -> None:
        """Benchmark bandpass filter (500-5000 Hz)."""
        result = benchmark(
            audio_transform_service.apply_bandpass,
            benchmark_audio_data,
            low_hz=500.0,
            high_hz=5000.0,
        )
        assert result.success

    def test_benchmark_lowpass(
        self,
        benchmark,
        audio_transform_service: AudioTransformService,
        benchmark_audio_data: AudioData,
    ) -> None:
        """Benchmark lowpass filter (4000 Hz)."""
        result = benchmark(
            audio_transform_service.apply_lowpass,
            benchmark_audio_data,
            cutoff_hz=4000.0,
        )
        assert result.success

    def test_benchmark_highpass(
        self,
        benchmark,
        audio_transform_service: AudioTransformService,
        benchmark_audio_data: AudioData,
    ) -> None:
        """Benchmark highpass filter (500 Hz)."""
        result = benchmark(
            audio_transform_service.apply_highpass,
            benchmark_audio_data,
            cutoff_hz=500.0,
        )
        assert result.success


@pytest.mark.benchmark
class TestNormalizeBenchmarks:
    """Benchmarks for normalization operations."""

    def test_benchmark_normalize_peak(
        self,
        benchmark,
        audio_transform_service: AudioTransformService,
        benchmark_audio_data: AudioData,
    ) -> None:
        """Benchmark peak normalization."""
        result = benchmark(
            audio_transform_service.normalize_peak,
            benchmark_audio_data,
            target_peak=0.9,
        )
        assert result.success

    def test_benchmark_normalize_loudness(
        self,
        benchmark,
        audio_transform_service: AudioTransformService,
        benchmark_audio_data: AudioData,
    ) -> None:
        """Benchmark loudness normalization."""
        result = benchmark(
            audio_transform_service.normalize_loudness,
            benchmark_audio_data,
            target_db=-20.0,
        )
        assert result.success


@pytest.mark.benchmark
class TestDenoiseBenchmarks:
    """Benchmarks for noise reduction."""

    def test_benchmark_denoise(
        self,
        benchmark,
        audio_transform_service: AudioTransformService,
    ) -> None:
        """Benchmark spectral denoise on noisy audio."""
        # Create noisy audio
        sample_rate = 44100
        duration = 5.0  # Shorter for denoise as it's slow
        t = np.linspace(0, duration, int(sample_rate * duration), dtype=np.float32)
        signal = 0.5 * np.sin(2 * np.pi * 440.0 * t)
        noise = 0.1 * np.random.randn(len(t)).astype(np.float32)
        samples = signal + noise

        audio = AudioData(samples=samples, sample_rate=sample_rate, channels=1)

        result = benchmark(
            audio_transform_service.denoise,
            audio,
            strength=1.0,
        )
        assert result.success


@pytest.mark.benchmark
class TestChainBenchmarks:
    """Benchmarks for chained operations (typical ML preprocessing pipeline)."""

    def test_benchmark_ml_preprocessing_chain(
        self,
        benchmark,
        audio_transform_service: AudioTransformService,
        benchmark_audio_data: AudioData,
    ) -> None:
        """Benchmark typical ML preprocessing: bandpass -> resample -> normalize."""
        result = benchmark(
            audio_transform_service.chain,
            benchmark_audio_data,
            [
                ("apply_bandpass", {"low_hz": 500.0, "high_hz": 8000.0}),
                ("resample", {"target_sample_rate": 16000}),
                ("normalize_peak", {"target_peak": 0.9}),
            ],
        )
        assert result.success
