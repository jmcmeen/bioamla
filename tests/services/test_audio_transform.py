"""Tests for AudioTransformService - core paths for pre-migration verification."""

import numpy as np

from bioamla.models.audio import AudioData
from bioamla.repository.local import LocalFileRepository
from bioamla.services.audio_transform import AudioTransformService


class TestAudioTransformServiceFiltering:
    """Tests for filtering operations."""

    def test_apply_bandpass(
        self, mock_repository, sample_audio_data: AudioData
    ) -> None:
        """Test bandpass filter applies without error and modifies audio."""
        service = AudioTransformService(mock_repository)

        result = service.apply_bandpass(
            sample_audio_data, low_hz=200.0, high_hz=2000.0
        )

        assert result.success, f"Bandpass filter failed: {result.error}"
        assert result.data is not None
        assert result.data.sample_rate == sample_audio_data.sample_rate
        assert result.data.is_modified is True
        assert "bandpass" in result.data.metadata.get("last_operation", "")

    def test_apply_bandpass_preserves_duration(
        self, mock_repository, sample_audio_data: AudioData
    ) -> None:
        """Test bandpass filter preserves audio duration."""
        service = AudioTransformService(mock_repository)

        result = service.apply_bandpass(
            sample_audio_data, low_hz=200.0, high_hz=2000.0
        )

        assert result.success
        assert abs(result.data.duration - sample_audio_data.duration) < 0.001

    def test_apply_lowpass(
        self, mock_repository, sample_audio_data: AudioData
    ) -> None:
        """Test lowpass filter applies without error."""
        service = AudioTransformService(mock_repository)

        result = service.apply_lowpass(sample_audio_data, cutoff_hz=1000.0)

        assert result.success, f"Lowpass filter failed: {result.error}"
        assert result.data.is_modified is True

    def test_apply_highpass(
        self, mock_repository, sample_audio_data: AudioData
    ) -> None:
        """Test highpass filter applies without error."""
        service = AudioTransformService(mock_repository)

        result = service.apply_highpass(sample_audio_data, cutoff_hz=500.0)

        assert result.success, f"Highpass filter failed: {result.error}"
        assert result.data.is_modified is True


class TestAudioTransformServiceResampling:
    """Tests for resampling operations."""

    def test_resample_downsample(
        self, mock_repository, sample_audio_data_44100: AudioData
    ) -> None:
        """Test downsampling from 44100 Hz to 16000 Hz."""
        service = AudioTransformService(mock_repository)

        result = service.resample(sample_audio_data_44100, target_sample_rate=16000)

        assert result.success, f"Resample failed: {result.error}"
        assert result.data.sample_rate == 16000
        assert result.data.is_modified is True

    def test_resample_upsample(
        self, mock_repository, sample_audio_data: AudioData
    ) -> None:
        """Test upsampling from 16000 Hz to 44100 Hz."""
        service = AudioTransformService(mock_repository)

        result = service.resample(sample_audio_data, target_sample_rate=44100)

        assert result.success, f"Resample failed: {result.error}"
        assert result.data.sample_rate == 44100
        assert result.data.is_modified is True

    def test_resample_preserves_duration(
        self, mock_repository, sample_audio_data_44100: AudioData
    ) -> None:
        """Test that resampling preserves audio duration."""
        service = AudioTransformService(mock_repository)

        result = service.resample(sample_audio_data_44100, target_sample_rate=16000)

        assert result.success
        # Allow small tolerance for rounding differences
        assert abs(result.data.duration - sample_audio_data_44100.duration) < 0.01

    def test_resample_same_rate_no_op(
        self, mock_repository, sample_audio_data: AudioData
    ) -> None:
        """Test resampling to same rate is a no-op."""
        service = AudioTransformService(mock_repository)

        result = service.resample(sample_audio_data, target_sample_rate=16000)

        assert result.success
        # Should return original data when already at target rate
        assert result.data.sample_rate == 16000


class TestAudioTransformServiceNormalization:
    """Tests for normalization operations."""

    def test_normalize_peak(
        self, mock_repository, sample_audio_data: AudioData
    ) -> None:
        """Test peak normalization."""
        service = AudioTransformService(mock_repository)

        result = service.normalize_peak(sample_audio_data, target_peak=0.9)

        assert result.success, f"Peak normalize failed: {result.error}"
        assert result.data.is_modified is True
        # Check peak is close to target
        actual_peak = np.max(np.abs(result.data.samples))
        assert abs(actual_peak - 0.9) < 0.01

    def test_normalize_loudness(
        self, mock_repository, sample_audio_data: AudioData
    ) -> None:
        """Test loudness normalization."""
        service = AudioTransformService(mock_repository)

        result = service.normalize_loudness(sample_audio_data, target_db=-20.0)

        assert result.success, f"Loudness normalize failed: {result.error}"
        assert result.data.is_modified is True


class TestAudioTransformServiceTrimming:
    """Tests for trimming operations."""

    def test_trim_start_end(
        self, mock_repository, sample_audio_data_44100: AudioData
    ) -> None:
        """Test time-based trimming."""
        service = AudioTransformService(mock_repository)

        result = service.trim(
            sample_audio_data_44100, start_time=0.2, end_time=0.8
        )

        assert result.success, f"Trim failed: {result.error}"
        assert result.data.is_modified is True
        # Check duration is approximately 0.6 seconds
        assert abs(result.data.duration - 0.6) < 0.01

    def test_trim_only_start(
        self, mock_repository, sample_audio_data: AudioData
    ) -> None:
        """Test trimming with only start time."""
        service = AudioTransformService(mock_repository)

        result = service.trim(sample_audio_data, start_time=0.3)

        assert result.success
        assert abs(result.data.duration - 0.7) < 0.01


class TestAudioTransformServiceDenoise:
    """Tests for noise reduction operations."""

    def test_denoise_basic(
        self, mock_repository, sample_audio_with_noise: AudioData
    ) -> None:
        """Test basic noise reduction."""
        service = AudioTransformService(mock_repository)

        result = service.denoise(sample_audio_with_noise, strength=1.0)

        assert result.success, f"Denoise failed: {result.error}"
        assert result.data.is_modified is True
        assert result.data.sample_rate == sample_audio_with_noise.sample_rate

    def test_denoise_strength_parameter(
        self, mock_repository, sample_audio_with_noise: AudioData
    ) -> None:
        """Test that different denoise strengths produce different results."""
        service = AudioTransformService(mock_repository)

        result_low = service.denoise(sample_audio_with_noise, strength=0.5)
        result_high = service.denoise(sample_audio_with_noise, strength=1.5)

        assert result_low.success
        assert result_high.success
        # Results should be different
        assert not np.array_equal(result_low.data.samples, result_high.data.samples)


class TestAudioTransformServiceFileOperations:
    """Tests for file-based operations using real filesystem."""

    def test_segment_file(self, test_audio_path_3s: str, tmp_path) -> None:
        """Test file segmentation creates expected segments."""
        repository = LocalFileRepository()
        service = AudioTransformService(repository)
        output_dir = str(tmp_path / "segments")

        result = service.segment_file(
            input_path=test_audio_path_3s,
            output_dir=output_dir,
            duration=1.0,
            overlap=0.0,
        )

        assert result.success, f"Segment failed: {result.error}"
        # 3 second file with 1 second segments = 3 segments
        batch_result = result.data["batch_result"]
        assert batch_result.processed == 3
        assert batch_result.failed == 0

    def test_resample_file(self, test_audio_path: str, tmp_path) -> None:
        """Test file-based resampling."""
        repository = LocalFileRepository()
        service = AudioTransformService(repository)
        output_path = str(tmp_path / "resampled.wav")

        result = service.resample_file(
            input_path=test_audio_path,
            output_path=output_path,
            target_rate=8000,
        )

        assert result.success, f"Resample file failed: {result.error}"
        assert result.data.sample_rate == 8000

    def test_filter_file_bandpass(self, test_audio_path: str, tmp_path) -> None:
        """Test file-based bandpass filtering."""
        repository = LocalFileRepository()
        service = AudioTransformService(repository)
        output_path = str(tmp_path / "filtered.wav")

        result = service.filter_file(
            input_path=test_audio_path,
            output_path=output_path,
            bandpass=(200.0, 2000.0),
        )

        assert result.success, f"Filter file failed: {result.error}"


class TestAudioTransformServiceChaining:
    """Tests for operation chaining."""

    def test_chain_multiple_operations(
        self, mock_repository, sample_audio_data_44100: AudioData
    ) -> None:
        """Test chaining multiple transform operations."""
        service = AudioTransformService(mock_repository)

        result = service.chain(
            sample_audio_data_44100,
            [
                ("apply_bandpass", {"low_hz": 500, "high_hz": 4000}),
                ("resample", {"target_sample_rate": 16000}),
                ("normalize_peak", {"target_peak": 0.9}),
            ],
        )

        assert result.success, f"Chain failed: {result.error}"
        assert result.data.sample_rate == 16000
        assert result.data.is_modified is True


class TestAudioTransformServiceChannels:
    """Tests for channel operations."""

    def test_to_mono(
        self, mock_repository, sample_audio_stereo: AudioData
    ) -> None:
        """Test stereo to mono conversion."""
        service = AudioTransformService(mock_repository)

        result = service.to_mono(sample_audio_stereo)

        assert result.success, f"To mono failed: {result.error}"
        assert result.data.channels == 1
        assert result.data.samples.ndim == 1
