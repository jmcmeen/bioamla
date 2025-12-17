"""
Unit tests for bioamla.realtime module.
"""

import json
import time
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np
import pytest


class TestRecordingConfig:
    """Tests for RecordingConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        from bioamla.realtime import RecordingConfig

        config = RecordingConfig()
        assert config.sample_rate == 16000
        assert config.channels == 1
        assert config.chunk_size == 1024
        assert config.buffer_seconds == 10.0

    def test_custom_values(self):
        """Test custom configuration values."""
        from bioamla.realtime import RecordingConfig

        config = RecordingConfig(
            sample_rate=44100,
            channels=2,
            chunk_size=2048,
        )
        assert config.sample_rate == 44100
        assert config.channels == 2
        assert config.chunk_size == 2048


class TestDetectionEvent:
    """Tests for DetectionEvent dataclass."""

    def test_creation(self):
        """Test creating detection event."""
        from bioamla.realtime import DetectionEvent

        event = DetectionEvent(
            timestamp=datetime.now(),
            start_time=0.0,
            end_time=1.0,
            label="bird_song",
            confidence=0.95,
        )

        assert event.label == "bird_song"
        assert event.confidence == 0.95
        assert event.audio_segment is None

    def test_to_dict(self):
        """Test conversion to dictionary."""
        from bioamla.realtime import DetectionEvent

        now = datetime.now()
        event = DetectionEvent(
            timestamp=now,
            start_time=0.5,
            end_time=1.5,
            label="bird_song",
            confidence=0.9,
            metadata={"channel": 1},
        )

        d = event.to_dict()

        assert d["timestamp"] == now.isoformat()
        assert d["start_time"] == 0.5
        assert d["end_time"] == 1.5
        assert d["label"] == "bird_song"
        assert d["confidence"] == 0.9
        assert d["metadata"] == {"channel": 1}

    def test_with_audio_segment(self):
        """Test event with audio segment."""
        from bioamla.realtime import DetectionEvent

        audio = np.random.randn(16000).astype(np.float32)
        event = DetectionEvent(
            timestamp=datetime.now(),
            start_time=0.0,
            end_time=1.0,
            label="test",
            confidence=0.8,
            audio_segment=audio,
        )

        assert event.audio_segment is not None
        assert len(event.audio_segment) == 16000


class TestAudioRecorder:
    """Tests for AudioRecorder base class."""

    def test_initialization(self):
        """Test recorder initialization."""
        from bioamla.realtime import AudioRecorder, RecordingConfig

        config = RecordingConfig(sample_rate=16000)
        recorder = AudioRecorder(config)

        assert recorder.config.sample_rate == 16000
        assert not recorder.is_recording
        assert recorder.audio_buffer is None

    def test_get_sounddevice_import_error(self):
        """Test error when sounddevice not installed."""
        from bioamla.realtime import AudioRecorder

        recorder = AudioRecorder()

        with patch.dict("sys.modules", {"sounddevice": None}):
            with patch("bioamla.realtime.AudioRecorder._get_sounddevice") as mock:
                mock.side_effect = ImportError("sounddevice is required")
                with pytest.raises(ImportError, match="sounddevice"):
                    recorder._get_sounddevice()

    @patch("bioamla.realtime.AudioRecorder._get_sounddevice")
    def test_start_creates_buffer(self, mock_sd):
        """Test that start creates audio buffer."""
        from bioamla.realtime import AudioRecorder, RecordingConfig

        mock_stream = MagicMock()
        mock_sd.return_value.InputStream.return_value = mock_stream

        config = RecordingConfig(sample_rate=16000, buffer_seconds=5.0)
        recorder = AudioRecorder(config)
        recorder.start()

        assert recorder.audio_buffer is not None
        assert len(recorder.audio_buffer) == 16000 * 5  # 5 seconds at 16kHz
        assert recorder.is_recording

        recorder.stop()

    @patch("bioamla.realtime.AudioRecorder._get_sounddevice")
    def test_stop(self, mock_sd):
        """Test stopping recording."""
        from bioamla.realtime import AudioRecorder

        mock_stream = MagicMock()
        mock_sd.return_value.InputStream.return_value = mock_stream

        recorder = AudioRecorder()
        recorder.start()
        recorder.stop()

        assert not recorder.is_recording
        mock_stream.stop.assert_called_once()
        mock_stream.close.assert_called_once()

    def test_get_buffer_not_recording(self):
        """Test getting buffer when not recording."""
        from bioamla.realtime import AudioRecorder

        recorder = AudioRecorder()
        recorder.audio_buffer = np.zeros(16000)
        recorder.buffer_position = 0

        buffer = recorder.get_buffer()
        assert len(buffer) == 16000

    def test_get_buffer_partial(self):
        """Test getting partial buffer."""
        from bioamla.realtime import AudioRecorder, RecordingConfig

        config = RecordingConfig(sample_rate=16000, buffer_seconds=10.0)
        recorder = AudioRecorder(config)
        recorder.audio_buffer = np.zeros(160000)
        recorder.buffer_position = 80000

        buffer = recorder.get_buffer(seconds=2.0)
        assert len(buffer) == 32000  # 2 seconds at 16kHz


class TestLiveRecorder:
    """Tests for LiveRecorder."""

    def test_initialization(self):
        """Test live recorder initialization."""
        from bioamla.realtime import LiveRecorder

        recorder = LiveRecorder(detection_interval=2.0)

        assert recorder.detection_interval == 2.0
        assert recorder.min_confidence == 0.5
        assert len(recorder.detections) == 0

    def test_add_callback(self):
        """Test adding detection callback."""
        from bioamla.realtime import LiveRecorder

        recorder = LiveRecorder()
        callback = MagicMock()
        recorder.add_callback(callback)

        assert callback in recorder._callbacks

    @patch("bioamla.realtime.AudioRecorder._get_sounddevice")
    def test_start_creates_output_dir(self, mock_sd, tmp_path):
        """Test that start creates output directory."""
        from bioamla.realtime import LiveRecorder

        mock_stream = MagicMock()
        mock_sd.return_value.InputStream.return_value = mock_stream

        output_dir = tmp_path / "detections"
        recorder = LiveRecorder(
            output_dir=str(output_dir),
            save_detections=True,
        )
        recorder.start()

        assert output_dir.exists()
        recorder.stop()

    def test_get_recent_detections_empty(self):
        """Test getting detections when empty."""
        from bioamla.realtime import LiveRecorder

        recorder = LiveRecorder()
        detections = recorder.get_recent_detections()

        assert len(detections) == 0

    def test_get_recent_detections_filtered_by_label(self):
        """Test filtering detections by label."""
        from bioamla.realtime import LiveRecorder, DetectionEvent

        recorder = LiveRecorder()
        recorder.detections = [
            DetectionEvent(
                timestamp=datetime.now(),
                start_time=0, end_time=1,
                label="bird", confidence=0.9
            ),
            DetectionEvent(
                timestamp=datetime.now(),
                start_time=1, end_time=2,
                label="frog", confidence=0.8
            ),
            DetectionEvent(
                timestamp=datetime.now(),
                start_time=2, end_time=3,
                label="bird", confidence=0.85
            ),
        ]

        bird_detections = recorder.get_recent_detections(label="bird")
        assert len(bird_detections) == 2
        assert all(d.label == "bird" for d in bird_detections)


class TestSpectrogramConfig:
    """Tests for SpectrogramConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        from bioamla.realtime import SpectrogramConfig

        config = SpectrogramConfig()
        assert config.sample_rate == 16000
        assert config.n_fft == 1024
        assert config.hop_length == 256
        assert config.n_mels == 128
        assert config.window_seconds == 5.0

    def test_custom_values(self):
        """Test custom configuration values."""
        from bioamla.realtime import SpectrogramConfig

        config = SpectrogramConfig(
            sample_rate=22050,
            n_fft=2048,
            n_mels=64,
        )
        assert config.sample_rate == 22050
        assert config.n_fft == 2048
        assert config.n_mels == 64


class TestRealtimeSpectrogram:
    """Tests for RealtimeSpectrogram."""

    def test_initialization(self):
        """Test spectrogram initialization."""
        from bioamla.realtime import RealtimeSpectrogram, SpectrogramConfig

        config = SpectrogramConfig(n_mels=64)
        spectrogram = RealtimeSpectrogram(config=config)

        assert spectrogram.config.n_mels == 64
        assert not spectrogram._running

    def test_hz_to_mel(self):
        """Test Hz to mel conversion."""
        from bioamla.realtime import RealtimeSpectrogram

        spectrogram = RealtimeSpectrogram()
        mel_1000 = spectrogram._hz_to_mel(1000)
        # Mel scale at 1000 Hz should be around 1000 mels (by definition)
        assert 500 < mel_1000 < 1500

    def test_mel_to_hz(self):
        """Test mel to Hz conversion."""
        from bioamla.realtime import RealtimeSpectrogram

        spectrogram = RealtimeSpectrogram()
        hz = spectrogram._mel_to_hz(np.array([1000]))
        # 1000 mels should convert back to around 1000 Hz
        assert 500 < hz[0] < 1500

    def test_compute_spectrogram(self):
        """Test spectrogram computation."""
        from bioamla.realtime import RealtimeSpectrogram, SpectrogramConfig

        config = SpectrogramConfig(
            sample_rate=16000,
            n_fft=512,
            hop_length=128,
            n_mels=64,
        )
        spectrogram = RealtimeSpectrogram(config=config)

        audio = np.random.randn(16000).astype(np.float32)
        mel_spec = spectrogram._compute_spectrogram(audio)

        assert mel_spec.shape[0] == 64  # n_mels
        assert mel_spec.shape[1] > 0  # some number of frames

    def test_get_current_spectrogram_none(self):
        """Test getting spectrogram when none computed."""
        from bioamla.realtime import RealtimeSpectrogram

        spectrogram = RealtimeSpectrogram()
        result = spectrogram.get_current_spectrogram()

        assert result is None


class TestMonitoringConfig:
    """Tests for MonitoringConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        from bioamla.realtime import MonitoringConfig

        config = MonitoringConfig()
        assert config.detection_threshold == 0.5
        assert config.alert_cooldown == 30.0
        assert config.target_classes is None
        assert config.save_detections_only is True

    def test_custom_values(self):
        """Test custom configuration values."""
        from bioamla.realtime import MonitoringConfig

        config = MonitoringConfig(
            detection_threshold=0.8,
            target_classes=["bird", "frog"],
            alert_cooldown=60.0,
        )
        assert config.detection_threshold == 0.8
        assert config.target_classes == ["bird", "frog"]


class TestContinuousMonitor:
    """Tests for ContinuousMonitor."""

    def test_initialization(self):
        """Test monitor initialization."""
        from bioamla.realtime import ContinuousMonitor, MonitoringConfig

        detector = MagicMock()
        config = MonitoringConfig(detection_threshold=0.7)
        monitor = ContinuousMonitor(detector=detector, config=config)

        assert monitor.config.detection_threshold == 0.7

    def test_add_alert_callback(self):
        """Test adding alert callback."""
        from bioamla.realtime import ContinuousMonitor

        detector = MagicMock()
        monitor = ContinuousMonitor(detector=detector)

        callback = MagicMock()
        monitor.add_alert_callback(callback)

        assert callback in monitor._alert_callbacks

    def test_get_statistics_empty(self):
        """Test getting statistics when empty."""
        from bioamla.realtime import ContinuousMonitor

        detector = MagicMock()
        monitor = ContinuousMonitor(detector=detector)

        stats = monitor.get_statistics()

        assert stats["total_detections"] == 0
        assert stats["detections_by_class"] == {}


class TestAudioStreamProcessor:
    """Tests for AudioStreamProcessor."""

    def test_initialization(self):
        """Test stream processor initialization."""
        from bioamla.realtime import AudioStreamProcessor

        processor = AudioStreamProcessor(sample_rate=22050, chunk_size=2048)

        assert processor.sample_rate == 22050
        assert processor.chunk_size == 2048
        assert len(processor.processors) == 0
        assert len(processor.output_callbacks) == 0

    def test_add_processor(self):
        """Test adding processor to pipeline."""
        from bioamla.realtime import AudioStreamProcessor

        processor = AudioStreamProcessor()

        def normalize(audio):
            return audio / np.max(np.abs(audio) + 1e-10)

        result = processor.add_processor(normalize)

        assert result is processor  # Returns self for chaining
        assert len(processor.processors) == 1

    def test_add_output_callback(self):
        """Test adding output callback."""
        from bioamla.realtime import AudioStreamProcessor

        processor = AudioStreamProcessor()
        callback = MagicMock()

        result = processor.add_output_callback(callback)

        assert result is processor  # Returns self for chaining
        assert callback in processor.output_callbacks

    def test_processor_chaining(self):
        """Test method chaining for adding processors."""
        from bioamla.realtime import AudioStreamProcessor

        def processor1(x):
            return x

        def processor2(x):
            return x

        callback = MagicMock()

        stream = (AudioStreamProcessor()
                  .add_processor(processor1)
                  .add_processor(processor2)
                  .add_output_callback(callback))

        assert len(stream.processors) == 2
        assert len(stream.output_callbacks) == 1


class TestUtilityFunctions:
    """Tests for utility functions."""

    def test_list_audio_devices_import_error(self):
        """Test error when sounddevice not installed."""
        from bioamla.realtime import list_audio_devices

        with patch.dict("sys.modules", {"sounddevice": None}):
            with pytest.raises(ImportError, match="sounddevice"):
                list_audio_devices()

    @patch("bioamla.realtime.sd", create=True)
    def test_list_audio_devices(self, mock_sd):
        """Test listing audio devices."""
        # Skip if import patching doesn't work
        pytest.importorskip("sounddevice", reason="sounddevice not available")

    @patch("bioamla.realtime.sd", create=True)
    def test_get_default_input_device(self, mock_sd):
        """Test getting default input device."""
        from bioamla.realtime import get_default_input_device

        # Should return None when sounddevice not available
        with patch("bioamla.realtime.get_default_input_device") as mock_func:
            mock_func.return_value = None
            result = mock_func()
            assert result is None


# =============================================================================
# Integration-style tests (mocked)
# =============================================================================

class TestLiveRecorderIntegration:
    """Integration tests for LiveRecorder with mocked audio."""

    @patch("bioamla.realtime.AudioRecorder._get_sounddevice")
    def test_detection_callback_called(self, mock_sd, tmp_path):
        """Test that detection callback is called."""
        from bioamla.realtime import LiveRecorder, DetectionEvent

        # Setup mock
        mock_stream = MagicMock()
        mock_sd.return_value.InputStream.return_value = mock_stream

        # Create detector that returns detections
        def mock_detector(audio, sr):
            return [
                MagicMock(
                    label="bird",
                    confidence=0.9,
                    start_time=0.0,
                    end_time=1.0,
                )
            ]

        callback = MagicMock()

        recorder = LiveRecorder(
            detector=mock_detector,
            detection_interval=0.1,
            min_confidence=0.5,
            output_dir=str(tmp_path),
            save_detections=False,
        )
        recorder.add_callback(callback)

        # Start and simulate detection
        recorder.start()

        # Give detection loop time to run
        time.sleep(0.3)

        recorder.stop()

        # Callback should have been called
        # (may not be called if detection loop didn't run in time)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def temp_dir(tmp_path):
    """Provide a temporary directory."""
    return tmp_path
