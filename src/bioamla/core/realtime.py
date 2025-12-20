"""
Real-Time Processing Module
===========================

This module provides real-time audio processing capabilities:
- Live audio recording with detection
- Real-time spectrogram streaming
- Continuous monitoring and alerting

Example:
    >>> from bioamla.realtime import LiveRecorder, RealtimeSpectrogram
    >>> recorder = LiveRecorder(detector=my_detector)
    >>> recorder.start()
    >>> # Processing happens in background
    >>> recorder.stop()
    >>>
    >>> stream = RealtimeSpectrogram(callback=display_fn)
    >>> stream.start()
"""

import json
import logging
import queue
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from bioamla.core.files import TextFile

import numpy as np

logger = logging.getLogger(__name__)

__all__ = [
    # Configuration
    "RecordingConfig",
    "DetectionEvent",
    "SpectrogramConfig",
    "MonitoringConfig",
    # Core classes (thread-safe)
    "AudioRecorder",
    "LiveRecorder",
    "RealtimeSpectrogram",
    "ContinuousMonitor",
    "AudioStreamProcessor",
    # Utility functions
    "list_audio_devices",
    "get_default_input_device",
    "test_recording",
]


# =============================================================================
# Audio Recording
# =============================================================================

@dataclass
class RecordingConfig:
    """Configuration for live recording."""

    sample_rate: int = 16000
    channels: int = 1
    chunk_size: int = 1024
    buffer_seconds: float = 10.0
    device_index: Optional[int] = None
    format: str = "float32"


@dataclass
class DetectionEvent:
    """Represents a detection event during live recording."""

    timestamp: datetime
    start_time: float
    end_time: float
    label: str
    confidence: float
    audio_segment: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "start_time": self.start_time,
            "end_time": self.end_time,
            "label": self.label,
            "confidence": self.confidence,
            "metadata": self.metadata,
        }


class AudioRecorder:
    """
    Base audio recorder using sounddevice.

    Handles audio capture from microphone or audio interface.

    Thread Safety:
        This class is thread-safe. Audio capture runs in a separate callback
        thread managed by sounddevice. The internal circular buffer is protected
        by ``threading.Lock`` (``buffer_lock``), ensuring safe concurrent access
        to ``get_buffer()`` while recording is active.
    """

    def __init__(self, config: Optional[RecordingConfig] = None):
        """
        Initialize audio recorder.

        Args:
            config: Recording configuration
        """
        self.config = config or RecordingConfig()
        self.is_recording = False
        self.audio_buffer = None
        self.buffer_lock = threading.Lock()
        self._stream = None

    def _get_sounddevice(self):
        """Import and return sounddevice module."""
        try:
            import sounddevice as sd
            return sd
        except ImportError as err:
            raise ImportError(
                "sounddevice is required for recording. "
                "Install with: pip install sounddevice"
            ) from err

    def start(self) -> None:
        """Start recording."""
        sd = self._get_sounddevice()

        buffer_samples = int(self.config.buffer_seconds * self.config.sample_rate)
        self.audio_buffer = np.zeros(buffer_samples, dtype=np.float32)
        self.buffer_position = 0

        def callback(indata, frames, time_info, status):
            if status:
                logger.warning(f"Recording status: {status}")

            audio = indata[:, 0] if indata.ndim > 1 else indata.flatten()

            with self.buffer_lock:
                # Circular buffer
                end_pos = self.buffer_position + len(audio)
                if end_pos <= len(self.audio_buffer):
                    self.audio_buffer[self.buffer_position:end_pos] = audio
                else:
                    first_part = len(self.audio_buffer) - self.buffer_position
                    self.audio_buffer[self.buffer_position:] = audio[:first_part]
                    self.audio_buffer[:len(audio) - first_part] = audio[first_part:]
                self.buffer_position = end_pos % len(self.audio_buffer)

        self._stream = sd.InputStream(
            samplerate=self.config.sample_rate,
            channels=self.config.channels,
            blocksize=self.config.chunk_size,
            device=self.config.device_index,
            callback=callback,
        )
        self._stream.start()
        self.is_recording = True
        logger.info("Recording started")

    def stop(self) -> None:
        """Stop recording."""
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        self.is_recording = False
        logger.info("Recording stopped")

    def get_buffer(self, seconds: Optional[float] = None) -> np.ndarray:
        """
        Get audio from buffer.

        Args:
            seconds: Number of seconds to get (all if None)

        Returns:
            Audio array
        """
        with self.buffer_lock:
            if seconds is None:
                return self.audio_buffer.copy()

            samples = int(seconds * self.config.sample_rate)
            samples = min(samples, len(self.audio_buffer))

            # Get most recent samples
            end = self.buffer_position
            start = (end - samples) % len(self.audio_buffer)

            if start < end:
                return self.audio_buffer[start:end].copy()
            else:
                return np.concatenate([
                    self.audio_buffer[start:],
                    self.audio_buffer[:end]
                ])


class LiveRecorder:
    """
    Live audio recorder with real-time detection.

    Continuously records audio and runs detection in background.

    Thread Safety:
        This class manages multiple threads internally: an audio recording
        thread (via AudioRecorder) and a detection processing thread. Detection
        events are communicated via a thread-safe ``queue.Queue``. The class is
        safe to use from the main thread while background processing occurs.
    """

    def __init__(
        self,
        detector: Optional[Callable] = None,
        config: Optional[RecordingConfig] = None,
        detection_interval: float = 1.0,
        min_confidence: float = 0.5,
        save_detections: bool = True,
        output_dir: str = "./detections",
    ):
        """
        Initialize live recorder.

        Args:
            detector: Detection function (audio -> detections)
            config: Recording configuration
            detection_interval: Seconds between detection runs
            min_confidence: Minimum confidence threshold
            save_detections: Whether to save detected audio segments
            output_dir: Output directory for saved detections
        """
        self.detector = detector
        self.config = config or RecordingConfig()
        self.detection_interval = detection_interval
        self.min_confidence = min_confidence
        self.save_detections = save_detections
        self.output_dir = Path(output_dir)

        self.recorder = AudioRecorder(self.config)
        self.detections: List[DetectionEvent] = []
        self.detection_queue: queue.Queue = queue.Queue()

        self._detection_thread = None
        self._running = False
        self._callbacks: List[Callable[[DetectionEvent], None]] = []

    def add_callback(self, callback: Callable[[DetectionEvent], None]) -> None:
        """Add callback for detection events."""
        self._callbacks.append(callback)

    def start(self) -> None:
        """Start recording and detection."""
        self._running = True
        self.recorder.start()

        if self.detector is not None:
            self._detection_thread = threading.Thread(
                target=self._detection_loop,
                daemon=True
            )
            self._detection_thread.start()

        if self.save_detections:
            self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Live recorder started")

    def stop(self) -> None:
        """Stop recording and detection."""
        self._running = False
        self.recorder.stop()

        if self._detection_thread is not None:
            self._detection_thread.join(timeout=5.0)

        logger.info(f"Live recorder stopped. Total detections: {len(self.detections)}")

    def _detection_loop(self) -> None:
        """Background detection loop."""
        last_detection_time = 0.0

        while self._running:
            current_time = time.time()

            if current_time - last_detection_time >= self.detection_interval:
                try:
                    audio = self.recorder.get_buffer(self.detection_interval * 2)
                    results = self.detector(audio, self.config.sample_rate)

                    for result in results:
                        if hasattr(result, "confidence") and result.confidence < self.min_confidence:
                            continue

                        event = DetectionEvent(
                            timestamp=datetime.now(),
                            start_time=getattr(result, "start_time", 0.0),
                            end_time=getattr(result, "end_time", 0.0),
                            label=getattr(result, "label", str(result)),
                            confidence=getattr(result, "confidence", 1.0),
                            audio_segment=audio if self.save_detections else None,
                        )

                        self.detections.append(event)
                        self.detection_queue.put(event)

                        for callback in self._callbacks:
                            try:
                                callback(event)
                            except Exception as e:
                                logger.error(f"Callback error: {e}")

                        if self.save_detections:
                            self._save_detection(event)

                except Exception as e:
                    logger.error(f"Detection error: {e}")

                last_detection_time = current_time

            time.sleep(0.1)

    def _save_detection(self, event: DetectionEvent) -> None:
        """Save detection to file."""
        try:
            import soundfile as sf

            timestamp_str = event.timestamp.strftime("%Y%m%d_%H%M%S_%f")
            filename = f"{event.label}_{timestamp_str}.wav"
            filepath = self.output_dir / filename

            if event.audio_segment is not None:
                sf.write(filepath, event.audio_segment, self.config.sample_rate)

            # Save metadata
            meta_path = filepath.with_suffix(".json")
            with TextFile(meta_path, mode="w", encoding="utf-8") as f:
                json.dump(event.to_dict(), f.handle, indent=2)

        except Exception as e:
            logger.error(f"Error saving detection: {e}")

    def get_recent_detections(
        self,
        seconds: Optional[float] = None,
        label: Optional[str] = None
    ) -> List[DetectionEvent]:
        """
        Get recent detections.

        Args:
            seconds: Time window (all if None)
            label: Filter by label

        Returns:
            List of detection events
        """
        cutoff = None
        if seconds is not None:
            cutoff = datetime.now().timestamp() - seconds

        filtered = []
        for det in self.detections:
            if cutoff is not None and det.timestamp.timestamp() < cutoff:
                continue
            if label is not None and det.label != label:
                continue
            filtered.append(det)

        return filtered


# =============================================================================
# Real-Time Spectrogram
# =============================================================================

@dataclass
class SpectrogramConfig:
    """Configuration for real-time spectrogram."""

    sample_rate: int = 16000
    n_fft: int = 1024
    hop_length: int = 256
    n_mels: int = 128
    fmin: float = 0.0
    fmax: Optional[float] = None
    window_seconds: float = 5.0
    update_interval: float = 0.1


class RealtimeSpectrogram:
    """
    Real-time spectrogram streaming.

    Continuously computes and streams spectrogram data.

    Thread Safety:
        This class runs spectrogram computation in a background thread while
        audio recording occurs in a separate callback thread. The callback
        function is invoked from the computation thread. Safe for GUI integration
        where callbacks update display from a worker thread.
    """

    def __init__(
        self,
        config: Optional[SpectrogramConfig] = None,
        callback: Optional[Callable[[np.ndarray, float], None]] = None,
        recording_config: Optional[RecordingConfig] = None,
    ):
        """
        Initialize real-time spectrogram.

        Args:
            config: Spectrogram configuration
            callback: Function called with (spectrogram, timestamp)
            recording_config: Audio recording configuration
        """
        self.config = config or SpectrogramConfig()
        self.callback = callback

        if recording_config is None:
            recording_config = RecordingConfig(
                sample_rate=self.config.sample_rate
            )
        self.recorder = AudioRecorder(recording_config)

        self._running = False
        self._thread = None
        self._spectrogram_buffer = None
        self._mel_filterbank = None

    def _create_mel_filterbank(self) -> np.ndarray:
        """Create mel filterbank matrix."""
        try:
            import librosa
            return librosa.filters.mel(
                sr=self.config.sample_rate,
                n_fft=self.config.n_fft,
                n_mels=self.config.n_mels,
                fmin=self.config.fmin,
                fmax=self.config.fmax or self.config.sample_rate // 2,
            )
        except ImportError:
            # Simple mel filterbank approximation
            n_freq = self.config.n_fft // 2 + 1
            mel_points = np.linspace(
                self._hz_to_mel(self.config.fmin),
                self._hz_to_mel(self.config.fmax or self.config.sample_rate // 2),
                self.config.n_mels + 2
            )
            hz_points = self._mel_to_hz(mel_points)
            bin_points = np.floor((self.config.n_fft + 1) * hz_points / self.config.sample_rate).astype(int)

            filterbank = np.zeros((self.config.n_mels, n_freq))
            for i in range(self.config.n_mels):
                left = bin_points[i]
                center = bin_points[i + 1]
                right = bin_points[i + 2]

                for j in range(left, center):
                    if center > left:
                        filterbank[i, j] = (j - left) / (center - left)
                for j in range(center, right):
                    if right > center:
                        filterbank[i, j] = (right - j) / (right - center)

            return filterbank

    @staticmethod
    def _hz_to_mel(hz: float) -> float:
        return 2595 * np.log10(1 + hz / 700)

    @staticmethod
    def _mel_to_hz(mel: np.ndarray) -> np.ndarray:
        return 700 * (10 ** (mel / 2595) - 1)

    def _compute_spectrogram(self, audio: np.ndarray) -> np.ndarray:
        """Compute mel spectrogram from audio."""
        # Apply window
        n_frames = (len(audio) - self.config.n_fft) // self.config.hop_length + 1
        frames = np.zeros((n_frames, self.config.n_fft))

        for i in range(n_frames):
            start = i * self.config.hop_length
            frame = audio[start:start + self.config.n_fft]
            window = np.hanning(len(frame))
            frames[i] = frame * window

        # FFT
        spectrogram = np.abs(np.fft.rfft(frames, axis=1)) ** 2

        # Apply mel filterbank
        if self._mel_filterbank is None:
            self._mel_filterbank = self._create_mel_filterbank()

        mel_spec = np.dot(spectrogram, self._mel_filterbank.T)

        # Log scale
        mel_spec = 10 * np.log10(mel_spec + 1e-10)

        return mel_spec.T  # (n_mels, n_frames)

    def start(self) -> None:
        """Start spectrogram streaming."""
        self._running = True
        self.recorder.start()

        self._thread = threading.Thread(
            target=self._stream_loop,
            daemon=True
        )
        self._thread.start()

        logger.info("Real-time spectrogram started")

    def stop(self) -> None:
        """Stop spectrogram streaming."""
        self._running = False
        self.recorder.stop()

        if self._thread is not None:
            self._thread.join(timeout=5.0)

        logger.info("Real-time spectrogram stopped")

    def _stream_loop(self) -> None:
        """Background streaming loop."""
        while self._running:
            try:
                audio = self.recorder.get_buffer(self.config.window_seconds)

                if len(audio) > self.config.n_fft:
                    spectrogram = self._compute_spectrogram(audio)
                    timestamp = time.time()

                    self._spectrogram_buffer = spectrogram

                    if self.callback is not None:
                        self.callback(spectrogram, timestamp)

            except Exception as e:
                logger.error(f"Spectrogram error: {e}")

            time.sleep(self.config.update_interval)

    def get_current_spectrogram(self) -> Optional[np.ndarray]:
        """Get the current spectrogram buffer."""
        return self._spectrogram_buffer


# =============================================================================
# Continuous Monitoring
# =============================================================================

@dataclass
class MonitoringConfig:
    """Configuration for continuous monitoring."""

    detection_threshold: float = 0.5
    alert_cooldown: float = 30.0  # Seconds between alerts for same class
    target_classes: Optional[List[str]] = None
    exclude_classes: Optional[List[str]] = None
    save_all: bool = False
    save_detections_only: bool = True


class ContinuousMonitor:
    """
    Continuous monitoring with alerting.

    Monitors audio stream and triggers alerts based on detections.

    Thread Safety:
        This class coordinates multiple background threads for audio capture
        and detection processing. Alert callbacks and event handlers are invoked
        from the monitoring thread. Safe to start/stop from the main thread.
    """

    def __init__(
        self,
        detector: Callable,
        config: Optional[MonitoringConfig] = None,
        recording_config: Optional[RecordingConfig] = None,
    ):
        """
        Initialize continuous monitor.

        Args:
            detector: Detection function
            config: Monitoring configuration
            recording_config: Recording configuration
        """
        self.detector = detector
        self.config = config or MonitoringConfig()

        self.live_recorder = LiveRecorder(
            detector=self._wrapped_detector,
            config=recording_config,
            min_confidence=self.config.detection_threshold,
            save_detections=self.config.save_detections_only,
        )

        self._alert_callbacks: List[Callable[[DetectionEvent], None]] = []
        self._last_alert_times: Dict[str, float] = {}
        self._statistics: Dict[str, int] = {}

    def _wrapped_detector(
        self,
        audio: np.ndarray,
        sample_rate: int
    ) -> List[Any]:
        """Wrapper to filter detections."""
        results = self.detector(audio, sample_rate)
        filtered = []

        for result in results:
            label = getattr(result, "label", str(result))

            # Filter by target classes
            if self.config.target_classes is not None:
                if label not in self.config.target_classes:
                    continue

            # Filter by excluded classes
            if self.config.exclude_classes is not None:
                if label in self.config.exclude_classes:
                    continue

            filtered.append(result)

            # Update statistics
            self._statistics[label] = self._statistics.get(label, 0) + 1

            # Check alert cooldown
            current_time = time.time()
            last_alert = self._last_alert_times.get(label, 0)

            if current_time - last_alert >= self.config.alert_cooldown:
                self._last_alert_times[label] = current_time
                self._trigger_alert(result)

        return filtered

    def _trigger_alert(self, detection) -> None:
        """Trigger alert callbacks."""
        event = DetectionEvent(
            timestamp=datetime.now(),
            start_time=getattr(detection, "start_time", 0.0),
            end_time=getattr(detection, "end_time", 0.0),
            label=getattr(detection, "label", str(detection)),
            confidence=getattr(detection, "confidence", 1.0),
        )

        for callback in self._alert_callbacks:
            try:
                callback(event)
            except Exception as e:
                logger.error(f"Alert callback error: {e}")

    def add_alert_callback(
        self,
        callback: Callable[[DetectionEvent], None]
    ) -> None:
        """Add callback for alerts."""
        self._alert_callbacks.append(callback)

    def start(self) -> None:
        """Start monitoring."""
        self.live_recorder.start()
        logger.info("Continuous monitoring started")

    def stop(self) -> None:
        """Stop monitoring."""
        self.live_recorder.stop()
        logger.info("Continuous monitoring stopped")

    def get_statistics(self) -> Dict[str, Any]:
        """Get monitoring statistics."""
        return {
            "total_detections": sum(self._statistics.values()),
            "detections_by_class": self._statistics.copy(),
            "is_running": self.live_recorder._running,
        }


# =============================================================================
# Audio Stream Processing
# =============================================================================

class AudioStreamProcessor:
    """
    Process audio streams with custom processing pipeline.

    Allows chaining multiple processing steps.

    Thread Safety:
        This class processes audio in a background thread. Processor functions
        and output callbacks are invoked sequentially from this processing thread.
        Adding processors or callbacks while running is not recommended.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        chunk_size: int = 1024,
    ):
        """
        Initialize stream processor.

        Args:
            sample_rate: Sample rate
            chunk_size: Processing chunk size
        """
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.processors: List[Callable[[np.ndarray], np.ndarray]] = []
        self.output_callbacks: List[Callable[[np.ndarray], None]] = []

        self.recorder = AudioRecorder(RecordingConfig(
            sample_rate=sample_rate,
            chunk_size=chunk_size,
        ))

        self._running = False
        self._thread = None

    def add_processor(
        self,
        processor: Callable[[np.ndarray], np.ndarray]
    ) -> "AudioStreamProcessor":
        """
        Add processing step to pipeline.

        Args:
            processor: Function that processes audio chunk

        Returns:
            self for chaining
        """
        self.processors.append(processor)
        return self

    def add_output_callback(
        self,
        callback: Callable[[np.ndarray], None]
    ) -> "AudioStreamProcessor":
        """
        Add output callback.

        Args:
            callback: Function called with processed audio

        Returns:
            self for chaining
        """
        self.output_callbacks.append(callback)
        return self

    def start(self) -> None:
        """Start stream processing."""
        self._running = True
        self.recorder.start()

        self._thread = threading.Thread(
            target=self._process_loop,
            daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        """Stop stream processing."""
        self._running = False
        self.recorder.stop()

        if self._thread is not None:
            self._thread.join(timeout=5.0)

    def _process_loop(self) -> None:
        """Background processing loop."""
        last_position = 0

        while self._running:
            try:
                # Get new audio
                with self.recorder.buffer_lock:
                    current_position = self.recorder.buffer_position
                    if current_position != last_position:
                        if current_position > last_position:
                            audio = self.recorder.audio_buffer[last_position:current_position].copy()
                        else:
                            audio = np.concatenate([
                                self.recorder.audio_buffer[last_position:],
                                self.recorder.audio_buffer[:current_position]
                            ])
                        last_position = current_position
                    else:
                        audio = None

                if audio is not None and len(audio) > 0:
                    # Apply processors
                    processed = audio
                    for processor in self.processors:
                        processed = processor(processed)

                    # Call output callbacks
                    for callback in self.output_callbacks:
                        callback(processed)

            except Exception as e:
                logger.error(f"Processing error: {e}")

            time.sleep(0.01)


# =============================================================================
# Utility Functions
# =============================================================================

def list_audio_devices() -> List[Dict[str, Any]]:
    """
    List available audio input devices.

    Returns:
        List of device info dictionaries
    """
    try:
        import sounddevice as sd
    except ImportError as err:
        raise ImportError(
            "sounddevice is required. Install with: pip install sounddevice"
        ) from err

    devices = []
    for i, device in enumerate(sd.query_devices()):
        if device["max_input_channels"] > 0:
            devices.append({
                "index": i,
                "name": device["name"],
                "channels": device["max_input_channels"],
                "sample_rate": device["default_samplerate"],
            })

    return devices


def get_default_input_device() -> Optional[int]:
    """Get default input device index."""
    try:
        import sounddevice as sd
        return sd.default.device[0]
    except Exception:
        return None


def test_recording(duration: float = 2.0, device: Optional[int] = None) -> np.ndarray:
    """
    Test audio recording.

    Args:
        duration: Recording duration in seconds
        device: Device index

    Returns:
        Recorded audio array
    """
    try:
        import sounddevice as sd
    except ImportError as err:
        raise ImportError(
            "sounddevice is required. Install with: pip install sounddevice"
        ) from err

    sample_rate = 16000
    audio = sd.rec(
        int(duration * sample_rate),
        samplerate=sample_rate,
        channels=1,
        device=device,
        dtype=np.float32,
    )
    sd.wait()

    return audio.flatten()
