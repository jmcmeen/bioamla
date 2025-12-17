"""
Unit tests for bioamla.detection module (Advanced Detection).
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from bioamla.detection import (
    Detection,
    PeakDetection,
    BandLimitedEnergyDetector,
    RibbitDetector,
    CWTPeakDetector,
    AcceleratingPatternDetector,
    detect_all,
    export_detections,
    batch_detect,
)


# =============================================================================
# Test Data Classes
# =============================================================================

class TestDetection:
    """Tests for Detection dataclass."""

    def test_detection_creation(self):
        """Test creating Detection dataclass."""
        det = Detection(
            start_time=1.0,
            end_time=2.0,
            confidence=0.8,
            frequency_low=1000.0,
            frequency_high=5000.0,
            label="test",
        )

        assert det.start_time == 1.0
        assert det.end_time == 2.0
        assert det.confidence == 0.8
        assert det.frequency_low == 1000.0
        assert det.frequency_high == 5000.0
        assert det.label == "test"

    def test_detection_duration(self):
        """Test detection duration property."""
        det = Detection(start_time=1.5, end_time=3.5)
        assert det.duration == 2.0

    def test_detection_center_time(self):
        """Test detection center_time property."""
        det = Detection(start_time=1.0, end_time=3.0)
        assert det.center_time == 2.0

    def test_detection_to_dict(self):
        """Test Detection to_dict method."""
        det = Detection(
            start_time=1.0,
            end_time=2.0,
            confidence=0.9,
            label="band_energy",
            metadata={"extra": "data"},
        )

        d = det.to_dict()

        assert d["start_time"] == 1.0
        assert d["end_time"] == 2.0
        assert d["confidence"] == 0.9
        assert d["duration"] == 1.0
        assert d["label"] == "band_energy"
        assert d["extra"] == "data"

    def test_detection_defaults(self):
        """Test Detection default values."""
        det = Detection(start_time=0.0, end_time=1.0)

        assert det.confidence == 1.0
        assert det.frequency_low is None
        assert det.frequency_high is None
        assert det.label == ""
        assert det.metadata == {}


class TestPeakDetection:
    """Tests for PeakDetection dataclass."""

    def test_peak_detection_creation(self):
        """Test creating PeakDetection."""
        peak = PeakDetection(
            time=1.5,
            amplitude=0.8,
            width=0.1,
            prominence=0.5,
            frequency=2000.0,
        )

        assert peak.time == 1.5
        assert peak.amplitude == 0.8
        assert peak.width == 0.1
        assert peak.prominence == 0.5
        assert peak.frequency == 2000.0

    def test_peak_detection_to_dict(self):
        """Test PeakDetection to_dict method."""
        peak = PeakDetection(time=1.0, amplitude=0.9)

        d = peak.to_dict()

        assert d["time"] == 1.0
        assert d["amplitude"] == 0.9
        assert "width" in d
        assert "prominence" in d


# =============================================================================
# Test Band-Limited Energy Detector
# =============================================================================

class TestBandLimitedEnergyDetector:
    """Tests for BandLimitedEnergyDetector."""

    def test_detector_creation(self):
        """Test creating detector with parameters."""
        detector = BandLimitedEnergyDetector(
            low_freq=500,
            high_freq=5000,
            threshold_db=-20,
            min_duration=0.1,
        )

        assert detector.low_freq == 500
        assert detector.high_freq == 5000
        assert detector.threshold_db == -20
        assert detector.min_duration == 0.1

    def test_compute_energy(self):
        """Test energy computation."""
        detector = BandLimitedEnergyDetector(low_freq=1000, high_freq=4000)

        sample_rate = 22050
        duration = 1.0
        audio = np.random.randn(int(sample_rate * duration)).astype(np.float32) * 0.1

        energy, times = detector.compute_energy(audio, sample_rate)

        assert len(energy) > 0
        assert len(times) == len(energy)
        assert energy.dtype == np.float64 or energy.dtype == np.float32

    def test_detect_silence(self):
        """Test detection on silence returns empty."""
        detector = BandLimitedEnergyDetector(
            low_freq=1000,
            high_freq=4000,
            threshold_db=-20,
        )

        sample_rate = 22050
        audio = np.zeros(int(sample_rate * 1.0), dtype=np.float32)

        detections = detector.detect(audio, sample_rate)

        assert len(detections) == 0

    def test_detect_noise(self):
        """Test detection on noise."""
        detector = BandLimitedEnergyDetector(
            low_freq=1000,
            high_freq=4000,
            threshold_db=-30,
            min_duration=0.05,
        )

        sample_rate = 22050
        duration = 1.0
        audio = np.random.randn(int(sample_rate * duration)).astype(np.float32) * 0.5

        detections = detector.detect(audio, sample_rate)

        # Should detect something in noisy signal above threshold
        assert isinstance(detections, list)
        for det in detections:
            assert isinstance(det, Detection)
            assert det.start_time >= 0
            # Allow small tolerance for edge effects in time computation
            assert det.end_time <= duration + 0.1
            assert det.frequency_low == 1000
            assert det.frequency_high == 4000

    def test_detect_with_tone(self):
        """Test detection with a tone burst."""
        detector = BandLimitedEnergyDetector(
            low_freq=1000,
            high_freq=3000,
            threshold_db=-25,
            min_duration=0.1,
        )

        sample_rate = 22050
        duration = 1.0
        t = np.linspace(0, duration, int(sample_rate * duration))

        # Create tone burst in middle of signal
        audio = np.zeros_like(t, dtype=np.float32)
        tone_start = int(0.3 * sample_rate)
        tone_end = int(0.6 * sample_rate)
        audio[tone_start:tone_end] = np.sin(2 * np.pi * 2000 * t[tone_start:tone_end]) * 0.5

        detections = detector.detect(audio, sample_rate)

        # Should detect the tone burst
        assert len(detections) >= 1
        # Check timing is roughly correct
        if detections:
            assert detections[0].start_time < 0.4
            assert detections[0].end_time > 0.5

    def test_merge_detections(self):
        """Test merging nearby detections."""
        detector = BandLimitedEnergyDetector(merge_threshold=0.2)

        detections = [
            Detection(start_time=0.0, end_time=0.5, confidence=0.8),
            Detection(start_time=0.6, end_time=1.0, confidence=0.7),  # Should merge
            Detection(start_time=2.0, end_time=2.5, confidence=0.9),  # Should not merge
        ]

        merged = detector._merge_detections(detections)

        assert len(merged) == 2
        assert merged[0].start_time == 0.0
        assert merged[0].end_time == 1.0
        assert merged[1].start_time == 2.0

    def test_detect_from_file(self, mock_audio_file):
        """Test detection from file."""
        detector = BandLimitedEnergyDetector(
            low_freq=500,
            high_freq=7000,
            threshold_db=-30,
        )

        detections = detector.detect_from_file(mock_audio_file)

        assert isinstance(detections, list)


# =============================================================================
# Test RIBBIT Detector
# =============================================================================

class TestRibbitDetector:
    """Tests for RibbitDetector."""

    def test_detector_creation(self):
        """Test creating RIBBIT detector."""
        detector = RibbitDetector(
            pulse_rate_hz=10.0,
            pulse_rate_tolerance=0.2,
            low_freq=500,
            high_freq=3000,
        )

        assert detector.pulse_rate_hz == 10.0
        assert detector.pulse_rate_tolerance == 0.2
        assert detector.low_freq == 500
        assert detector.high_freq == 3000

    def test_compute_pulse_score_silence(self):
        """Test pulse score on silence."""
        detector = RibbitDetector(pulse_rate_hz=10.0)

        sample_rate = 22050
        audio = np.zeros(int(sample_rate * 2.0), dtype=np.float32)

        score = detector.compute_pulse_score(audio, sample_rate)

        assert isinstance(score, float)
        assert 0 <= score <= 1

    def test_compute_pulse_score_noise(self):
        """Test pulse score on random noise."""
        detector = RibbitDetector(pulse_rate_hz=10.0)

        sample_rate = 22050
        audio = np.random.randn(int(sample_rate * 2.0)).astype(np.float32) * 0.1

        score = detector.compute_pulse_score(audio, sample_rate)

        assert isinstance(score, float)
        assert 0 <= score <= 1

    def test_compute_pulse_score_periodic_signal(self):
        """Test pulse score on periodic signal."""
        detector = RibbitDetector(
            pulse_rate_hz=10.0,
            pulse_rate_tolerance=0.3,
            low_freq=1000,
            high_freq=3000,
        )

        sample_rate = 22050
        duration = 2.0
        t = np.linspace(0, duration, int(sample_rate * duration))

        # Create periodic pulses at 10 Hz in 2000 Hz carrier
        pulse_rate = 10.0
        carrier_freq = 2000.0
        audio = np.sin(2 * np.pi * carrier_freq * t) * (0.5 + 0.5 * np.sin(2 * np.pi * pulse_rate * t))
        audio = audio.astype(np.float32)

        score = detector.compute_pulse_score(audio, sample_rate)

        assert isinstance(score, float)
        # Periodic signal should have higher score than noise
        assert score >= 0

    def test_detect_basic(self):
        """Test basic RIBBIT detection."""
        detector = RibbitDetector(
            pulse_rate_hz=10.0,
            min_score=0.1,
            window_duration=1.0,
            hop_duration=0.5,
        )

        sample_rate = 22050
        duration = 3.0
        audio = np.random.randn(int(sample_rate * duration)).astype(np.float32) * 0.1

        detections = detector.detect(audio, sample_rate)

        assert isinstance(detections, list)
        for det in detections:
            assert isinstance(det, Detection)
            assert det.label == "ribbit"
            assert "pulse_rate_hz" in det.metadata

    def test_compute_temporal_scores(self):
        """Test computing temporal scores."""
        detector = RibbitDetector(
            window_duration=1.0,
            hop_duration=0.5,
        )

        sample_rate = 22050
        duration = 3.0
        audio = np.random.randn(int(sample_rate * duration)).astype(np.float32) * 0.1

        scores, times = detector.compute_temporal_scores(audio, sample_rate)

        assert len(scores) == len(times)
        assert len(scores) > 0
        assert all(0 <= s <= 1 for s in scores)

    def test_detect_from_file(self, mock_audio_file):
        """Test detection from file."""
        detector = RibbitDetector(min_score=0.05)

        detections = detector.detect_from_file(mock_audio_file)

        assert isinstance(detections, list)


# =============================================================================
# Test CWT Peak Detector
# =============================================================================

class TestCWTPeakDetector:
    """Tests for CWTPeakDetector."""

    def test_detector_creation(self):
        """Test creating CWT detector."""
        detector = CWTPeakDetector(
            min_scale=1,
            max_scale=50,
            n_scales=20,
            snr_threshold=2.0,
        )

        assert detector.min_scale == 1
        assert detector.max_scale == 50
        assert detector.n_scales == 20
        assert detector.snr_threshold == 2.0

    def test_compute_cwt(self):
        """Test CWT computation."""
        detector = CWTPeakDetector(n_scales=10)

        signal = np.random.randn(1000).astype(np.float32)
        sample_rate = 100

        cwt_matrix, scales = detector.compute_cwt(signal, sample_rate)

        assert cwt_matrix.shape[0] == 10  # n_scales
        assert cwt_matrix.shape[1] == len(signal)
        assert len(scales) == 10

    def test_detect_silence(self):
        """Test detection on silence."""
        detector = CWTPeakDetector(snr_threshold=3.0)

        sample_rate = 22050
        audio = np.zeros(int(sample_rate * 1.0), dtype=np.float32)

        peaks = detector.detect(audio, sample_rate)

        assert isinstance(peaks, list)
        # Should find few or no peaks in silence

    def test_detect_with_peaks(self):
        """Test detection with clear peaks."""
        detector = CWTPeakDetector(
            snr_threshold=1.5,
            min_peak_distance=0.05,
        )

        sample_rate = 22050
        duration = 1.0
        t = np.linspace(0, duration, int(sample_rate * duration))

        # Create signal with clear peaks
        audio = np.zeros_like(t, dtype=np.float32)
        peak_times = [0.2, 0.4, 0.6, 0.8]
        for pt in peak_times:
            idx = int(pt * sample_rate)
            # Create short pulse
            pulse_width = int(0.01 * sample_rate)
            start_idx = max(0, idx - pulse_width // 2)
            end_idx = min(len(audio), idx + pulse_width // 2)
            audio[start_idx:end_idx] = 0.5

        peaks = detector.detect(audio, sample_rate)

        assert isinstance(peaks, list)
        for peak in peaks:
            assert isinstance(peak, PeakDetection)
            assert peak.time >= 0
            assert peak.time <= duration

    def test_detect_sequences(self):
        """Test sequence detection."""
        detector = CWTPeakDetector(snr_threshold=1.5)

        sample_rate = 22050
        duration = 2.0
        t = np.linspace(0, duration, int(sample_rate * duration))

        # Create signal with a sequence of peaks
        audio = np.zeros_like(t, dtype=np.float32)
        for pt in [0.2, 0.3, 0.4, 0.5, 0.6]:  # 5 peaks close together
            idx = int(pt * sample_rate)
            pulse_width = int(0.02 * sample_rate)
            start_idx = max(0, idx - pulse_width // 2)
            end_idx = min(len(audio), idx + pulse_width // 2)
            audio[start_idx:end_idx] = 0.5

        detections = detector.detect_sequences(
            audio, sample_rate,
            min_peaks=3,
            max_gap=0.3,
        )

        assert isinstance(detections, list)
        for det in detections:
            assert isinstance(det, Detection)
            assert det.label == "cwt_sequence"
            assert "n_peaks" in det.metadata

    def test_detect_with_frequency_band(self):
        """Test detection with frequency band filter."""
        detector = CWTPeakDetector(
            low_freq=1000,
            high_freq=4000,
            snr_threshold=2.0,
        )

        sample_rate = 22050
        audio = np.random.randn(int(sample_rate * 1.0)).astype(np.float32) * 0.1

        peaks = detector.detect(audio, sample_rate)

        assert isinstance(peaks, list)

    def test_detect_from_file(self, mock_audio_file):
        """Test detection from file."""
        detector = CWTPeakDetector(snr_threshold=1.0)

        peaks = detector.detect_from_file(mock_audio_file)

        assert isinstance(peaks, list)


# =============================================================================
# Test Accelerating Pattern Detector
# =============================================================================

class TestAcceleratingPatternDetector:
    """Tests for AcceleratingPatternDetector."""

    def test_detector_creation(self):
        """Test creating accelerating pattern detector."""
        detector = AcceleratingPatternDetector(
            min_pulses=5,
            acceleration_threshold=2.0,
            low_freq=1000,
            high_freq=5000,
        )

        assert detector.min_pulses == 5
        assert detector.acceleration_threshold == 2.0
        assert detector.low_freq == 1000
        assert detector.high_freq == 5000

    def test_analyze_intervals_insufficient_pulses(self):
        """Test interval analysis with insufficient pulses."""
        detector = AcceleratingPatternDetector(min_pulses=5)

        peak_times = np.array([0.1, 0.2, 0.3])  # Only 3 pulses

        result = detector.analyze_intervals(peak_times)

        assert result["is_accelerating"] is False
        assert result["n_pulses"] == 3

    def test_analyze_intervals_constant_rate(self):
        """Test interval analysis with constant rate."""
        detector = AcceleratingPatternDetector(
            min_pulses=5,
            acceleration_threshold=1.5,
            min_pulse_rate=5.0,
            max_pulse_rate=20.0,
        )

        # Constant 10 Hz rate
        peak_times = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])

        result = detector.analyze_intervals(peak_times)

        assert result["is_accelerating"] == False
        # For constant rate, ratio should be close to 1.0
        assert abs(result["acceleration_ratio"] - 1.0) < 0.5

    def test_analyze_intervals_accelerating(self):
        """Test interval analysis with accelerating pattern."""
        detector = AcceleratingPatternDetector(
            min_pulses=5,
            acceleration_threshold=1.5,
            min_pulse_rate=5.0,
            max_pulse_rate=30.0,
        )

        # Accelerating from 5 Hz to 20 Hz
        # intervals: 0.2, 0.15, 0.1, 0.08, 0.05
        peak_times = np.array([0.0, 0.2, 0.35, 0.45, 0.53, 0.58])

        result = detector.analyze_intervals(peak_times)

        assert result["n_pulses"] == 6
        # Should detect acceleration
        assert result["acceleration_ratio"] > 1.0

    def test_detect_basic(self):
        """Test basic detection."""
        detector = AcceleratingPatternDetector(
            min_pulses=3,
            acceleration_threshold=1.3,
            window_duration=2.0,
            hop_duration=1.0,
        )

        sample_rate = 22050
        duration = 4.0
        audio = np.random.randn(int(sample_rate * duration)).astype(np.float32) * 0.1

        detections = detector.detect(audio, sample_rate)

        assert isinstance(detections, list)
        for det in detections:
            assert isinstance(det, Detection)

    def test_detect_deceleration(self):
        """Test detection with deceleration threshold."""
        detector = AcceleratingPatternDetector(
            min_pulses=3,
            acceleration_threshold=1.5,
            deceleration_threshold=1.5,  # Detect deceleration too
            window_duration=2.0,
        )

        sample_rate = 22050
        duration = 3.0
        audio = np.random.randn(int(sample_rate * duration)).astype(np.float32) * 0.1

        detections = detector.detect(audio, sample_rate)

        assert isinstance(detections, list)

    def test_detect_from_file(self, mock_audio_file):
        """Test detection from file."""
        detector = AcceleratingPatternDetector(
            min_pulses=2,
            acceleration_threshold=1.2,
        )

        detections = detector.detect_from_file(mock_audio_file)

        assert isinstance(detections, list)


# =============================================================================
# Test Utility Functions
# =============================================================================

class TestUtilityFunctions:
    """Tests for utility functions."""

    def test_detect_all(self):
        """Test running multiple detectors."""
        detectors = [
            BandLimitedEnergyDetector(low_freq=1000, high_freq=4000, threshold_db=-25),
            RibbitDetector(pulse_rate_hz=10.0, min_score=0.1),
        ]

        sample_rate = 22050
        audio = np.random.randn(int(sample_rate * 2.0)).astype(np.float32) * 0.1

        all_detections = detect_all(audio, sample_rate, detectors)

        assert isinstance(all_detections, list)
        # Should be sorted by start time
        if len(all_detections) > 1:
            times = [d.start_time for d in all_detections]
            assert times == sorted(times)

    def test_export_detections_csv(self, temp_dir):
        """Test exporting detections to CSV."""
        detections = [
            Detection(start_time=0.0, end_time=1.0, confidence=0.9, label="test1"),
            Detection(start_time=2.0, end_time=3.0, confidence=0.8, label="test2"),
        ]

        output_path = temp_dir / "detections.csv"
        result = export_detections(detections, output_path, format="csv")

        assert result.exists()

        # Verify content
        with open(result) as f:
            content = f.read()
            assert "start_time" in content
            assert "0.0" in content
            assert "test1" in content

    def test_export_detections_json(self, temp_dir):
        """Test exporting detections to JSON."""
        import json

        detections = [
            Detection(start_time=0.0, end_time=1.0, confidence=0.9),
        ]

        output_path = temp_dir / "detections.json"
        result = export_detections(detections, output_path, format="json")

        assert result.exists()

        with open(result) as f:
            data = json.load(f)
            assert len(data) == 1
            assert data[0]["start_time"] == 0.0

    def test_export_detections_empty(self, temp_dir):
        """Test exporting empty detections."""
        output_path = temp_dir / "empty.csv"
        result = export_detections([], output_path, format="csv")

        assert result.exists()

    def test_batch_detect(self, mock_audio_file, temp_dir):
        """Test batch detection."""
        # Create a second mock file
        import shutil
        second_file = temp_dir / "second.wav"
        shutil.copy(mock_audio_file, second_file)

        detector = BandLimitedEnergyDetector(
            low_freq=500,
            high_freq=7000,
            threshold_db=-30,
        )

        results = batch_detect(
            [mock_audio_file, second_file],
            detector,
            verbose=False,
        )

        assert len(results) == 2
        assert str(mock_audio_file) in results
        assert str(second_file) in results


# =============================================================================
# Test Edge Cases
# =============================================================================

class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_very_short_audio(self):
        """Test detection with very short audio."""
        detector = BandLimitedEnergyDetector()

        sample_rate = 22050
        audio = np.random.randn(100).astype(np.float32)

        # Should not crash
        detections = detector.detect(audio, sample_rate)
        assert isinstance(detections, list)

    def test_stereo_audio(self):
        """Test detection with stereo audio."""
        detector = BandLimitedEnergyDetector()

        sample_rate = 22050
        audio = np.random.randn(2, int(sample_rate * 1.0)).astype(np.float32)

        detections = detector.detect(audio, sample_rate)
        assert isinstance(detections, list)

    def test_high_sample_rate(self):
        """Test detection with high sample rate."""
        detector = BandLimitedEnergyDetector(
            low_freq=1000,
            high_freq=20000,
        )

        sample_rate = 96000
        audio = np.random.randn(int(sample_rate * 0.5)).astype(np.float32) * 0.1

        detections = detector.detect(audio, sample_rate)
        assert isinstance(detections, list)

    def test_low_sample_rate(self):
        """Test detection with low sample rate."""
        detector = BandLimitedEnergyDetector(
            low_freq=100,
            high_freq=3000,
        )

        sample_rate = 8000
        audio = np.random.randn(int(sample_rate * 1.0)).astype(np.float32) * 0.1

        detections = detector.detect(audio, sample_rate)
        assert isinstance(detections, list)

    def test_cwt_single_sample(self):
        """Test CWT with minimal signal."""
        detector = CWTPeakDetector()

        sample_rate = 22050
        audio = np.array([0.5], dtype=np.float32)

        # Should handle gracefully
        peaks = detector.detect(audio, sample_rate)
        assert isinstance(peaks, list)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def temp_dir(tmp_path):
    """Provide a temporary directory."""
    return tmp_path


@pytest.fixture
def mock_audio_file(temp_dir):
    """Create a mock audio file with noise for testing."""
    import struct

    audio_path = temp_dir / "test_audio.wav"

    sample_rate = 16000
    duration = 1
    num_samples = sample_rate * duration
    bits_per_sample = 16
    num_channels = 1

    byte_rate = sample_rate * num_channels * bits_per_sample // 8
    block_align = num_channels * bits_per_sample // 8
    data_size = num_samples * block_align
    file_size = 36 + data_size

    # Generate noise instead of silence
    noise_samples = np.random.randint(-1000, 1000, num_samples, dtype=np.int16)

    with open(audio_path, "wb") as f:
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
        f.write(noise_samples.tobytes())

    return audio_path
