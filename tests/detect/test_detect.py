"""Tests for the detect domain (flattened, exception-based API)."""

import importlib.util
import json

import numpy as np
import pytest

from bioamla.detect import (
    RIBBIT_PRESETS,
    AcceleratingPatternDetector,
    BandLimitedEnergyDetector,
    CWTPeakDetector,
    Detection,
    DetectionError,
    InvalidDetectionParams,
    PeakDetection,
    RibbitDetection,
    RibbitDetector,
    batch_detect,
    create_ribbit_profile,
    detect_all,
    export_detections,
    get_ribbit_preset,
    list_ribbit_presets,
)
from bioamla.detect.batch import batch_detect_dir
from bioamla.exceptions import DependencyError

OPENSOUNDSCAPE_AVAILABLE = importlib.util.find_spec("opensoundscape") is not None


@pytest.fixture
def burst_signal() -> tuple:
    """A 3-second 16kHz signal with periodic ~440Hz bursts (so detectors fire)."""
    sample_rate = 16000
    duration = 3.0
    t = np.linspace(0, duration, int(sample_rate * duration), dtype=np.float32)
    signal = np.zeros_like(t)
    # Add 6 bursts at 10 Hz pulse rate within the band 500-5000 Hz.
    for start in np.arange(0.0, duration, 0.1):
        s = int(start * sample_rate)
        e = int((start + 0.04) * sample_rate)
        tt = t[s:e] - t[s]
        signal[s:e] = 0.8 * np.sin(2 * np.pi * 1000 * tt)
    return signal, sample_rate


# =============================================================================
# Band-limited energy detector
# =============================================================================


class TestBandLimitedEnergyDetector:
    def test_detect_returns_detections(self, burst_signal) -> None:
        audio, sr = burst_signal
        detector = BandLimitedEnergyDetector(low_freq=500, high_freq=5000)
        detections = detector.detect(audio, sr)
        assert isinstance(detections, list)
        assert all(isinstance(d, Detection) for d in detections)
        assert len(detections) > 0

    def test_detect_from_file(self, test_audio_path_3s) -> None:
        detector = BandLimitedEnergyDetector(low_freq=300, high_freq=5000)
        detections = detector.detect_from_file(test_audio_path_3s)
        assert isinstance(detections, list)
        assert all(isinstance(d, Detection) for d in detections)

    def test_invalid_freq_band_raises(self) -> None:
        with pytest.raises(InvalidDetectionParams, match="high_freq"):
            BandLimitedEnergyDetector(low_freq=5000, high_freq=500)

    def test_negative_freq_raises(self) -> None:
        with pytest.raises(InvalidDetectionParams):
            BandLimitedEnergyDetector(low_freq=-100, high_freq=500)


# =============================================================================
# RIBBIT (native autocorrelation) detector
# =============================================================================


class TestRibbitDetector:
    def test_detect_returns_list(self, burst_signal) -> None:
        audio, sr = burst_signal
        detector = RibbitDetector(pulse_rate_hz=10.0, low_freq=500, high_freq=5000, min_score=0.0)
        detections = detector.detect(audio, sr)
        assert isinstance(detections, list)
        assert all(isinstance(d, Detection) for d in detections)

    def test_pulse_score_in_range(self, burst_signal) -> None:
        audio, sr = burst_signal
        detector = RibbitDetector(pulse_rate_hz=10.0, low_freq=500, high_freq=5000)
        score = detector.compute_pulse_score(audio, sr)
        assert 0.0 <= score <= 1.0

    def test_invalid_freq_band_raises(self) -> None:
        with pytest.raises(InvalidDetectionParams):
            RibbitDetector(low_freq=5000, high_freq=500)


# =============================================================================
# CWT peak detector
# =============================================================================


class TestCWTPeakDetector:
    def test_detect_returns_peaks(self, burst_signal) -> None:
        audio, sr = burst_signal
        detector = CWTPeakDetector(snr_threshold=1.0)
        peaks = detector.detect(audio, sr)
        assert isinstance(peaks, list)
        assert all(isinstance(p, PeakDetection) for p in peaks)

    def test_detect_sequences(self, burst_signal) -> None:
        audio, sr = burst_signal
        detector = CWTPeakDetector(snr_threshold=1.0)
        seqs = detector.detect_sequences(audio, sr, min_peaks=2)
        assert isinstance(seqs, list)
        assert all(isinstance(d, Detection) for d in seqs)

    def test_invalid_freq_band_raises(self) -> None:
        with pytest.raises(InvalidDetectionParams):
            CWTPeakDetector(low_freq=5000, high_freq=500)

    def test_no_validation_when_bands_unset(self) -> None:
        # Both bounds None: should not validate / raise.
        detector = CWTPeakDetector()
        assert detector.low_freq is None


# =============================================================================
# Accelerating pattern detector
# =============================================================================


class TestAcceleratingPatternDetector:
    def test_detect_returns_list(self, burst_signal) -> None:
        audio, sr = burst_signal
        detector = AcceleratingPatternDetector(low_freq=500, high_freq=5000)
        detections = detector.detect(audio, sr)
        assert isinstance(detections, list)
        assert all(isinstance(d, Detection) for d in detections)

    def test_invalid_freq_band_raises(self) -> None:
        with pytest.raises(InvalidDetectionParams):
            AcceleratingPatternDetector(low_freq=5000, high_freq=500)


# =============================================================================
# detect_from_file error handling
# =============================================================================


class TestDetectFromFileErrors:
    def test_missing_file_raises_audio_load_error(self) -> None:
        from bioamla.exceptions import AudioLoadError

        detector = BandLimitedEnergyDetector(low_freq=500, high_freq=5000)
        with pytest.raises(AudioLoadError):
            detector.detect_from_file("/nonexistent/path/file.wav")

    def test_detection_error_is_bioamla_error(self) -> None:
        from bioamla.exceptions import BioamlaError

        assert issubclass(DetectionError, BioamlaError)


# =============================================================================
# Convenience functions
# =============================================================================


class TestDetectAll:
    def test_detect_all_combines(self, burst_signal) -> None:
        audio, sr = burst_signal
        detectors = [
            BandLimitedEnergyDetector(low_freq=500, high_freq=5000),
            CWTPeakDetector(snr_threshold=1.0),
        ]
        detections = detect_all(audio, sr, detectors)
        assert isinstance(detections, list)
        assert all(isinstance(d, Detection) for d in detections)
        # Sorted by start time
        times = [d.start_time for d in detections]
        assert times == sorted(times)


class TestExportDetections:
    def test_export_csv(self, tmp_path) -> None:
        detections = [
            Detection(start_time=0.0, end_time=0.5, confidence=0.9, label="x"),
            Detection(start_time=1.0, end_time=1.5, confidence=0.8, label="y"),
        ]
        out = tmp_path / "out.csv"
        result = export_detections(detections, out, format="csv")
        assert result.exists()
        content = out.read_text()
        assert "start_time" in content
        assert "0.0" in content

    def test_export_json(self, tmp_path) -> None:
        detections = [Detection(start_time=0.0, end_time=0.5, confidence=0.9)]
        out = tmp_path / "out.json"
        export_detections(detections, out, format="json")
        data = json.loads(out.read_text())
        assert len(data) == 1
        assert data[0]["start_time"] == 0.0

    def test_export_empty_csv_writes_header(self, tmp_path) -> None:
        out = tmp_path / "empty.csv"
        export_detections([], out, format="csv")
        assert "start_time" in out.read_text()


class TestBatchDetect:
    def test_batch_detect_dict(self, test_audio_path_3s) -> None:
        detector = BandLimitedEnergyDetector(low_freq=500, high_freq=5000)
        results = batch_detect([test_audio_path_3s], detector, verbose=False)
        assert test_audio_path_3s in results
        assert isinstance(results[test_audio_path_3s], list)

    def test_batch_detect_bad_file_collected(self) -> None:
        detector = BandLimitedEnergyDetector(low_freq=500, high_freq=5000)
        results = batch_detect(["/nope/missing.wav"], detector, verbose=False)
        assert results["/nope/missing.wav"] == []


class TestBatchDetectDir:
    def test_batch_dir_energy(self, test_audio_dir, tmp_path) -> None:
        out_dir = tmp_path / "out"
        result = batch_detect_dir(
            test_audio_dir, out_dir, method="energy", low_freq=300, high_freq=5000
        )
        assert result.total_files == 3
        assert result.metadata["method"] == "energy"
        out_file = out_dir / "detections_energy.json"
        assert out_file.exists()
        data = json.loads(out_file.read_text())
        assert len(data) == 3

    def test_batch_dir_unknown_method(self, test_audio_dir, tmp_path) -> None:
        with pytest.raises(InvalidDetectionParams):
            batch_detect_dir(test_audio_dir, tmp_path, method="bogus")


# =============================================================================
# OpenSoundscape RIBBIT path
# =============================================================================


class TestRibbitPresets:
    def test_list_presets(self) -> None:
        presets = list_ribbit_presets()
        assert "american_bullfrog" in presets
        assert "spring_peeper" in presets

    def test_get_preset(self) -> None:
        params = get_ribbit_preset("spring_peeper")
        assert "signal_band" in params
        assert params == RIBBIT_PRESETS["spring_peeper"]

    def test_get_preset_unknown(self) -> None:
        with pytest.raises(InvalidDetectionParams, match="Unknown preset"):
            get_ribbit_preset("not_a_preset")

    def test_create_profile_valid(self) -> None:
        profile = create_ribbit_profile(
            "custom", signal_band=(500, 2000), pulse_rate_range=(3.0, 15.0)
        )
        assert profile["name"] == "custom"

    def test_create_profile_invalid_band(self) -> None:
        with pytest.raises(InvalidDetectionParams):
            create_ribbit_profile("bad", signal_band=(2000, 500), pulse_rate_range=(3.0, 15.0))


class TestRibbitDetectOpenSoundscape:
    @pytest.mark.skipif(not OPENSOUNDSCAPE_AVAILABLE, reason="opensoundscape not installed")
    def test_ribbit_detect_preset_runs(self, test_audio_path_3s) -> None:
        from bioamla.detect import ribbit_detect_preset

        detections, metadata = ribbit_detect_preset(test_audio_path_3s, "generic_mid_freq")
        assert isinstance(detections, list)
        assert all(isinstance(d, RibbitDetection) for d in detections)
        assert "duration" in metadata

    @pytest.mark.skipif(OPENSOUNDSCAPE_AVAILABLE, reason="opensoundscape is installed")
    def test_ribbit_detect_missing_dependency(self, test_audio_path_3s) -> None:
        from bioamla.detect import ribbit_detect

        with pytest.raises(DependencyError, match="opensoundscape"):
            ribbit_detect(test_audio_path_3s, signal_band=(500, 2000), pulse_rate_range=(3.0, 15.0))
