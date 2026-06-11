"""Coverage tests for :mod:`bioamla.datasets.measurements`."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.io.wavfile as wav

from bioamla.datasets.annotations import Annotation
from bioamla.datasets.measurements import DEFAULT_METRICS, compute_measurements
from bioamla.exceptions import AnnotationError, NotFoundError


@pytest.fixture
def chirp_audio_path(tmp_path) -> str:
    """A 2-second 16 kHz file with a 1 kHz tone for spectral measurements."""
    sample_rate = 16000
    t = np.linspace(0, 2.0, int(sample_rate * 2.0), dtype=np.float32)
    samples = (0.5 * np.sin(2 * np.pi * 1000.0 * t) * 32767).astype(np.int16)
    path = tmp_path / "chirp.wav"
    wav.write(str(path), sample_rate, samples)
    return str(path)


def _ann(**kw) -> Annotation:
    base = {"start_time": 0.2, "end_time": 1.0, "low_freq": 200.0, "high_freq": 4000.0}
    base.update(kw)
    return Annotation(**base)


class TestComputeMeasurements:
    def test_default_metrics(self, chirp_audio_path: str) -> None:
        m = compute_measurements(_ann(), chirp_audio_path)
        for key in DEFAULT_METRICS:
            assert key in m
        assert m["duration"] == pytest.approx(0.8)
        assert m["bandwidth"] == pytest.approx(3800.0)
        assert m["rms"] > 0.0
        assert m["peak"] > 0.0
        # Tone centroid should sit near 1 kHz.
        assert 700.0 < m["centroid"] < 1500.0

    def test_all_spectral_metrics(self, chirp_audio_path: str) -> None:
        metrics = ["crest_factor", "centroid", "bandwidth_spectral", "rolloff"]
        m = compute_measurements(_ann(), chirp_audio_path, metrics=metrics)
        assert set(m) == set(metrics)
        assert m["rolloff"] > 0.0
        assert m["bandwidth_spectral"] >= 0.0

    def test_bandwidth_skipped_when_freqs_missing(self, chirp_audio_path: str) -> None:
        ann = _ann(low_freq=None, high_freq=None)
        m = compute_measurements(ann, chirp_audio_path, metrics=["bandwidth", "rms"])
        assert "bandwidth" not in m  # bandwidth property is None -> omitted
        assert "rms" in m

    def test_spectral_without_freq_bounds(self, chirp_audio_path: str) -> None:
        # No low/high freq -> the masking branch is skipped.
        ann = _ann(low_freq=None, high_freq=None)
        m = compute_measurements(ann, chirp_audio_path, metrics=["centroid"])
        assert "centroid" in m

    def test_crest_factor_zero_rms(self, tmp_path) -> None:
        sample_rate = 16000
        samples = np.zeros(sample_rate, dtype=np.int16)
        path = tmp_path / "silent.wav"
        wav.write(str(path), sample_rate, samples)
        ann = _ann(start_time=0.1, end_time=0.5)
        m = compute_measurements(ann, str(path), metrics=["crest_factor"])
        assert m["crest_factor"] == 0.0

    def test_missing_file_raises(self, tmp_path) -> None:
        with pytest.raises(NotFoundError, match="Audio file not found"):
            compute_measurements(_ann(), str(tmp_path / "nope.wav"))

    def test_load_failure_wrapped(self, tmp_path) -> None:
        bad = tmp_path / "bad.wav"
        bad.write_bytes(b"not a wav file at all")
        with pytest.raises(AnnotationError, match="Failed to load audio"):
            compute_measurements(_ann(), str(bad))

    def test_channel_clamped_to_available(self, chirp_audio_path: str) -> None:
        # Request channel 5 on mono audio -> clamps to channel 0 without error.
        ann = _ann(channel=5)
        m = compute_measurements(ann, chirp_audio_path, metrics=["rms"])
        assert "rms" in m
