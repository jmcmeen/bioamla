"""Coverage tests for :mod:`bioamla.datasets.measurements`."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.io.wavfile as wav

from bioamla.datasets.annotations import Annotation
from bioamla.datasets.measurements import (
    ALL_METRICS,
    DEFAULT_METRICS,
    compute_measurements,
)
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


@pytest.fixture
def sweep_audio_path(tmp_path) -> str:
    """A 2-second 16 kHz linear frequency sweep (500 -> 3000 Hz) for contour tests."""
    from scipy.signal import chirp

    sample_rate = 16000
    t = np.linspace(0, 2.0, int(sample_rate * 2.0), dtype=np.float32)
    samples = (0.5 * chirp(t, f0=500.0, f1=3000.0, t1=2.0, method="linear") * 32767).astype(
        np.int16
    )
    path = tmp_path / "sweep.wav"
    wav.write(str(path), sample_rate, samples)
    return str(path)


@pytest.fixture
def two_tone_audio_path(tmp_path) -> str:
    """A 16 kHz file with a weak 1 kHz tone and a strong 6 kHz tone."""
    sample_rate = 16000
    t = np.linspace(0, 2.0, int(sample_rate * 2.0), dtype=np.float32)
    signal = 0.2 * np.sin(2 * np.pi * 1000.0 * t) + 0.6 * np.sin(2 * np.pi * 6000.0 * t)
    samples = (signal * 32767).astype(np.int16)
    path = tmp_path / "two_tone.wav"
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


class TestExpandedMetrics:
    """Coverage for the expanded time/amplitude/power/frequency/entropy/contour metrics."""

    def test_metrics_all_returns_full_set(self, chirp_audio_path: str) -> None:
        # A clean tone with freq bounds -> nothing is uncomputable, so "all" is complete.
        m = compute_measurements(_ann(), chirp_audio_path, metrics="all")
        assert set(m) == set(ALL_METRICS)

    def test_invalid_metrics_string_raises(self, chirp_audio_path: str) -> None:
        with pytest.raises(AnnotationError, match="metrics must be a list"):
            compute_measurements(_ann(), chirp_audio_path, metrics="rms")  # not a list, not "all"

    def test_time_domain(self, chirp_audio_path: str) -> None:
        m = compute_measurements(
            _ann(), chirp_audio_path, metrics=["zero_crossing_rate", "peak_time"]
        )
        # 1 kHz tone at 16 kHz -> 2000 zero crossings/s -> 0.125 crossings per sample.
        assert m["zero_crossing_rate"] == pytest.approx(0.125, abs=0.01)
        # Peak of a sine occurs within its first cycle (< 1 ms in).
        assert 0.0 <= m["peak_time"] < 0.01

    def test_amplitude_db_metrics(self, chirp_audio_path: str) -> None:
        metrics = ["peak_db", "rms_db", "crest_factor_db", "dynamic_range"]
        m = compute_measurements(_ann(), chirp_audio_path, metrics=metrics)
        assert set(m) == set(metrics)
        assert m["peak_db"] < 0.0  # below full scale
        assert m["rms_db"] < m["peak_db"]
        # Crest factor of a sine is ~3.01 dB; dynamic_range mirrors it here.
        assert m["crest_factor_db"] == pytest.approx(3.01, abs=0.3)
        assert m["dynamic_range"] == pytest.approx(m["crest_factor_db"], abs=1e-6)

    def test_power_metrics(self, chirp_audio_path: str) -> None:
        m = compute_measurements(
            _ann(), chirp_audio_path, metrics=["avg_power", "max_power", "energy", "rms", "peak"]
        )
        # avg_power == rms**2, max_power == peak**2.
        assert m["avg_power"] == pytest.approx(m["rms"] ** 2, rel=1e-5)
        assert m["max_power"] == pytest.approx(m["peak"] ** 2, rel=1e-5)
        assert m["energy"] > 0.0

    def test_frequency_percentiles_band_limited(self, chirp_audio_path: str) -> None:
        metrics = [
            "peak_frequency",
            "freq_q1",
            "freq_q3",
            "freq_5",
            "freq_95",
            "bandwidth_90",
            "bandwidth_iqr",
        ]
        m = compute_measurements(_ann(), chirp_audio_path, metrics=metrics)
        assert set(m) == set(metrics)
        # Energy of a 1 kHz tone sits at ~1 kHz; percentiles bracket it tightly.
        assert m["peak_frequency"] == pytest.approx(1000.0, abs=30.0)
        assert m["freq_5"] <= 1000.0 <= m["freq_95"]
        assert m["bandwidth_90"] >= 0.0
        assert m["bandwidth_iqr"] >= 0.0

    def test_new_freq_metrics_respect_annotation_band(self, two_tone_audio_path: str) -> None:
        # Strong 6 kHz tone is outside a 200-4000 Hz box -> peak_frequency tracks the
        # in-band 1 kHz tone, not the louder out-of-band one.
        ann = _ann(low_freq=200.0, high_freq=4000.0)
        m = compute_measurements(ann, two_tone_audio_path, metrics=["peak_frequency"])
        assert m["peak_frequency"] == pytest.approx(1000.0, abs=50.0)

    def test_new_freq_metrics_full_band_when_no_bounds(self, two_tone_audio_path: str) -> None:
        # Without freq bounds the louder 6 kHz tone wins.
        ann = _ann(low_freq=None, high_freq=None)
        m = compute_measurements(ann, two_tone_audio_path, metrics=["peak_frequency"])
        assert m["peak_frequency"] == pytest.approx(6000.0, abs=50.0)

    def test_entropy_metrics(self, chirp_audio_path: str) -> None:
        m = compute_measurements(
            _ann(), chirp_audio_path, metrics=["spectral_entropy", "temporal_entropy"]
        )
        assert m["spectral_entropy"] >= 0.0
        assert m["temporal_entropy"] >= 0.0

    def test_contour_on_constant_tone(self, chirp_audio_path: str) -> None:
        metrics = ["pfc_min", "pfc_max", "pfc_mean", "pfc_start", "pfc_end", "pfc_slope"]
        m = compute_measurements(_ann(), chirp_audio_path, metrics=metrics)
        assert set(m) == set(metrics)
        # Constant 1 kHz tone -> flat contour near 1 kHz, ~zero slope.
        assert m["pfc_mean"] == pytest.approx(1000.0, abs=70.0)
        assert m["pfc_slope"] == pytest.approx(0.0, abs=50.0)

    def test_contour_tracks_rising_sweep(self, sweep_audio_path: str) -> None:
        # Whole 500 -> 3000 Hz sweep; band 200-4000 covers it.
        ann = _ann(start_time=0.0, end_time=2.0)
        metrics = ["pfc_start", "pfc_end", "pfc_slope"]
        m = compute_measurements(ann, sweep_audio_path, metrics=metrics)
        assert m["pfc_end"] > m["pfc_start"]  # rising contour
        assert m["pfc_slope"] > 0.0

    def test_db_metrics_omitted_for_silence(self, tmp_path) -> None:
        # dBFS of digital silence is -inf and dynamic_range is NaN -> omit, don't emit.
        sample_rate = 16000
        samples = np.zeros(sample_rate, dtype=np.int16)
        path = tmp_path / "silent.wav"
        wav.write(str(path), sample_rate, samples)
        ann = _ann(start_time=0.1, end_time=0.5)
        m = compute_measurements(
            ann, str(path), metrics=["rms_db", "peak_db", "dynamic_range", "avg_power"]
        )
        assert "rms_db" not in m
        assert "peak_db" not in m
        assert "dynamic_range" not in m
        assert m["avg_power"] == 0.0  # power metrics are still well-defined (zero)

    def test_contour_omitted_for_short_clip(self, chirp_audio_path: str) -> None:
        # A sub-millisecond region is too short to frame -> contour metrics omitted.
        ann = _ann(start_time=0.2, end_time=0.2005)
        m = compute_measurements(ann, chirp_audio_path, metrics=["pfc_mean", "rms"])
        assert "pfc_mean" not in m
        assert "rms" in m  # other metrics still computed
