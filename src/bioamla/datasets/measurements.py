"""Compute acoustic measurements for an annotated audio region.

Measurements span five domains — **time**, **amplitude**, **power**,
**frequency**, and **entropy** — plus a summarized **peak-frequency contour**.
Each metric is a single scalar so the result is a flat ``dict[str, float]`` that
merges cleanly into a measurements CSV/Parquet table (one column per metric).

Frequency-domain metrics are computed from a single Welch PSD that is restricted
to the annotation's ``low_freq``/``high_freq`` box when both bounds are set
(matching ``centroid``/``rolloff`` behaviour); the per-frame peak-frequency
contour honours the same band. Amplitude (dB) and entropy metrics reuse the
domain primitives in :mod:`bioamla.audio` and :mod:`bioamla.indices`.

Metrics that cannot be computed for a given region (e.g. a frequency metric when
the PSD is empty, or the contour for a clip too short to frame) are **omitted**
from the result rather than emitted as ``NaN``.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from bioamla.datasets.annotations import Annotation
from bioamla.exceptions import AnnotationError, NotFoundError

logger = logging.getLogger(__name__)

# Backward-compatible default set — kept byte-identical: existing callers and
# tests rely on exactly these five metrics being returned when ``metrics`` is None.
DEFAULT_METRICS = ["duration", "bandwidth", "rms", "peak", "centroid"]

# Full metric vocabulary, grouped by domain. ``metrics="all"`` expands to ALL_METRICS.
TIME_METRICS = ["duration", "zero_crossing_rate", "peak_time"]
AMPLITUDE_METRICS = [
    "rms",
    "peak",
    "crest_factor",
    "rms_db",
    "peak_db",
    "crest_factor_db",
    "dynamic_range",
]
POWER_METRICS = ["avg_power", "max_power", "energy"]
FREQUENCY_METRICS = [
    "bandwidth",
    "centroid",
    "bandwidth_spectral",
    "rolloff",
    "peak_frequency",
    "freq_q1",
    "freq_q3",
    "freq_5",
    "freq_95",
    "bandwidth_90",
    "bandwidth_iqr",
]
ENTROPY_METRICS = ["spectral_entropy", "temporal_entropy"]
CONTOUR_METRICS = ["pfc_min", "pfc_max", "pfc_mean", "pfc_start", "pfc_end", "pfc_slope"]

# Every supported metric, in a stable, domain-grouped order.
ALL_METRICS = [
    *TIME_METRICS,
    *AMPLITUDE_METRICS,
    *POWER_METRICS,
    *FREQUENCY_METRICS,
    *ENTROPY_METRICS,
    *CONTOUR_METRICS,
]

# Pre-built membership sets, so the per-region ``_measure_*`` helpers test
# ``requested & <group>`` without re-allocating a set on every call.
_ALL_METRICS = frozenset(ALL_METRICS)
_POWER_METRICS = frozenset(POWER_METRICS)
_ENTROPY_METRICS = frozenset(ENTROPY_METRICS)
_CONTOUR_METRICS = frozenset(CONTOUR_METRICS)
# Frequency metrics derived from the (band-masked) Welch PSD.
_PSD_METRICS = frozenset(FREQUENCY_METRICS) - {"bandwidth"}
_AMPLITUDE_DB_METRICS = frozenset({"rms_db", "peak_db", "crest_factor_db", "dynamic_range"})


def compute_measurements(
    annotation: Annotation,
    audio_path: str,
    metrics: list[str] | str | None = None,
) -> dict[str, float]:
    """Compute acoustic measurements for one annotation region.

    Args:
        annotation: The annotation to measure.
        audio_path: Path to the source audio file.
        metrics: Metrics to compute. ``None`` uses :data:`DEFAULT_METRICS`; the
            string ``"all"`` expands to :data:`ALL_METRICS`; otherwise pass a list
            of metric names. Supported names, grouped by domain:

            * **time**: ``duration``, ``zero_crossing_rate`` (crossings per
              sample, 0–1), ``peak_time`` (s from region start)
            * **amplitude**: ``rms``, ``peak`` (linear), ``crest_factor``
              (linear peak/rms), ``rms_db``, ``peak_db`` (dBFS),
              ``crest_factor_db`` (dB), ``dynamic_range`` (dB)
            * **power**: ``avg_power``, ``max_power``, ``energy`` (Σ x²)
            * **frequency** (band-limited Welch PSD): ``bandwidth`` (annotation
              box width), ``centroid``, ``bandwidth_spectral``, ``rolloff``
              (85%), ``peak_frequency``, ``freq_q1``/``freq_q3`` (25/75%),
              ``freq_5``/``freq_95`` (5/95%), ``bandwidth_90`` (95−5%),
              ``bandwidth_iqr`` (75−25%)
            * **entropy**: ``spectral_entropy``, ``temporal_entropy``
            * **peak-frequency contour**: ``pfc_min``, ``pfc_max``, ``pfc_mean``,
              ``pfc_start``, ``pfc_end``, ``pfc_slope`` (Hz/s)

    Returns:
        Dict mapping metric name to value. Metrics that cannot be computed for
        the region are omitted.

    Raises:
        NotFoundError: If the audio file doesn't exist.
        AnnotationError: If ``metrics`` is malformed, or the audio cannot be
            loaded or measured.
    """
    if not Path(audio_path).exists():
        raise NotFoundError(f"Audio file not found: {audio_path}")

    requested = _resolve_metrics(metrics)

    from bioamla.audio import load_audio

    try:
        audio_data, sample_rate = load_audio(audio_path)
    except Exception as e:
        raise AnnotationError(f"Failed to load audio {audio_path}: {e}") from e

    if audio_data.ndim == 1:
        audio_data = audio_data.reshape(-1, 1)

    start_sample = int(annotation.start_time * sample_rate)
    end_sample = int(annotation.end_time * sample_rate)

    channel_idx = min(annotation.channel - 1, audio_data.shape[1] - 1)
    clip = audio_data[start_sample:end_sample, channel_idx]
    n = len(clip)

    measurements: dict[str, float] = {}

    try:
        _measure_time(measurements, requested, annotation, clip, n, sample_rate)
        _measure_amplitude(measurements, requested, clip, n)
        _measure_power(measurements, requested, clip, n)
        _measure_frequency(measurements, requested, annotation, clip, n, sample_rate)
        _measure_entropy(measurements, requested, clip, n, sample_rate)
        _measure_contour(measurements, requested, annotation, clip, n, sample_rate)
    except Exception as e:
        raise AnnotationError(f"Failed to compute measurements: {e}") from e

    return measurements


def _resolve_metrics(metrics: list[str] | str | None) -> set[str]:
    """Normalize and validate the ``metrics`` argument into a set of metric names.

    Raises:
        AnnotationError: If ``metrics`` is a string other than ``"all"``, or a list
            containing names that aren't in :data:`ALL_METRICS` (so typos fail fast
            rather than being silently dropped).
    """
    if metrics is None:
        return set(DEFAULT_METRICS)
    if isinstance(metrics, str):
        if metrics == "all":
            return set(ALL_METRICS)
        raise AnnotationError(
            f"metrics must be a list of names or the string 'all', got {metrics!r}"
        )

    resolved = set(metrics)
    unknown = resolved - _ALL_METRICS
    if unknown:
        raise AnnotationError(
            f"Unknown metric name(s): {sorted(unknown)}. "
            f"Choose from {ALL_METRICS} or pass metrics='all'."
        )
    return resolved


def _measure_time(
    out: dict[str, float],
    req: set[str],
    annotation: Annotation,
    clip: np.ndarray,
    n: int,
    sample_rate: int,
) -> None:
    if "duration" in req:
        out["duration"] = annotation.duration
    if n == 0:
        return
    if "zero_crossing_rate" in req:
        if n > 1:
            crossings = int(np.count_nonzero(np.diff(np.signbit(clip))))
            out["zero_crossing_rate"] = float(crossings / (n - 1))
        else:
            out["zero_crossing_rate"] = 0.0
    if "peak_time" in req:
        out["peak_time"] = float(int(np.argmax(np.abs(clip))) / sample_rate)


def _measure_amplitude(out: dict[str, float], req: set[str], clip: np.ndarray, n: int) -> None:
    if n == 0:
        return
    if "rms" in req:
        out["rms"] = float(np.sqrt(np.mean(clip**2)))
    if "peak" in req:
        out["peak"] = float(np.max(np.abs(clip)))
    if "crest_factor" in req:
        rms = np.sqrt(np.mean(clip**2))
        peak = np.max(np.abs(clip))
        out["crest_factor"] = float(peak / rms) if rms > 0 else 0.0
    # dB metrics are undefined for digital silence (dBFS -> -inf, dynamic_range
    # -> NaN), so omit them rather than emit non-finite values.
    if req & _AMPLITUDE_DB_METRICS and np.max(np.abs(clip)) > 0:
        from bioamla.audio import get_amplitude_stats

        # NB: AmplitudeStats.crest_factor is the dB form (peak_db − rms_db); the
        # linear ``crest_factor`` above is a distinct metric, kept for compat.
        stats = get_amplitude_stats(clip)
        if "rms_db" in req:
            out["rms_db"] = stats.rms_db
        if "peak_db" in req:
            out["peak_db"] = stats.peak_db
        if "crest_factor_db" in req:
            out["crest_factor_db"] = stats.crest_factor
        if "dynamic_range" in req:
            out["dynamic_range"] = stats.dynamic_range


def _measure_power(out: dict[str, float], req: set[str], clip: np.ndarray, n: int) -> None:
    if n == 0 or not (req & _POWER_METRICS):
        return
    squared = clip**2
    if "avg_power" in req:
        out["avg_power"] = float(np.mean(squared))
    if "max_power" in req:
        out["max_power"] = float(np.max(squared))
    if "energy" in req:
        out["energy"] = float(np.sum(squared))


def _measure_frequency(
    out: dict[str, float],
    req: set[str],
    annotation: Annotation,
    clip: np.ndarray,
    n: int,
    sample_rate: int,
) -> None:
    if "bandwidth" in req and annotation.bandwidth is not None:
        out["bandwidth"] = annotation.bandwidth

    if n == 0 or not (req & _PSD_METRICS):
        return

    from scipy import signal as scipy_signal

    n_fft = min(2048, n)
    freqs, psd = scipy_signal.welch(clip, sample_rate, nperseg=n_fft)

    if annotation.low_freq is not None and annotation.high_freq is not None:
        mask = (freqs >= annotation.low_freq) & (freqs <= annotation.high_freq)
        freqs = freqs[mask]
        psd = psd[mask]

    if len(psd) == 0 or np.sum(psd) <= 0:
        return

    total = float(np.sum(psd))
    cumsum = np.cumsum(psd)

    if "peak_frequency" in req:
        out["peak_frequency"] = float(freqs[int(np.argmax(psd))])
    if "centroid" in req:
        out["centroid"] = float(np.sum(freqs * psd) / total)
    if "bandwidth_spectral" in req:
        centroid = np.sum(freqs * psd) / total
        out["bandwidth_spectral"] = float(np.sqrt(np.sum((freqs - centroid) ** 2 * psd) / total))
    if "rolloff" in req:
        out["rolloff"] = _percentile_freq(freqs, cumsum, 0.85)
    if "freq_q1" in req:
        out["freq_q1"] = _percentile_freq(freqs, cumsum, 0.25)
    if "freq_q3" in req:
        out["freq_q3"] = _percentile_freq(freqs, cumsum, 0.75)
    if "freq_5" in req:
        out["freq_5"] = _percentile_freq(freqs, cumsum, 0.05)
    if "freq_95" in req:
        out["freq_95"] = _percentile_freq(freqs, cumsum, 0.95)
    if "bandwidth_90" in req:
        out["bandwidth_90"] = _percentile_freq(freqs, cumsum, 0.95) - _percentile_freq(
            freqs, cumsum, 0.05
        )
    if "bandwidth_iqr" in req:
        out["bandwidth_iqr"] = _percentile_freq(freqs, cumsum, 0.75) - _percentile_freq(
            freqs, cumsum, 0.25
        )


def _measure_entropy(
    out: dict[str, float], req: set[str], clip: np.ndarray, n: int, sample_rate: int
) -> None:
    if n == 0 or not (req & _ENTROPY_METRICS):
        return
    from bioamla.indices import spectral_entropy, temporal_entropy

    if "spectral_entropy" in req:
        out["spectral_entropy"] = spectral_entropy(clip, sample_rate)
    if "temporal_entropy" in req:
        out["temporal_entropy"] = temporal_entropy(clip, sample_rate)


def _measure_contour(
    out: dict[str, float],
    req: set[str],
    annotation: Annotation,
    clip: np.ndarray,
    n: int,
    sample_rate: int,
) -> None:
    if n == 0 or not (req & _CONTOUR_METRICS):
        return

    contour = _peak_freq_contour(clip, sample_rate, annotation.low_freq, annotation.high_freq)
    if contour is None:
        return
    times, peak_freqs = contour

    if "pfc_min" in req:
        out["pfc_min"] = float(np.min(peak_freqs))
    if "pfc_max" in req:
        out["pfc_max"] = float(np.max(peak_freqs))
    if "pfc_mean" in req:
        out["pfc_mean"] = float(np.mean(peak_freqs))
    if "pfc_start" in req:
        out["pfc_start"] = float(peak_freqs[0])
    if "pfc_end" in req:
        out["pfc_end"] = float(peak_freqs[-1])
    if "pfc_slope" in req:
        if len(peak_freqs) >= 2 and (times[-1] - times[0]) > 0:
            out["pfc_slope"] = float(np.polyfit(times, peak_freqs, 1)[0])
        else:
            out["pfc_slope"] = 0.0


def _percentile_freq(freqs: np.ndarray, cumsum: np.ndarray, fraction: float) -> float:
    """Frequency below which ``fraction`` of the cumulative PSD energy lies.

    Keyed off ``cumsum[-1]`` (the sequential running total) so ``rolloff`` stays
    bit-identical to its pre-expansion definition.
    """
    idx = int(np.searchsorted(cumsum, fraction * cumsum[-1]))
    return float(freqs[min(idx, len(freqs) - 1)])


def _peak_freq_contour(
    clip: np.ndarray,
    sample_rate: int,
    low_freq: float | None,
    high_freq: float | None,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Per-frame peak-frequency track of the clip (band-limited when bounds set).

    Returns ``(frame_times, peak_freqs)`` over frames with non-zero energy, or
    ``None`` if the clip is too short to frame or has no signal.
    """
    n = len(clip)
    if n < 32:
        return None

    from scipy import signal as scipy_signal

    nperseg = min(256, n)
    freqs, times, zxx = scipy_signal.stft(
        clip, fs=sample_rate, nperseg=nperseg, noverlap=nperseg // 2
    )
    mag = np.abs(zxx)

    if low_freq is not None and high_freq is not None:
        mask = (freqs >= low_freq) & (freqs <= high_freq)
        if np.any(mask):
            freqs = freqs[mask]
            mag = mag[mask, :]

    # Drop silent frames so the contour reflects the signal, not the noise floor.
    keep = mag.max(axis=0) > 0
    if not np.any(keep):
        return None

    mag = mag[:, keep]
    frame_times = times[keep]
    peak_freqs = freqs[np.argmax(mag, axis=0)]
    return frame_times, peak_freqs
