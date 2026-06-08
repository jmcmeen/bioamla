"""Compute acoustic measurements for an annotated audio region."""

from __future__ import annotations

import logging
from pathlib import Path

from bioamla.datasets.annotations import Annotation
from bioamla.exceptions import AnnotationError, DependencyError, NotFoundError

logger = logging.getLogger(__name__)

DEFAULT_METRICS = ["duration", "bandwidth", "rms", "peak", "centroid"]


def compute_measurements(
    annotation: Annotation,
    audio_path: str,
    metrics: list[str] | None = None,
) -> dict[str, float]:
    """Compute acoustic measurements for one annotation region.

    Args:
        annotation: The annotation to measure.
        audio_path: Path to the source audio file.
        metrics: Metrics to compute. If None, uses :data:`DEFAULT_METRICS`.
            Supported: duration, bandwidth, rms, peak, crest_factor, centroid,
            bandwidth_spectral, rolloff.

    Returns:
        Dict mapping metric name to value.

    Raises:
        NotFoundError: If the audio file doesn't exist.
        DependencyError: If numpy/scipy are not installed.
        AnnotationError: If the audio cannot be loaded or measured.
    """
    if not Path(audio_path).exists():
        raise NotFoundError(f"Audio file not found: {audio_path}")

    try:
        import numpy as np
        from scipy import signal as scipy_signal
    except ImportError as e:
        raise DependencyError(
            "Acoustic measurements require numpy and scipy — install bioamla[detect]"
        ) from e

    from bioamla.adapters.pydub import load_audio

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

    if metrics is None:
        metrics = list(DEFAULT_METRICS)

    measurements: dict[str, float] = {}

    try:
        if "duration" in metrics:
            measurements["duration"] = annotation.duration

        if "bandwidth" in metrics and annotation.bandwidth is not None:
            measurements["bandwidth"] = annotation.bandwidth

        if "rms" in metrics:
            measurements["rms"] = float(np.sqrt(np.mean(clip**2)))

        if "peak" in metrics:
            measurements["peak"] = float(np.max(np.abs(clip)))

        if "crest_factor" in metrics:
            rms = np.sqrt(np.mean(clip**2))
            peak = np.max(np.abs(clip))
            measurements["crest_factor"] = float(peak / rms) if rms > 0 else 0.0

        if any(m in metrics for m in ["centroid", "bandwidth_spectral", "rolloff"]):
            n_fft = min(2048, len(clip))
            freqs, psd = scipy_signal.welch(clip, sample_rate, nperseg=n_fft)

            if annotation.low_freq is not None and annotation.high_freq is not None:
                mask = (freqs >= annotation.low_freq) & (freqs <= annotation.high_freq)
                freqs = freqs[mask]
                psd = psd[mask]

            if len(psd) > 0 and np.sum(psd) > 0:
                if "centroid" in metrics:
                    measurements["centroid"] = float(np.sum(freqs * psd) / np.sum(psd))

                if "bandwidth_spectral" in metrics:
                    centroid = np.sum(freqs * psd) / np.sum(psd)
                    measurements["bandwidth_spectral"] = float(
                        np.sqrt(np.sum((freqs - centroid) ** 2 * psd) / np.sum(psd))
                    )

                if "rolloff" in metrics:
                    cumsum = np.cumsum(psd)
                    rolloff_idx = np.searchsorted(cumsum, 0.85 * cumsum[-1])
                    measurements["rolloff"] = float(freqs[min(rolloff_idx, len(freqs) - 1)])
    except Exception as e:
        raise AnnotationError(f"Failed to compute measurements: {e}") from e

    return measurements
