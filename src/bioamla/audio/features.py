"""Frame-level spectral descriptors, log-mel spectrogram and MFCCs.

:func:`frame_features` describes a recording one short frame at a time, so you can
follow how a call's pitch, bandwidth or tonality moves rather than summarising the
whole clip (which :func:`~bioamla.audio.get_frequency_stats` does).

The analysis:

1. A Hann-windowed STFT (``n_fft`` 1024, or 2048 above 60 kHz; hop ``n_fft / 4``).
2. Optional stationary noise-floor subtraction: for each frequency bin, a low
   percentile of its magnitude over (up to) 300 evenly spaced frames, scaled up a
   little, is subtracted from every frame. Wind, hiss and hum stay out of the
   descriptors; transient calls come through.
3. Per-frame descriptors over the ``f_min``–``f_max`` band, from the denoised
   magnitude spectrum (flatness from the raw one, where silence is still noise-like):

   | Descriptor | Meaning |
   | --- | --- |
   | `level_db` | frame power in dB relative to the loudest frame |
   | `centroid_hz` | magnitude-weighted mean frequency |
   | `spread_hz` | magnitude-weighted standard deviation around the centroid |
   | `flatness` | geometric / arithmetic mean of the magnitudes (1 = noise-like) |
   | `crest` | peak / mean magnitude (high = tonal) |
   | `rolloff_hz` | frequency below which 85% of the magnitude lies |
   | `flux` | change from the previous frame, normalised by both frames' level (0 for the first frame) |
   | `peak_hz` | frequency of the strongest bin |
   | `low_hz` | lowest frequency within 20 dB of the frame's peak |
   | `high_hz` | highest frequency within 20 dB of the frame's peak |

4. A log-mel spectrogram (triangular filters over ``f_min``–``f_max``, power in dB
   relative to the loudest band, floored at -40 dB so leftover noise can't swing the
   cepstrum) and MFCCs (an unnormalised DCT-II of the log-mel frames).

Only numpy is needed, so this works on the base install.
"""

from __future__ import annotations

import logging
import math
from dataclasses import asdict, dataclass, field, replace
from typing import TYPE_CHECKING, Any

import numpy as np

from bioamla.common.config import config_dataclass
from bioamla.exceptions import InvalidInputError

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)

#: Descriptor names, in the column order of :meth:`FrameFeatures.to_dataframe`.
DESCRIPTORS: tuple[str, ...] = (
    "level_db",
    "centroid_hz",
    "spread_hz",
    "flatness",
    "crest",
    "rolloff_hz",
    "flux",
    "peak_hz",
    "low_hz",
    "high_hz",
)

_MIN_FRAMES = 8


@config_dataclass
class FrameFeatureConfig:
    """Settings for :func:`frame_features`.

    ``n_fft``, ``hop`` and ``f_max`` default to values derived from the sample rate:
    ``n_fft`` 1024 (2048 above 60 kHz), ``hop`` ``n_fft // 4``, and ``f_max`` the
    lower of 92% of Nyquist and 16 kHz.

    Attributes:
        n_fft: FFT size in samples.
        hop: Samples between frame starts.
        n_mels: Number of mel bands.
        n_mfcc: Number of MFCCs, including c0.
        f_min: Lower edge of the analysis band in Hz.
        f_max: Upper edge of the analysis band in Hz.
        denoise: Subtract the stationary noise floor before measuring.
        noise_percentile: Percentile (0-1) of each bin's magnitude taken as its noise floor.
        noise_scale: Factor applied to that percentile before subtracting.
    """

    n_fft: int | None = None
    hop: int | None = None
    n_mels: int = 40
    n_mfcc: int = 13
    f_min: float = 250.0
    f_max: float | None = None
    denoise: bool = True
    noise_percentile: float = 0.2
    noise_scale: float = 1.8

    def __post_init__(self) -> None:
        checks = [
            (
                self.n_fft is None or self.n_fft >= 16,
                f"n_fft must be at least 16, got {self.n_fft}",
            ),
            (self.hop is None or self.hop >= 1, f"hop must be at least 1, got {self.hop}"),
            (self.n_mels >= 2, f"n_mels must be at least 2, got {self.n_mels}"),
            (
                1 <= self.n_mfcc <= self.n_mels,
                f"n_mfcc must be between 1 and n_mels ({self.n_mels}), got {self.n_mfcc}",
            ),
            (self.f_min >= 0, f"f_min must not be negative, got {self.f_min}"),
            (
                self.f_max is None or self.f_max > self.f_min,
                f"f_max must be above f_min ({self.f_min:g}), got {self.f_max}",
            ),
            (
                0 <= self.noise_percentile <= 1,
                f"noise_percentile must be between 0 and 1, got {self.noise_percentile}",
            ),
            (self.noise_scale >= 0, f"noise_scale must not be negative, got {self.noise_scale}"),
        ]
        for ok, message in checks:
            if not ok:
                raise InvalidInputError(message)


@dataclass
class FrameFeatures:
    """Per-frame analysis of a recording, from :func:`frame_features`.

    Attributes:
        sample_rate: Sample rate of the analysed audio in Hz.
        n_fft: FFT size used.
        hop: Hop between frames in samples.
        f_min: Lower edge of the analysis band in Hz.
        f_max: Upper edge of the analysis band in Hz.
        duration: Length of the audio in seconds.
        times: Centre time of each frame in seconds, shape ``(frames,)``.
        descriptors: Descriptor name -> per-frame values, see :data:`DESCRIPTORS`.
        mel_db: Log-mel spectrogram in dB, shape ``(frames, n_mels)``.
        mfcc: MFCCs, shape ``(frames, n_mfcc)``.
        config: The settings used, with derived values filled in.
    """

    sample_rate: float
    n_fft: int
    hop: int
    f_min: float
    f_max: float
    duration: float
    times: np.ndarray
    descriptors: dict[str, np.ndarray]
    mel_db: np.ndarray
    mfcc: np.ndarray
    config: FrameFeatureConfig = field(repr=False)

    @property
    def n_frames(self) -> int:
        """Number of frames."""
        return len(self.times)

    def to_dataframe(self) -> pd.DataFrame:
        """One row per frame: ``time``, the descriptors, then ``mfcc_0``..``mfcc_<n-1>``."""
        import pandas as pd

        columns: dict[str, np.ndarray] = {"time": self.times}
        columns.update({name: self.descriptors[name] for name in DESCRIPTORS})
        for k in range(self.mfcc.shape[1]):
            columns[f"mfcc_{k}"] = self.mfcc[:, k]
        return pd.DataFrame(columns)

    def to_dict(self) -> dict[str, Any]:
        """The analysis settings (``config`` included) plus per-frame values as lists."""
        return {
            "sample_rate": self.sample_rate,
            "n_fft": self.n_fft,
            "hop": self.hop,
            "f_min": self.f_min,
            "f_max": self.f_max,
            "duration": self.duration,
            "n_frames": self.n_frames,
            "times": self.times.tolist(),
            "descriptors": {k: v.tolist() for k, v in self.descriptors.items()},
            "mel_db": self.mel_db.tolist(),
            "mfcc": self.mfcc.tolist(),
            "config": asdict(self.config),
        }


def _resolve(config: FrameFeatureConfig, sample_rate: float) -> FrameFeatureConfig:
    """Fill in the sample-rate dependent defaults and check them against Nyquist."""
    if not (math.isfinite(sample_rate) and sample_rate > 0):
        raise InvalidInputError(f"sample_rate must be a positive number, got {sample_rate}")
    n_fft = config.n_fft if config.n_fft is not None else (2048 if sample_rate > 60000 else 1024)
    hop = config.hop if config.hop is not None else n_fft // 4
    f_max = config.f_max if config.f_max is not None else min(sample_rate / 2 * 0.92, 16000.0)
    if not config.f_min < f_max <= sample_rate / 2:
        raise InvalidInputError(
            f"need f_min < f_max <= Nyquist ({sample_rate / 2:g} Hz), "
            f"got f_min={config.f_min:g}, f_max={f_max:g}"
        )
    return replace(config, n_fft=n_fft, hop=hop, f_max=float(f_max))


def _mel_filterbank(n_mels: int, bin_idx: np.ndarray, bin_hz: float, f_min: float, f_max: float):
    """Triangular mel filters over the given FFT bins, shape ``(n_mels, len(bin_idx))``."""
    lo, hi = (2595 * np.log10(1 + f / 700) for f in (f_min, f_max))
    mels = lo + (hi - lo) * np.arange(n_mels + 2) / (n_mels + 1)
    edges = 700 * (10 ** (mels / 2595) - 1) / bin_hz  # in bins
    a, c, b = edges[:-2, None], edges[1:-1, None], edges[2:, None]
    k = bin_idx[None, :]
    return np.clip(np.minimum((k - a) / (c - a), (b - k) / (b - c)), 0, None)


def _to_mono(audio: np.ndarray) -> np.ndarray:
    """1-D float64 samples; 2-D input is averaged over its channel axis.

    ``(samples, channels)`` is the usual layout; a short first axis (at most 8, and
    shorter than the second) is read as ``(channels, samples)``.
    """
    audio = np.asarray(audio, dtype=np.float64)
    if audio.ndim == 1:
        return audio
    if audio.ndim == 2:
        channels_first = audio.shape[0] <= 8 and audio.shape[0] < audio.shape[1]
        return audio.mean(axis=0 if channels_first else 1)
    raise InvalidInputError(f"audio must be 1-D or 2-D (samples x channels), got {audio.shape}")


_CHUNK_FRAMES = 2048  # frames transformed at a time, to bound memory on long recordings


def frame_features(
    audio: np.ndarray, sample_rate: float, config: FrameFeatureConfig | None = None
) -> FrameFeatures:
    """Spectral descriptors, log-mel spectrogram and MFCCs for every frame.

    See the module docstring for what is measured and how. The recording is
    processed in blocks of frames, so memory stays bounded on long files.

    Args:
        audio: Samples, shape ``(samples,)``. 2-D input (``(samples, channels)``, or
            ``(channels, samples)`` with at most 8 channels) is averaged to mono.
        sample_rate: Sample rate in Hz.
        config: Analysis settings. ``None`` uses the defaults.

    Returns:
        :class:`FrameFeatures`.

    Raises:
        InvalidInputError: If the audio is too short (fewer than 8 frames), the
            sample rate is not positive, or a setting is out of range.

    Example:
        >>> from bioamla.audio import frame_features, load_audio
        >>> audio, sr = load_audio("wren.wav")
        >>> features = frame_features(audio, sr)
        >>> features.to_dataframe()[["time", "centroid_hz", "flatness"]].head()
    """
    config = _resolve(config or FrameFeatureConfig(), sample_rate)
    audio = _to_mono(audio)
    n_fft, hop, f_max = config.n_fft, config.hop, config.f_max
    assert n_fft is not None and hop is not None and f_max is not None  # resolved

    n_frames = (len(audio) - n_fft) // hop + 1 if len(audio) >= n_fft else 0
    if n_frames < _MIN_FRAMES:
        raise InvalidInputError(
            f"audio is too short to analyze: {len(audio) / sample_rate:.3f} s gives "
            f"{n_frames} frames of {n_fft} samples, need at least {_MIN_FRAMES}"
        )

    bins = n_fft // 2 + 1
    bin_hz = sample_rate / n_fft
    b_lo = max(1, math.ceil(config.f_min / bin_hz))
    b_hi = min(bins - 1, math.floor(f_max / bin_hz))
    if b_hi < b_lo:
        raise InvalidInputError(
            f"the band {config.f_min:g}-{f_max:g} Hz holds no FFT bins at n_fft={n_fft}"
        )
    band = slice(b_lo, b_hi + 1)
    bin_idx = np.arange(b_lo, b_hi + 1, dtype=np.float64)
    count = len(bin_idx)

    window = np.hanning(n_fft)  # symmetric: 0.5 - 0.5 cos(2 pi i / (N - 1))
    framed = np.lib.stride_tricks.sliding_window_view(audio, n_fft)[::hop][:n_frames]  # a view

    def band_magnitude(rows: np.ndarray) -> np.ndarray:
        return np.abs(np.fft.rfft(rows * window, axis=1))[:, band]

    if config.denoise:
        probes = min(n_frames, 300)
        idx = (np.arange(probes) * (n_frames - 1)) // max(1, probes - 1)
        rank = min(int(probes * config.noise_percentile), probes - 1)
        noise = np.sort(band_magnitude(framed[idx]), axis=0)[rank] * config.noise_scale
    else:
        noise = np.zeros(count)

    filters = _mel_filterbank(config.n_mels, bin_idx, bin_hz, config.f_min, f_max)
    out = {name: np.empty(n_frames) for name in DESCRIPTORS}
    power = np.empty(n_frames)
    mel = np.empty((n_frames, config.n_mels))
    prev = np.zeros(count)
    prev_level = 0.0

    for lo in range(0, n_frames, _CHUNK_FRAMES):
        hi = min(lo + _CHUNK_FRAMES, n_frames)
        raw = band_magnitude(framed[lo:hi])
        m = np.maximum(raw - noise, 0.0)

        total = m.sum(axis=1)
        pw = (m * m).sum(axis=1)
        peak = m.max(axis=1)
        peak_bin = np.where(peak > 0, b_lo + m.argmax(axis=1), b_lo)
        has = total > 0
        safe_total = np.where(has, total, 1.0)
        centroid = np.where(has, (m * bin_idx).sum(axis=1) / safe_total, peak_bin)
        spread = np.where(
            has, np.sqrt((m * (bin_idx - centroid[:, None]) ** 2).sum(axis=1) / safe_total), 0.0
        )
        # Flatness from the raw spectrum, where silence still looks like noise.
        raw_mean = raw.mean(axis=1)
        flatness = np.where(
            raw_mean > 0,
            np.exp(np.log(raw + 1e-9).mean(axis=1)) / np.where(raw_mean > 0, raw_mean, 1.0),
            0.0,
        )
        mean = total / count
        crest = np.where(mean > 0, peak / np.where(mean > 0, mean, 1.0), 0.0)
        rolloff = b_lo + np.argmax(np.cumsum(m, axis=1) >= (total * 0.85)[:, None], axis=1)
        above = m > (peak * 0.1)[:, None]  # within 20 dB of the frame's peak magnitude
        any_above = above.any(axis=1)
        low = np.where(any_above, b_lo + above.argmax(axis=1), peak_bin)
        high = np.where(any_above, b_hi - above[:, ::-1].argmax(axis=1), peak_bin)

        before = np.vstack([prev[None, :], m[:-1]])
        level = np.sqrt(pw)
        level_before = np.concatenate([[prev_level], level[:-1]])
        flux = np.sqrt(((m - before) ** 2).sum(axis=1)) / (level + level_before + 1e-9)

        out["centroid_hz"][lo:hi] = centroid * bin_hz
        out["spread_hz"][lo:hi] = spread * bin_hz
        out["flatness"][lo:hi] = flatness
        out["crest"][lo:hi] = crest
        out["rolloff_hz"][lo:hi] = rolloff * bin_hz
        out["flux"][lo:hi] = flux
        out["peak_hz"][lo:hi] = peak_bin * bin_hz
        out["low_hz"][lo:hi] = low * bin_hz
        out["high_hz"][lo:hi] = high * bin_hz
        power[lo:hi] = pw
        mel[lo:hi] = (m * m) @ filters.T
        prev, prev_level = m[-1], level[-1]

    out["flux"][0] = 0.0  # the first frame has nothing before it to change from
    out["level_db"] = 10 * np.log10(power / (power.max() or 1.0) + 1e-10)
    mel_db = 10 * np.log10(mel / (mel.max() or 1.0) + 1e-4)
    k = np.arange(config.n_mfcc)[:, None]
    n = np.arange(config.n_mels)[None, :]
    mfcc = mel_db @ np.cos(np.pi * k * (n + 0.5) / config.n_mels).T

    logger.debug("Analyzed %d frames (n_fft=%d, hop=%d)", n_frames, n_fft, hop)
    return FrameFeatures(
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop=hop,
        f_min=float(config.f_min),
        f_max=float(f_max),
        duration=len(audio) / sample_rate,
        times=(np.arange(n_frames) * hop + n_fft / 2) / sample_rate,
        descriptors=out,
        mel_db=mel_db,
        mfcc=mfcc,
        config=config,
    )


__all__ = ["DESCRIPTORS", "FrameFeatureConfig", "FrameFeatures", "frame_features"]
