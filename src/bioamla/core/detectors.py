"""
Advanced Detection
==================

Provides specialized audio detection algorithms for bioacoustic analysis,
including energy detection, periodic call detection (RIBBIT), peak sequence
detection via CWT, and accelerating pattern detection.

Features:
- Band-limited energy detector for frequency-specific sounds
- RIBBIT periodic call detector for repetitive vocalizations
- Peak sequence detection via Continuous Wavelet Transform (CWT)
- Accelerating pattern detection for species with increasing call rates

References:
- RIBBIT algorithm: Lapp et al. (2021) - opensoundscape
- CWT peak detection: Du et al. (2006) - improved peak detection in mass spectra

Example:
    >>> from bioamla.detection import (
    ...     BandLimitedEnergyDetector,
    ...     RibbitDetector,
    ...     CWTPeakDetector,
    ...     AcceleratingPatternDetector,
    ... )
    >>>
    >>> # Detect frog calls in specific frequency band
    >>> detector = BandLimitedEnergyDetector(low_freq=500, high_freq=3000)
    >>> detections = detector.detect(audio, sample_rate)
    >>>
    >>> # Detect periodic calls like cricket chirps
    >>> ribbit = RibbitDetector(pulse_rate_hz=10.0)
    >>> scores = ribbit.detect(audio, sample_rate)
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import librosa
import numpy as np
from scipy import ndimage
from scipy.signal import find_peaks

from bioamla.core.signal import bandpass_filter
from bioamla.core.files import TextFile

logger = logging.getLogger(__name__)

__all__ = [
    # Data classes
    "Detection",
    "PeakDetection",
    # Detector classes
    "BandLimitedEnergyDetector",
    "RibbitDetector",
    "CWTPeakDetector",
    "AcceleratingPatternDetector",
    # Convenience functions
    "detect_all",
    "export_detections",
    "batch_detect",
]


# =============================================================================
# Detection Result Classes
# =============================================================================


@dataclass
class Detection:
    """
    Represents a detected acoustic event.

    Attributes:
        start_time: Start time in seconds.
        end_time: End time in seconds.
        confidence: Detection confidence/score (0-1).
        frequency_low: Lower frequency bound in Hz.
        frequency_high: Upper frequency bound in Hz.
        label: Optional detection label.
        metadata: Additional metadata.
    """

    start_time: float
    end_time: float
    confidence: float = 1.0
    frequency_low: Optional[float] = None
    frequency_high: Optional[float] = None
    label: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def duration(self) -> float:
        """Duration of detection in seconds."""
        return self.end_time - self.start_time

    @property
    def center_time(self) -> float:
        """Center time of detection in seconds."""
        return (self.start_time + self.end_time) / 2

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "start_time": self.start_time,
            "end_time": self.end_time,
            "confidence": self.confidence,
            "duration": self.duration,
            "frequency_low": self.frequency_low,
            "frequency_high": self.frequency_high,
            "label": self.label,
            **self.metadata,
        }


@dataclass
class PeakDetection:
    """
    Represents a detected peak in a signal.

    Attributes:
        time: Time of peak in seconds.
        amplitude: Peak amplitude/intensity.
        width: Peak width in seconds.
        prominence: Peak prominence.
        frequency: Associated frequency in Hz (if applicable).
    """

    time: float
    amplitude: float
    width: float = 0.0
    prominence: float = 0.0
    frequency: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "time": self.time,
            "amplitude": self.amplitude,
            "width": self.width,
            "prominence": self.prominence,
            "frequency": self.frequency,
        }


# =============================================================================
# Band-Limited Energy Detector
# =============================================================================


class BandLimitedEnergyDetector:
    """
    Band-limited energy detector for frequency-specific sound detection.

    This detector filters audio to a specific frequency band and detects
    regions where the energy exceeds a threshold. Useful for detecting
    vocalizations that occur in a known frequency range.

    Args:
        low_freq: Lower frequency bound in Hz.
        high_freq: Upper frequency bound in Hz.
        threshold_db: Detection threshold in dB relative to max energy.
        min_duration: Minimum detection duration in seconds.
        merge_threshold: Merge detections within this many seconds.
        smoothing_window: Energy smoothing window in seconds.

    Example:
        >>> detector = BandLimitedEnergyDetector(low_freq=1000, high_freq=4000)
        >>> detections = detector.detect(audio, sample_rate)
        >>> for d in detections:
        ...     print(f"{d.start_time:.2f}s - {d.end_time:.2f}s: {d.confidence:.2f}")
    """

    def __init__(
        self,
        low_freq: float = 500.0,
        high_freq: float = 5000.0,
        threshold_db: float = -20.0,
        min_duration: float = 0.05,
        merge_threshold: float = 0.1,
        smoothing_window: float = 0.02,
    ):
        self.low_freq = low_freq
        self.high_freq = high_freq
        self.threshold_db = threshold_db
        self.min_duration = min_duration
        self.merge_threshold = merge_threshold
        self.smoothing_window = smoothing_window

    def compute_energy(
        self,
        audio: np.ndarray,
        sample_rate: int,
        hop_length: int = 256,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute band-limited energy envelope.

        Args:
            audio: Audio signal.
            sample_rate: Sample rate in Hz.
            hop_length: Hop length for energy computation.

        Returns:
            Tuple of (energy envelope, time axis).
        """
        # Bandpass filter
        filtered = bandpass_filter(audio, sample_rate, self.low_freq, self.high_freq)

        # Compute energy envelope using RMS
        frame_length = hop_length * 2
        energy = librosa.feature.rms(
            y=filtered,
            frame_length=frame_length,
            hop_length=hop_length,
        )[0]

        # Smooth energy
        if self.smoothing_window > 0:
            smooth_samples = max(1, int(self.smoothing_window * sample_rate / hop_length))
            kernel = np.ones(smooth_samples) / smooth_samples
            energy = np.convolve(energy, kernel, mode="same")

        # Time axis
        times = librosa.frames_to_time(
            np.arange(len(energy)),
            sr=sample_rate,
            hop_length=hop_length,
        )

        return energy, times

    def detect(
        self,
        audio: np.ndarray,
        sample_rate: int,
        hop_length: int = 256,
    ) -> List[Detection]:
        """
        Detect sounds in the specified frequency band.

        Args:
            audio: Audio signal as numpy array.
            sample_rate: Sample rate in Hz.
            hop_length: Hop length for analysis.

        Returns:
            List of Detection objects.
        """
        # Ensure mono
        if audio.ndim > 1:
            audio = audio.mean(axis=0)

        energy, times = self.compute_energy(audio, sample_rate, hop_length)

        if len(energy) == 0 or np.max(energy) == 0:
            return []

        # Convert to dB
        energy_db = librosa.amplitude_to_db(energy, ref=np.max)

        # Find regions above threshold
        above_threshold = energy_db > self.threshold_db

        # Find contiguous regions
        labeled, num_features = ndimage.label(above_threshold)

        detections = []
        time_step = times[1] - times[0] if len(times) > 1 else hop_length / sample_rate

        for region_id in range(1, num_features + 1):
            region_indices = np.where(labeled == region_id)[0]

            if len(region_indices) == 0:
                continue

            start_idx = region_indices[0]
            end_idx = region_indices[-1]

            start_time = times[start_idx]
            end_time = times[end_idx] + time_step

            # Check minimum duration
            if end_time - start_time < self.min_duration:
                continue

            # Calculate confidence as normalized energy
            region_energy = energy[region_indices]
            confidence = float(np.mean(region_energy) / np.max(energy))

            detections.append(
                Detection(
                    start_time=float(start_time),
                    end_time=float(end_time),
                    confidence=confidence,
                    frequency_low=self.low_freq,
                    frequency_high=self.high_freq,
                    label="band_energy",
                )
            )

        # Merge nearby detections
        detections = self._merge_detections(detections)

        return detections

    def _merge_detections(self, detections: List[Detection]) -> List[Detection]:
        """Merge detections that are close together."""
        if len(detections) <= 1:
            return detections

        # Sort by start time
        detections = sorted(detections, key=lambda d: d.start_time)

        merged = []
        current = detections[0]

        for det in detections[1:]:
            if det.start_time - current.end_time <= self.merge_threshold:
                # Merge
                current = Detection(
                    start_time=current.start_time,
                    end_time=max(current.end_time, det.end_time),
                    confidence=max(current.confidence, det.confidence),
                    frequency_low=current.frequency_low,
                    frequency_high=current.frequency_high,
                    label=current.label,
                )
            else:
                merged.append(current)
                current = det

        merged.append(current)
        return merged

    def detect_from_file(
        self,
        filepath: Union[str, Path],
        **kwargs,
    ) -> List[Detection]:
        """
        Detect sounds from an audio file.

        Args:
            filepath: Path to audio file.
            **kwargs: Additional arguments for detect().

        Returns:
            List of Detection objects.
        """
        from bioamla.adapters.pydub import load_audio

        audio, sample_rate = load_audio(str(filepath))
        return self.detect(audio, sample_rate, **kwargs)


# =============================================================================
# RIBBIT Periodic Call Detector
# =============================================================================


class RibbitDetector:
    """
    RIBBIT (Repeat-Interval-Based Bioacoustic Identification Tool) detector.

    Detects periodic vocalizations by analyzing the autocorrelation of
    the spectrogram at different pulse rates. Particularly effective for
    detecting frog calls, insect sounds, and other repetitive vocalizations.

    Based on the opensoundscape RIBBIT algorithm.

    Args:
        pulse_rate_hz: Expected pulse rate in Hz (pulses per second).
        pulse_rate_tolerance: Tolerance around expected rate (fraction).
        low_freq: Lower frequency bound in Hz.
        high_freq: Upper frequency bound in Hz.
        window_duration: Analysis window duration in seconds.
        hop_duration: Hop between analysis windows in seconds.
        min_score: Minimum detection score threshold.
        n_fft: FFT size for spectrogram.

    Example:
        >>> # Detect frog calls with ~10 pulses per second
        >>> detector = RibbitDetector(
        ...     pulse_rate_hz=10.0,
        ...     low_freq=500,
        ...     high_freq=3000,
        ... )
        >>> detections = detector.detect(audio, sample_rate)
    """

    def __init__(
        self,
        pulse_rate_hz: float = 10.0,
        pulse_rate_tolerance: float = 0.2,
        low_freq: float = 500.0,
        high_freq: float = 5000.0,
        window_duration: float = 2.0,
        hop_duration: float = 0.5,
        min_score: float = 0.3,
        n_fft: int = 1024,
    ):
        self.pulse_rate_hz = pulse_rate_hz
        self.pulse_rate_tolerance = pulse_rate_tolerance
        self.low_freq = low_freq
        self.high_freq = high_freq
        self.window_duration = window_duration
        self.hop_duration = hop_duration
        self.min_score = min_score
        self.n_fft = n_fft

    def compute_pulse_score(
        self,
        audio: np.ndarray,
        sample_rate: int,
    ) -> float:
        """
        Compute RIBBIT pulse score for an audio segment.

        The score measures how well the audio matches the expected
        periodic pulse rate by analyzing spectrogram autocorrelation.

        Args:
            audio: Audio segment.
            sample_rate: Sample rate in Hz.

        Returns:
            Pulse score (0-1).
        """
        # Bandpass filter
        filtered = bandpass_filter(audio, sample_rate, self.low_freq, self.high_freq)

        # Compute spectrogram
        hop_length = self.n_fft // 4
        spec = np.abs(
            librosa.stft(
                filtered,
                n_fft=self.n_fft,
                hop_length=hop_length,
            )
        )

        # Get frequency bins within band
        frequencies = librosa.fft_frequencies(sr=sample_rate, n_fft=self.n_fft)
        freq_mask = (frequencies >= self.low_freq) & (frequencies <= self.high_freq)

        if not np.any(freq_mask):
            return 0.0

        # Sum energy across frequency band
        band_energy = np.sum(spec[freq_mask, :], axis=0)

        if len(band_energy) < 10:
            return 0.0

        # Normalize
        band_energy = band_energy - np.mean(band_energy)
        if np.std(band_energy) > 0:
            band_energy = band_energy / np.std(band_energy)

        # Compute autocorrelation
        autocorr = np.correlate(band_energy, band_energy, mode="full")
        autocorr = autocorr[len(autocorr) // 2 :]  # Take positive lags

        # Convert lag to time
        time_per_frame = hop_length / sample_rate
        np.arange(len(autocorr)) * time_per_frame

        # Expected period and tolerance
        expected_period = 1.0 / self.pulse_rate_hz
        period_tolerance = expected_period * self.pulse_rate_tolerance

        # Find peaks in autocorrelation at expected period
        period_min = expected_period - period_tolerance
        period_max = expected_period + period_tolerance

        # Find lag range
        lag_min = int(period_min / time_per_frame)
        lag_max = int(period_max / time_per_frame)

        if lag_max >= len(autocorr):
            lag_max = len(autocorr) - 1
        if lag_min >= lag_max:
            return 0.0

        # Find max autocorrelation in expected range
        search_range = autocorr[lag_min : lag_max + 1]
        if len(search_range) == 0:
            return 0.0

        # Score is the normalized autocorrelation at expected period
        max_in_range = np.max(search_range)
        score = max_in_range / (autocorr[0] + 1e-10)  # Normalize by zero-lag

        # Also check harmonics (2x, 3x the period)
        harmonic_scores = []
        for harmonic in [2, 3]:
            h_period = expected_period * harmonic
            h_lag_min = int((h_period - period_tolerance) / time_per_frame)
            h_lag_max = int((h_period + period_tolerance) / time_per_frame)

            if h_lag_max < len(autocorr) and h_lag_min < h_lag_max:
                h_range = autocorr[h_lag_min : h_lag_max + 1]
                if len(h_range) > 0:
                    h_score = np.max(h_range) / (autocorr[0] + 1e-10)
                    harmonic_scores.append(h_score)

        # Boost score if harmonics are present
        if harmonic_scores:
            score = score + 0.3 * np.mean(harmonic_scores)

        return float(np.clip(score, 0, 1))

    def detect(
        self,
        audio: np.ndarray,
        sample_rate: int,
    ) -> List[Detection]:
        """
        Detect periodic calls using RIBBIT algorithm.

        Args:
            audio: Audio signal as numpy array.
            sample_rate: Sample rate in Hz.

        Returns:
            List of Detection objects.
        """
        # Ensure mono
        if audio.ndim > 1:
            audio = audio.mean(axis=0)

        len(audio) / sample_rate
        window_samples = int(self.window_duration * sample_rate)
        hop_samples = int(self.hop_duration * sample_rate)

        detections = []
        position = 0

        while position + window_samples <= len(audio):
            window = audio[position : position + window_samples]
            score = self.compute_pulse_score(window, sample_rate)

            if score >= self.min_score:
                start_time = position / sample_rate
                end_time = (position + window_samples) / sample_rate

                detections.append(
                    Detection(
                        start_time=float(start_time),
                        end_time=float(end_time),
                        confidence=float(score),
                        frequency_low=self.low_freq,
                        frequency_high=self.high_freq,
                        label="ribbit",
                        metadata={
                            "pulse_rate_hz": self.pulse_rate_hz,
                        },
                    )
                )

            position += hop_samples

        # Merge overlapping detections
        detections = self._merge_overlapping(detections)

        return detections

    def _merge_overlapping(self, detections: List[Detection]) -> List[Detection]:
        """Merge overlapping detections."""
        if len(detections) <= 1:
            return detections

        detections = sorted(detections, key=lambda d: d.start_time)
        merged = []
        current = detections[0]

        for det in detections[1:]:
            if det.start_time <= current.end_time:
                # Merge overlapping
                current = Detection(
                    start_time=current.start_time,
                    end_time=max(current.end_time, det.end_time),
                    confidence=max(current.confidence, det.confidence),
                    frequency_low=current.frequency_low,
                    frequency_high=current.frequency_high,
                    label=current.label,
                    metadata=current.metadata,
                )
            else:
                merged.append(current)
                current = det

        merged.append(current)
        return merged

    def compute_temporal_scores(
        self,
        audio: np.ndarray,
        sample_rate: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute RIBBIT scores over time.

        Args:
            audio: Audio signal.
            sample_rate: Sample rate in Hz.

        Returns:
            Tuple of (scores array, time axis).
        """
        if audio.ndim > 1:
            audio = audio.mean(axis=0)

        window_samples = int(self.window_duration * sample_rate)
        hop_samples = int(self.hop_duration * sample_rate)

        scores = []
        times = []
        position = 0

        while position + window_samples <= len(audio):
            window = audio[position : position + window_samples]
            score = self.compute_pulse_score(window, sample_rate)

            center_time = (position + window_samples / 2) / sample_rate
            scores.append(score)
            times.append(center_time)

            position += hop_samples

        return np.array(scores), np.array(times)

    def detect_from_file(
        self,
        filepath: Union[str, Path],
    ) -> List[Detection]:
        """Detect periodic calls from an audio file."""
        from bioamla.adapters.pydub import load_audio

        audio, sample_rate = load_audio(str(filepath))
        return self.detect(audio, sample_rate)


# =============================================================================
# CWT Peak Detector
# =============================================================================


class CWTPeakDetector:
    """
    Peak sequence detector using Continuous Wavelet Transform (CWT).

    Uses CWT to identify peaks in the audio energy envelope at multiple
    scales, providing robust peak detection that's less sensitive to
    noise and baseline drift.

    Args:
        min_scale: Minimum wavelet scale.
        max_scale: Maximum wavelet scale.
        n_scales: Number of scales to analyze.
        snr_threshold: Signal-to-noise ratio threshold for peaks.
        min_peak_distance: Minimum distance between peaks in seconds.
        low_freq: Optional frequency band lower bound.
        high_freq: Optional frequency band upper bound.

    Example:
        >>> detector = CWTPeakDetector(snr_threshold=3.0)
        >>> peaks = detector.detect(audio, sample_rate)
        >>> for p in peaks:
        ...     print(f"Peak at {p.time:.3f}s, amplitude: {p.amplitude:.2f}")
    """

    def __init__(
        self,
        min_scale: int = 1,
        max_scale: int = 50,
        n_scales: int = 20,
        snr_threshold: float = 2.0,
        min_peak_distance: float = 0.01,
        low_freq: Optional[float] = None,
        high_freq: Optional[float] = None,
    ):
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.n_scales = n_scales
        self.snr_threshold = snr_threshold
        self.min_peak_distance = min_peak_distance
        self.low_freq = low_freq
        self.high_freq = high_freq

    def compute_cwt(
        self,
        signal: np.ndarray,
        sample_rate: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute Continuous Wavelet Transform using convolution.

        Args:
            signal: Input signal.
            sample_rate: Sample rate in Hz.

        Returns:
            Tuple of (CWT coefficients, scales).
        """
        # Create log-spaced scales
        scales = np.geomspace(self.min_scale, self.max_scale, self.n_scales)

        # Compute CWT using convolution with Ricker (Mexican hat) wavelet
        # This replaces the deprecated scipy.signal.cwt
        cwt_matrix = np.zeros((len(scales), len(signal)))

        for i, scale in enumerate(scales):
            # Create Ricker wavelet at this scale
            width = int(10 * scale)
            if width < 1:
                width = 1
            t = np.arange(width) - (width - 1) / 2
            norm = 2 / (np.sqrt(3 * scale) * np.pi**0.25)
            wavelet = norm * (1 - (t / scale) ** 2) * np.exp(-(t**2) / (2 * scale**2))

            # Convolve signal with wavelet
            if len(signal) >= len(wavelet):
                cwt_matrix[i, :] = np.convolve(signal, wavelet, mode="same")
            else:
                # Handle case where signal is shorter than wavelet
                cwt_matrix[i, :] = np.convolve(signal, wavelet[: len(signal)], mode="same")

        return np.abs(cwt_matrix), scales

    def find_ridge_peaks(
        self,
        cwt_matrix: np.ndarray,
        sample_rate: int,
    ) -> List[PeakDetection]:
        """
        Find peaks along ridges in CWT matrix.

        Args:
            cwt_matrix: CWT coefficient matrix.
            sample_rate: Sample rate in Hz.

        Returns:
            List of PeakDetection objects.
        """
        # Sum across scales for overall peak detection
        ridge_line = np.sum(cwt_matrix, axis=0)

        # Estimate noise level
        noise_level = np.median(ridge_line)
        signal_std = np.std(ridge_line)

        # Threshold
        threshold = noise_level + self.snr_threshold * signal_std

        # Find peaks
        min_distance_samples = int(self.min_peak_distance * sample_rate)
        min_distance_samples = max(1, min_distance_samples)

        peak_indices, properties = find_peaks(
            ridge_line,
            height=threshold,
            distance=min_distance_samples,
            prominence=signal_std * 0.5,
            width=1,
        )

        peaks = []
        for i, idx in enumerate(peak_indices):
            time = idx / sample_rate
            amplitude = float(ridge_line[idx])

            # Get width from properties
            width = 0.0
            if "widths" in properties and i < len(properties["widths"]):
                width = float(properties["widths"][i]) / sample_rate

            # Get prominence
            prominence = 0.0
            if "prominences" in properties and i < len(properties["prominences"]):
                prominence = float(properties["prominences"][i])

            peaks.append(
                PeakDetection(
                    time=time,
                    amplitude=amplitude,
                    width=width,
                    prominence=prominence,
                )
            )

        return peaks

    def detect(
        self,
        audio: np.ndarray,
        sample_rate: int,
        hop_length: int = 256,
    ) -> List[PeakDetection]:
        """
        Detect peaks using CWT analysis.

        Args:
            audio: Audio signal as numpy array.
            sample_rate: Sample rate in Hz.
            hop_length: Hop length for envelope computation.

        Returns:
            List of PeakDetection objects.
        """
        # Ensure mono
        if audio.ndim > 1:
            audio = audio.mean(axis=0)

        # Apply frequency band filter if specified
        if self.low_freq is not None and self.high_freq is not None:
            audio = bandpass_filter(audio, sample_rate, self.low_freq, self.high_freq)

        # Compute energy envelope
        envelope = librosa.feature.rms(
            y=audio,
            frame_length=hop_length * 2,
            hop_length=hop_length,
        )[0]

        # Effective sample rate for envelope
        envelope_sr = sample_rate / hop_length

        # Apply CWT
        cwt_matrix, scales = self.compute_cwt(envelope, int(envelope_sr))

        # Find peaks
        peaks = self.find_ridge_peaks(cwt_matrix, int(envelope_sr))

        # Adjust times for hop length
        for peak in peaks:
            peak.time *= (hop_length / sample_rate) * envelope_sr

        return peaks

    def detect_sequences(
        self,
        audio: np.ndarray,
        sample_rate: int,
        min_peaks: int = 3,
        max_gap: float = 1.0,
    ) -> List[Detection]:
        """
        Detect sequences of peaks as detections.

        Args:
            audio: Audio signal.
            sample_rate: Sample rate in Hz.
            min_peaks: Minimum peaks to form a sequence.
            max_gap: Maximum gap between peaks in a sequence.

        Returns:
            List of Detection objects.
        """
        peaks = self.detect(audio, sample_rate)

        if len(peaks) < min_peaks:
            return []

        # Sort by time
        peaks = sorted(peaks, key=lambda p: p.time)

        # Group peaks into sequences
        sequences = []
        current_sequence = [peaks[0]]

        for peak in peaks[1:]:
            if peak.time - current_sequence[-1].time <= max_gap:
                current_sequence.append(peak)
            else:
                if len(current_sequence) >= min_peaks:
                    sequences.append(current_sequence)
                current_sequence = [peak]

        if len(current_sequence) >= min_peaks:
            sequences.append(current_sequence)

        # Convert sequences to detections
        detections = []
        for seq in sequences:
            start_time = seq[0].time
            end_time = seq[-1].time
            avg_confidence = np.mean([p.amplitude for p in seq])

            detections.append(
                Detection(
                    start_time=float(start_time),
                    end_time=float(end_time),
                    confidence=float(
                        avg_confidence / (np.max([p.amplitude for p in peaks]) + 1e-10)
                    ),
                    frequency_low=self.low_freq,
                    frequency_high=self.high_freq,
                    label="cwt_sequence",
                    metadata={
                        "n_peaks": len(seq),
                        "mean_interval": np.mean(np.diff([p.time for p in seq]))
                        if len(seq) > 1
                        else 0,
                    },
                )
            )

        return detections

    def detect_from_file(
        self,
        filepath: Union[str, Path],
        **kwargs,
    ) -> List[PeakDetection]:
        """Detect peaks from an audio file."""
        from bioamla.adapters.pydub import load_audio

        audio, sample_rate = load_audio(str(filepath))
        return self.detect(audio, sample_rate, **kwargs)


# =============================================================================
# Accelerating Pattern Detector
# =============================================================================


class AcceleratingPatternDetector:
    """
    Detector for accelerating call patterns.

    Many species produce vocalizations with increasing or decreasing
    pulse rates (e.g., tree frog advertisement calls that speed up).
    This detector identifies such patterns by analyzing inter-pulse
    intervals.

    Args:
        min_pulses: Minimum number of pulses to detect pattern.
        acceleration_threshold: Minimum acceleration ratio (final_rate/initial_rate).
        low_freq: Lower frequency bound in Hz.
        high_freq: Upper frequency bound in Hz.
        min_pulse_rate: Minimum expected pulse rate in Hz.
        max_pulse_rate: Maximum expected pulse rate in Hz.
        window_duration: Analysis window duration in seconds.

    Example:
        >>> # Detect calls that accelerate from 5 to 15+ pulses/sec
        >>> detector = AcceleratingPatternDetector(
        ...     min_pulses=5,
        ...     acceleration_threshold=2.0,  # 2x speedup
        ...     low_freq=1000,
        ...     high_freq=4000,
        ... )
        >>> detections = detector.detect(audio, sample_rate)
    """

    def __init__(
        self,
        min_pulses: int = 5,
        acceleration_threshold: float = 1.5,
        deceleration_threshold: Optional[float] = None,
        low_freq: float = 500.0,
        high_freq: float = 5000.0,
        min_pulse_rate: float = 2.0,
        max_pulse_rate: float = 50.0,
        window_duration: float = 3.0,
        hop_duration: float = 0.5,
    ):
        self.min_pulses = min_pulses
        self.acceleration_threshold = acceleration_threshold
        self.deceleration_threshold = deceleration_threshold
        self.low_freq = low_freq
        self.high_freq = high_freq
        self.min_pulse_rate = min_pulse_rate
        self.max_pulse_rate = max_pulse_rate
        self.window_duration = window_duration
        self.hop_duration = hop_duration

        # Use CWT peak detector for finding pulses
        self._peak_detector = CWTPeakDetector(
            snr_threshold=2.0,
            min_peak_distance=1.0 / max_pulse_rate,
            low_freq=low_freq,
            high_freq=high_freq,
        )

    def analyze_intervals(
        self,
        peak_times: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Analyze inter-peak intervals for acceleration pattern.

        Args:
            peak_times: Array of peak times in seconds.

        Returns:
            Dictionary with interval analysis results.
        """
        if len(peak_times) < self.min_pulses:
            return {
                "is_accelerating": False,
                "is_decelerating": False,
                "acceleration_ratio": 1.0,
                "initial_rate": 0.0,
                "final_rate": 0.0,
                "n_pulses": len(peak_times),
            }

        # Compute intervals
        intervals = np.diff(peak_times)

        if len(intervals) < 2:
            return {
                "is_accelerating": False,
                "is_decelerating": False,
                "acceleration_ratio": 1.0,
                "initial_rate": 0.0,
                "final_rate": 0.0,
                "n_pulses": len(peak_times),
            }

        # Filter out unrealistic intervals
        min_interval = 1.0 / self.max_pulse_rate
        max_interval = 1.0 / self.min_pulse_rate
        valid_mask = (intervals >= min_interval) & (intervals <= max_interval)
        valid_intervals = intervals[valid_mask]

        if len(valid_intervals) < 2:
            return {
                "is_accelerating": False,
                "is_decelerating": False,
                "acceleration_ratio": 1.0,
                "initial_rate": 0.0,
                "final_rate": 0.0,
                "n_pulses": len(peak_times),
            }

        # Estimate initial and final rates
        n_samples = min(3, len(valid_intervals) // 2)
        initial_intervals = valid_intervals[:n_samples]
        final_intervals = valid_intervals[-n_samples:]

        initial_rate = 1.0 / np.mean(initial_intervals)
        final_rate = 1.0 / np.mean(final_intervals)

        acceleration_ratio = final_rate / initial_rate if initial_rate > 0 else 1.0

        # Check for acceleration
        is_accelerating = acceleration_ratio >= self.acceleration_threshold

        # Check for deceleration
        is_decelerating = False
        if self.deceleration_threshold is not None:
            is_decelerating = acceleration_ratio <= (1.0 / self.deceleration_threshold)

        return {
            "is_accelerating": is_accelerating,
            "is_decelerating": is_decelerating,
            "acceleration_ratio": float(acceleration_ratio),
            "initial_rate": float(initial_rate),
            "final_rate": float(final_rate),
            "n_pulses": len(peak_times),
            "intervals": valid_intervals.tolist(),
        }

    def detect(
        self,
        audio: np.ndarray,
        sample_rate: int,
    ) -> List[Detection]:
        """
        Detect accelerating/decelerating call patterns.

        Args:
            audio: Audio signal as numpy array.
            sample_rate: Sample rate in Hz.

        Returns:
            List of Detection objects.
        """
        # Ensure mono
        if audio.ndim > 1:
            audio = audio.mean(axis=0)

        len(audio) / sample_rate
        window_samples = int(self.window_duration * sample_rate)
        hop_samples = int(self.hop_duration * sample_rate)

        detections = []
        position = 0

        while position + window_samples <= len(audio):
            window = audio[position : position + window_samples]
            window_start = position / sample_rate

            # Detect peaks in window
            peaks = self._peak_detector.detect(window, sample_rate)

            if len(peaks) >= self.min_pulses:
                peak_times = np.array([p.time for p in peaks])

                # Analyze intervals
                analysis = self.analyze_intervals(peak_times)

                if analysis["is_accelerating"] or analysis["is_decelerating"]:
                    pattern_type = "accelerating" if analysis["is_accelerating"] else "decelerating"

                    # Calculate confidence based on number of pulses and consistency
                    confidence = min(1.0, analysis["n_pulses"] / (self.min_pulses * 2))

                    detections.append(
                        Detection(
                            start_time=float(window_start),
                            end_time=float(window_start + self.window_duration),
                            confidence=confidence,
                            frequency_low=self.low_freq,
                            frequency_high=self.high_freq,
                            label=f"pattern_{pattern_type}",
                            metadata={
                                "pattern_type": pattern_type,
                                "acceleration_ratio": analysis["acceleration_ratio"],
                                "initial_rate": analysis["initial_rate"],
                                "final_rate": analysis["final_rate"],
                                "n_pulses": analysis["n_pulses"],
                            },
                        )
                    )

            position += hop_samples

        # Merge overlapping detections
        detections = self._merge_overlapping(detections)

        return detections

    def _merge_overlapping(self, detections: List[Detection]) -> List[Detection]:
        """Merge overlapping detections of the same type."""
        if len(detections) <= 1:
            return detections

        detections = sorted(detections, key=lambda d: d.start_time)
        merged = []
        current = detections[0]

        for det in detections[1:]:
            if det.start_time <= current.end_time and det.label == current.label:
                # Merge overlapping same-type detections
                current = Detection(
                    start_time=current.start_time,
                    end_time=max(current.end_time, det.end_time),
                    confidence=max(current.confidence, det.confidence),
                    frequency_low=current.frequency_low,
                    frequency_high=current.frequency_high,
                    label=current.label,
                    metadata={
                        **current.metadata,
                        "acceleration_ratio": max(
                            current.metadata.get("acceleration_ratio", 1),
                            det.metadata.get("acceleration_ratio", 1),
                        ),
                    },
                )
            else:
                merged.append(current)
                current = det

        merged.append(current)
        return merged

    def detect_from_file(
        self,
        filepath: Union[str, Path],
    ) -> List[Detection]:
        """Detect accelerating patterns from an audio file."""
        from bioamla.adapters.pydub import load_audio

        audio, sample_rate = load_audio(str(filepath))
        return self.detect(audio, sample_rate)


# =============================================================================
# Utility Functions
# =============================================================================


def detect_all(
    audio: np.ndarray,
    sample_rate: int,
    detectors: List[
        Union[
            BandLimitedEnergyDetector, RibbitDetector, CWTPeakDetector, AcceleratingPatternDetector
        ]
    ],
) -> List[Detection]:
    """
    Run multiple detectors and combine results.

    Args:
        audio: Audio signal.
        sample_rate: Sample rate in Hz.
        detectors: List of detector instances.

    Returns:
        Combined list of detections.
    """
    all_detections = []

    for detector in detectors:
        if isinstance(detector, CWTPeakDetector):
            # Convert peaks to detections
            peaks = detector.detect(audio, sample_rate)
            for peak in peaks:
                all_detections.append(
                    Detection(
                        start_time=max(0, peak.time - peak.width / 2),
                        end_time=peak.time + peak.width / 2,
                        confidence=float(
                            peak.prominence / (np.max([p.prominence for p in peaks]) + 1e-10)
                        )
                        if peaks
                        else 0,
                        label="cwt_peak",
                    )
                )
        else:
            detections = detector.detect(audio, sample_rate)
            all_detections.extend(detections)

    # Sort by start time
    all_detections.sort(key=lambda d: d.start_time)

    return all_detections


def export_detections(
    detections: List[Detection],
    output_path: Union[str, Path],
    format: str = "csv",
) -> Path:
    """
    Export detections to file.

    Args:
        detections: List of Detection objects.
        output_path: Output file path.
        format: Output format ('csv' or 'json').

    Returns:
        Path to output file.
    """
    import csv
    import json

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if format == "json":
        with TextFile(output_path, mode="w") as f:
            json.dump([d.to_dict() for d in detections], f.handle, indent=2)
    else:
        if detections:
            fieldnames = list(detections[0].to_dict().keys())
            with TextFile(output_path, mode="w", newline="") as f:
                writer = csv.DictWriter(f.handle, fieldnames=fieldnames)
                writer.writeheader()
                for d in detections:
                    writer.writerow(d.to_dict())
        else:
            # Write empty file with header
            with TextFile(output_path, mode="w", newline="") as f:
                f.write(
                    "start_time,end_time,confidence,duration,frequency_low,frequency_high,label\n"
                )

    return output_path


def batch_detect(
    filepaths: List[Union[str, Path]],
    detector: Union[
        BandLimitedEnergyDetector, RibbitDetector, CWTPeakDetector, AcceleratingPatternDetector
    ],
    verbose: bool = True,
) -> Dict[str, List[Detection]]:
    """
    Run detection on multiple files.

    Args:
        filepaths: List of audio file paths.
        detector: Detector instance to use.
        verbose: Print progress.

    Returns:
        Dictionary mapping filepath to list of detections.
    """
    results = {}

    for i, filepath in enumerate(filepaths, 1):
        if verbose:
            print(f"[{i}/{len(filepaths)}] Processing {filepath}")

        try:
            if hasattr(detector, "detect_from_file"):
                detections = detector.detect_from_file(filepath)
            else:
                from bioamla.adapters.pydub import load_audio

                audio, sr = load_audio(str(filepath))
                detections = detector.detect(audio, sr)

            results[str(filepath)] = detections

            if verbose:
                print(f"  Found {len(detections)} detections")

        except Exception as e:
            logger.error(f"Error processing {filepath}: {e}")
            results[str(filepath)] = []

    return results
