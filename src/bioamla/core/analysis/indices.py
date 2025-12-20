"""
Acoustic Indices
================

Calculate acoustic indices commonly used in ecoacoustics and soundscape ecology
for characterizing biodiversity and anthropogenic impacts on soundscapes.

Implemented indices:
- ACI: Acoustic Complexity Index
- ADI: Acoustic Diversity Index
- AEI: Acoustic Evenness Index
- BIO: Bioacoustic Index
- NDSI: Normalized Difference Soundscape Index

References:
- Pieretti et al. (2011) - ACI
- Villanueva-Rivera et al. (2011) - ADI, AEI
- Boelman et al. (2007) - BIO
- Kasten et al. (2012) - NDSI

Example:
    >>> from bioamla.indices import compute_aci, compute_ndsi, compute_all_indices
    >>>
    >>> # Compute single index
    >>> aci = compute_aci(audio, sample_rate)
    >>>
    >>> # Compute all indices at once
    >>> indices = compute_all_indices(audio, sample_rate)
    >>> print(f"NDSI: {indices['ndsi']:.3f}")
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import librosa
import numpy as np
from scipy import signal as scipy_signal

import logging

logger = logging.getLogger(__name__)


@dataclass
class AcousticIndices:
    """
    Container for all acoustic indices computed from an audio signal.

    Attributes:
        aci: Acoustic Complexity Index
        adi: Acoustic Diversity Index
        aei: Acoustic Evenness Index
        bio: Bioacoustic Index
        ndsi: Normalized Difference Soundscape Index
        anthrophony: Anthrophony component (1-2 kHz)
        biophony: Biophony component (2-8 kHz)
        sample_rate: Sample rate used for computation
        duration: Duration of audio in seconds
    """

    aci: float
    adi: float
    aei: float
    bio: float
    ndsi: float
    anthrophony: float = 0.0
    biophony: float = 0.0
    sample_rate: int = 0
    duration: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "aci": self.aci,
            "adi": self.adi,
            "aei": self.aei,
            "bio": self.bio,
            "ndsi": self.ndsi,
            "anthrophony": self.anthrophony,
            "biophony": self.biophony,
            "sample_rate": self.sample_rate,
            "duration": self.duration,
        }


def _compute_spectrogram(
    audio: np.ndarray,
    sample_rate: int,
    n_fft: int = 512,
    hop_length: Optional[int] = None,
    window: str = "hann",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute spectrogram for acoustic index calculations.

    Args:
        audio: Audio signal as numpy array.
        sample_rate: Sample rate in Hz.
        n_fft: FFT window size.
        hop_length: Hop length (default: n_fft // 2).
        window: Window function.

    Returns:
        Tuple of (spectrogram, frequencies, times).
    """
    if hop_length is None:
        hop_length = n_fft // 2

    # Compute STFT
    stft = librosa.stft(
        audio,
        n_fft=n_fft,
        hop_length=hop_length,
        window=window,
        center=True,
    )

    # Get magnitude spectrogram
    spectrogram = np.abs(stft)

    # Get frequency bins
    frequencies = librosa.fft_frequencies(sr=sample_rate, n_fft=n_fft)

    # Get time bins
    times = librosa.frames_to_time(
        np.arange(spectrogram.shape[1]),
        sr=sample_rate,
        hop_length=hop_length,
    )

    return spectrogram, frequencies, times


def _get_frequency_band_indices(
    frequencies: np.ndarray,
    low_freq: float,
    high_freq: float,
) -> np.ndarray:
    """Get indices of frequency bins within a band."""
    return np.where((frequencies >= low_freq) & (frequencies <= high_freq))[0]


def compute_aci(
    audio: np.ndarray,
    sample_rate: int,
    n_fft: int = 512,
    hop_length: Optional[int] = None,
    min_freq: float = 0.0,
    max_freq: Optional[float] = None,
    j: int = 5,
) -> float:
    """
    Compute the Acoustic Complexity Index (ACI).

    ACI measures the variability of sound intensities within frequency bands
    over time. Higher values indicate more complex acoustic environments,
    often associated with higher biodiversity.

    The index is computed by calculating the absolute difference between
    adjacent amplitude values in each frequency bin, then summing across
    time and dividing by the total amplitude.

    Reference:
        Pieretti, N., Farina, A., & Morri, D. (2011). A new methodology to
        infer the singing activity of an avian community: The Acoustic
        Complexity Index (ACI). Ecological Indicators, 11(3), 868-873.

    Args:
        audio: Audio signal as numpy array.
        sample_rate: Sample rate in Hz.
        n_fft: FFT window size (default: 512).
        hop_length: Hop length (default: n_fft // 2).
        min_freq: Minimum frequency to consider in Hz.
        max_freq: Maximum frequency to consider in Hz (default: Nyquist).
        j: Number of temporal steps to cluster (default: 5).

    Returns:
        Acoustic Complexity Index value.

    Example:
        >>> aci = compute_aci(audio, 22050, min_freq=2000, max_freq=8000)
    """
    if max_freq is None:
        max_freq = sample_rate / 2

    spectrogram, frequencies, _ = _compute_spectrogram(
        audio, sample_rate, n_fft, hop_length
    )

    # Get frequency band indices
    freq_indices = _get_frequency_band_indices(frequencies, min_freq, max_freq)

    if len(freq_indices) == 0:
        return 0.0

    # Filter spectrogram to frequency band
    spec_band = spectrogram[freq_indices, :]

    # Number of temporal frames
    n_frames = spec_band.shape[1]

    if n_frames < 2:
        return 0.0

    # Calculate ACI for each frequency bin
    aci_values = []

    # Process in clusters of j frames
    n_clusters = n_frames // j

    for cluster_idx in range(n_clusters):
        start_idx = cluster_idx * j
        end_idx = start_idx + j

        cluster = spec_band[:, start_idx:end_idx]

        # Calculate absolute differences for each frequency bin
        for freq_idx in range(cluster.shape[0]):
            freq_values = cluster[freq_idx, :]

            # Sum of absolute differences
            d = np.sum(np.abs(np.diff(freq_values)))

            # Total intensity
            total = np.sum(freq_values)

            if total > 0:
                aci_values.append(d / total)

    if not aci_values:
        return 0.0

    return float(np.sum(aci_values))


def compute_adi(
    audio: np.ndarray,
    sample_rate: int,
    n_fft: int = 512,
    hop_length: Optional[int] = None,
    max_freq: float = 10000.0,
    freq_step: float = 1000.0,
    db_threshold: float = -50.0,
) -> float:
    """
    Compute the Acoustic Diversity Index (ADI).

    ADI is based on the Shannon diversity index applied to frequency bands.
    It measures the evenness of sound distribution across frequency bands.
    Higher values indicate more evenly distributed acoustic energy across
    frequency bands, suggesting higher acoustic diversity.

    Reference:
        Villanueva-Rivera, L. J., Pijanowski, B. C., Doucette, J., &
        Pekin, B. (2011). A primer of acoustic analysis for landscape
        ecologists. Landscape Ecology, 26(9), 1233-1246.

    Args:
        audio: Audio signal as numpy array.
        sample_rate: Sample rate in Hz.
        n_fft: FFT window size.
        hop_length: Hop length (default: n_fft // 2).
        max_freq: Maximum frequency to analyze in Hz.
        freq_step: Frequency band width in Hz.
        db_threshold: Threshold in dB for considering sound present.

    Returns:
        Acoustic Diversity Index value.

    Example:
        >>> adi = compute_adi(audio, 22050, max_freq=10000, freq_step=1000)
    """
    spectrogram, frequencies, _ = _compute_spectrogram(
        audio, sample_rate, n_fft, hop_length
    )

    # Convert to dB
    spectrogram_db = librosa.amplitude_to_db(spectrogram, ref=np.max)

    # Define frequency bands
    n_bands = int(max_freq / freq_step)
    band_proportions = []

    for i in range(n_bands):
        low_freq = i * freq_step
        high_freq = (i + 1) * freq_step

        freq_indices = _get_frequency_band_indices(frequencies, low_freq, high_freq)

        if len(freq_indices) == 0:
            continue

        # Get band spectrogram
        band_spec = spectrogram_db[freq_indices, :]

        # Calculate proportion of values above threshold
        above_threshold = band_spec > db_threshold
        proportion = np.mean(above_threshold)

        if proportion > 0:
            band_proportions.append(proportion)

    if not band_proportions:
        return 0.0

    # Normalize proportions
    total = np.sum(band_proportions)
    if total == 0:
        return 0.0

    proportions = np.array(band_proportions) / total

    # Calculate Shannon diversity index
    # H' = -sum(p * ln(p))
    adi = -np.sum(proportions * np.log(proportions + 1e-10))

    return float(adi)


def compute_aei(
    audio: np.ndarray,
    sample_rate: int,
    n_fft: int = 512,
    hop_length: Optional[int] = None,
    max_freq: float = 10000.0,
    freq_step: float = 1000.0,
    db_threshold: float = -50.0,
) -> float:
    """
    Compute the Acoustic Evenness Index (AEI).

    AEI is based on the Gini coefficient applied to frequency bands.
    It measures the evenness of sound energy distribution across frequency
    bands. Lower values indicate more even distribution (higher evenness),
    while higher values indicate more uneven distribution.

    Reference:
        Villanueva-Rivera, L. J., Pijanowski, B. C., Doucette, J., &
        Pekin, B. (2011). A primer of acoustic analysis for landscape
        ecologists. Landscape Ecology, 26(9), 1233-1246.

    Args:
        audio: Audio signal as numpy array.
        sample_rate: Sample rate in Hz.
        n_fft: FFT window size.
        hop_length: Hop length (default: n_fft // 2).
        max_freq: Maximum frequency to analyze in Hz.
        freq_step: Frequency band width in Hz.
        db_threshold: Threshold in dB for considering sound present.

    Returns:
        Acoustic Evenness Index value (Gini coefficient).

    Example:
        >>> aei = compute_aei(audio, 22050, max_freq=10000)
    """
    spectrogram, frequencies, _ = _compute_spectrogram(
        audio, sample_rate, n_fft, hop_length
    )

    # Convert to dB
    spectrogram_db = librosa.amplitude_to_db(spectrogram, ref=np.max)

    # Define frequency bands
    n_bands = int(max_freq / freq_step)
    band_proportions = []

    for i in range(n_bands):
        low_freq = i * freq_step
        high_freq = (i + 1) * freq_step

        freq_indices = _get_frequency_band_indices(frequencies, low_freq, high_freq)

        if len(freq_indices) == 0:
            band_proportions.append(0.0)
            continue

        # Get band spectrogram
        band_spec = spectrogram_db[freq_indices, :]

        # Calculate proportion of values above threshold
        above_threshold = band_spec > db_threshold
        proportion = np.mean(above_threshold)
        band_proportions.append(proportion)

    if not band_proportions or sum(band_proportions) == 0:
        return 0.0

    # Calculate Gini coefficient
    proportions = np.array(band_proportions)
    proportions = np.sort(proportions)
    n = len(proportions)

    # Gini coefficient formula
    index = np.arange(1, n + 1)
    gini = (np.sum((2 * index - n - 1) * proportions)) / (n * np.sum(proportions) + 1e-10)

    return float(gini)


def compute_bio(
    audio: np.ndarray,
    sample_rate: int,
    n_fft: int = 512,
    hop_length: Optional[int] = None,
    min_freq: float = 2000.0,
    max_freq: float = 8000.0,
    db_threshold: float = -50.0,
) -> float:
    """
    Compute the Bioacoustic Index (BIO).

    BIO calculates the area under the mean spectrum curve within a specified
    frequency range, typically 2-8 kHz where most bird and insect sounds occur.
    Higher values indicate more acoustic activity in this biologically
    relevant frequency range.

    Reference:
        Boelman, N. T., Asner, G. P., Hart, P. J., & Martin, R. E. (2007).
        Multi-trophic invasion resistance in Hawaii: Bioacoustics, field
        surveys, and airborne remote sensing. Ecological Applications,
        17(8), 2137-2144.

    Args:
        audio: Audio signal as numpy array.
        sample_rate: Sample rate in Hz.
        n_fft: FFT window size.
        hop_length: Hop length (default: n_fft // 2).
        min_freq: Minimum frequency in Hz (default: 2000).
        max_freq: Maximum frequency in Hz (default: 8000).
        db_threshold: Threshold in dB for baseline.

    Returns:
        Bioacoustic Index value.

    Example:
        >>> bio = compute_bio(audio, 22050, min_freq=2000, max_freq=11000)
    """
    spectrogram, frequencies, _ = _compute_spectrogram(
        audio, sample_rate, n_fft, hop_length
    )

    # Get frequency band indices
    freq_indices = _get_frequency_band_indices(frequencies, min_freq, max_freq)

    if len(freq_indices) == 0:
        return 0.0

    # Get band spectrogram
    spec_band = spectrogram[freq_indices, :]

    # Convert to dB
    spec_db = librosa.amplitude_to_db(spec_band, ref=np.max)

    # Calculate mean spectrum
    mean_spectrum = np.mean(spec_db, axis=1)

    # Normalize to minimum value as baseline
    baseline = np.min(mean_spectrum)
    normalized_spectrum = mean_spectrum - baseline

    # Calculate area under curve (sum of values above threshold)
    area = np.sum(normalized_spectrum[normalized_spectrum > 0])

    # Scale by frequency resolution
    freq_resolution = frequencies[1] - frequencies[0] if len(frequencies) > 1 else 1.0
    bio = area * freq_resolution / 1000  # Scale factor for reasonable values

    return float(bio)


def compute_ndsi(
    audio: np.ndarray,
    sample_rate: int,
    n_fft: int = 1024,
    hop_length: Optional[int] = None,
    anthro_min: float = 1000.0,
    anthro_max: float = 2000.0,
    bio_min: float = 2000.0,
    bio_max: float = 8000.0,
) -> Tuple[float, float, float]:
    """
    Compute the Normalized Difference Soundscape Index (NDSI).

    NDSI compares the amount of anthropogenic sound (typically 1-2 kHz) to
    biological sound (typically 2-8 kHz). Values range from -1 to +1, where:
    - +1 indicates pure biophony (biological sounds only)
    - -1 indicates pure anthrophony (human sounds only)
    - 0 indicates equal amounts of both

    Reference:
        Kasten, E. P., Gage, S. H., Fox, J., & Joo, W. (2012). The remote
        environmental assessment laboratory's acoustic library: An archive
        for studying soundscape ecology. Ecological Informatics, 12, 50-67.

    Args:
        audio: Audio signal as numpy array.
        sample_rate: Sample rate in Hz.
        n_fft: FFT window size.
        hop_length: Hop length (default: n_fft // 2).
        anthro_min: Minimum frequency for anthrophony band (Hz).
        anthro_max: Maximum frequency for anthrophony band (Hz).
        bio_min: Minimum frequency for biophony band (Hz).
        bio_max: Maximum frequency for biophony band (Hz).

    Returns:
        Tuple of (NDSI value, anthrophony, biophony).

    Example:
        >>> ndsi, anthro, bio = compute_ndsi(audio, 22050)
        >>> print(f"NDSI: {ndsi:.3f}, Anthrophony: {anthro:.3f}, Biophony: {bio:.3f}")
    """
    spectrogram, frequencies, _ = _compute_spectrogram(
        audio, sample_rate, n_fft, hop_length
    )

    # Get anthrophony frequency indices (typically 1-2 kHz)
    anthro_indices = _get_frequency_band_indices(frequencies, anthro_min, anthro_max)

    # Get biophony frequency indices (typically 2-8 kHz)
    bio_indices = _get_frequency_band_indices(frequencies, bio_min, bio_max)

    if len(anthro_indices) == 0 or len(bio_indices) == 0:
        return 0.0, 0.0, 0.0

    # Calculate energy in each band
    anthrophony = np.sum(spectrogram[anthro_indices, :] ** 2)
    biophony = np.sum(spectrogram[bio_indices, :] ** 2)

    # Calculate NDSI
    total = biophony + anthrophony
    if total == 0:
        return 0.0, 0.0, 0.0

    ndsi = (biophony - anthrophony) / total

    return float(ndsi), float(anthrophony), float(biophony)


def compute_all_indices(
    audio: np.ndarray,
    sample_rate: int,
    n_fft: int = 512,
    hop_length: Optional[int] = None,
    aci_min_freq: float = 0.0,
    aci_max_freq: Optional[float] = None,
    adi_max_freq: float = 10000.0,
    adi_freq_step: float = 1000.0,
    bio_min_freq: float = 2000.0,
    bio_max_freq: float = 8000.0,
    db_threshold: float = -50.0,
) -> AcousticIndices:
    """
    Compute all acoustic indices at once.

    This is more efficient than computing indices individually as the
    spectrogram is only computed once.

    Args:
        audio: Audio signal as numpy array.
        sample_rate: Sample rate in Hz.
        n_fft: FFT window size.
        hop_length: Hop length (default: n_fft // 2).
        aci_min_freq: Minimum frequency for ACI in Hz.
        aci_max_freq: Maximum frequency for ACI in Hz.
        adi_max_freq: Maximum frequency for ADI/AEI in Hz.
        adi_freq_step: Frequency band width for ADI/AEI in Hz.
        bio_min_freq: Minimum frequency for BIO in Hz.
        bio_max_freq: Maximum frequency for BIO in Hz.
        db_threshold: Threshold in dB for ADI/AEI/BIO.

    Returns:
        AcousticIndices dataclass with all computed values.

    Example:
        >>> indices = compute_all_indices(audio, 22050)
        >>> print(f"ACI: {indices.aci:.2f}")
        >>> print(f"ADI: {indices.adi:.2f}")
        >>> print(f"NDSI: {indices.ndsi:.3f}")
    """
    # Ensure audio is mono
    if audio.ndim > 1:
        audio = audio.mean(axis=0) if audio.shape[0] <= 2 else audio[0]

    duration = len(audio) / sample_rate

    # Compute ACI
    aci = compute_aci(
        audio, sample_rate, n_fft, hop_length,
        min_freq=aci_min_freq, max_freq=aci_max_freq
    )

    # Compute ADI
    adi = compute_adi(
        audio, sample_rate, n_fft, hop_length,
        max_freq=adi_max_freq, freq_step=adi_freq_step, db_threshold=db_threshold
    )

    # Compute AEI
    aei = compute_aei(
        audio, sample_rate, n_fft, hop_length,
        max_freq=adi_max_freq, freq_step=adi_freq_step, db_threshold=db_threshold
    )

    # Compute BIO
    bio = compute_bio(
        audio, sample_rate, n_fft, hop_length,
        min_freq=bio_min_freq, max_freq=bio_max_freq, db_threshold=db_threshold
    )

    # Compute NDSI
    ndsi, anthrophony, biophony = compute_ndsi(
        audio, sample_rate, n_fft=1024, hop_length=hop_length
    )

    return AcousticIndices(
        aci=aci,
        adi=adi,
        aei=aei,
        bio=bio,
        ndsi=ndsi,
        anthrophony=anthrophony,
        biophony=biophony,
        sample_rate=sample_rate,
        duration=duration,
    )


def compute_indices_from_file(
    filepath: Union[str, Path],
    **kwargs,
) -> AcousticIndices:
    """
    Compute all acoustic indices from an audio file.

    Args:
        filepath: Path to audio file.
        **kwargs: Additional arguments passed to compute_all_indices.

    Returns:
        AcousticIndices dataclass.

    Example:
        >>> indices = compute_indices_from_file("recording.wav")
        >>> print(indices.to_dict())
    """
    audio, sample_rate = librosa.load(str(filepath), sr=None, mono=True)
    return compute_all_indices(audio, sample_rate, **kwargs)


def batch_compute_indices(
    filepaths: List[Union[str, Path]],
    verbose: bool = True,
    **kwargs,
) -> List[Dict[str, Any]]:
    """
    Compute acoustic indices for multiple audio files.

    Args:
        filepaths: List of paths to audio files.
        verbose: Print progress.
        **kwargs: Additional arguments passed to compute_all_indices.

    Returns:
        List of dictionaries with indices and file information.

    Example:
        >>> files = ["file1.wav", "file2.wav", "file3.wav"]
        >>> results = batch_compute_indices(files)
        >>> for r in results:
        ...     print(f"{r['filepath']}: NDSI={r['ndsi']:.3f}")
    """
    results = []

    for i, filepath in enumerate(filepaths, 1):
        if verbose:
            print(f"[{i}/{len(filepaths)}] Processing {filepath}")

        try:
            indices = compute_indices_from_file(filepath, **kwargs)
            # Build result with filepath first, success last for better CSV column order
            result = {"filepath": str(filepath)}
            result.update(indices.to_dict())
            result["success"] = True
            results.append(result)
        except Exception as e:
            if verbose:
                print(f"  Error: {e}")
            results.append({
                "filepath": str(filepath),
                "success": False,
                "error": str(e),
            })

    return results


def temporal_indices(
    audio: np.ndarray,
    sample_rate: int,
    window_duration: float = 60.0,
    hop_duration: Optional[float] = None,
    **kwargs,
) -> List[Dict[str, Any]]:
    """
    Compute acoustic indices over time windows.

    Useful for analyzing how soundscape characteristics change over time.

    Args:
        audio: Audio signal as numpy array.
        sample_rate: Sample rate in Hz.
        window_duration: Duration of each window in seconds.
        hop_duration: Hop between windows in seconds (default: window_duration).
        **kwargs: Additional arguments passed to compute_all_indices.

    Returns:
        List of dictionaries with indices for each time window.

    Example:
        >>> # Compute indices every minute for a long recording
        >>> results = temporal_indices(audio, 22050, window_duration=60)
        >>> for r in results:
        ...     print(f"Time {r['start_time']:.0f}s: NDSI={r['ndsi']:.3f}")
    """
    if hop_duration is None:
        hop_duration = window_duration

    # Ensure mono
    if audio.ndim > 1:
        audio = audio.mean(axis=0)

    window_samples = int(window_duration * sample_rate)
    hop_samples = int(hop_duration * sample_rate)
    total_samples = len(audio)

    results = []
    position = 0

    while position + window_samples <= total_samples:
        window = audio[position:position + window_samples]
        start_time = position / sample_rate
        end_time = (position + window_samples) / sample_rate

        indices = compute_all_indices(window, sample_rate, **kwargs)
        result = indices.to_dict()
        result["start_time"] = start_time
        result["end_time"] = end_time
        results.append(result)

        position += hop_samples

    return results


def spectral_entropy(
    audio: np.ndarray,
    sample_rate: int,
    n_fft: int = 512,
    hop_length: Optional[int] = None,
) -> float:
    """
    Compute spectral entropy of the audio signal.

    Spectral entropy measures the uniformity of the power spectrum.
    High entropy indicates a flat (noise-like) spectrum, while low entropy
    indicates concentration of energy in specific frequencies.

    Args:
        audio: Audio signal as numpy array.
        sample_rate: Sample rate in Hz.
        n_fft: FFT window size.
        hop_length: Hop length (default: n_fft // 2).

    Returns:
        Spectral entropy value (0 to log(n_fft/2)).

    Example:
        >>> entropy = spectral_entropy(audio, 22050)
    """
    spectrogram, _, _ = _compute_spectrogram(audio, sample_rate, n_fft, hop_length)

    # Calculate power spectrum
    power_spectrum = np.mean(spectrogram ** 2, axis=1)

    # Normalize to probability distribution
    total_power = np.sum(power_spectrum)
    if total_power == 0:
        return 0.0

    prob = power_spectrum / total_power

    # Calculate entropy
    entropy = -np.sum(prob * np.log2(prob + 1e-10))

    return float(entropy)


def temporal_entropy(
    audio: np.ndarray,
    sample_rate: int,
    n_fft: int = 512,
    hop_length: Optional[int] = None,
) -> float:
    """
    Compute temporal entropy of the audio signal.

    Temporal entropy measures the uniformity of energy distribution over time.
    High entropy indicates constant sound levels, while low entropy indicates
    intermittent or varying sound patterns.

    Args:
        audio: Audio signal as numpy array.
        sample_rate: Sample rate in Hz.
        n_fft: FFT window size.
        hop_length: Hop length (default: n_fft // 2).

    Returns:
        Temporal entropy value.

    Example:
        >>> entropy = temporal_entropy(audio, 22050)
    """
    spectrogram, _, _ = _compute_spectrogram(audio, sample_rate, n_fft, hop_length)

    # Calculate energy envelope over time
    energy = np.sum(spectrogram ** 2, axis=0)

    # Normalize to probability distribution
    total_energy = np.sum(energy)
    if total_energy == 0:
        return 0.0

    prob = energy / total_energy

    # Calculate entropy
    entropy = -np.sum(prob * np.log2(prob + 1e-10))

    return float(entropy)
