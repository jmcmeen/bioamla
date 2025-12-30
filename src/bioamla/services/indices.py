# services/indices.py
"""
Service for acoustic index calculations.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from bioamla.core.indices import (
    AcousticIndices,
    compute_aci,
    compute_adi,
    compute_aei,
    compute_all_indices,
    compute_bio,
    compute_ndsi,
    spectral_entropy,
    temporal_entropy,
    temporal_indices,
)
from bioamla.repository.protocol import FileRepositoryProtocol

from .audio_file import AudioData
from .base import BaseService, ServiceResult

# Available index names for selection
AVAILABLE_INDICES = ["aci", "adi", "aei", "bio", "ndsi", "h_spectral", "h_temporal"]


@dataclass
class IndicesResult:
    """Result containing computed acoustic indices."""

    indices: AcousticIndices
    source_path: Optional[str] = None
    h_spectral: Optional[float] = None
    h_temporal: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = self.indices.to_dict()
        if self.source_path:
            result["filepath"] = self.source_path
        if self.h_spectral is not None:
            result["h_spectral"] = self.h_spectral
        if self.h_temporal is not None:
            result["h_temporal"] = self.h_temporal
        return result


@dataclass
class TemporalIndicesResult:
    """Result containing temporal indices analysis."""

    windows: List[Dict[str, Any]]
    source_path: Optional[str] = None
    window_duration: float = 60.0
    hop_duration: float = 60.0
    total_duration: float = 0.0

    @property
    def num_windows(self) -> int:
        """Number of analysis windows."""
        return len(self.windows)


@dataclass
class BatchIndicesResult:
    """Result containing batch indices computation."""

    results: List[Dict[str, Any]]
    successful: int = 0
    failed: int = 0
    output_path: Optional[str] = None

    @property
    def total(self) -> int:
        """Total number of files processed."""
        return self.successful + self.failed


class IndicesService(BaseService):
    """
    Service for acoustic index calculations.

    Computes standard ecoacoustic indices:
    - ACI: Acoustic Complexity Index
    - ADI: Acoustic Diversity Index
    - AEI: Acoustic Evenness Index
    - BIO: Bioacoustic Index
    - NDSI: Normalized Difference Soundscape Index
    - H (spectral): Spectral entropy
    - H (temporal): Temporal entropy
    """

    # Default parameters for index calculations
    DEFAULT_N_FFT = 512
    DEFAULT_HOP_LENGTH = None  # Uses n_fft // 2
    DEFAULT_ACI_MIN_FREQ = 0.0
    DEFAULT_ACI_MAX_FREQ = None  # Nyquist
    DEFAULT_ADI_MAX_FREQ = 10000.0
    DEFAULT_ADI_FREQ_STEP = 1000.0
    DEFAULT_BIO_MIN_FREQ = 2000.0
    DEFAULT_BIO_MAX_FREQ = 8000.0
    DEFAULT_DB_THRESHOLD = -50.0

    def __init__(self, file_repository: FileRepositoryProtocol) -> None:
        """Initialize the service.

        Args:
            file_repository: File repository for all file I/O operations (required).
        """
        super().__init__(file_repository)

    # =========================================================================
    # Single File Operations
    # =========================================================================

    def calculate(
        self,
        audio: AudioData,
        indices: Optional[List[str]] = None,
        include_entropy: bool = True,
        **kwargs,
    ) -> ServiceResult[IndicesResult]:
        """
        Calculate acoustic indices for audio data.

        Args:
            audio: AudioData object
            indices: List of indices to compute (default: all)
            include_entropy: Include spectral/temporal entropy
            **kwargs: Additional parameters for index calculations

        Returns:
            ServiceResult containing IndicesResult
        """
        try:
            # Ensure mono
            samples = audio.samples
            if samples.ndim > 1:
                samples = samples.mean(axis=-1)

            # Get parameters with defaults
            n_fft = kwargs.get("n_fft", self.DEFAULT_N_FFT)
            hop_length = kwargs.get("hop_length", self.DEFAULT_HOP_LENGTH)
            aci_min_freq = kwargs.get("aci_min_freq", self.DEFAULT_ACI_MIN_FREQ)
            aci_max_freq = kwargs.get("aci_max_freq", self.DEFAULT_ACI_MAX_FREQ)
            adi_max_freq = kwargs.get("adi_max_freq", self.DEFAULT_ADI_MAX_FREQ)
            adi_freq_step = kwargs.get("adi_freq_step", self.DEFAULT_ADI_FREQ_STEP)
            bio_min_freq = kwargs.get("bio_min_freq", self.DEFAULT_BIO_MIN_FREQ)
            bio_max_freq = kwargs.get("bio_max_freq", self.DEFAULT_BIO_MAX_FREQ)
            db_threshold = kwargs.get("db_threshold", self.DEFAULT_DB_THRESHOLD)

            # Calculate all indices
            acoustic_indices = compute_all_indices(
                samples,
                audio.sample_rate,
                n_fft=n_fft,
                hop_length=hop_length,
                aci_min_freq=aci_min_freq,
                aci_max_freq=aci_max_freq,
                adi_max_freq=adi_max_freq,
                adi_freq_step=adi_freq_step,
                bio_min_freq=bio_min_freq,
                bio_max_freq=bio_max_freq,
                db_threshold=db_threshold,
            )

            # Calculate entropy if requested
            h_spectral = None
            h_temporal = None
            if include_entropy:
                h_spectral = spectral_entropy(samples, audio.sample_rate, n_fft)
                h_temporal = temporal_entropy(samples, audio.sample_rate, n_fft)

            result = IndicesResult(
                indices=acoustic_indices,
                source_path=audio.source_path,
                h_spectral=h_spectral,
                h_temporal=h_temporal,
            )

            return ServiceResult.ok(
                data=result,
                message=f"Computed indices for {audio.duration:.1f}s audio",
            )

        except Exception as e:
            return ServiceResult.fail(f"Index calculation failed: {e}")

    def calculate_single_index(
        self,
        audio: AudioData,
        index_name: str,
        **kwargs,
    ) -> ServiceResult[float]:
        """
        Calculate a single acoustic index.

        Args:
            audio: AudioData object
            index_name: Name of index (aci, adi, aei, bio, ndsi, h_spectral, h_temporal)
            **kwargs: Parameters for index calculation

        Returns:
            ServiceResult containing the index value
        """
        try:
            samples = audio.samples
            if samples.ndim > 1:
                samples = samples.mean(axis=-1)

            n_fft = kwargs.get("n_fft", self.DEFAULT_N_FFT)
            hop_length = kwargs.get("hop_length", self.DEFAULT_HOP_LENGTH)

            index_name = index_name.lower()

            if index_name == "aci":
                value = compute_aci(
                    samples,
                    audio.sample_rate,
                    n_fft,
                    hop_length,
                    min_freq=kwargs.get("min_freq", self.DEFAULT_ACI_MIN_FREQ),
                    max_freq=kwargs.get("max_freq", self.DEFAULT_ACI_MAX_FREQ),
                )
            elif index_name == "adi":
                value = compute_adi(
                    samples,
                    audio.sample_rate,
                    n_fft,
                    hop_length,
                    max_freq=kwargs.get("max_freq", self.DEFAULT_ADI_MAX_FREQ),
                    freq_step=kwargs.get("freq_step", self.DEFAULT_ADI_FREQ_STEP),
                    db_threshold=kwargs.get("db_threshold", self.DEFAULT_DB_THRESHOLD),
                )
            elif index_name == "aei":
                value = compute_aei(
                    samples,
                    audio.sample_rate,
                    n_fft,
                    hop_length,
                    max_freq=kwargs.get("max_freq", self.DEFAULT_ADI_MAX_FREQ),
                    freq_step=kwargs.get("freq_step", self.DEFAULT_ADI_FREQ_STEP),
                    db_threshold=kwargs.get("db_threshold", self.DEFAULT_DB_THRESHOLD),
                )
            elif index_name == "bio":
                value = compute_bio(
                    samples,
                    audio.sample_rate,
                    n_fft,
                    hop_length,
                    min_freq=kwargs.get("min_freq", self.DEFAULT_BIO_MIN_FREQ),
                    max_freq=kwargs.get("max_freq", self.DEFAULT_BIO_MAX_FREQ),
                    db_threshold=kwargs.get("db_threshold", self.DEFAULT_DB_THRESHOLD),
                )
            elif index_name == "ndsi":
                value, _, _ = compute_ndsi(
                    samples,
                    audio.sample_rate,
                    n_fft=1024,
                    hop_length=hop_length,
                )
            elif index_name == "h_spectral":
                value = spectral_entropy(samples, audio.sample_rate, n_fft, hop_length)
            elif index_name == "h_temporal":
                value = temporal_entropy(samples, audio.sample_rate, n_fft, hop_length)
            else:
                return ServiceResult.fail(
                    f"Unknown index: {index_name}. Available: {AVAILABLE_INDICES}"
                )

            return ServiceResult.ok(
                data=value,
                message=f"{index_name.upper()}: {value:.4f}",
            )

        except Exception as e:
            return ServiceResult.fail(f"Index calculation failed: {e}")

    # =========================================================================
    # Temporal Analysis
    # =========================================================================

    def calculate_temporal(
        self,
        audio: AudioData,
        window_duration: float = 60.0,
        hop_duration: Optional[float] = None,
        **kwargs,
    ) -> ServiceResult[TemporalIndicesResult]:
        """
        Calculate acoustic indices over sliding time windows.

        Args:
            audio: AudioData object
            window_duration: Duration of each window in seconds
            hop_duration: Hop between windows (default: same as window_duration)
            **kwargs: Parameters for index calculations

        Returns:
            ServiceResult containing TemporalIndicesResult
        """
        try:
            samples = audio.samples
            if samples.ndim > 1:
                samples = samples.mean(axis=-1)

            if hop_duration is None:
                hop_duration = window_duration

            # Use core temporal_indices function
            windows = temporal_indices(
                samples,
                audio.sample_rate,
                window_duration=window_duration,
                hop_duration=hop_duration,
                **kwargs,
            )

            result = TemporalIndicesResult(
                windows=windows,
                source_path=audio.source_path,
                window_duration=window_duration,
                hop_duration=hop_duration,
                total_duration=audio.duration,
            )

            return ServiceResult.ok(
                data=result,
                message=f"Computed {len(windows)} windows ({window_duration}s each)",
            )

        except Exception as e:
            return ServiceResult.fail(f"Temporal analysis failed: {e}")


    def _save_results(
        self,
        results: List[Dict[str, Any]],
        output_path: str,
    ) -> ServiceResult[str]:
        """Save results to CSV or Parquet."""
        try:
            import pandas as pd

            df = pd.DataFrame(results)
            path = Path(output_path)
            self.file_repository.mkdir(path.parent, parents=True)

            if path.suffix.lower() == ".parquet":
                df.to_parquet(path, index=False)
            else:
                df.to_csv(path, index=False)

            return ServiceResult.ok(
                data=str(path),
                message=f"Saved results to {path}",
            )

        except Exception as e:
            return ServiceResult.fail(f"Failed to save results: {e}")
            return ServiceResult.fail(f"Failed to save results: {e}")

    # =========================================================================
    # Convenience Methods
    # =========================================================================

    def get_available_indices(self) -> List[str]:
        """Get list of available index names."""
        return AVAILABLE_INDICES.copy()

    def describe_index(self, index_name: str) -> Optional[str]:
        """Get description of an acoustic index."""
        descriptions = {
            "aci": "Acoustic Complexity Index - measures variability of sound intensities",
            "adi": "Acoustic Diversity Index - Shannon diversity across frequency bands",
            "aei": "Acoustic Evenness Index - Gini coefficient of frequency distribution",
            "bio": "Bioacoustic Index - area under spectrum in 2-8 kHz range",
            "ndsi": "Normalized Difference Soundscape Index - biophony vs anthrophony ratio",
            "h_spectral": "Spectral Entropy - uniformity of power spectrum",
            "h_temporal": "Temporal Entropy - uniformity of energy over time",
        }
        return descriptions.get(index_name.lower())
