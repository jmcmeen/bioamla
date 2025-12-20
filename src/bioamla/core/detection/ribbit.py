# core/detection/ribbit.py
"""
RIBBIT Detection Module
=======================

Wrapper for OpenSoundscape RIBBIT algorithm for detecting periodic vocalizations.

RIBBIT (Repeat-Interval Based Bioacoustic Identification Tool) is designed to
detect animal vocalizations that have a consistent pulse rate, such as:
- Frog and toad calls
- Insect songs
- Other periodic bioacoustic signals

This module provides:
- RibbitProfile: Configuration for species-specific detection
- RibbitDetector: Main detector class
- Preset profiles for common amphibian species
- Batch processing utilities

Example:
    from bioamla.core.detection.ribbit import RibbitDetector, RibbitProfile

    # Use a preset profile
    detector = RibbitDetector.from_preset("american_bullfrog")
    detections = detector.detect("audio.wav")

    # Custom profile
    profile = RibbitProfile(
        name="custom_frog",
        signal_band=(500, 2000),
        pulse_rate_range=(2.0, 8.0),
        noise_bands=[(100, 400), (2500, 5000)],
    )
    detector = RibbitDetector(profile)
    detections = detector.detect("audio.wav")

References:
    - OpenSoundscape RIBBIT: https://opensoundscape.org/
    - Lapp et al. 2021: https://doi.org/10.1111/2041-210X.13718
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from bioamla.core.logger import get_logger

logger = get_logger(__name__)

__all__ = [
    "RibbitProfile",
    "RibbitDetection",
    "RibbitResult",
    "RibbitDetector",
    "get_preset_profiles",
    "list_preset_names",
]


@dataclass
class RibbitProfile:
    """
    Configuration profile for RIBBIT detection.

    Defines the acoustic parameters for detecting a specific type of
    periodic vocalization.

    Attributes:
        name: Profile name/identifier
        signal_band: Frequency range (low_hz, high_hz) of the target signal
        pulse_rate_range: Expected pulse rate range (min_hz, max_hz)
        noise_bands: List of frequency bands for noise estimation
        window_length: Analysis window length in seconds
        overlap: Window overlap fraction (0-1)
        score_threshold: Minimum score for detection
        min_detection_duration: Minimum duration for valid detection (seconds)
    """

    name: str
    signal_band: Tuple[float, float]
    pulse_rate_range: Tuple[float, float]
    noise_bands: List[Tuple[float, float]] = field(default_factory=list)
    window_length: float = 2.0
    overlap: float = 0.5
    score_threshold: float = 0.5
    min_detection_duration: float = 0.5
    description: str = ""
    species: Optional[str] = None

    def validate(self) -> Optional[str]:
        """Validate profile parameters. Returns error message or None."""
        if self.signal_band[0] >= self.signal_band[1]:
            return f"Invalid signal band: {self.signal_band}"
        if self.pulse_rate_range[0] >= self.pulse_rate_range[1]:
            return f"Invalid pulse rate range: {self.pulse_rate_range}"
        if not (0 <= self.overlap < 1):
            return f"Overlap must be in [0, 1): {self.overlap}"
        if not (0 <= self.score_threshold <= 1):
            return f"Score threshold must be in [0, 1]: {self.score_threshold}"
        return None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "signal_band": list(self.signal_band),
            "pulse_rate_range": list(self.pulse_rate_range),
            "noise_bands": [list(b) for b in self.noise_bands],
            "window_length": self.window_length,
            "overlap": self.overlap,
            "score_threshold": self.score_threshold,
            "min_detection_duration": self.min_detection_duration,
            "description": self.description,
            "species": self.species,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RibbitProfile":
        """Create from dictionary."""
        return cls(
            name=data["name"],
            signal_band=tuple(data["signal_band"]),
            pulse_rate_range=tuple(data["pulse_rate_range"]),
            noise_bands=[tuple(b) for b in data.get("noise_bands", [])],
            window_length=data.get("window_length", 2.0),
            overlap=data.get("overlap", 0.5),
            score_threshold=data.get("score_threshold", 0.5),
            min_detection_duration=data.get("min_detection_duration", 0.5),
            description=data.get("description", ""),
            species=data.get("species"),
        )


@dataclass
class RibbitDetection:
    """A single RIBBIT detection."""

    start_time: float
    end_time: float
    score: float
    pulse_rate: Optional[float] = None
    signal_power: Optional[float] = None
    snr: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def duration(self) -> float:
        """Detection duration in seconds."""
        return self.end_time - self.start_time

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration": self.duration,
            "score": self.score,
            "pulse_rate": self.pulse_rate,
            "signal_power": self.signal_power,
            "snr": self.snr,
            **self.metadata,
        }


@dataclass
class RibbitResult:
    """Result from RIBBIT detection on a file."""

    filepath: str
    profile_name: str
    detections: List[RibbitDetection]
    duration: float
    sample_rate: int
    processing_time: float = 0.0
    error: Optional[str] = None

    @property
    def num_detections(self) -> int:
        """Number of detections."""
        return len(self.detections)

    @property
    def total_detection_time(self) -> float:
        """Total time with detections (seconds)."""
        return sum(d.duration for d in self.detections)

    @property
    def detection_percentage(self) -> float:
        """Percentage of file with detections."""
        return (self.total_detection_time / self.duration * 100) if self.duration > 0 else 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "filepath": self.filepath,
            "profile_name": self.profile_name,
            "num_detections": self.num_detections,
            "total_detection_time": self.total_detection_time,
            "detection_percentage": self.detection_percentage,
            "duration": self.duration,
            "sample_rate": self.sample_rate,
            "processing_time": self.processing_time,
            "detections": [d.to_dict() for d in self.detections],
            "error": self.error,
        }


class RibbitDetector:
    """
    RIBBIT detector for periodic bioacoustic signals.

    This class wraps the OpenSoundscape RIBBIT algorithm, providing a
    simplified interface for bioamla workflows.

    Example:
        # From preset
        detector = RibbitDetector.from_preset("spring_peeper")
        result = detector.detect("audio.wav")

        # Custom profile
        profile = RibbitProfile(
            name="my_frog",
            signal_band=(800, 1500),
            pulse_rate_range=(5.0, 15.0),
        )
        detector = RibbitDetector(profile)
        result = detector.detect("audio.wav")
    """

    def __init__(self, profile: RibbitProfile):
        """
        Initialize RIBBIT detector.

        Args:
            profile: Detection profile configuration
        """
        self.profile = profile
        self._ribbit = None

        # Validate profile
        error = profile.validate()
        if error:
            raise ValueError(f"Invalid profile: {error}")

    @classmethod
    def from_preset(cls, preset_name: str) -> "RibbitDetector":
        """
        Create detector from a preset profile.

        Args:
            preset_name: Name of the preset profile

        Returns:
            Configured RibbitDetector
        """
        profiles = get_preset_profiles()
        if preset_name not in profiles:
            available = ", ".join(profiles.keys())
            raise ValueError(f"Unknown preset: {preset_name}. Available: {available}")
        return cls(profiles[preset_name])

    def _get_ribbit(self):
        """Lazy-load OpenSoundscape RIBBIT."""
        if self._ribbit is not None:
            return self._ribbit

        try:
            from opensoundscape.ribbit import ribbit

            self._ribbit = ribbit
            return self._ribbit
        except ImportError as err:
            raise ImportError(
                "OpenSoundscape is required for RIBBIT detection. "
                "Install with: pip install opensoundscape"
            ) from err

    def detect(
        self,
        audio: Union[str, np.ndarray],
        sample_rate: Optional[int] = None,
    ) -> RibbitResult:
        """
        Run RIBBIT detection on audio.

        Args:
            audio: Audio file path or numpy array
            sample_rate: Sample rate (required if audio is array)

        Returns:
            RibbitResult with detections
        """
        import time

        start_time = time.time()

        # Load audio
        filepath = audio if isinstance(audio, str) else "<array>"
        try:
            if isinstance(audio, str):
                import soundfile as sf

                audio_data, sr = sf.read(audio, dtype="float32")
            else:
                audio_data = audio
                sr = sample_rate
                if sr is None:
                    raise ValueError("sample_rate required when audio is array")

            # Ensure mono
            if audio_data.ndim > 1:
                audio_data = audio_data.mean(axis=-1)

            duration = len(audio_data) / sr

        except Exception as e:
            return RibbitResult(
                filepath=filepath,
                profile_name=self.profile.name,
                detections=[],
                duration=0.0,
                sample_rate=0,
                processing_time=time.time() - start_time,
                error=str(e),
            )

        # Run detection
        try:
            detections = self._run_detection(audio_data, sr)
        except Exception as e:
            logger.warning(f"RIBBIT detection failed: {e}")
            return RibbitResult(
                filepath=filepath,
                profile_name=self.profile.name,
                detections=[],
                duration=duration,
                sample_rate=sr,
                processing_time=time.time() - start_time,
                error=str(e),
            )

        processing_time = time.time() - start_time

        return RibbitResult(
            filepath=filepath,
            profile_name=self.profile.name,
            detections=detections,
            duration=duration,
            sample_rate=sr,
            processing_time=processing_time,
        )

    def _run_detection(
        self,
        audio: np.ndarray,
        sample_rate: int,
    ) -> List[RibbitDetection]:
        """Run RIBBIT algorithm on audio data."""
        try:
            # Try OpenSoundscape first
            return self._run_opensoundscape(audio, sample_rate)
        except ImportError:
            # Fall back to native implementation
            logger.info("OpenSoundscape not available, using native RIBBIT implementation")
            return self._run_native(audio, sample_rate)

    def _run_opensoundscape(
        self,
        audio: np.ndarray,
        sample_rate: int,
    ) -> List[RibbitDetection]:
        """Run detection using OpenSoundscape RIBBIT."""
        from opensoundscape.ribbit import ribbit

        # Prepare spectrogram parameters
        window_samples = int(self.profile.window_length * sample_rate)
        hop_samples = int(window_samples * (1 - self.profile.overlap))

        # Run RIBBIT
        scores, times = ribbit(
            audio,
            sr=sample_rate,
            signal_band=self.profile.signal_band,
            pulse_rate_range=self.profile.pulse_rate_range,
            noise_bands=self.profile.noise_bands if self.profile.noise_bands else None,
            window_samples=window_samples,
            hop_samples=hop_samples,
        )

        # Convert to detections
        return self._scores_to_detections(scores, times)

    def _run_native(
        self,
        audio: np.ndarray,
        sample_rate: int,
    ) -> List[RibbitDetection]:
        """
        Native RIBBIT implementation (simplified).

        This is a fallback when OpenSoundscape is not available.
        """
        from scipy import signal as scipy_signal

        # Compute spectrogram
        window_samples = int(self.profile.window_length * sample_rate)
        hop_samples = int(window_samples * (1 - self.profile.overlap))

        frequencies, times, spectrogram = scipy_signal.spectrogram(
            audio,
            fs=sample_rate,
            nperseg=window_samples,
            noverlap=window_samples - hop_samples,
        )

        # Find signal band indices
        sig_low, sig_high = self.profile.signal_band
        sig_idx = (frequencies >= sig_low) & (frequencies <= sig_high)

        if not sig_idx.any():
            return []

        # Extract signal power
        signal_power = spectrogram[sig_idx, :].mean(axis=0)

        # Compute noise power if noise bands specified
        if self.profile.noise_bands:
            noise_powers = []
            for low, high in self.profile.noise_bands:
                noise_idx = (frequencies >= low) & (frequencies <= high)
                if noise_idx.any():
                    noise_powers.append(spectrogram[noise_idx, :].mean(axis=0))
            if noise_powers:
                noise_power = np.mean(noise_powers, axis=0)
                snr = signal_power / (noise_power + 1e-10)
            else:
                snr = signal_power
        else:
            snr = signal_power

        # Analyze pulse rate using autocorrelation
        scores = []
        for i in range(len(times)):
            # Get local window
            start_idx = max(0, i - 5)
            end_idx = min(len(times), i + 6)
            window = signal_power[start_idx:end_idx]

            if len(window) < 3:
                scores.append(0.0)
                continue

            # Autocorrelation
            acf = np.correlate(window - window.mean(), window - window.mean(), mode="full")
            acf = acf[len(acf) // 2 :]
            acf = acf / (acf[0] + 1e-10)

            # Find peaks in expected pulse rate range
            min_lag = int(1 / self.profile.pulse_rate_range[1] / (times[1] - times[0] + 1e-10))
            max_lag = int(1 / self.profile.pulse_rate_range[0] / (times[1] - times[0] + 1e-10))

            min_lag = max(1, min(min_lag, len(acf) - 1))
            max_lag = max(min_lag + 1, min(max_lag, len(acf)))

            if max_lag > min_lag:
                peak_score = acf[min_lag:max_lag].max() if len(acf) > max_lag else 0
            else:
                peak_score = 0

            # Combine with SNR
            score = peak_score * np.log1p(snr[i])
            scores.append(float(score))

        scores = np.array(scores)

        # Normalize scores
        if scores.max() > 0:
            scores = scores / scores.max()

        return self._scores_to_detections(scores, times)

    def _scores_to_detections(
        self,
        scores: np.ndarray,
        times: np.ndarray,
    ) -> List[RibbitDetection]:
        """Convert score array to detection objects."""
        detections = []
        threshold = self.profile.score_threshold
        min_dur = self.profile.min_detection_duration

        # Find contiguous regions above threshold
        above_threshold = scores >= threshold
        changes = np.diff(above_threshold.astype(int))

        starts = np.where(changes == 1)[0] + 1
        ends = np.where(changes == -1)[0] + 1

        # Handle edge cases
        if above_threshold[0]:
            starts = np.concatenate([[0], starts])
        if above_threshold[-1]:
            ends = np.concatenate([ends, [len(scores)]])

        # Create detections
        time_step = times[1] - times[0] if len(times) > 1 else 0.1

        for start_idx, end_idx in zip(starts, ends):
            start_time = times[start_idx] if start_idx < len(times) else times[-1]
            end_time = (
                times[end_idx - 1] + time_step if end_idx <= len(times) else times[-1] + time_step
            )
            duration = end_time - start_time

            if duration >= min_dur:
                segment_scores = scores[start_idx:end_idx]
                avg_score = segment_scores.mean() if len(segment_scores) > 0 else 0

                detections.append(
                    RibbitDetection(
                        start_time=float(start_time),
                        end_time=float(end_time),
                        score=float(avg_score),
                    )
                )

        return detections

    def detect_batch(
        self,
        audio_files: List[str],
        progress_callback=None,
    ) -> List[RibbitResult]:
        """
        Run detection on multiple files.

        Args:
            audio_files: List of audio file paths
            progress_callback: Optional callback(current, total)

        Returns:
            List of RibbitResult
        """
        results = []
        total = len(audio_files)

        for i, filepath in enumerate(audio_files):
            result = self.detect(filepath)
            results.append(result)

            if progress_callback:
                progress_callback(i + 1, total)

        return results


# =============================================================================
# Preset Profiles
# =============================================================================


def get_preset_profiles() -> Dict[str, RibbitProfile]:
    """
    Get all preset RIBBIT profiles.

    Returns:
        Dictionary of profile name -> RibbitProfile
    """
    return {
        # North American frogs
        "american_bullfrog": RibbitProfile(
            name="american_bullfrog",
            species="Lithobates catesbeianus",
            signal_band=(100, 400),
            pulse_rate_range=(1.0, 4.0),
            noise_bands=[(50, 80), (500, 1000)],
            window_length=3.0,
            score_threshold=0.4,
            description="American Bullfrog - deep jug-o-rum call",
        ),
        "spring_peeper": RibbitProfile(
            name="spring_peeper",
            species="Pseudacris crucifer",
            signal_band=(2500, 3500),
            pulse_rate_range=(15.0, 25.0),
            noise_bands=[(1000, 2000), (4000, 5000)],
            window_length=1.0,
            score_threshold=0.5,
            description="Spring Peeper - high pitched peeping",
        ),
        "green_frog": RibbitProfile(
            name="green_frog",
            species="Lithobates clamitans",
            signal_band=(200, 600),
            pulse_rate_range=(2.0, 6.0),
            noise_bands=[(50, 150), (800, 1500)],
            window_length=2.0,
            score_threshold=0.45,
            description="Green Frog - banjo-like twang",
        ),
        "wood_frog": RibbitProfile(
            name="wood_frog",
            species="Lithobates sylvaticus",
            signal_band=(500, 1500),
            pulse_rate_range=(3.0, 8.0),
            noise_bands=[(200, 400), (2000, 3000)],
            window_length=2.0,
            score_threshold=0.45,
            description="Wood Frog - duck-like quacking",
        ),
        "pacific_treefrog": RibbitProfile(
            name="pacific_treefrog",
            species="Pseudacris regilla",
            signal_band=(1000, 2500),
            pulse_rate_range=(8.0, 20.0),
            noise_bands=[(500, 800), (3000, 4000)],
            window_length=1.5,
            score_threshold=0.5,
            description="Pacific Treefrog - kreck-ek call",
        ),
        "gray_treefrog": RibbitProfile(
            name="gray_treefrog",
            species="Hyla versicolor",
            signal_band=(1000, 2000),
            pulse_rate_range=(15.0, 35.0),
            noise_bands=[(500, 800), (2500, 4000)],
            window_length=1.0,
            score_threshold=0.5,
            description="Gray Treefrog - musical trill",
        ),
        "american_toad": RibbitProfile(
            name="american_toad",
            species="Anaxyrus americanus",
            signal_band=(1000, 2000),
            pulse_rate_range=(25.0, 40.0),
            noise_bands=[(500, 800), (2500, 4000)],
            window_length=3.0,
            score_threshold=0.45,
            description="American Toad - long musical trill",
        ),
        "fowlers_toad": RibbitProfile(
            name="fowlers_toad",
            species="Anaxyrus fowleri",
            signal_band=(1500, 2500),
            pulse_rate_range=(30.0, 50.0),
            noise_bands=[(800, 1200), (3000, 4000)],
            window_length=2.0,
            score_threshold=0.45,
            description="Fowler's Toad - nasal waaah",
        ),
        # Tropical frogs
        "coqui": RibbitProfile(
            name="coqui",
            species="Eleutherodactylus coqui",
            signal_band=(1500, 2500),
            pulse_rate_range=(1.5, 4.0),
            noise_bands=[(800, 1200), (3000, 4000)],
            window_length=1.0,
            score_threshold=0.5,
            description="Coquí - co-QUI two-note call",
        ),
        # Generic profiles
        "generic_high_freq": RibbitProfile(
            name="generic_high_freq",
            signal_band=(2000, 4000),
            pulse_rate_range=(10.0, 30.0),
            noise_bands=[(1000, 1500), (5000, 6000)],
            window_length=1.5,
            score_threshold=0.5,
            description="Generic high-frequency periodic calls",
        ),
        "generic_low_freq": RibbitProfile(
            name="generic_low_freq",
            signal_band=(100, 500),
            pulse_rate_range=(1.0, 5.0),
            noise_bands=[(50, 80), (700, 1000)],
            window_length=3.0,
            score_threshold=0.45,
            description="Generic low-frequency periodic calls",
        ),
        "generic_mid_freq": RibbitProfile(
            name="generic_mid_freq",
            signal_band=(500, 2000),
            pulse_rate_range=(3.0, 15.0),
            noise_bands=[(200, 400), (2500, 4000)],
            window_length=2.0,
            score_threshold=0.45,
            description="Generic mid-frequency periodic calls",
        ),
    }


def list_preset_names() -> List[str]:
    """Get list of available preset profile names."""
    return list(get_preset_profiles().keys())
