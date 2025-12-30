from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class RibbitDetection:
    """A single RIBBIT detection result.

    Attributes:
        start_time: Start time in seconds.
        end_time: End time in seconds.
        score: Detection score (0-1).
    """

    start_time: float
    end_time: float
    score: float

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
        }


def ribbit_detect(
    audio_path: str,
    signal_band: Tuple[float, float],
    pulse_rate_range: Tuple[float, float],
    noise_bands: Optional[List[Tuple[float, float]]] = None,
    clip_duration: float = 2.0,
    clip_overlap: float = 0.5,
    score_threshold: float = 0.5,
    min_detection_duration: float = 0.5,
) -> Tuple[List[RibbitDetection], Dict[str, Any]]:
    """Run RIBBIT detection on an audio file using OpenSoundscape.

    RIBBIT (Repeat-Interval Based Bioacoustic Identification Tool) detects
    periodic vocalizations by analyzing pulse rates in specific frequency bands.

    Args:
        audio_path: Path to audio file.
        signal_band: Frequency range (low_hz, high_hz) of the target signal.
        pulse_rate_range: Expected pulse rate range (min_hz, max_hz).
        noise_bands: List of frequency bands for noise estimation.
        clip_duration: Analysis clip length in seconds.
        clip_overlap: Overlap between clips in seconds.
        score_threshold: Minimum score for detection (0-1).
        min_detection_duration: Minimum duration for valid detection (seconds).

    Returns:
        Tuple of (list of RibbitDetection, metadata dict with scores/times/duration).

    Example:
        >>> detections, metadata = ribbit_detect(
        ...     "frog_audio.wav",
        ...     signal_band=(500, 2000),
        ...     pulse_rate_range=(3.0, 15.0),
        ... )
        >>> for d in detections:
        ...     print(f"{d.start_time:.1f}s - {d.end_time:.1f}s: {d.score:.2f}")
    """
    from opensoundscape import Audio, Spectrogram
    from opensoundscape.ribbit import ribbit

    # Load audio and create spectrogram
    audio = Audio.from_file(audio_path)
    spectrogram = Spectrogram.from_audio(audio)
    duration = audio.duration
    sample_rate = audio.sample_rate

    # Run RIBBIT - returns DataFrame with columns ['start_time', 'end_time', 'score']
    result_df = ribbit(
        spectrogram=spectrogram,
        signal_band=signal_band,
        pulse_rate_range=pulse_rate_range,
        clip_duration=clip_duration,
        clip_overlap=clip_overlap,
        noise_bands=noise_bands,
    )

    # Convert DataFrame to detections
    detections = _dataframe_to_detections(
        result_df=result_df,
        threshold=score_threshold,
        min_duration=min_detection_duration,
    )

    metadata = {
        "scores": result_df["score"].tolist() if len(result_df) > 0 else [],
        "times": result_df["start_time"].tolist() if len(result_df) > 0 else [],
        "duration": duration,
        "sample_rate": sample_rate,
        "num_detections": len(detections),
    }

    return detections, metadata


def ribbit_detect_samples(
    samples: np.ndarray,
    sample_rate: int,
    signal_band: Tuple[float, float],
    pulse_rate_range: Tuple[float, float],
    noise_bands: Optional[List[Tuple[float, float]]] = None,
    clip_duration: float = 2.0,
    clip_overlap: float = 0.5,
    score_threshold: float = 0.5,
    min_detection_duration: float = 0.5,
) -> Tuple[List[RibbitDetection], Dict[str, Any]]:
    """Run RIBBIT detection on audio samples using OpenSoundscape.

    Args:
        samples: Audio samples as numpy array.
        sample_rate: Sample rate in Hz.
        signal_band: Frequency range (low_hz, high_hz) of the target signal.
        pulse_rate_range: Expected pulse rate range (min_hz, max_hz).
        noise_bands: List of frequency bands for noise estimation.
        clip_duration: Analysis clip length in seconds.
        clip_overlap: Overlap between clips in seconds.
        score_threshold: Minimum score for detection (0-1).
        min_detection_duration: Minimum duration for valid detection (seconds).

    Returns:
        Tuple of (list of RibbitDetection, metadata dict).
    """
    from opensoundscape import Audio, Spectrogram
    from opensoundscape.ribbit import ribbit

    # Ensure mono
    if samples.ndim > 1:
        samples = samples.mean(axis=-1)

    duration = len(samples) / sample_rate

    # Create Audio and Spectrogram objects
    audio = Audio(samples, sample_rate)
    spectrogram = Spectrogram.from_audio(audio)

    # Run RIBBIT - returns DataFrame with columns ['start_time', 'end_time', 'score']
    result_df = ribbit(
        spectrogram=spectrogram,
        signal_band=signal_band,
        pulse_rate_range=pulse_rate_range,
        clip_duration=clip_duration,
        clip_overlap=clip_overlap,
        noise_bands=noise_bands,
    )

    # Convert DataFrame to detections
    detections = _dataframe_to_detections(
        result_df=result_df,
        threshold=score_threshold,
        min_duration=min_detection_duration,
    )

    metadata = {
        "scores": result_df["score"].tolist() if len(result_df) > 0 else [],
        "times": result_df["start_time"].tolist() if len(result_df) > 0 else [],
        "duration": duration,
        "sample_rate": sample_rate,
        "num_detections": len(detections),
    }

    return detections, metadata


def _dataframe_to_detections(
    result_df,
    threshold: float,
    min_duration: float,
) -> List[RibbitDetection]:
    """Convert RIBBIT result DataFrame to detection objects.

    Args:
        result_df: DataFrame with columns ['start_time', 'end_time', 'score'].
        threshold: Minimum score threshold.
        min_duration: Minimum detection duration.

    Returns:
        List of RibbitDetection objects.
    """

    detections = []

    if len(result_df) == 0:
        return detections

    # Filter by threshold
    filtered_df = result_df[result_df["score"] >= threshold]

    for _, row in filtered_df.iterrows():
        start_time = float(row["start_time"])
        end_time = float(row["end_time"])
        duration = end_time - start_time

        if duration >= min_duration:
            detections.append(
                RibbitDetection(
                    start_time=start_time,
                    end_time=end_time,
                    score=float(row["score"]),
                )
            )

    return detections


# Preset profiles for common species
RIBBIT_PRESETS: Dict[str, Dict[str, Any]] = {
    "american_bullfrog": {
        "signal_band": (100, 400),
        "pulse_rate_range": (1.0, 4.0),
        "noise_bands": [(50, 80), (500, 1000)],
        "clip_duration": 3.0,
        "score_threshold": 0.4,
    },
    "spring_peeper": {
        "signal_band": (2500, 3500),
        "pulse_rate_range": (15.0, 25.0),
        "noise_bands": [(1000, 2000), (4000, 5000)],
        "clip_duration": 1.0,
        "score_threshold": 0.5,
    },
    "green_frog": {
        "signal_band": (200, 600),
        "pulse_rate_range": (2.0, 6.0),
        "noise_bands": [(50, 150), (800, 1500)],
        "clip_duration": 2.0,
        "score_threshold": 0.45,
    },
    "pacific_treefrog": {
        "signal_band": (1000, 2500),
        "pulse_rate_range": (8.0, 20.0),
        "noise_bands": [(500, 800), (3000, 4000)],
        "clip_duration": 1.5,
        "score_threshold": 0.5,
    },
    "generic_mid_freq": {
        "signal_band": (500, 2000),
        "pulse_rate_range": (3.0, 15.0),
        "noise_bands": [(200, 400), (2500, 4000)],
        "clip_duration": 2.0,
        "score_threshold": 0.45,
    },
    "generic_low_freq": {
        "signal_band": (100, 500),
        "pulse_rate_range": (1.0, 5.0),
        "noise_bands": [(50, 80), (700, 1000)],
        "clip_duration": 3.0,
        "score_threshold": 0.45,
    },
    "generic_high_freq": {
        "signal_band": (2000, 4000),
        "pulse_rate_range": (10.0, 30.0),
        "noise_bands": [(1000, 1500), (5000, 6000)],
        "clip_duration": 1.5,
        "score_threshold": 0.5,
    },
}


def ribbit_detect_preset(
    audio_path: str,
    preset: str,
    score_threshold: Optional[float] = None,
    min_detection_duration: float = 0.5,
) -> Tuple[List[RibbitDetection], Dict[str, Any]]:
    """Run RIBBIT detection using a preset profile.

    Args:
        audio_path: Path to audio file.
        preset: Name of preset profile (e.g., "american_bullfrog", "spring_peeper").
        score_threshold: Override preset's score threshold.
        min_detection_duration: Minimum duration for valid detection.

    Returns:
        Tuple of (list of RibbitDetection, metadata dict).

    Raises:
        ValueError: If preset name is not recognized.
    """
    if preset not in RIBBIT_PRESETS:
        available = ", ".join(RIBBIT_PRESETS.keys())
        raise ValueError(f"Unknown preset: {preset}. Available: {available}")

    params = RIBBIT_PRESETS[preset].copy()

    # Override threshold if provided
    if score_threshold is not None:
        params["score_threshold"] = score_threshold

    return ribbit_detect(
        audio_path=audio_path,
        signal_band=params["signal_band"],
        pulse_rate_range=params["pulse_rate_range"],
        noise_bands=params.get("noise_bands"),
        clip_duration=params.get("clip_duration", 2.0),
        clip_overlap=params.get("clip_overlap", 0.5),
        score_threshold=params.get("score_threshold", 0.5),
        min_detection_duration=min_detection_duration,
    )


def list_ribbit_presets() -> List[str]:
    """Get list of available RIBBIT preset names."""
    return list(RIBBIT_PRESETS.keys())


def get_ribbit_preset(preset: str) -> Dict[str, Any]:
    """Get parameters for a RIBBIT preset.

    Args:
        preset: Preset name.

    Returns:
        Dictionary of preset parameters.

    Raises:
        ValueError: If preset name is not recognized.
    """
    if preset not in RIBBIT_PRESETS:
        available = ", ".join(RIBBIT_PRESETS.keys())
        raise ValueError(f"Unknown preset: {preset}. Available: {available}")

    return RIBBIT_PRESETS[preset].copy()
