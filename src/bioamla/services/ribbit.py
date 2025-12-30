# services/ribbit.py
"""
Service for RIBBIT periodic vocalization detection operations.

Uses OpenSoundscape RIBBIT adapter for detection.
"""

from typing import Any, Dict, List, Optional, Tuple

from bioamla.models.ribbit import DetectionSummary
from bioamla.repository.protocol import FileRepositoryProtocol

from .base import BaseService, ServiceResult


class RibbitService(BaseService):
    """
    Service for RIBBIT periodic vocalization detection.

    Uses OpenSoundscape RIBBIT adapter for high-quality detection.

    Provides high-level methods for:
    - Single file detection with preset or custom profiles
    - Profile management and listing
    - Result export to CSV/JSON
    """

    def __init__(self, file_repository: FileRepositoryProtocol) -> None:
        """Initialize RIBBIT service.

        Args:
            file_repository: Repository for file operations (required)
        """
        super().__init__(file_repository=file_repository)

    # =========================================================================
    # Single File Detection
    # =========================================================================

    def detect(
        self,
        filepath: str,
        preset: Optional[str] = None,
        profile: Optional[Dict[str, Any]] = None,
    ) -> ServiceResult[DetectionSummary]:
        """
        Run RIBBIT detection on a single audio file.

        Uses OpenSoundscape RIBBIT implementation via adapter.

        Args:
            filepath: Path to audio file
            preset: Name of preset profile to use (e.g., "american_bullfrog", "spring_peeper")
            profile: Custom profile dictionary with keys:
                - signal_band: (low_hz, high_hz)
                - pulse_rate_range: (min_hz, max_hz)
                - noise_bands: [(low, high), ...] (optional)
                - clip_duration: float (optional)
                - score_threshold: float (optional)

        Returns:
            Result with detection summary
        """
        error = self._validate_input_path(filepath)
        if error:
            return ServiceResult.fail(error)

        try:
            # Lazy import adapter to avoid loading OSS at service init
            import time

            from bioamla.adapters.opensoundscape.ribbit import (
                ribbit_detect,
                ribbit_detect_preset,
            )

            start_time = time.time()

            if preset:
                # Use preset profile
                score_threshold = profile.get("score_threshold") if profile else None
                detections, metadata = ribbit_detect_preset(
                    audio_path=filepath,
                    preset=preset,
                    score_threshold=score_threshold,
                )
                profile_name = preset
            elif profile:
                # Use custom profile
                detections, metadata = ribbit_detect(
                    audio_path=filepath,
                    signal_band=profile["signal_band"],
                    pulse_rate_range=profile["pulse_rate_range"],
                    noise_bands=profile.get("noise_bands"),
                    clip_duration=profile.get("clip_duration", 2.0),
                    score_threshold=profile.get("score_threshold", 0.5),
                )
                profile_name = profile.get("name", "custom")
            else:
                # Default to generic mid-frequency
                detections, metadata = ribbit_detect_preset(
                    audio_path=filepath,
                    preset="generic_mid_freq",
                )
                profile_name = "generic_mid_freq"

            processing_time = time.time() - start_time

            # Calculate detection metrics
            total_detection_time = sum(d.duration for d in detections)
            duration = metadata.get("duration", 0.0)
            detection_percentage = (total_detection_time / duration * 100) if duration > 0 else 0.0

            summary = DetectionSummary(
                filepath=filepath,
                profile_name=profile_name,
                num_detections=len(detections),
                total_detection_time=total_detection_time,
                detection_percentage=detection_percentage,
                duration=duration,
                processing_time=processing_time,
            )

            return ServiceResult.ok(
                data=summary,
                message=f"Found {len(detections)} detections",
                detections=[d.to_dict() for d in detections],
                metadata=metadata,
            )
        except Exception as e:
            return ServiceResult.fail(str(e))

    # =========================================================================
    # Profile Management
    # =========================================================================

    def list_presets(self) -> ServiceResult[List[Dict[str, Any]]]:
        """
        List all available preset profiles.

        Returns:
            Result with list of preset info
        """
        try:
            from bioamla.adapters.opensoundscape.ribbit import (
                RIBBIT_PRESETS,
                list_ribbit_presets,
            )

            preset_list = []
            for name in list_ribbit_presets():
                params = RIBBIT_PRESETS[name]
                preset_list.append(
                    {
                        "name": name,
                        "signal_band": list(params["signal_band"]),
                        "pulse_rate_range": list(params["pulse_rate_range"]),
                        "noise_bands": params.get("noise_bands", []),
                        "clip_duration": params.get("clip_duration", 2.0),
                        "score_threshold": params.get("score_threshold", 0.5),
                    }
                )

            return ServiceResult.ok(
                data=preset_list,
                message=f"Found {len(preset_list)} preset profiles",
            )
        except Exception as e:
            return ServiceResult.fail(str(e))

    def get_preset(self, preset_name: str) -> ServiceResult[Dict[str, Any]]:
        """
        Get details of a specific preset profile.

        Args:
            preset_name: Name of the preset

        Returns:
            Result with profile details
        """
        try:
            from bioamla.adapters.opensoundscape.ribbit import get_ribbit_preset

            params = get_ribbit_preset(preset_name)
            return ServiceResult.ok(
                data=params,
                message=f"Profile: {preset_name}",
            )
        except ValueError as e:
            return ServiceResult.fail(str(e))
        except Exception as e:
            return ServiceResult.fail(str(e))

    def create_profile(
        self,
        name: str,
        signal_band: Tuple[float, float],
        pulse_rate_range: Tuple[float, float],
        noise_bands: Optional[List[Tuple[float, float]]] = None,
        clip_duration: float = 2.0,
        score_threshold: float = 0.5,
        description: str = "",
        species: Optional[str] = None,
    ) -> ServiceResult[Dict[str, Any]]:
        """
        Create a custom RIBBIT profile.

        Args:
            name: Profile name
            signal_band: Frequency range (low_hz, high_hz)
            pulse_rate_range: Pulse rate range (min_hz, max_hz)
            noise_bands: List of noise estimation bands
            clip_duration: Analysis clip length (seconds)
            score_threshold: Detection threshold (0-1)
            description: Profile description
            species: Species name

        Returns:
            Result with created profile
        """
        try:
            # Validate signal band
            if signal_band[0] >= signal_band[1]:
                return ServiceResult.fail("signal_band low must be less than high")
            if signal_band[0] < 0:
                return ServiceResult.fail("signal_band frequencies must be positive")

            # Validate pulse rate range
            if pulse_rate_range[0] >= pulse_rate_range[1]:
                return ServiceResult.fail("pulse_rate_range min must be less than max")
            if pulse_rate_range[0] <= 0:
                return ServiceResult.fail("pulse_rate_range must be positive")

            # Validate threshold
            if not 0 <= score_threshold <= 1:
                return ServiceResult.fail("score_threshold must be between 0 and 1")

            profile = {
                "name": name,
                "signal_band": signal_band,
                "pulse_rate_range": pulse_rate_range,
                "noise_bands": noise_bands or [],
                "clip_duration": clip_duration,
                "score_threshold": score_threshold,
                "description": description,
                "species": species,
            }

            return ServiceResult.ok(
                data=profile,
                message=f"Created profile: {name}",
            )
        except Exception as e:
            return ServiceResult.fail(str(e))

    # =========================================================================
    # Visualization Helpers
    # =========================================================================

    def get_detection_timeline(
        self,
        detections: List[Dict[str, Any]],
        duration: float,
        resolution: float = 1.0,
    ) -> ServiceResult[Dict[str, Any]]:
        """
        Generate timeline data for visualization.

        Args:
            detections: List of detection dicts from detect() result
            duration: Total audio duration in seconds
            resolution: Time resolution in seconds

        Returns:
            Result with timeline data
        """
        try:
            import numpy as np

            n_bins = int(np.ceil(duration / resolution))
            timeline = np.zeros(n_bins)
            times = np.arange(n_bins) * resolution

            for detection in detections:
                start_bin = int(detection["start_time"] / resolution)
                end_bin = int(np.ceil(detection["end_time"] / resolution))
                end_bin = min(end_bin, n_bins)  # Clamp to array size
                timeline[start_bin:end_bin] = detection.get("score", 1.0)

            return ServiceResult.ok(
                data={
                    "times": times.tolist(),
                    "scores": timeline.tolist(),
                    "resolution": resolution,
                    "duration": duration,
                },
                message="Generated timeline",
            )
        except Exception as e:
            return ServiceResult.fail(str(e))
