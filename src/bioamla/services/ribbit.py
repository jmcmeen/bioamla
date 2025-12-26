# services/ribbit.py
"""
Service for RIBBIT periodic vocalization detection operations.
"""

from typing import Any, Dict, List, Optional, Tuple

from bioamla.models.ribbit import DetectionSummary
from bioamla.repository.protocol import FileRepositoryProtocol

from .base import BaseService, ServiceResult


class RibbitService(BaseService):
    """
    Service for RIBBIT periodic vocalization detection.

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
        self._detector = None
        self._current_profile = None

    def _get_detector(
        self,
        preset: Optional[str] = None,
        profile: Optional[Dict[str, Any]] = None,
    ) -> "RibbitDetector":
        """Get or create detector with specified profile."""
        from bioamla.core.audio.ribbit import RibbitDetector, RibbitProfile

        if preset:
            # Use preset profile
            if self._current_profile != preset:
                self._detector = RibbitDetector.from_preset(preset)
                self._current_profile = preset
            return self._detector

        if profile:
            # Use custom profile
            ribbit_profile = RibbitProfile.from_dict(profile)
            self._detector = RibbitDetector(ribbit_profile)
            self._current_profile = profile.get("name", "custom")
            return self._detector

        # Default to generic mid-frequency
        if self._detector is None:
            self._detector = RibbitDetector.from_preset("generic_mid_freq")
            self._current_profile = "generic_mid_freq"

        return self._detector

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

        Args:
            filepath: Path to audio file
            preset: Name of preset profile to use
            profile: Custom profile dictionary

        Returns:
            Result with detection summary
        """
        error = self._validate_input_path(filepath)
        if error:
            return ServiceResult.fail(error)

        try:
            detector = self._get_detector(preset=preset, profile=profile)
            result = detector.detect(filepath)

            if result.error:
                return ServiceResult.fail(result.error)

            summary = DetectionSummary(
                filepath=result.filepath,
                profile_name=result.profile_name,
                num_detections=result.num_detections,
                total_detection_time=result.total_detection_time,
                detection_percentage=result.detection_percentage,
                duration=result.duration,
                processing_time=result.processing_time,
            )

            return ServiceResult.ok(
                data=summary,
                message=f"Found {result.num_detections} detections",
                detections=[d.to_dict() for d in result.detections],
                result=result,
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
            from bioamla.core.audio.ribbit import get_preset_profiles

            profiles = get_preset_profiles()
            preset_list = []

            for name, profile in profiles.items():
                preset_list.append(
                    {
                        "name": name,
                        "species": profile.species,
                        "description": profile.description,
                        "signal_band": list(profile.signal_band),
                        "pulse_rate_range": list(profile.pulse_rate_range),
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
            from bioamla.core.audio.ribbit import get_preset_profiles

            profiles = get_preset_profiles()

            if preset_name not in profiles:
                available = ", ".join(profiles.keys())
                return ServiceResult.fail(
                    f"Unknown preset: {preset_name}. Available: {available}"
                )

            profile = profiles[preset_name]
            return ServiceResult.ok(
                data=profile.to_dict(),
                message=f"Profile: {preset_name}",
            )
        except Exception as e:
            return ServiceResult.fail(str(e))

    def create_profile(
        self,
        name: str,
        signal_band: Tuple[float, float],
        pulse_rate_range: Tuple[float, float],
        noise_bands: Optional[List[Tuple[float, float]]] = None,
        window_length: float = 2.0,
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
            window_length: Analysis window length (seconds)
            score_threshold: Detection threshold (0-1)
            description: Profile description
            species: Species name

        Returns:
            Result with created profile
        """
        try:
            from bioamla.core.audio.ribbit import RibbitProfile

            profile = RibbitProfile(
                name=name,
                signal_band=signal_band,
                pulse_rate_range=pulse_rate_range,
                noise_bands=noise_bands or [],
                window_length=window_length,
                score_threshold=score_threshold,
                description=description,
                species=species,
            )

            error = profile.validate()
            if error:
                return ServiceResult.fail(f"Invalid profile: {error}")

            return ServiceResult.ok(
                data=profile.to_dict(),
                message=f"Created profile: {name}",
                profile=profile,
            )
        except Exception as e:
            return ServiceResult.fail(str(e))

    # =========================================================================
    # Visualization Helpers
    # =========================================================================

    def get_detection_timeline(
        self,
        result,  # RibbitResult
        resolution: float = 1.0,
    ) -> ServiceResult[Dict[str, Any]]:
        """
        Generate timeline data for visualization.

        Args:
            result: RibbitResult from detection
            resolution: Time resolution in seconds

        Returns:
            Result with timeline data
        """
        try:
            import numpy as np

            duration = result.duration
            n_bins = int(np.ceil(duration / resolution))
            timeline = np.zeros(n_bins)
            times = np.arange(n_bins) * resolution

            for detection in result.detections:
                start_bin = int(detection.start_time / resolution)
                end_bin = int(np.ceil(detection.end_time / resolution))
                timeline[start_bin:end_bin] = detection.score

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
