# controllers/ribbit.py
"""
RIBBIT Controller
=================

Controller for RIBBIT periodic vocalization detection operations.

Orchestrates between CLI/API views and core RIBBIT detection functions.
Handles profile management, batch processing, and result formatting.

Example:
    from bioamla.controllers.ribbit import RibbitController

    controller = RibbitController()

    # List available presets
    presets = controller.list_presets()

    # Detect using preset
    result = controller.detect("audio.wav", preset="spring_peeper")

    # Batch detection
    result = controller.detect_batch(
        directory="./audio",
        preset="american_bullfrog",
        output_csv="detections.csv",
    )
"""

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .base import BaseController, ControllerResult, ToDictMixin


@dataclass
class DetectionSummary(ToDictMixin):
    """Summary of RIBBIT detection results."""

    filepath: str
    profile_name: str
    num_detections: int
    total_detection_time: float
    detection_percentage: float
    duration: float
    processing_time: float


@dataclass
class BatchDetectionSummary(ToDictMixin):
    """Summary of batch RIBBIT detection."""

    total_files: int
    files_with_detections: int
    total_detections: int
    total_duration: float
    total_detection_time: float
    detection_percentage: float
    output_path: Optional[str]
    errors: List[str] = field(default_factory=list)


class RibbitController(BaseController):
    """
    Controller for RIBBIT periodic vocalization detection.

    Provides high-level methods for:
    - Single file detection with preset or custom profiles
    - Batch detection with progress reporting
    - Profile management and listing
    - Result export to CSV/JSON
    """

    def __init__(self):
        """Initialize RIBBIT controller."""
        super().__init__()
        self._detector = None
        self._current_profile = None

    def _get_detector(
        self,
        preset: Optional[str] = None,
        profile: Optional[Dict[str, Any]] = None,
    ):
        """Get or create detector with specified profile."""
        from bioamla.core.detection.ribbit import RibbitDetector, RibbitProfile

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
    ) -> ControllerResult[DetectionSummary]:
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
            return ControllerResult.fail(error)

        try:
            detector = self._get_detector(preset=preset, profile=profile)
            result = detector.detect(filepath)

            if result.error:
                return ControllerResult.fail(result.error)

            summary = DetectionSummary(
                filepath=result.filepath,
                profile_name=result.profile_name,
                num_detections=result.num_detections,
                total_detection_time=result.total_detection_time,
                detection_percentage=result.detection_percentage,
                duration=result.duration,
                processing_time=result.processing_time,
            )

            return ControllerResult.ok(
                data=summary,
                message=f"Found {result.num_detections} detections",
                detections=[d.to_dict() for d in result.detections],
                result=result,
            )
        except Exception as e:
            return ControllerResult.fail(str(e))

    # =========================================================================
    # Batch Detection
    # =========================================================================

    def detect_batch(
        self,
        directory: str,
        preset: Optional[str] = None,
        profile: Optional[Dict[str, Any]] = None,
        output_csv: Optional[str] = None,
        recursive: bool = True,
    ) -> ControllerResult[BatchDetectionSummary]:
        """
        Run RIBBIT detection on multiple audio files.

        Args:
            directory: Directory containing audio files
            preset: Name of preset profile to use
            profile: Custom profile dictionary
            output_csv: Optional CSV output path
            recursive: Search subdirectories

        Returns:
            Result with batch detection summary
        """
        error = self._validate_input_path(directory)
        if error:
            return ControllerResult.fail(error)

        # Start run tracking
        run_id = self._start_run(
            name=f"RIBBIT batch detection: {directory}",
            action="ribbit",
            input_path=directory,
            output_path=output_csv or "",
            parameters={
                "preset": preset,
                "profile": profile,
                "recursive": recursive,
            },
        )

        try:
            from bioamla.core.files import TextFile

            detector = self._get_detector(preset=preset, profile=profile)
            files = self._get_audio_files(directory, recursive=recursive)

            if not files:
                self._fail_run("No audio files found")
                return ControllerResult.fail(f"No audio files found in {directory}")

            all_detections = []
            all_results = []
            errors = []
            total_duration = 0.0
            total_detection_time = 0.0
            files_with_detections = 0

            def process_file(filepath: Path):
                result = detector.detect(str(filepath))
                return result

            for filepath, result, error in self._process_batch(files, process_file):
                if error:
                    errors.append(f"{filepath.name}: {error}")
                elif result is not None:
                    all_results.append(result)
                    total_duration += result.duration
                    total_detection_time += result.total_detection_time

                    if result.num_detections > 0:
                        files_with_detections += 1

                    for detection in result.detections:
                        all_detections.append(
                            {
                                "filepath": str(filepath),
                                "profile": result.profile_name,
                                "start_time": detection.start_time,
                                "end_time": detection.end_time,
                                "duration": detection.duration,
                                "score": detection.score,
                                "pulse_rate": detection.pulse_rate,
                            }
                        )

            # Write CSV if requested
            saved_path = None
            if output_csv and all_detections:
                output_path = Path(output_csv)
                output_path.parent.mkdir(parents=True, exist_ok=True)

                with TextFile(output_path, mode="w", newline="") as f:
                    fieldnames = [
                        "filepath",
                        "profile",
                        "start_time",
                        "end_time",
                        "duration",
                        "score",
                        "pulse_rate",
                    ]
                    writer = csv.DictWriter(f.handle, fieldnames=fieldnames)
                    writer.writeheader()
                    for det in all_detections:
                        writer.writerow(det)

                saved_path = str(output_path)

            detection_percentage = (
                total_detection_time / total_duration * 100 if total_duration > 0 else 0
            )

            summary = BatchDetectionSummary(
                total_files=len(files),
                files_with_detections=files_with_detections,
                total_detections=len(all_detections),
                total_duration=total_duration,
                total_detection_time=total_detection_time,
                detection_percentage=detection_percentage,
                output_path=saved_path,
                errors=errors,
            )

            # Complete run with results
            self._complete_run(
                results={
                    "total_files": len(files),
                    "files_with_detections": files_with_detections,
                    "total_detections": len(all_detections),
                    "detection_percentage": detection_percentage,
                    "errors_count": len(errors),
                },
                output_files=[saved_path] if saved_path else None,
            )

            return ControllerResult.ok(
                data=summary,
                message=f"Found {len(all_detections)} detections in {files_with_detections} files",
                detections=all_detections,
                results=all_results,
            )
        except Exception as e:
            self._fail_run(str(e))
            return ControllerResult.fail(str(e))

    # =========================================================================
    # Profile Management
    # =========================================================================

    def list_presets(self) -> ControllerResult[List[Dict[str, Any]]]:
        """
        List all available preset profiles.

        Returns:
            Result with list of preset info
        """
        try:
            from bioamla.core.detection.ribbit import get_preset_profiles

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

            return ControllerResult.ok(
                data=preset_list,
                message=f"Found {len(preset_list)} preset profiles",
            )
        except Exception as e:
            return ControllerResult.fail(str(e))

    def get_preset(self, preset_name: str) -> ControllerResult[Dict[str, Any]]:
        """
        Get details of a specific preset profile.

        Args:
            preset_name: Name of the preset

        Returns:
            Result with profile details
        """
        try:
            from bioamla.core.detection.ribbit import get_preset_profiles

            profiles = get_preset_profiles()

            if preset_name not in profiles:
                available = ", ".join(profiles.keys())
                return ControllerResult.fail(
                    f"Unknown preset: {preset_name}. Available: {available}"
                )

            profile = profiles[preset_name]
            return ControllerResult.ok(
                data=profile.to_dict(),
                message=f"Profile: {preset_name}",
            )
        except Exception as e:
            return ControllerResult.fail(str(e))

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
    ) -> ControllerResult[Dict[str, Any]]:
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
            from bioamla.core.detection.ribbit import RibbitProfile

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
                return ControllerResult.fail(f"Invalid profile: {error}")

            return ControllerResult.ok(
                data=profile.to_dict(),
                message=f"Created profile: {name}",
                profile=profile,
            )
        except Exception as e:
            return ControllerResult.fail(str(e))

    # =========================================================================
    # Visualization Helpers
    # =========================================================================

    def get_detection_timeline(
        self,
        result,  # RibbitResult
        resolution: float = 1.0,
    ) -> ControllerResult[Dict[str, Any]]:
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

            return ControllerResult.ok(
                data={
                    "times": times.tolist(),
                    "scores": timeline.tolist(),
                    "resolution": resolution,
                    "duration": duration,
                },
                message="Generated timeline",
            )
        except Exception as e:
            return ControllerResult.fail(str(e))
