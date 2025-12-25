# services/annotation.py
"""
Service for managing audio annotations with support for:
- File-based CRUD operations on annotations
- Raven selection table import/export
- CSV/Parquet/JSON export
- Audio clip extraction from annotations
- Measurement computation for annotations
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from bioamla.core.audio.annotations import (
    Annotation as CoreAnnotation,
)
from bioamla.core.audio.annotations import (
    load_csv_annotations,
    load_raven_selection_table,
    save_csv_annotations,
    save_raven_selection_table,
    summarize_annotations,
)
from bioamla.repository.protocol import FileRepositoryProtocol
from bioamla.services.base import BaseService, ServiceResult

logger = logging.getLogger(__name__)


@dataclass
class AnnotationResult:
    """Result of annotation operations."""

    annotations: List[CoreAnnotation]
    file_path: Optional[str] = None
    summary: Optional[Dict[str, Any]] = None


@dataclass
class ClipExtractionResult:
    """Result of clip extraction operation."""

    total_clips: int = 0
    extracted_clips: List[str] = field(default_factory=list)
    failed_clips: List[str] = field(default_factory=list)
    output_directory: Optional[str] = None


@dataclass
class MeasurementResult:
    """Result of measurement computation."""

    annotation_id: Optional[str] = None
    measurements: Dict[str, float] = field(default_factory=dict)


class AnnotationService(BaseService):
    """
    Service for annotation operations.

    Provides a unified interface for:
    - Loading and saving annotations in various formats
    - Creating annotations in memory
    - Extracting audio clips from annotations
    - Computing measurements for annotations

    All file I/O operations are delegated to the file repository.

    Usage:
        from bioamla.repository.local import LocalFileRepository

        svc = AnnotationService(LocalFileRepository())
        result = svc.import_raven("selections.txt")
        annotations = result.data.annotations
    """

    def __init__(self, file_repository: FileRepositoryProtocol) -> None:
        """Initialize AnnotationService.

        Args:
            file_repository: File repository for all file I/O operations (required).
        """
        super().__init__(file_repository)
        self._annotations: List[CoreAnnotation] = []

    # =========================================================================
    # Import Operations
    # =========================================================================

    def import_raven(
        self,
        filepath: str,
        label_column: Optional[str] = None,
    ) -> ServiceResult[AnnotationResult]:
        """
        Import annotations from a Raven selection table.

        Args:
            filepath: Path to the Raven selection table (.txt)
            label_column: Optional column name to use for labels

        Returns:
            ServiceResult containing AnnotationResult with imported annotations
        """
        error = self._validate_input_path(filepath)
        if error:
            return ServiceResult.fail(error)

        try:
            annotations = load_raven_selection_table(filepath, label_column=label_column)
            summary = summarize_annotations(annotations)

            return ServiceResult.ok(
                data=AnnotationResult(annotations=annotations, file_path=filepath, summary=summary),
                message=f"Imported {len(annotations)} annotations from Raven table",
                count=len(annotations),
            )

        except Exception as e:
            logger.exception(f"Failed to import Raven table: {e}")
            return ServiceResult.fail(f"Failed to import Raven table: {e}")

    def import_csv(
        self,
        filepath: str,
        start_time_col: str = "start_time",
        end_time_col: str = "end_time",
        low_freq_col: str = "low_freq",
        high_freq_col: str = "high_freq",
        label_col: str = "label",
    ) -> ServiceResult[AnnotationResult]:
        """
        Import annotations from a CSV file.

        Args:
            filepath: Path to the CSV file
            start_time_col: Column name for start time
            end_time_col: Column name for end time
            low_freq_col: Column name for low frequency
            high_freq_col: Column name for high frequency
            label_col: Column name for label

        Returns:
            ServiceResult containing AnnotationResult with imported annotations
        """
        error = self._validate_input_path(filepath)
        if error:
            return ServiceResult.fail(error)

        try:
            annotations = load_csv_annotations(
                filepath,
                start_time_col=start_time_col,
                end_time_col=end_time_col,
                low_freq_col=low_freq_col,
                high_freq_col=high_freq_col,
                label_col=label_col,
            )
            summary = summarize_annotations(annotations)

            return ServiceResult.ok(
                data=AnnotationResult(annotations=annotations, file_path=filepath, summary=summary),
                message=f"Imported {len(annotations)} annotations from CSV",
                count=len(annotations),
            )

        except Exception as e:
            logger.exception(f"Failed to import CSV: {e}")
            return ServiceResult.fail(f"Failed to import CSV: {e}")

    # =========================================================================
    # Export Operations
    # =========================================================================

    def export_raven(
        self,
        annotations: List[CoreAnnotation],
        output_path: str,
        include_custom_fields: bool = True,
    ) -> ServiceResult[str]:
        """
        Export annotations to a Raven selection table.

        Args:
            annotations: List of annotations to export
            output_path: Output file path (.txt)
            include_custom_fields: Include custom fields as additional columns

        Returns:
            ServiceResult containing the output file path
        """
        error = self._validate_output_path(output_path)
        if error:
            return ServiceResult.fail(error)

        try:
            saved_path = save_raven_selection_table(
                annotations, output_path, include_custom_fields=include_custom_fields
            )

            return ServiceResult.ok(
                data=saved_path,
                message=f"Exported {len(annotations)} annotations to Raven table",
                count=len(annotations),
            )

        except Exception as e:
            logger.exception(f"Failed to export Raven table: {e}")
            return ServiceResult.fail(f"Failed to export Raven table: {e}")

    def export_csv(
        self,
        annotations: List[CoreAnnotation],
        output_path: str,
        include_custom_fields: bool = True,
    ) -> ServiceResult[str]:
        """
        Export annotations to a CSV file.

        Args:
            annotations: List of annotations to export
            output_path: Output file path (.csv)
            include_custom_fields: Include custom fields as additional columns

        Returns:
            ServiceResult containing the output file path
        """
        error = self._validate_output_path(output_path)
        if error:
            return ServiceResult.fail(error)

        try:
            saved_path = save_csv_annotations(
                annotations, output_path, include_custom_fields=include_custom_fields
            )

            return ServiceResult.ok(
                data=saved_path,
                message=f"Exported {len(annotations)} annotations to CSV",
                count=len(annotations),
            )

        except Exception as e:
            logger.exception(f"Failed to export CSV: {e}")
            return ServiceResult.fail(f"Failed to export CSV: {e}")

    def export_parquet(
        self,
        annotations: List[CoreAnnotation],
        output_path: str,
    ) -> ServiceResult[str]:
        """
        Export annotations to a Parquet file.

        Args:
            annotations: List of annotations to export
            output_path: Output file path (.parquet)

        Returns:
            ServiceResult containing the output file path
        """
        error = self._validate_output_path(output_path)
        if error:
            return ServiceResult.fail(error)

        try:
            import pandas as pd

            # Convert annotations to list of dicts
            data = [ann.to_dict() for ann in annotations]
            df = pd.DataFrame(data)

            # Ensure output path
            path = Path(output_path)
            self.file_repository.mkdir(path.parent, parents=True)

            df.to_parquet(str(path), index=False)

            return ServiceResult.ok(
                data=str(path),
                message=f"Exported {len(annotations)} annotations to Parquet",
                count=len(annotations),
            )

        except ImportError:
            return ServiceResult.fail("pandas and pyarrow required for Parquet export")
        except Exception as e:
            logger.exception(f"Failed to export Parquet: {e}")
            return ServiceResult.fail(f"Failed to export Parquet: {e}")

    def export_json(
        self,
        annotations: List[CoreAnnotation],
        output_path: str,
    ) -> ServiceResult[str]:
        """
        Export annotations to a JSON file.

        Args:
            annotations: List of annotations to export
            output_path: Output file path (.json)

        Returns:
            ServiceResult containing the output file path
        """
        error = self._validate_output_path(output_path)
        if error:
            return ServiceResult.fail(error)

        try:
            path = Path(output_path)
            self.file_repository.mkdir(path.parent, parents=True)

            data = [ann.to_dict() for ann in annotations]
            content = json.dumps(data, indent=2, default=str)
            self.file_repository.write_text(path, content)

            return ServiceResult.ok(
                data=str(path),
                message=f"Exported {len(annotations)} annotations to JSON",
                count=len(annotations),
            )

        except Exception as e:
            logger.exception(f"Failed to export JSON: {e}")
            return ServiceResult.fail(f"Failed to export JSON: {e}")

    # =========================================================================
    # Annotation Creation
    # =========================================================================

    def create_annotation(
        self,
        start_time: float,
        end_time: float,
        label: str = "",
        low_freq: Optional[float] = None,
        high_freq: Optional[float] = None,
        channel: int = 1,
        confidence: Optional[float] = None,
        notes: str = "",
    ) -> ServiceResult[CoreAnnotation]:
        """
        Create a new annotation.

        Args:
            start_time: Start time in seconds
            end_time: End time in seconds
            label: Label/class for the annotation
            low_freq: Lower frequency bound in Hz
            high_freq: Upper frequency bound in Hz
            channel: Audio channel (1-indexed)
            confidence: Confidence score (0.0-1.0)
            notes: Additional notes

        Returns:
            ServiceResult containing the created annotation
        """
        if end_time <= start_time:
            return ServiceResult.fail("end_time must be greater than start_time")

        if low_freq is not None and high_freq is not None and high_freq <= low_freq:
            return ServiceResult.fail("high_freq must be greater than low_freq")

        try:
            annotation = CoreAnnotation(
                start_time=start_time,
                end_time=end_time,
                low_freq=low_freq,
                high_freq=high_freq,
                label=label,
                channel=channel,
                confidence=confidence,
                notes=notes,
            )

            return ServiceResult.ok(data=annotation, message="Created annotation")

        except Exception as e:
            logger.exception(f"Failed to create annotation: {e}")
            return ServiceResult.fail(f"Failed to create annotation: {e}")

    # =========================================================================
    # Clip Extraction
    # =========================================================================

    def extract_clips(
        self,
        annotations: List[CoreAnnotation],
        audio_path: str,
        output_dir: str,
        padding_ms: float = 0.0,
        format: str = "wav",
        include_label_in_filename: bool = True,
    ) -> ServiceResult[ClipExtractionResult]:
        """
        Extract audio clips from annotations.

        Args:
            annotations: List of annotations to extract
            audio_path: Path to the source audio file
            output_dir: Directory for output clips
            padding_ms: Padding in milliseconds to add before/after each clip
            format: Output audio format (wav, flac, etc.)
            include_label_in_filename: Include label in clip filename

        Returns:
            ServiceResult containing ClipExtractionResult
        """
        error = self._validate_input_path(audio_path)
        if error:
            return ServiceResult.fail(error)

        error = self._validate_output_path(output_dir)
        if error:
            return ServiceResult.fail(error)

        try:
            import soundfile as sf

            # Load audio file
            audio_data, sample_rate = sf.read(audio_path)

            # Handle mono/stereo
            if audio_data.ndim == 1:
                audio_data = audio_data.reshape(-1, 1)

            output_path = Path(output_dir)
            self.file_repository.mkdir(output_path, parents=True)

            padding_samples = int(padding_ms * sample_rate / 1000)
            total_samples = len(audio_data)

            result = ClipExtractionResult(
                total_clips=len(annotations), output_directory=str(output_path)
            )

            audio_stem = Path(audio_path).stem

            for i, ann in enumerate(annotations):
                try:
                    # Calculate sample indices
                    start_sample = max(0, int(ann.start_time * sample_rate) - padding_samples)
                    end_sample = min(
                        total_samples, int(ann.end_time * sample_rate) + padding_samples
                    )

                    # Extract clip
                    clip = audio_data[start_sample:end_sample]

                    # Build filename
                    if include_label_in_filename and ann.label:
                        safe_label = ann.label.replace(" ", "_").replace("/", "-")
                        filename = f"{audio_stem}_{i:04d}_{safe_label}.{format}"
                    else:
                        filename = f"{audio_stem}_{i:04d}.{format}"

                    clip_path = output_path / filename

                    # Write clip
                    sf.write(str(clip_path), clip, sample_rate)
                    result.extracted_clips.append(str(clip_path))

                except Exception as e:
                    logger.warning(f"Failed to extract clip {i}: {e}")
                    result.failed_clips.append(f"Clip {i}: {e}")

            return ServiceResult.ok(
                data=result,
                message=f"Extracted {len(result.extracted_clips)} clips",
                extracted=len(result.extracted_clips),
                failed=len(result.failed_clips),
            )

        except Exception as e:
            logger.exception(f"Failed to extract clips: {e}")
            return ServiceResult.fail(f"Failed to extract clips: {e}")

    # =========================================================================
    # Measurement Operations
    # =========================================================================

    def compute_measurements(
        self,
        annotation: CoreAnnotation,
        audio_path: str,
        metrics: Optional[List[str]] = None,
    ) -> ServiceResult[MeasurementResult]:
        """
        Compute acoustic measurements for an annotation.

        Args:
            annotation: The annotation to measure
            audio_path: Path to the source audio file
            metrics: List of metrics to compute. If None, computes default set.
                     Options: duration, bandwidth, rms, peak, centroid, etc.

        Returns:
            ServiceResult containing MeasurementResult
        """
        error = self._validate_input_path(audio_path)
        if error:
            return ServiceResult.fail(error)

        try:
            import soundfile as sf
            from scipy import signal as scipy_signal

            # Load audio region
            audio_data, sample_rate = sf.read(audio_path)

            if audio_data.ndim == 1:
                audio_data = audio_data.reshape(-1, 1)

            # Extract the annotation region
            start_sample = int(annotation.start_time * sample_rate)
            end_sample = int(annotation.end_time * sample_rate)

            # Use specified channel or first channel
            channel_idx = min(annotation.channel - 1, audio_data.shape[1] - 1)
            clip = audio_data[start_sample:end_sample, channel_idx]

            # Default metrics
            if metrics is None:
                metrics = ["duration", "bandwidth", "rms", "peak", "centroid"]

            measurements = {}

            # Basic temporal measurements
            if "duration" in metrics:
                measurements["duration"] = annotation.duration

            if "bandwidth" in metrics and annotation.bandwidth is not None:
                measurements["bandwidth"] = annotation.bandwidth

            # Amplitude measurements
            if "rms" in metrics:
                measurements["rms"] = float(np.sqrt(np.mean(clip**2)))

            if "peak" in metrics:
                measurements["peak"] = float(np.max(np.abs(clip)))

            if "crest_factor" in metrics:
                rms = np.sqrt(np.mean(clip**2))
                peak = np.max(np.abs(clip))
                measurements["crest_factor"] = float(peak / rms) if rms > 0 else 0.0

            # Spectral measurements
            if any(m in metrics for m in ["centroid", "bandwidth_spectral", "rolloff"]):
                # Compute spectrum
                n_fft = min(2048, len(clip))
                freqs, psd = scipy_signal.welch(clip, sample_rate, nperseg=n_fft)

                # Apply frequency bounds if specified
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

            return ServiceResult.ok(
                data=MeasurementResult(measurements=measurements),
                message=f"Computed {len(measurements)} measurements",
            )

        except Exception as e:
            logger.exception(f"Failed to compute measurements: {e}")
            return ServiceResult.fail(f"Failed to compute measurements: {e}")

    # =========================================================================
    # Batch Operations
    # =========================================================================

    def import_batch(
        self,
        directory: str,
        file_pattern: str = "*.txt",
        format: str = "raven",
    ) -> ServiceResult[AnnotationResult]:
        """
        Import annotations from all matching files in a directory.

        Args:
            directory: Path to directory containing annotation files
            file_pattern: Glob pattern for matching files
            format: Annotation format ('raven' or 'csv')

        Returns:
            ServiceResult containing combined AnnotationResult
        """
        error = self._validate_input_path(directory)
        if error:
            return ServiceResult.fail(error)

        try:
            from bioamla.core.audio.annotations import load_annotations_from_directory

            all_annotations_dict = load_annotations_from_directory(
                directory, file_pattern=file_pattern, format=format
            )

            # Flatten all annotations
            all_annotations = []
            for _filename, annotations in all_annotations_dict.items():
                all_annotations.extend(annotations)

            summary = summarize_annotations(all_annotations)

            return ServiceResult.ok(
                data=AnnotationResult(
                    annotations=all_annotations, file_path=directory, summary=summary
                ),
                message=f"Imported {len(all_annotations)} annotations from {len(all_annotations_dict)} files",
                count=len(all_annotations),
                files=len(all_annotations_dict),
            )

        except Exception as e:
            logger.exception(f"Failed to batch import: {e}")
            return ServiceResult.fail(f"Failed to batch import: {e}")
