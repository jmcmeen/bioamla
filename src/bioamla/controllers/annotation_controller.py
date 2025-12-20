# controllers/annotation_controller.py
"""
Annotation Controller
=====================

Controller for managing audio annotations with support for:
- CRUD operations on annotations
- Raven selection table import/export
- CSV/Parquet/JSON export
- Audio clip extraction from annotations
- Integration with database layer for persistence
- Measurement computation for annotations

This controller bridges the core annotation system (bioamla.core.annotations)
with the database layer (bioamla.database) and provides a clean interface
for CLI and GUI applications.
"""
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union
from uuid import UUID

import numpy as np

from bioamla.controllers.base import BaseController, BatchProgress, ControllerResult
from bioamla.core.annotations import (
    Annotation as CoreAnnotation,
    AnnotationSet,
    load_csv_annotations,
    load_raven_selection_table,
    save_csv_annotations,
    save_raven_selection_table,
    summarize_annotations,
)

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


class AnnotationController(BaseController):
    """
    Controller for annotation operations.

    Provides a unified interface for:
    - Loading and saving annotations in various formats
    - Creating, updating, and deleting annotations
    - Extracting audio clips from annotations
    - Computing measurements for annotations
    - Database persistence (when available)

    Usage:
        # Basic usage without database
        ctrl = AnnotationController()
        result = ctrl.import_raven("selections.txt")
        annotations = result.data.annotations

        # With database persistence
        from bioamla.database import DatabaseConnection, UnitOfWork
        db = DatabaseConnection("sqlite:///project.db")
        ctrl = AnnotationController(database=db)

        with ctrl.begin_transaction() as uow:
            result = ctrl.import_raven("selections.txt", recording_id=rec_id, uow=uow)
            uow.commit()
    """

    def __init__(self, database=None):
        """
        Initialize AnnotationController.

        Args:
            database: Optional DatabaseConnection for persistence.
                      If not provided, operations are in-memory only.
        """
        super().__init__()
        self._database = database
        self._annotations: List[CoreAnnotation] = []

    @property
    def has_database(self) -> bool:
        """Check if database is available."""
        return self._database is not None

    def begin_transaction(self):
        """
        Begin a database transaction.

        Returns:
            UnitOfWork context manager

        Raises:
            RuntimeError: If no database is configured
        """
        if not self.has_database:
            raise RuntimeError("No database configured for AnnotationController")

        from bioamla.database import UnitOfWork

        return UnitOfWork(self._database)

    # =========================================================================
    # Import Operations
    # =========================================================================

    def import_raven(
        self,
        filepath: str,
        label_column: Optional[str] = None,
        recording_id: Optional[UUID] = None,
        uow=None,
    ) -> ControllerResult[AnnotationResult]:
        """
        Import annotations from a Raven selection table.

        Args:
            filepath: Path to the Raven selection table (.txt)
            label_column: Optional column name to use for labels
            recording_id: Optional recording UUID for database persistence
            uow: Optional UnitOfWork for database transaction

        Returns:
            ControllerResult containing AnnotationResult with imported annotations
        """
        error = self._validate_input_path(filepath)
        if error:
            return ControllerResult.fail(error)

        try:
            annotations = load_raven_selection_table(filepath, label_column=label_column)

            # Persist to database if UoW provided
            if uow and recording_id:
                self._persist_annotations(annotations, recording_id, "import", filepath, uow)

            summary = summarize_annotations(annotations)

            return ControllerResult.ok(
                data=AnnotationResult(
                    annotations=annotations, file_path=filepath, summary=summary
                ),
                message=f"Imported {len(annotations)} annotations from Raven table",
                count=len(annotations),
            )

        except Exception as e:
            logger.exception(f"Failed to import Raven table: {e}")
            return ControllerResult.fail(f"Failed to import Raven table: {e}")

    def import_csv(
        self,
        filepath: str,
        start_time_col: str = "start_time",
        end_time_col: str = "end_time",
        low_freq_col: str = "low_freq",
        high_freq_col: str = "high_freq",
        label_col: str = "label",
        recording_id: Optional[UUID] = None,
        uow=None,
    ) -> ControllerResult[AnnotationResult]:
        """
        Import annotations from a CSV file.

        Args:
            filepath: Path to the CSV file
            start_time_col: Column name for start time
            end_time_col: Column name for end time
            low_freq_col: Column name for low frequency
            high_freq_col: Column name for high frequency
            label_col: Column name for label
            recording_id: Optional recording UUID for database persistence
            uow: Optional UnitOfWork for database transaction

        Returns:
            ControllerResult containing AnnotationResult with imported annotations
        """
        error = self._validate_input_path(filepath)
        if error:
            return ControllerResult.fail(error)

        try:
            annotations = load_csv_annotations(
                filepath,
                start_time_col=start_time_col,
                end_time_col=end_time_col,
                low_freq_col=low_freq_col,
                high_freq_col=high_freq_col,
                label_col=label_col,
            )

            # Persist to database if UoW provided
            if uow and recording_id:
                self._persist_annotations(annotations, recording_id, "import", filepath, uow)

            summary = summarize_annotations(annotations)

            return ControllerResult.ok(
                data=AnnotationResult(
                    annotations=annotations, file_path=filepath, summary=summary
                ),
                message=f"Imported {len(annotations)} annotations from CSV",
                count=len(annotations),
            )

        except Exception as e:
            logger.exception(f"Failed to import CSV: {e}")
            return ControllerResult.fail(f"Failed to import CSV: {e}")

    # =========================================================================
    # Export Operations
    # =========================================================================

    def export_raven(
        self,
        annotations: List[CoreAnnotation],
        output_path: str,
        include_custom_fields: bool = True,
    ) -> ControllerResult[str]:
        """
        Export annotations to a Raven selection table.

        Args:
            annotations: List of annotations to export
            output_path: Output file path (.txt)
            include_custom_fields: Include custom fields as additional columns

        Returns:
            ControllerResult containing the output file path
        """
        error = self._validate_output_path(output_path)
        if error:
            return ControllerResult.fail(error)

        try:
            saved_path = save_raven_selection_table(
                annotations, output_path, include_custom_fields=include_custom_fields
            )

            return ControllerResult.ok(
                data=saved_path,
                message=f"Exported {len(annotations)} annotations to Raven table",
                count=len(annotations),
            )

        except Exception as e:
            logger.exception(f"Failed to export Raven table: {e}")
            return ControllerResult.fail(f"Failed to export Raven table: {e}")

    def export_csv(
        self,
        annotations: List[CoreAnnotation],
        output_path: str,
        include_custom_fields: bool = True,
    ) -> ControllerResult[str]:
        """
        Export annotations to a CSV file.

        Args:
            annotations: List of annotations to export
            output_path: Output file path (.csv)
            include_custom_fields: Include custom fields as additional columns

        Returns:
            ControllerResult containing the output file path
        """
        error = self._validate_output_path(output_path)
        if error:
            return ControllerResult.fail(error)

        try:
            saved_path = save_csv_annotations(
                annotations, output_path, include_custom_fields=include_custom_fields
            )

            return ControllerResult.ok(
                data=saved_path,
                message=f"Exported {len(annotations)} annotations to CSV",
                count=len(annotations),
            )

        except Exception as e:
            logger.exception(f"Failed to export CSV: {e}")
            return ControllerResult.fail(f"Failed to export CSV: {e}")

    def export_parquet(
        self,
        annotations: List[CoreAnnotation],
        output_path: str,
    ) -> ControllerResult[str]:
        """
        Export annotations to a Parquet file.

        Args:
            annotations: List of annotations to export
            output_path: Output file path (.parquet)

        Returns:
            ControllerResult containing the output file path
        """
        error = self._validate_output_path(output_path)
        if error:
            return ControllerResult.fail(error)

        try:
            import pandas as pd

            # Convert annotations to list of dicts
            data = [ann.to_dict() for ann in annotations]
            df = pd.DataFrame(data)

            # Ensure output path
            path = Path(output_path)
            path.parent.mkdir(parents=True, exist_ok=True)

            df.to_parquet(str(path), index=False)

            return ControllerResult.ok(
                data=str(path),
                message=f"Exported {len(annotations)} annotations to Parquet",
                count=len(annotations),
            )

        except ImportError:
            return ControllerResult.fail("pandas and pyarrow required for Parquet export")
        except Exception as e:
            logger.exception(f"Failed to export Parquet: {e}")
            return ControllerResult.fail(f"Failed to export Parquet: {e}")

    def export_json(
        self,
        annotations: List[CoreAnnotation],
        output_path: str,
    ) -> ControllerResult[str]:
        """
        Export annotations to a JSON file.

        Args:
            annotations: List of annotations to export
            output_path: Output file path (.json)

        Returns:
            ControllerResult containing the output file path
        """
        error = self._validate_output_path(output_path)
        if error:
            return ControllerResult.fail(error)

        try:
            path = Path(output_path)
            path.parent.mkdir(parents=True, exist_ok=True)

            data = [ann.to_dict() for ann in annotations]

            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, default=str)

            return ControllerResult.ok(
                data=str(path),
                message=f"Exported {len(annotations)} annotations to JSON",
                count=len(annotations),
            )

        except Exception as e:
            logger.exception(f"Failed to export JSON: {e}")
            return ControllerResult.fail(f"Failed to export JSON: {e}")

    # =========================================================================
    # CRUD Operations
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
        recording_id: Optional[UUID] = None,
        uow=None,
    ) -> ControllerResult[CoreAnnotation]:
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
            recording_id: Recording UUID for database persistence
            uow: UnitOfWork for database transaction

        Returns:
            ControllerResult containing the created annotation
        """
        if end_time <= start_time:
            return ControllerResult.fail("end_time must be greater than start_time")

        if low_freq is not None and high_freq is not None and high_freq <= low_freq:
            return ControllerResult.fail("high_freq must be greater than low_freq")

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

            # Persist to database if UoW provided
            if uow and recording_id:
                db_annotation = self._persist_annotation(
                    annotation, recording_id, "manual", None, uow
                )
                return ControllerResult.ok(
                    data=annotation,
                    message="Created annotation",
                    db_id=str(db_annotation.id),
                )

            return ControllerResult.ok(data=annotation, message="Created annotation")

        except Exception as e:
            logger.exception(f"Failed to create annotation: {e}")
            return ControllerResult.fail(f"Failed to create annotation: {e}")

    def update_annotation(
        self,
        annotation_id: UUID,
        uow,
        **fields,
    ) -> ControllerResult[CoreAnnotation]:
        """
        Update an existing annotation in the database.

        Args:
            annotation_id: UUID of the annotation to update
            uow: UnitOfWork for database transaction
            **fields: Fields to update (start_time, end_time, label, etc.)

        Returns:
            ControllerResult containing the updated annotation
        """
        try:
            from bioamla.database.models import AnnotationUpdate

            update_data = AnnotationUpdate(**fields)
            db_annotation = uow.annotations.update_by_id(annotation_id, update_data)

            if db_annotation is None:
                return ControllerResult.fail(f"Annotation {annotation_id} not found")

            # Convert back to core annotation
            annotation = self._db_to_core_annotation(db_annotation)

            return ControllerResult.ok(data=annotation, message="Updated annotation")

        except Exception as e:
            logger.exception(f"Failed to update annotation: {e}")
            return ControllerResult.fail(f"Failed to update annotation: {e}")

    def delete_annotation(
        self,
        annotation_id: UUID,
        uow,
    ) -> ControllerResult[bool]:
        """
        Delete an annotation from the database.

        Args:
            annotation_id: UUID of the annotation to delete
            uow: UnitOfWork for database transaction

        Returns:
            ControllerResult containing True if deleted
        """
        try:
            deleted = uow.annotations.delete_by_id(annotation_id)

            if not deleted:
                return ControllerResult.fail(f"Annotation {annotation_id} not found")

            return ControllerResult.ok(data=True, message="Deleted annotation")

        except Exception as e:
            logger.exception(f"Failed to delete annotation: {e}")
            return ControllerResult.fail(f"Failed to delete annotation: {e}")

    def get_annotations(
        self,
        recording_id: UUID,
        uow,
        skip: int = 0,
        limit: int = 1000,
    ) -> ControllerResult[List[CoreAnnotation]]:
        """
        Get all annotations for a recording from the database.

        Args:
            recording_id: UUID of the recording
            uow: UnitOfWork for database query
            skip: Number of records to skip
            limit: Maximum records to return

        Returns:
            ControllerResult containing list of annotations
        """
        try:
            db_annotations = uow.annotations.get_by_recording(
                recording_id, skip=skip, limit=limit
            )

            annotations = [self._db_to_core_annotation(a) for a in db_annotations]

            return ControllerResult.ok(
                data=annotations,
                message=f"Retrieved {len(annotations)} annotations",
                count=len(annotations),
            )

        except Exception as e:
            logger.exception(f"Failed to get annotations: {e}")
            return ControllerResult.fail(f"Failed to get annotations: {e}")

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
    ) -> ControllerResult[ClipExtractionResult]:
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
            ControllerResult containing ClipExtractionResult
        """
        error = self._validate_input_path(audio_path)
        if error:
            return ControllerResult.fail(error)

        error = self._validate_output_path(output_dir)
        if error:
            return ControllerResult.fail(error)

        try:
            import soundfile as sf

            # Load audio file
            audio_data, sample_rate = sf.read(audio_path)

            # Handle mono/stereo
            if audio_data.ndim == 1:
                audio_data = audio_data.reshape(-1, 1)

            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

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

            return ControllerResult.ok(
                data=result,
                message=f"Extracted {len(result.extracted_clips)} clips",
                extracted=len(result.extracted_clips),
                failed=len(result.failed_clips),
            )

        except Exception as e:
            logger.exception(f"Failed to extract clips: {e}")
            return ControllerResult.fail(f"Failed to extract clips: {e}")

    # =========================================================================
    # Measurement Operations
    # =========================================================================

    def compute_measurements(
        self,
        annotation: CoreAnnotation,
        audio_path: str,
        metrics: Optional[List[str]] = None,
    ) -> ControllerResult[MeasurementResult]:
        """
        Compute acoustic measurements for an annotation.

        Args:
            annotation: The annotation to measure
            audio_path: Path to the source audio file
            metrics: List of metrics to compute. If None, computes default set.
                     Options: duration, bandwidth, rms, peak, centroid, etc.

        Returns:
            ControllerResult containing MeasurementResult
        """
        error = self._validate_input_path(audio_path)
        if error:
            return ControllerResult.fail(error)

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

            return ControllerResult.ok(
                data=MeasurementResult(measurements=measurements),
                message=f"Computed {len(measurements)} measurements",
            )

        except Exception as e:
            logger.exception(f"Failed to compute measurements: {e}")
            return ControllerResult.fail(f"Failed to compute measurements: {e}")

    # =========================================================================
    # Batch Operations
    # =========================================================================

    def import_batch(
        self,
        directory: str,
        file_pattern: str = "*.txt",
        format: str = "raven",
        recording_id: Optional[UUID] = None,
        uow=None,
    ) -> ControllerResult[AnnotationResult]:
        """
        Import annotations from all matching files in a directory.

        Args:
            directory: Path to directory containing annotation files
            file_pattern: Glob pattern for matching files
            format: Annotation format ('raven' or 'csv')
            recording_id: Optional recording UUID for database persistence
            uow: Optional UnitOfWork for database transaction

        Returns:
            ControllerResult containing combined AnnotationResult
        """
        error = self._validate_input_path(directory)
        if error:
            return ControllerResult.fail(error)

        try:
            from bioamla.core.annotations import load_annotations_from_directory

            all_annotations_dict = load_annotations_from_directory(
                directory, file_pattern=file_pattern, format=format
            )

            # Flatten all annotations
            all_annotations = []
            for filename, annotations in all_annotations_dict.items():
                all_annotations.extend(annotations)

            # Persist to database if UoW provided
            if uow and recording_id:
                self._persist_annotations(
                    all_annotations, recording_id, "import", directory, uow
                )

            summary = summarize_annotations(all_annotations)

            return ControllerResult.ok(
                data=AnnotationResult(
                    annotations=all_annotations, file_path=directory, summary=summary
                ),
                message=f"Imported {len(all_annotations)} annotations from {len(all_annotations_dict)} files",
                count=len(all_annotations),
                files=len(all_annotations_dict),
            )

        except Exception as e:
            logger.exception(f"Failed to batch import: {e}")
            return ControllerResult.fail(f"Failed to batch import: {e}")

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _persist_annotations(
        self,
        annotations: List[CoreAnnotation],
        recording_id: UUID,
        source: str,
        source_file: Optional[str],
        uow,
    ) -> List:
        """Persist annotations to database."""
        from bioamla.database.models import AnnotationCreate

        db_annotations = []
        for ann in annotations:
            create_data = AnnotationCreate(
                recording_id=recording_id,
                start_time=ann.start_time,
                end_time=ann.end_time,
                low_freq=ann.low_freq,
                high_freq=ann.high_freq,
                label=ann.label,
                channel=ann.channel,
                confidence=ann.confidence,
                notes=ann.notes,
                source=source,
                source_file=source_file,
                custom_fields=ann.custom_fields if ann.custom_fields else None,
            )
            db_ann = uow.annotations.create(create_data)
            db_annotations.append(db_ann)

        return db_annotations

    def _persist_annotation(
        self,
        annotation: CoreAnnotation,
        recording_id: UUID,
        source: str,
        source_file: Optional[str],
        uow,
    ):
        """Persist a single annotation to database."""
        from bioamla.database.models import AnnotationCreate

        create_data = AnnotationCreate(
            recording_id=recording_id,
            start_time=annotation.start_time,
            end_time=annotation.end_time,
            low_freq=annotation.low_freq,
            high_freq=annotation.high_freq,
            label=annotation.label,
            channel=annotation.channel,
            confidence=annotation.confidence,
            notes=annotation.notes,
            source=source,
            source_file=source_file,
            custom_fields=annotation.custom_fields if annotation.custom_fields else None,
        )
        return uow.annotations.create(create_data)

    def _db_to_core_annotation(self, db_annotation) -> CoreAnnotation:
        """Convert database annotation to core annotation."""
        return CoreAnnotation(
            start_time=db_annotation.start_time,
            end_time=db_annotation.end_time,
            low_freq=db_annotation.low_freq,
            high_freq=db_annotation.high_freq,
            label=db_annotation.label,
            channel=db_annotation.channel,
            confidence=db_annotation.confidence,
            notes=db_annotation.notes or "",
            custom_fields=db_annotation.custom_fields or {},
        )
