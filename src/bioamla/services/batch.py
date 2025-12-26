# services/batch.py
"""
Service for batch processing operations across multiple files.

"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from .base import BaseService, ProgressCallback, ServiceResult


@dataclass
class BatchResult:
    """Result of a batch processing operation."""

    files_processed: int = 0
    files_failed: int = 0
    errors: List[str] = field(default_factory=list)
    output_dir: Optional[str] = None


class BatchService(BaseService):
    """
    Service for batch processing operations.

    Orchestrates other services for iterative file processing with
    progress tracking and error handling.
    """

    def __init__(self) -> None:
        super().__init__()

    def audio_convert(
        self,
        input_dir: str,
        output_dir: str,
        format: str = "wav",
        sample_rate: Optional[int] = None,
        channels: Optional[int] = None,
        recursive: bool = True,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> ServiceResult[BatchResult]:
        """
        Batch convert audio files.

        Args:
            input_dir: Input directory containing audio files
            output_dir: Output directory for converted files
            format: Output format (wav, mp3, flac, ogg)
            sample_rate: Target sample rate (optional)
            channels: Target number of channels (optional)
            recursive: Search subdirectories
            progress_callback: Optional callback for progress updates

        Returns:
            Result with batch processing summary
        """
        from bioamla.repository.local import LocalFileRepository
        from bioamla.services.audio_file import AudioFileService

        error = self._validate_input_path(input_dir)
        if error:
            return ServiceResult.fail(error)

        try:
            import numpy as np

            # Get audio files
            files = self._get_audio_files(input_dir, recursive=recursive)
            if not files:
                return ServiceResult.fail(f"No audio files found in {input_dir}")

            # Create output directory
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            # Initialize file service with repository
            repository = LocalFileRepository()
            file_svc = AudioFileService(file_repository=repository)

            processed = 0
            failed = 0
            errors = []

            for filepath in files:
                try:
                    # Calculate relative path for output
                    try:
                        rel_path = filepath.relative_to(input_dir)
                    except ValueError:
                        rel_path = Path(filepath.name)

                    out_file = output_path / rel_path.with_suffix(f".{format}")
                    out_file.parent.mkdir(parents=True, exist_ok=True)

                    # Load audio
                    open_result = file_svc.open(str(filepath))
                    if not open_result.success:
                        errors.append(f"{filepath.name}: {open_result.error}")
                        failed += 1
                        continue

                    audio_data = open_result.data

                    # Handle channel conversion if requested
                    if channels is not None and channels != audio_data.channels:
                        if channels == 1 and audio_data.channels == 2:
                            if audio_data.samples.ndim == 2:
                                audio_data.samples = audio_data.samples.mean(axis=1)
                            audio_data.channels = 1
                        elif channels == 2 and audio_data.channels == 1:
                            audio_data.samples = np.column_stack(
                                [audio_data.samples, audio_data.samples]
                            )
                            audio_data.channels = 2

                    # Save with conversion
                    save_result = file_svc.save_as(
                        audio_data=audio_data,
                        output_path=str(out_file),
                        target_sample_rate=sample_rate,
                        format=format,
                    )

                    if save_result.success:
                        processed += 1
                    else:
                        errors.append(f"{filepath.name}: {save_result.error}")
                        failed += 1

                except Exception as e:
                    errors.append(f"{filepath.name}: {e}")
                    failed += 1

            return ServiceResult.ok(
                data=BatchResult(
                    files_processed=processed,
                    files_failed=failed,
                    errors=errors,
                    output_dir=str(output_path),
                ),
                message=f"Converted {processed} files, {failed} failed",
            )

        except Exception as e:
            return ServiceResult.fail(str(e))
