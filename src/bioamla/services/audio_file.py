# services/audio_file.py
"""
Service responsible for audio file I/O operations.
"""

import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np

from bioamla.repository.protocol import FileRepositoryProtocol

from .base import BaseService, ServiceResult


@dataclass
class AudioData:
    """
    Container for audio data with metadata.

    This is the primary data transfer object between services.
    AudioFileService produces AudioData, AudioTransformService transforms it,
    and AudioFileService persists it.
    """

    samples: np.ndarray
    sample_rate: int
    channels: int = 1
    source_path: Optional[str] = None
    is_modified: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def duration(self) -> float:
        """Duration in seconds."""
        return len(self.samples) / self.sample_rate

    @property
    def num_samples(self) -> int:
        """Total number of samples."""
        return len(self.samples)

    def copy(self) -> "AudioData":
        """Create a deep copy of the audio data."""
        return AudioData(
            samples=self.samples.copy(),
            sample_rate=self.sample_rate,
            channels=self.channels,
            source_path=self.source_path,
            is_modified=self.is_modified,
            metadata=self.metadata.copy(),
        )

    def mark_modified(self) -> "AudioData":
        """Return a copy marked as modified."""
        copy = self.copy()
        copy.is_modified = True
        return copy


class AudioFileService(BaseService):
    """
    Service for audio file I/O operations.

    Manages all file-based operations:
    - Opening audio files
    - Saving audio data
    - Writing transformed audio

    This service is the single point of responsibility for audio persistence.
    In-memory transforms should use AudioTransformService, which produces AudioData
    objects that can then be saved through this service.

    Example:
        from bioamla.repository.local import LocalFileRepository

        file_svc = AudioFileService(LocalFileRepository())

        # Open a file
        result = file_svc.open("recording.wav")
        if result.success:
            audio = result.data

        # After processing...
        save_result = file_svc.save(processed_audio, "output.wav")
    """

    def __init__(self, file_repository: FileRepositoryProtocol) -> None:
        """Initialize the service.

        Args:
            file_repository: File repository for all file I/O operations (required).
        """
        super().__init__(file_repository)
        self._temp_dir: Optional[Path] = None

    def _get_temp_dir(self) -> Path:
        """Get or create a temporary directory for intermediate files."""
        if self._temp_dir is None or not self._temp_dir.exists():
            self._temp_dir = Path(tempfile.mkdtemp(prefix="bioamla_audio_"))
        return self._temp_dir

    def cleanup(self) -> None:
        """Clean up temporary files."""
        if self._temp_dir and self._temp_dir.exists():
            shutil.rmtree(self._temp_dir)
            self._temp_dir = None

    # =========================================================================
    # File Operations
    # =========================================================================

    def open(self, filepath: str) -> ServiceResult[AudioData]:
        """
        Open an audio file and load its data.

        Args:
            filepath: Path to the audio file

        Returns:
            ServiceResult containing AudioData on success
        """
        from bioamla.adapters.pydub import load_audio

        error = self._validate_input_path(filepath)
        if error:
            return ServiceResult.fail(error)

        try:
            audio, sr = load_audio(filepath)

            # Ensure 1D for mono
            if audio.ndim == 1:
                channels = 1
            else:
                channels = audio.shape[1] if audio.shape[1] <= 2 else 1
                if channels == 1:
                    audio = audio.flatten()

            audio_data = AudioData(
                samples=audio,
                sample_rate=sr,
                channels=channels,
                source_path=str(Path(filepath).resolve()),
                is_modified=False,
                metadata={"original_duration": len(audio) / sr},
            )

            return ServiceResult.ok(
                data=audio_data,
                message=f"Loaded {filepath}",
                duration=audio_data.duration,
                sample_rate=sr,
            )

        except Exception as e:
            return ServiceResult.fail(f"Failed to open audio file: {e}")

    def save(
        self,
        audio_data: AudioData,
        output_path: str,
        format: Optional[str] = None,
    ) -> ServiceResult[str]:
        """
        Save audio data to a file.

        Args:
            audio_data: AudioData object to save
            output_path: Destination file path
            format: Audio format (auto-detected from extension if not specified)

        Returns:
            ServiceResult containing the output path on success
        """
        from bioamla.adapters.pydub import save_audio

        error = self._validate_output_path(output_path)
        if error:
            return ServiceResult.fail(error)

        try:
            output = Path(output_path)
            self.file_repository.mkdir(output.parent, parents=True)

            save_audio(
                str(output),
                audio_data.samples,
                audio_data.sample_rate,
                format=format,
            )

            return ServiceResult.ok(
                data=str(output),
                message=f"Saved audio to {output}",
            )

        except Exception as e:
            return ServiceResult.fail(f"Failed to save audio: {e}")

    def save_as(
        self,
        audio_data: AudioData,
        output_path: str,
        target_sample_rate: Optional[int] = None,
        format: Optional[str] = None,
    ) -> ServiceResult[str]:
        """
        Save audio data to a new file, optionally with format conversion.

        Args:
            audio_data: AudioData object to save
            output_path: Destination file path
            target_sample_rate: Resample to this rate (optional)
            format: Audio format (auto-detected from extension if not specified)

        Returns:
            ServiceResult containing the output path on success
        """
        # Handle resampling if requested
        data_to_save = audio_data
        if target_sample_rate and target_sample_rate != audio_data.sample_rate:
            try:
                from bioamla.core.signal import resample_audio

                resampled = resample_audio(
                    audio_data.samples,
                    audio_data.sample_rate,
                    target_sample_rate,
                )
                data_to_save = AudioData(
                    samples=resampled,
                    sample_rate=target_sample_rate,
                    channels=audio_data.channels,
                    source_path=audio_data.source_path,
                    is_modified=True,
                    metadata=audio_data.metadata.copy(),
                )
            except Exception as e:
                return ServiceResult.fail(f"Resampling failed: {e}")

        return self.save(data_to_save, output_path, format=format)

    def overwrite(
        self,
        audio_data: AudioData,
        target_path: Optional[str] = None,
    ) -> ServiceResult[str]:
        """
        Overwrite an existing audio file.

        If target_path is not specified, uses the audio_data's source_path.

        Args:
            audio_data: AudioData object with modified samples
            target_path: Path to overwrite (defaults to source_path)

        Returns:
            ServiceResult containing the overwritten path on success
        """
        path = target_path or audio_data.source_path
        if path is None:
            return ServiceResult.fail("No target path specified and audio has no source path")

        if not self.file_repository.exists(path):
            return ServiceResult.fail(f"Target file does not exist: {path}")

        return self.save(audio_data, path)

    def write_with_transform(
        self,
        input_path: str,
        output_path: str,
        transform: Callable[[np.ndarray, int], Tuple[np.ndarray, int]],
        transform_name: str = "transform",
    ) -> ServiceResult[str]:
        """
        Load, transform, and save audio in one operation.

        Args:
            input_path: Source audio file
            output_path: Destination audio file
            transform: Function (audio, sr) -> (processed_audio, new_sr)
            transform_name: Name for logging

        Returns:
            ServiceResult containing the output path on success
        """
        from bioamla.adapters.pydub import load_audio, save_audio

        input_error = self._validate_input_path(input_path)
        if input_error:
            return ServiceResult.fail(input_error)

        output_error = self._validate_output_path(output_path)
        if output_error:
            return ServiceResult.fail(output_error)

        try:
            # Load input
            audio, sr = load_audio(str(input_path))

            # Apply transform
            processed, out_sr = transform(audio, sr)

            # Ensure parent directory exists
            self.file_repository.mkdir(Path(output_path).parent, parents=True)

            # Save
            save_audio(str(output_path), processed, out_sr)

            return ServiceResult.ok(
                data=str(output_path),
                message=f"Applied {transform_name} and saved to {output_path}",
            )

        except Exception as e:
            return ServiceResult.fail(f"Failed to write audio: {e}")

    # =========================================================================
    # Temporary File Support
    # =========================================================================

    def create_temp_file(
        self,
        audio_data: AudioData,
        suffix: str = ".wav",
    ) -> ServiceResult[str]:
        """
        Create a temporary audio file.

        Useful when external tools require file paths.

        Args:
            audio_data: AudioData to write
            suffix: File extension

        Returns:
            ServiceResult containing the temporary file path
        """
        from bioamla.adapters.pydub import save_audio

        try:
            temp_dir = self._get_temp_dir()
            temp_file = tempfile.NamedTemporaryFile(
                dir=temp_dir,
                suffix=suffix,
                delete=False,
            )
            temp_path = temp_file.name
            temp_file.close()

            save_audio(temp_path, audio_data.samples, audio_data.sample_rate)

            return ServiceResult.ok(
                data=temp_path,
                message=f"Created temporary file: {temp_path}",
            )

        except Exception as e:
            return ServiceResult.fail(f"Failed to create temp file: {e}")
