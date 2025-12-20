# controllers/audio_file.py
"""
Audio File Controller
=====================

Controller responsible for audio file I/O operations.

This controller manages:
- Opening and loading audio files from disk
- Saving audio data to files
- Writing processed audio to new locations
- Integration with the command pattern for undo/redo

Design principle: AudioFileController is the ONLY component that should write
audio data to permanent storage. AudioController handles in-memory transforms
but must delegate to AudioFileController for persistence.

Usage:
    from bioamla.controllers import AudioFileController, AudioController

    file_ctrl = AudioFileController()
    audio_ctrl = AudioController()

    # Load audio
    result = file_ctrl.open("input.wav")
    audio_data = result.data

    # Process in memory
    processed = audio_ctrl.apply_bandpass(audio_data, 500, 8000)

    # Save through file controller
    file_ctrl.save(processed, "output.wav")

    # Undo the save
    file_ctrl.undo()
"""
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from bioamla.commands.base import (
    Command,
    CommandResult,
    FileBackupMixin,
    UndoManager,
)
from .base import BaseController, ControllerResult


@dataclass
class AudioData:
    """
    Container for audio data with metadata.

    This is the primary data transfer object between controllers.
    AudioFileController produces AudioData, AudioController transforms it,
    and AudioFileController persists it.
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


# =============================================================================
# File Commands (for undo/redo support)
# =============================================================================


class SaveAudioCommand(Command, FileBackupMixin):
    """
    Command to save audio data to a file.

    Supports undo by backing up existing files before overwriting.
    """

    def __init__(
        self,
        audio_data: AudioData,
        output_path: str,
        format: Optional[str] = None,
        subtype: Optional[str] = None,
    ):
        self.audio_data = audio_data
        self.output_path = Path(output_path)
        self.format = format
        self.subtype = subtype
        self._file_existed = False
        self._backup_path: Optional[Path] = None
        super().__init__()

    @property
    def name(self) -> str:
        return "Save Audio"

    @property
    def description(self) -> str:
        return f"Save audio to {self.output_path.name}"

    def execute(self) -> CommandResult:
        """Save the audio data to file."""
        import soundfile as sf

        try:
            # Backup existing file if it exists
            if self.output_path.exists():
                self._file_existed = True
                self._backup_path = self._backup_file(self.output_path)

            # Ensure parent directory exists
            self.output_path.parent.mkdir(parents=True, exist_ok=True)

            # Determine format from extension if not specified
            format_to_use = self.format
            if format_to_use is None:
                ext = self.output_path.suffix.lower()
                format_map = {
                    ".wav": "WAV",
                    ".flac": "FLAC",
                    ".ogg": "OGG",
                    ".mp3": "MP3",
                }
                format_to_use = format_map.get(ext, "WAV")

            # Save audio
            sf.write(
                str(self.output_path),
                self.audio_data.samples,
                self.audio_data.sample_rate,
                format=format_to_use,
                subtype=self.subtype,
            )

            return CommandResult.ok(
                data=str(self.output_path),
                message=f"Saved audio to {self.output_path}",
            )

        except Exception as e:
            # Restore backup if save failed
            if self._backup_path:
                self._restore_file(self.output_path)
            return CommandResult.fail(f"Failed to save audio: {e}")

    def undo(self) -> None:
        """Undo the save operation."""
        if self._file_existed and self._backup_path:
            # Restore the original file
            self._restore_file(self.output_path)
        elif self.output_path.exists():
            # Delete the newly created file
            self.output_path.unlink()

        self._cleanup_backups()


class WriteAudioCommand(Command, FileBackupMixin):
    """
    Command to write audio with a specific transformation applied.

    Used when applying transforms and saving in one operation.
    """

    def __init__(
        self,
        input_path: str,
        output_path: str,
        transform: Callable[[np.ndarray, int], Tuple[np.ndarray, int]],
        transform_name: str = "transform",
    ):
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)
        self.transform = transform
        self.transform_name = transform_name
        self._file_existed = False
        self._backup_path: Optional[Path] = None
        super().__init__()

    @property
    def name(self) -> str:
        return f"Write Audio ({self.transform_name})"

    @property
    def description(self) -> str:
        return f"Apply {self.transform_name} and save to {self.output_path.name}"

    def execute(self) -> CommandResult:
        """Load, transform, and save audio."""
        import soundfile as sf

        try:
            # Load input
            audio, sr = sf.read(str(self.input_path), dtype="float32")

            # Apply transform
            processed, out_sr = self.transform(audio, sr)

            # Backup existing output if it exists
            if self.output_path.exists():
                self._file_existed = True
                self._backup_path = self._backup_file(self.output_path)

            # Ensure parent directory exists
            self.output_path.parent.mkdir(parents=True, exist_ok=True)

            # Save
            sf.write(str(self.output_path), processed, out_sr)

            return CommandResult.ok(
                data=str(self.output_path),
                message=f"Applied {self.transform_name} and saved to {self.output_path}",
            )

        except Exception as e:
            if self._backup_path:
                self._restore_file(self.output_path)
            return CommandResult.fail(f"Failed to write audio: {e}")

    def undo(self) -> None:
        """Undo the write operation."""
        if self._file_existed and self._backup_path:
            self._restore_file(self.output_path)
        elif self.output_path.exists():
            self.output_path.unlink()

        self._cleanup_backups()


class OverwriteAudioCommand(Command, FileBackupMixin):
    """
    Command to overwrite an audio file in place.

    Useful for applying edits to the source file directly.
    """

    def __init__(
        self,
        audio_data: AudioData,
        target_path: Optional[str] = None,
    ):
        self.audio_data = audio_data
        # Use source path if no target specified
        self.target_path = Path(target_path) if target_path else None
        if self.target_path is None and audio_data.source_path:
            self.target_path = Path(audio_data.source_path)
        self._backup_path: Optional[Path] = None
        super().__init__()

    @property
    def name(self) -> str:
        return "Overwrite Audio"

    @property
    def description(self) -> str:
        if self.target_path:
            return f"Overwrite {self.target_path.name}"
        return "Overwrite source audio file"

    def validate(self) -> Optional[str]:
        """Validate the command can execute."""
        if self.target_path is None:
            return "No target path specified and audio has no source path"
        if not self.target_path.exists():
            return f"Target file does not exist: {self.target_path}"
        return None

    def execute(self) -> CommandResult:
        """Overwrite the audio file."""
        import soundfile as sf

        if self.target_path is None:
            return CommandResult.fail("No target path available")

        try:
            # Always backup before overwriting
            self._backup_path = self._backup_file(self.target_path)

            # Write new data
            sf.write(
                str(self.target_path),
                self.audio_data.samples,
                self.audio_data.sample_rate,
            )

            return CommandResult.ok(
                data=str(self.target_path),
                message=f"Overwrote {self.target_path}",
            )

        except Exception as e:
            if self._backup_path:
                self._restore_file(self.target_path)
            return CommandResult.fail(f"Failed to overwrite audio: {e}")

    def undo(self) -> None:
        """Restore the original file."""
        if self._backup_path and self.target_path:
            self._restore_file(self.target_path)
        self._cleanup_backups()


# =============================================================================
# Audio File Controller
# =============================================================================


class AudioFileController(BaseController):
    """
    Controller for audio file I/O operations.

    Manages all file-based operations with undo/redo support:
    - Opening audio files
    - Saving audio data
    - Writing transformed audio
    - Overwriting source files

    This controller is the single point of responsibility for audio persistence.
    In-memory transforms should use AudioController, which produces AudioData
    objects that can then be saved through this controller.

    Example:
        file_ctrl = AudioFileController()

        # Open a file
        result = file_ctrl.open("recording.wav")
        if result.success:
            audio = result.data

        # After processing with AudioController...
        save_result = file_ctrl.save(processed_audio, "output.wav")

        # Undo the save
        file_ctrl.undo()
    """

    def __init__(self, max_undo_levels: int = 100):
        super().__init__()
        self._undo_manager = UndoManager(max_history=max_undo_levels)
        self._temp_dir: Optional[Path] = None

    @property
    def can_undo(self) -> bool:
        """Check if undo is available."""
        return self._undo_manager.can_undo

    @property
    def can_redo(self) -> bool:
        """Check if redo is available."""
        return self._undo_manager.can_redo

    @property
    def undo_history(self) -> List[str]:
        """Get list of undoable operations."""
        return [info.description for info in self._undo_manager.history]

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

    def open(self, filepath: str) -> ControllerResult[AudioData]:
        """
        Open an audio file and load its data.

        Args:
            filepath: Path to the audio file

        Returns:
            ControllerResult containing AudioData on success
        """
        import soundfile as sf

        error = self._validate_input_path(filepath)
        if error:
            return ControllerResult.fail(error)

        try:
            audio, sr = sf.read(filepath, dtype="float32")

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

            return ControllerResult.ok(
                data=audio_data,
                message=f"Loaded {filepath}",
                duration=audio_data.duration,
                sample_rate=sr,
            )

        except Exception as e:
            return ControllerResult.fail(f"Failed to open audio file: {e}")

    def save(
        self,
        audio_data: AudioData,
        output_path: str,
        format: Optional[str] = None,
    ) -> ControllerResult[str]:
        """
        Save audio data to a file.

        This operation is undoable.

        Args:
            audio_data: AudioData object to save
            output_path: Destination file path
            format: Audio format (auto-detected from extension if not specified)

        Returns:
            ControllerResult containing the output path on success
        """
        error = self._validate_output_path(output_path)
        if error:
            return ControllerResult.fail(error)

        cmd = SaveAudioCommand(audio_data, output_path, format=format)
        result = self._undo_manager.execute(cmd)

        if result.success:
            return ControllerResult.ok(
                data=result.data,
                message=result.message,
            )
        return ControllerResult.fail(result.error)

    def save_as(
        self,
        audio_data: AudioData,
        output_path: str,
        target_sample_rate: Optional[int] = None,
        format: Optional[str] = None,
    ) -> ControllerResult[str]:
        """
        Save audio data to a new file, optionally with format conversion.

        This operation is undoable.

        Args:
            audio_data: AudioData object to save
            output_path: Destination file path
            target_sample_rate: Resample to this rate (optional)
            format: Audio format (auto-detected from extension if not specified)

        Returns:
            ControllerResult containing the output path on success
        """
        error = self._validate_output_path(output_path)
        if error:
            return ControllerResult.fail(error)

        # Handle resampling if requested
        data_to_save = audio_data
        if target_sample_rate and target_sample_rate != audio_data.sample_rate:
            try:
                from bioamla.core.audio.signal import resample_audio

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
                return ControllerResult.fail(f"Resampling failed: {e}")

        cmd = SaveAudioCommand(data_to_save, output_path, format=format)
        result = self._undo_manager.execute(cmd)

        if result.success:
            return ControllerResult.ok(
                data=result.data,
                message=result.message,
                sample_rate=data_to_save.sample_rate,
            )
        return ControllerResult.fail(result.error)

    def overwrite(
        self,
        audio_data: AudioData,
        target_path: Optional[str] = None,
    ) -> ControllerResult[str]:
        """
        Overwrite an existing audio file.

        If target_path is not specified, uses the audio_data's source_path.
        This operation is undoable.

        Args:
            audio_data: AudioData object with modified samples
            target_path: Path to overwrite (defaults to source_path)

        Returns:
            ControllerResult containing the overwritten path on success
        """
        cmd = OverwriteAudioCommand(audio_data, target_path)

        validation_error = cmd.validate()
        if validation_error:
            return ControllerResult.fail(validation_error)

        result = self._undo_manager.execute(cmd)

        if result.success:
            return ControllerResult.ok(
                data=result.data,
                message=result.message,
            )
        return ControllerResult.fail(result.error)

    def write_with_transform(
        self,
        input_path: str,
        output_path: str,
        transform: Callable[[np.ndarray, int], Tuple[np.ndarray, int]],
        transform_name: str = "transform",
    ) -> ControllerResult[str]:
        """
        Load, transform, and save audio in one operation.

        This is useful for CLI operations that need undo support without
        keeping audio in memory.

        Args:
            input_path: Source audio file
            output_path: Destination audio file
            transform: Function (audio, sr) -> (processed_audio, new_sr)
            transform_name: Name for undo history

        Returns:
            ControllerResult containing the output path on success
        """
        input_error = self._validate_input_path(input_path)
        if input_error:
            return ControllerResult.fail(input_error)

        output_error = self._validate_output_path(output_path)
        if output_error:
            return ControllerResult.fail(output_error)

        cmd = WriteAudioCommand(input_path, output_path, transform, transform_name)
        result = self._undo_manager.execute(cmd)

        if result.success:
            return ControllerResult.ok(
                data=result.data,
                message=result.message,
            )
        return ControllerResult.fail(result.error)

    # =========================================================================
    # Temporary File Support
    # =========================================================================

    def create_temp_file(
        self,
        audio_data: AudioData,
        suffix: str = ".wav",
    ) -> ControllerResult[str]:
        """
        Create a temporary audio file.

        Useful when AudioController needs to use external tools that require
        file paths. These files are NOT tracked for undo/redo.

        Args:
            audio_data: AudioData to write
            suffix: File extension

        Returns:
            ControllerResult containing the temporary file path
        """
        import soundfile as sf

        try:
            temp_dir = self._get_temp_dir()
            temp_file = tempfile.NamedTemporaryFile(
                dir=temp_dir,
                suffix=suffix,
                delete=False,
            )
            temp_path = temp_file.name
            temp_file.close()

            sf.write(temp_path, audio_data.samples, audio_data.sample_rate)

            return ControllerResult.ok(
                data=temp_path,
                message=f"Created temporary file: {temp_path}",
            )

        except Exception as e:
            return ControllerResult.fail(f"Failed to create temp file: {e}")

    # =========================================================================
    # Undo/Redo
    # =========================================================================

    def undo(self) -> ControllerResult[str]:
        """
        Undo the last file operation.

        Returns:
            ControllerResult with undo status
        """
        result = self._undo_manager.undo()
        if result is None:
            return ControllerResult.fail("Nothing to undo")
        if result.success:
            return ControllerResult.ok(message=result.message)
        return ControllerResult.fail(result.error)

    def redo(self) -> ControllerResult[str]:
        """
        Redo the last undone operation.

        Returns:
            ControllerResult with redo status
        """
        result = self._undo_manager.redo()
        if result is None:
            return ControllerResult.fail("Nothing to redo")
        if result.success:
            return ControllerResult.ok(message=result.message)
        return ControllerResult.fail(result.error)

    def clear_history(self) -> None:
        """Clear the undo/redo history."""
        self._undo_manager.clear()
