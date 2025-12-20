# commands/audio.py
"""
Audio Operation Commands
========================

Command pattern implementations for audio processing operations.

These commands provide undoable audio operations that can be executed
through the UndoManager. They integrate with the controller layer to
perform actual audio processing while maintaining undo/redo support.

Usage:
    from bioamla.commands import UndoManager
    from bioamla.commands.audio import (
        ResampleCommand,
        NormalizeCommand,
        FilterCommand,
        TrimCommand,
        DenoiseCommand,
    )

    manager = UndoManager()

    # Resample audio
    cmd = ResampleCommand(
        input_path="input.wav",
        output_path="output.wav",
        target_sample_rate=16000,
    )
    result = manager.execute(cmd)

    # Undo if needed
    manager.undo()

Design Notes:
    - Commands operate on files (input_path -> output_path) for undo support
    - Each command backs up the output file before overwriting (if it exists)
    - Undo restores the original file or deletes the newly created file
    - For in-memory transforms without persistence, use AudioTransformController
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .base import Command, CommandResult, FileBackupMixin


class ResampleCommand(Command, FileBackupMixin):
    """
    Command to resample an audio file to a different sample rate.

    This command loads an audio file, resamples it to the target sample rate,
    and saves it to the output path. Supports undo by backing up existing files.

    Args:
        input_path: Path to the input audio file
        output_path: Path to save the resampled audio
        target_sample_rate: Target sample rate in Hz

    Example:
        cmd = ResampleCommand("input.wav", "output.wav", 16000)
        manager.execute(cmd)
    """

    def __init__(
        self,
        input_path: str,
        output_path: str,
        target_sample_rate: int,
    ):
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)
        self.target_sample_rate = target_sample_rate
        self._file_existed = False
        self._backup_path: Optional[Path] = None
        super().__init__()

    @property
    def name(self) -> str:
        return "Resample Audio"

    @property
    def description(self) -> str:
        return f"Resample {self.input_path.name} to {self.target_sample_rate}Hz"

    def validate(self) -> Optional[str]:
        """Validate command parameters."""
        if not self.input_path.exists():
            return f"Input file does not exist: {self.input_path}"
        if self.target_sample_rate <= 0:
            return f"Invalid target sample rate: {self.target_sample_rate}"
        return None

    def execute(self) -> CommandResult:
        """Execute the resample operation."""
        import soundfile as sf

        from bioamla.core.audio.signal import resample_audio

        try:
            # Load input audio
            audio, sr = sf.read(str(self.input_path), dtype="float32")

            # Skip if already at target rate
            if sr == self.target_sample_rate:
                return CommandResult.ok(
                    data=str(self.output_path),
                    message=f"Audio already at {self.target_sample_rate}Hz",
                )

            # Backup existing output if it exists
            if self.output_path.exists():
                self._file_existed = True
                self._backup_path = self._backup_file(self.output_path)

            # Resample
            resampled = resample_audio(audio, sr, self.target_sample_rate)

            # Ensure parent directory exists
            self.output_path.parent.mkdir(parents=True, exist_ok=True)

            # Save
            sf.write(str(self.output_path), resampled, self.target_sample_rate)

            return CommandResult.ok(
                data=str(self.output_path),
                message=f"Resampled from {sr}Hz to {self.target_sample_rate}Hz",
                original_sample_rate=sr,
            )

        except Exception as e:
            if self._backup_path:
                self._restore_file(self.output_path)
            return CommandResult.fail(f"Resample failed: {e}")

    def undo(self) -> None:
        """Undo the resample operation."""
        if self._file_existed and self._backup_path:
            self._restore_file(self.output_path)
        elif self.output_path.exists():
            self.output_path.unlink()
        self._cleanup_backups()


class NormalizeCommand(Command, FileBackupMixin):
    """
    Command to normalize audio amplitude.

    Supports both peak normalization and loudness (RMS) normalization.

    Args:
        input_path: Path to the input audio file
        output_path: Path to save the normalized audio
        target_db: Target loudness in dBFS (for loudness mode)
        target_peak: Target peak amplitude 0.0-1.0 (for peak mode)
        mode: Normalization mode ("peak" or "loudness")

    Example:
        # Peak normalization
        cmd = NormalizeCommand("input.wav", "output.wav", target_peak=0.95, mode="peak")

        # Loudness normalization
        cmd = NormalizeCommand("input.wav", "output.wav", target_db=-20, mode="loudness")
    """

    def __init__(
        self,
        input_path: str,
        output_path: str,
        target_db: float = -20.0,
        target_peak: float = 0.99,
        mode: str = "loudness",
    ):
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)
        self.target_db = target_db
        self.target_peak = target_peak
        self.mode = mode
        self._file_existed = False
        self._backup_path: Optional[Path] = None
        super().__init__()

    @property
    def name(self) -> str:
        return "Normalize Audio"

    @property
    def description(self) -> str:
        if self.mode == "peak":
            return f"Normalize {self.input_path.name} to peak {self.target_peak}"
        return f"Normalize {self.input_path.name} to {self.target_db}dB"

    def validate(self) -> Optional[str]:
        """Validate command parameters."""
        if not self.input_path.exists():
            return f"Input file does not exist: {self.input_path}"
        if self.mode not in ("peak", "loudness"):
            return f"Invalid mode: {self.mode}. Must be 'peak' or 'loudness'"
        if self.mode == "peak" and not (0 < self.target_peak <= 1.0):
            return f"Target peak must be between 0 and 1: {self.target_peak}"
        return None

    def execute(self) -> CommandResult:
        """Execute the normalize operation."""
        import soundfile as sf

        from bioamla.core.audio.signal import normalize_loudness, peak_normalize

        try:
            # Load input audio
            audio, sr = sf.read(str(self.input_path), dtype="float32")

            # Backup existing output if it exists
            if self.output_path.exists():
                self._file_existed = True
                self._backup_path = self._backup_file(self.output_path)

            # Normalize based on mode
            if self.mode == "peak":
                normalized = peak_normalize(audio, target_peak=self.target_peak)
                message = f"Peak normalized to {self.target_peak}"
            else:
                normalized = normalize_loudness(audio, sr, target_db=self.target_db)
                message = f"Loudness normalized to {self.target_db}dB"

            # Ensure parent directory exists
            self.output_path.parent.mkdir(parents=True, exist_ok=True)

            # Save
            sf.write(str(self.output_path), normalized, sr)

            return CommandResult.ok(
                data=str(self.output_path),
                message=message,
            )

        except Exception as e:
            if self._backup_path:
                self._restore_file(self.output_path)
            return CommandResult.fail(f"Normalize failed: {e}")

    def undo(self) -> None:
        """Undo the normalize operation."""
        if self._file_existed and self._backup_path:
            self._restore_file(self.output_path)
        elif self.output_path.exists():
            self.output_path.unlink()
        self._cleanup_backups()


class FilterCommand(Command, FileBackupMixin):
    """
    Command to apply frequency filtering to audio.

    Supports lowpass, highpass, and bandpass filtering.

    Args:
        input_path: Path to the input audio file
        output_path: Path to save the filtered audio
        filter_type: Type of filter ("lowpass", "highpass", or "bandpass")
        low_freq: Low cutoff frequency (for bandpass/highpass)
        high_freq: High cutoff frequency (for bandpass/lowpass)
        order: Filter order (default: 5)

    Example:
        # Bandpass filter
        cmd = FilterCommand(
            "input.wav", "output.wav",
            filter_type="bandpass",
            low_freq=500,
            high_freq=8000
        )

        # Lowpass filter
        cmd = FilterCommand(
            "input.wav", "output.wav",
            filter_type="lowpass",
            high_freq=4000
        )
    """

    def __init__(
        self,
        input_path: str,
        output_path: str,
        filter_type: str = "bandpass",
        low_freq: Optional[float] = None,
        high_freq: Optional[float] = None,
        order: int = 5,
    ):
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)
        self.filter_type = filter_type
        self.low_freq = low_freq
        self.high_freq = high_freq
        self.order = order
        self._file_existed = False
        self._backup_path: Optional[Path] = None
        super().__init__()

    @property
    def name(self) -> str:
        return f"{self.filter_type.capitalize()} Filter"

    @property
    def description(self) -> str:
        if self.filter_type == "bandpass":
            return f"Bandpass filter {self.input_path.name} ({self.low_freq}-{self.high_freq}Hz)"
        elif self.filter_type == "lowpass":
            return f"Lowpass filter {self.input_path.name} (<{self.high_freq}Hz)"
        else:
            return f"Highpass filter {self.input_path.name} (>{self.low_freq}Hz)"

    def validate(self) -> Optional[str]:
        """Validate command parameters."""
        if not self.input_path.exists():
            return f"Input file does not exist: {self.input_path}"
        if self.filter_type not in ("lowpass", "highpass", "bandpass"):
            return f"Invalid filter type: {self.filter_type}"
        if self.filter_type == "bandpass" and (self.low_freq is None or self.high_freq is None):
            return "Bandpass filter requires both low_freq and high_freq"
        if self.filter_type == "lowpass" and self.high_freq is None:
            return "Lowpass filter requires high_freq"
        if self.filter_type == "highpass" and self.low_freq is None:
            return "Highpass filter requires low_freq"
        if self.order < 1:
            return f"Filter order must be at least 1: {self.order}"
        return None

    def execute(self) -> CommandResult:
        """Execute the filter operation."""
        import soundfile as sf

        from bioamla.core.audio.signal import bandpass_filter, highpass_filter, lowpass_filter

        try:
            # Load input audio
            audio, sr = sf.read(str(self.input_path), dtype="float32")

            # Backup existing output if it exists
            if self.output_path.exists():
                self._file_existed = True
                self._backup_path = self._backup_file(self.output_path)

            # Apply filter based on type
            if self.filter_type == "bandpass":
                filtered = bandpass_filter(audio, sr, self.low_freq, self.high_freq, self.order)
                message = f"Applied bandpass filter {self.low_freq}-{self.high_freq}Hz"
            elif self.filter_type == "lowpass":
                filtered = lowpass_filter(audio, sr, self.high_freq, self.order)
                message = f"Applied lowpass filter <{self.high_freq}Hz"
            else:
                filtered = highpass_filter(audio, sr, self.low_freq, self.order)
                message = f"Applied highpass filter >{self.low_freq}Hz"

            # Ensure parent directory exists
            self.output_path.parent.mkdir(parents=True, exist_ok=True)

            # Save
            sf.write(str(self.output_path), filtered, sr)

            return CommandResult.ok(
                data=str(self.output_path),
                message=message,
            )

        except Exception as e:
            if self._backup_path:
                self._restore_file(self.output_path)
            return CommandResult.fail(f"Filter failed: {e}")

    def undo(self) -> None:
        """Undo the filter operation."""
        if self._file_existed and self._backup_path:
            self._restore_file(self.output_path)
        elif self.output_path.exists():
            self.output_path.unlink()
        self._cleanup_backups()


class TrimCommand(Command, FileBackupMixin):
    """
    Command to trim audio to a specific time range.

    Args:
        input_path: Path to the input audio file
        output_path: Path to save the trimmed audio
        start_time: Start time in seconds (default: 0)
        end_time: End time in seconds (default: end of file)

    Example:
        # Trim to first 30 seconds
        cmd = TrimCommand("input.wav", "output.wav", end_time=30.0)

        # Extract segment from 10s to 25s
        cmd = TrimCommand("input.wav", "output.wav", start_time=10.0, end_time=25.0)
    """

    def __init__(
        self,
        input_path: str,
        output_path: str,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
    ):
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)
        self.start_time = start_time
        self.end_time = end_time
        self._file_existed = False
        self._backup_path: Optional[Path] = None
        self._original_duration: float = 0.0
        super().__init__()

    @property
    def name(self) -> str:
        return "Trim Audio"

    @property
    def description(self) -> str:
        start = self.start_time or 0.0
        end = self.end_time or "end"
        return f"Trim {self.input_path.name} ({start}s to {end})"

    def validate(self) -> Optional[str]:
        """Validate command parameters."""
        if not self.input_path.exists():
            return f"Input file does not exist: {self.input_path}"
        if self.start_time is not None and self.start_time < 0:
            return f"Start time cannot be negative: {self.start_time}"
        if self.end_time is not None and self.end_time < 0:
            return f"End time cannot be negative: {self.end_time}"
        if (
            self.start_time is not None
            and self.end_time is not None
            and self.start_time >= self.end_time
        ):
            return f"Start time must be less than end time: {self.start_time} >= {self.end_time}"
        return None

    def execute(self) -> CommandResult:
        """Execute the trim operation."""
        import soundfile as sf

        from bioamla.core.audio.signal import trim_audio

        try:
            # Load input audio
            audio, sr = sf.read(str(self.input_path), dtype="float32")
            self._original_duration = len(audio) / sr

            # Backup existing output if it exists
            if self.output_path.exists():
                self._file_existed = True
                self._backup_path = self._backup_file(self.output_path)

            # Trim
            trimmed = trim_audio(audio, sr, start_time=self.start_time, end_time=self.end_time)

            # Ensure parent directory exists
            self.output_path.parent.mkdir(parents=True, exist_ok=True)

            # Save
            sf.write(str(self.output_path), trimmed, sr)

            start = self.start_time or 0.0
            end = self.end_time or self._original_duration
            new_duration = len(trimmed) / sr

            return CommandResult.ok(
                data=str(self.output_path),
                message=f"Trimmed to {start:.2f}s-{end:.2f}s ({new_duration:.2f}s)",
                original_duration=self._original_duration,
                new_duration=new_duration,
            )

        except Exception as e:
            if self._backup_path:
                self._restore_file(self.output_path)
            return CommandResult.fail(f"Trim failed: {e}")

    def undo(self) -> None:
        """Undo the trim operation."""
        if self._file_existed and self._backup_path:
            self._restore_file(self.output_path)
        elif self.output_path.exists():
            self.output_path.unlink()
        self._cleanup_backups()


class TrimSilenceCommand(Command, FileBackupMixin):
    """
    Command to trim silence from the beginning and end of audio.

    Args:
        input_path: Path to the input audio file
        output_path: Path to save the trimmed audio
        threshold_db: Silence threshold in dBFS (default: -40)

    Example:
        cmd = TrimSilenceCommand("input.wav", "output.wav", threshold_db=-35)
    """

    def __init__(
        self,
        input_path: str,
        output_path: str,
        threshold_db: float = -40.0,
    ):
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)
        self.threshold_db = threshold_db
        self._file_existed = False
        self._backup_path: Optional[Path] = None
        super().__init__()

    @property
    def name(self) -> str:
        return "Trim Silence"

    @property
    def description(self) -> str:
        return f"Trim silence from {self.input_path.name}"

    def validate(self) -> Optional[str]:
        """Validate command parameters."""
        if not self.input_path.exists():
            return f"Input file does not exist: {self.input_path}"
        return None

    def execute(self) -> CommandResult:
        """Execute the trim silence operation."""
        import soundfile as sf

        from bioamla.core.audio.signal import trim_silence

        try:
            # Load input audio
            audio, sr = sf.read(str(self.input_path), dtype="float32")
            original_duration = len(audio) / sr

            # Backup existing output if it exists
            if self.output_path.exists():
                self._file_existed = True
                self._backup_path = self._backup_file(self.output_path)

            # Trim silence
            trimmed = trim_silence(audio, sr, threshold_db=self.threshold_db)

            # Ensure parent directory exists
            self.output_path.parent.mkdir(parents=True, exist_ok=True)

            # Save
            sf.write(str(self.output_path), trimmed, sr)

            new_duration = len(trimmed) / sr
            removed = original_duration - new_duration

            return CommandResult.ok(
                data=str(self.output_path),
                message=f"Removed {removed:.2f}s of silence",
                original_duration=original_duration,
                new_duration=new_duration,
                removed_duration=removed,
            )

        except Exception as e:
            if self._backup_path:
                self._restore_file(self.output_path)
            return CommandResult.fail(f"Trim silence failed: {e}")

    def undo(self) -> None:
        """Undo the trim silence operation."""
        if self._file_existed and self._backup_path:
            self._restore_file(self.output_path)
        elif self.output_path.exists():
            self.output_path.unlink()
        self._cleanup_backups()


class DenoiseCommand(Command, FileBackupMixin):
    """
    Command to apply spectral noise reduction to audio.

    Args:
        input_path: Path to the input audio file
        output_path: Path to save the denoised audio
        strength: Noise reduction strength (0.0 to 2.0, default: 1.0)

    Example:
        cmd = DenoiseCommand("input.wav", "output.wav", strength=1.5)
    """

    def __init__(
        self,
        input_path: str,
        output_path: str,
        strength: float = 1.0,
    ):
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)
        self.strength = strength
        self._file_existed = False
        self._backup_path: Optional[Path] = None
        super().__init__()

    @property
    def name(self) -> str:
        return "Denoise Audio"

    @property
    def description(self) -> str:
        return f"Denoise {self.input_path.name} (strength={self.strength})"

    def validate(self) -> Optional[str]:
        """Validate command parameters."""
        if not self.input_path.exists():
            return f"Input file does not exist: {self.input_path}"
        if not (0 <= self.strength <= 2.0):
            return f"Strength must be between 0.0 and 2.0: {self.strength}"
        return None

    def execute(self) -> CommandResult:
        """Execute the denoise operation."""
        import soundfile as sf

        from bioamla.core.audio.signal import spectral_denoise

        try:
            # Load input audio
            audio, sr = sf.read(str(self.input_path), dtype="float32")

            # Backup existing output if it exists
            if self.output_path.exists():
                self._file_existed = True
                self._backup_path = self._backup_file(self.output_path)

            # Denoise
            denoised = spectral_denoise(audio, sr, noise_reduce_factor=self.strength)

            # Ensure parent directory exists
            self.output_path.parent.mkdir(parents=True, exist_ok=True)

            # Save
            sf.write(str(self.output_path), denoised, sr)

            return CommandResult.ok(
                data=str(self.output_path),
                message=f"Applied noise reduction (strength={self.strength})",
            )

        except Exception as e:
            if self._backup_path:
                self._restore_file(self.output_path)
            return CommandResult.fail(f"Denoise failed: {e}")

    def undo(self) -> None:
        """Undo the denoise operation."""
        if self._file_existed and self._backup_path:
            self._restore_file(self.output_path)
        elif self.output_path.exists():
            self.output_path.unlink()
        self._cleanup_backups()


class GainCommand(Command, FileBackupMixin):
    """
    Command to apply gain (volume adjustment) to audio.

    Args:
        input_path: Path to the input audio file
        output_path: Path to save the adjusted audio
        gain_db: Gain in decibels (positive = louder, negative = quieter)

    Example:
        # Increase volume by 6dB
        cmd = GainCommand("input.wav", "output.wav", gain_db=6.0)

        # Decrease volume by 3dB
        cmd = GainCommand("input.wav", "output.wav", gain_db=-3.0)
    """

    def __init__(
        self,
        input_path: str,
        output_path: str,
        gain_db: float,
    ):
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)
        self.gain_db = gain_db
        self._file_existed = False
        self._backup_path: Optional[Path] = None
        super().__init__()

    @property
    def name(self) -> str:
        return "Adjust Gain"

    @property
    def description(self) -> str:
        sign = "+" if self.gain_db >= 0 else ""
        return f"Adjust gain on {self.input_path.name} ({sign}{self.gain_db}dB)"

    def validate(self) -> Optional[str]:
        """Validate command parameters."""
        if not self.input_path.exists():
            return f"Input file does not exist: {self.input_path}"
        return None

    def execute(self) -> CommandResult:
        """Execute the gain adjustment operation."""
        import soundfile as sf

        try:
            # Load input audio
            audio, sr = sf.read(str(self.input_path), dtype="float32")

            # Backup existing output if it exists
            if self.output_path.exists():
                self._file_existed = True
                self._backup_path = self._backup_file(self.output_path)

            # Apply gain
            gain_linear = 10 ** (self.gain_db / 20)
            adjusted = audio * gain_linear

            # Clip to prevent clipping
            adjusted = np.clip(adjusted, -1.0, 1.0)

            # Ensure parent directory exists
            self.output_path.parent.mkdir(parents=True, exist_ok=True)

            # Save
            sf.write(str(self.output_path), adjusted, sr)

            sign = "+" if self.gain_db >= 0 else ""
            return CommandResult.ok(
                data=str(self.output_path),
                message=f"Applied {sign}{self.gain_db}dB gain",
            )

        except Exception as e:
            if self._backup_path:
                self._restore_file(self.output_path)
            return CommandResult.fail(f"Gain adjustment failed: {e}")

    def undo(self) -> None:
        """Undo the gain adjustment operation."""
        if self._file_existed and self._backup_path:
            self._restore_file(self.output_path)
        elif self.output_path.exists():
            self.output_path.unlink()
        self._cleanup_backups()


class ConvertToMonoCommand(Command, FileBackupMixin):
    """
    Command to convert stereo audio to mono.

    Args:
        input_path: Path to the input audio file
        output_path: Path to save the mono audio

    Example:
        cmd = ConvertToMonoCommand("stereo.wav", "mono.wav")
    """

    def __init__(
        self,
        input_path: str,
        output_path: str,
    ):
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)
        self._file_existed = False
        self._backup_path: Optional[Path] = None
        super().__init__()

    @property
    def name(self) -> str:
        return "Convert to Mono"

    @property
    def description(self) -> str:
        return f"Convert {self.input_path.name} to mono"

    def validate(self) -> Optional[str]:
        """Validate command parameters."""
        if not self.input_path.exists():
            return f"Input file does not exist: {self.input_path}"
        return None

    def execute(self) -> CommandResult:
        """Execute the mono conversion operation."""
        import soundfile as sf

        try:
            # Load input audio
            audio, sr = sf.read(str(self.input_path), dtype="float32")

            # Check if already mono
            if audio.ndim == 1:
                return CommandResult.ok(
                    data=str(self.output_path),
                    message="Audio is already mono",
                )

            # Backup existing output if it exists
            if self.output_path.exists():
                self._file_existed = True
                self._backup_path = self._backup_file(self.output_path)

            # Convert to mono by averaging channels
            mono = audio.mean(axis=-1)

            # Ensure parent directory exists
            self.output_path.parent.mkdir(parents=True, exist_ok=True)

            # Save
            sf.write(str(self.output_path), mono, sr)

            return CommandResult.ok(
                data=str(self.output_path),
                message="Converted to mono",
                original_channels=audio.shape[-1] if audio.ndim > 1 else 1,
            )

        except Exception as e:
            if self._backup_path:
                self._restore_file(self.output_path)
            return CommandResult.fail(f"Mono conversion failed: {e}")

    def undo(self) -> None:
        """Undo the mono conversion operation."""
        if self._file_existed and self._backup_path:
            self._restore_file(self.output_path)
        elif self.output_path.exists():
            self.output_path.unlink()
        self._cleanup_backups()


@dataclass
class AudioProcessingPipeline:
    """
    A pipeline for chaining multiple audio commands.

    This is a convenience class for building complex audio processing
    workflows that can be executed as a single undoable operation.

    Example:
        pipeline = AudioProcessingPipeline("input.wav", "output.wav")
        pipeline.resample(16000)
        pipeline.bandpass(500, 8000)
        pipeline.normalize()

        from bioamla.commands import UndoManager
        manager = UndoManager()
        result = manager.execute(pipeline.build())
    """

    input_path: str
    output_path: str
    _operations: List[Dict[str, Any]] = field(default_factory=list)

    def resample(self, target_sample_rate: int) -> "AudioProcessingPipeline":
        """Add resample operation to pipeline."""
        self._operations.append(
            {
                "type": "resample",
                "target_sample_rate": target_sample_rate,
            }
        )
        return self

    def normalize(
        self,
        target_db: float = -20.0,
        target_peak: float = 0.99,
        mode: str = "loudness",
    ) -> "AudioProcessingPipeline":
        """Add normalize operation to pipeline."""
        self._operations.append(
            {
                "type": "normalize",
                "target_db": target_db,
                "target_peak": target_peak,
                "mode": mode,
            }
        )
        return self

    def bandpass(
        self, low_freq: float, high_freq: float, order: int = 5
    ) -> "AudioProcessingPipeline":
        """Add bandpass filter to pipeline."""
        self._operations.append(
            {
                "type": "filter",
                "filter_type": "bandpass",
                "low_freq": low_freq,
                "high_freq": high_freq,
                "order": order,
            }
        )
        return self

    def lowpass(self, high_freq: float, order: int = 5) -> "AudioProcessingPipeline":
        """Add lowpass filter to pipeline."""
        self._operations.append(
            {
                "type": "filter",
                "filter_type": "lowpass",
                "high_freq": high_freq,
                "order": order,
            }
        )
        return self

    def highpass(self, low_freq: float, order: int = 5) -> "AudioProcessingPipeline":
        """Add highpass filter to pipeline."""
        self._operations.append(
            {
                "type": "filter",
                "filter_type": "highpass",
                "low_freq": low_freq,
                "order": order,
            }
        )
        return self

    def trim(
        self,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
    ) -> "AudioProcessingPipeline":
        """Add trim operation to pipeline."""
        self._operations.append(
            {
                "type": "trim",
                "start_time": start_time,
                "end_time": end_time,
            }
        )
        return self

    def trim_silence(self, threshold_db: float = -40.0) -> "AudioProcessingPipeline":
        """Add trim silence operation to pipeline."""
        self._operations.append(
            {
                "type": "trim_silence",
                "threshold_db": threshold_db,
            }
        )
        return self

    def denoise(self, strength: float = 1.0) -> "AudioProcessingPipeline":
        """Add denoise operation to pipeline."""
        self._operations.append(
            {
                "type": "denoise",
                "strength": strength,
            }
        )
        return self

    def gain(self, gain_db: float) -> "AudioProcessingPipeline":
        """Add gain adjustment to pipeline."""
        self._operations.append(
            {
                "type": "gain",
                "gain_db": gain_db,
            }
        )
        return self

    def to_mono(self) -> "AudioProcessingPipeline":
        """Add mono conversion to pipeline."""
        self._operations.append(
            {
                "type": "to_mono",
            }
        )
        return self

    def build(self) -> "PipelineCommand":
        """Build the pipeline into an executable command."""
        return PipelineCommand(
            input_path=self.input_path,
            output_path=self.output_path,
            operations=self._operations,
        )


class PipelineCommand(Command, FileBackupMixin):
    """
    Command to execute an audio processing pipeline.

    This command applies a series of audio operations in sequence,
    storing the result in the output file. The entire pipeline is
    treated as a single undoable operation.
    """

    def __init__(
        self,
        input_path: str,
        output_path: str,
        operations: List[Dict[str, Any]],
    ):
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)
        self.operations = operations
        self._file_existed = False
        self._backup_path: Optional[Path] = None
        super().__init__()

    @property
    def name(self) -> str:
        return "Audio Pipeline"

    @property
    def description(self) -> str:
        op_names = [op["type"] for op in self.operations[:3]]
        if len(self.operations) > 3:
            op_names.append(f"...+{len(self.operations) - 3} more")
        return f"Pipeline: {' -> '.join(op_names)}"

    def validate(self) -> Optional[str]:
        """Validate command parameters."""
        if not self.input_path.exists():
            return f"Input file does not exist: {self.input_path}"
        if not self.operations:
            return "Pipeline has no operations"
        return None

    def execute(self) -> CommandResult:
        """Execute the pipeline."""
        import soundfile as sf

        from bioamla.core.audio.signal import (
            bandpass_filter,
            highpass_filter,
            lowpass_filter,
            normalize_loudness,
            peak_normalize,
            resample_audio,
            spectral_denoise,
            trim_audio,
            trim_silence,
        )

        try:
            # Load input audio
            audio, sr = sf.read(str(self.input_path), dtype="float32")

            # Backup existing output if it exists
            if self.output_path.exists():
                self._file_existed = True
                self._backup_path = self._backup_file(self.output_path)

            # Apply operations in sequence
            applied = []
            for op in self.operations:
                op_type = op["type"]

                if op_type == "resample":
                    target_sr = op["target_sample_rate"]
                    audio = resample_audio(audio, sr, target_sr)
                    sr = target_sr
                    applied.append(f"resample({target_sr}Hz)")

                elif op_type == "normalize":
                    if op["mode"] == "peak":
                        audio = peak_normalize(audio, target_peak=op["target_peak"])
                        applied.append(f"peak_norm({op['target_peak']})")
                    else:
                        audio = normalize_loudness(audio, sr, target_db=op["target_db"])
                        applied.append(f"loudness_norm({op['target_db']}dB)")

                elif op_type == "filter":
                    filter_type = op["filter_type"]
                    order = op.get("order", 5)
                    if filter_type == "bandpass":
                        audio = bandpass_filter(audio, sr, op["low_freq"], op["high_freq"], order)
                        applied.append(f"bandpass({op['low_freq']}-{op['high_freq']}Hz)")
                    elif filter_type == "lowpass":
                        audio = lowpass_filter(audio, sr, op["high_freq"], order)
                        applied.append(f"lowpass({op['high_freq']}Hz)")
                    else:
                        audio = highpass_filter(audio, sr, op["low_freq"], order)
                        applied.append(f"highpass({op['low_freq']}Hz)")

                elif op_type == "trim":
                    audio = trim_audio(
                        audio,
                        sr,
                        start_time=op.get("start_time"),
                        end_time=op.get("end_time"),
                    )
                    applied.append("trim")

                elif op_type == "trim_silence":
                    audio = trim_silence(audio, sr, threshold_db=op.get("threshold_db", -40.0))
                    applied.append("trim_silence")

                elif op_type == "denoise":
                    audio = spectral_denoise(audio, sr, noise_reduce_factor=op.get("strength", 1.0))
                    applied.append("denoise")

                elif op_type == "gain":
                    gain_linear = 10 ** (op["gain_db"] / 20)
                    audio = np.clip(audio * gain_linear, -1.0, 1.0)
                    applied.append(f"gain({op['gain_db']}dB)")

                elif op_type == "to_mono":
                    if audio.ndim > 1:
                        audio = audio.mean(axis=-1)
                    applied.append("mono")

            # Ensure parent directory exists
            self.output_path.parent.mkdir(parents=True, exist_ok=True)

            # Save
            sf.write(str(self.output_path), audio, sr)

            return CommandResult.ok(
                data=str(self.output_path),
                message=f"Applied pipeline: {' -> '.join(applied)}",
                operations_applied=applied,
            )

        except Exception as e:
            if self._backup_path:
                self._restore_file(self.output_path)
            return CommandResult.fail(f"Pipeline failed: {e}")

    def undo(self) -> None:
        """Undo the pipeline operation."""
        if self._file_existed and self._backup_path:
            self._restore_file(self.output_path)
        elif self.output_path.exists():
            self.output_path.unlink()
        self._cleanup_backups()
