# controllers/audio.py
"""
Audio Controller
================

Controller for audio processing operations.

Orchestrates between CLI/API views and core audio processing functions.
Handles file I/O, batch processing, and output formatting.
"""

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from .base import BaseController, ControllerResult


@dataclass
class AudioMetadata:
    """Metadata about an audio file."""

    filepath: str
    duration_seconds: float
    sample_rate: int
    channels: int
    bit_depth: Optional[int] = None
    format: Optional[str] = None


@dataclass
class ProcessedAudio:
    """Result of processing an audio file."""

    input_path: str
    output_path: str
    operation: str
    sample_rate: int
    duration_seconds: float


@dataclass
class AnalysisResult:
    """Result of audio analysis."""

    filepath: str
    duration_seconds: float
    sample_rate: int
    channels: int
    rms_db: float
    peak_db: float
    silence_ratio: float
    frequency_stats: Dict[str, float]


@dataclass
class BatchResult:
    """Result of a batch operation."""

    processed: int
    failed: int
    output_path: Optional[str] = None
    errors: List[str] = None

    def __post_init__(self):
        if self.errors is None:
            self.errors = []


class AudioController(BaseController):
    """
    Controller for audio processing operations.

    Provides high-level methods for:
    - File listing and metadata extraction
    - Audio format conversion
    - Signal processing (resample, filter, normalize, etc.)
    - Batch processing with progress
    - Analysis and visualization
    """

    # =========================================================================
    # File Operations
    # =========================================================================

    def list_files(
        self,
        directory: str,
        recursive: bool = True,
    ) -> ControllerResult[List[str]]:
        """
        List audio files in a directory.

        Args:
            directory: Directory path to search
            recursive: Search subdirectories

        Returns:
            Result with list of audio file paths
        """
        error = self._validate_input_path(directory)
        if error:
            return ControllerResult.fail(error)

        try:
            files = self._get_audio_files(directory, recursive=recursive)
            return ControllerResult.ok(
                data=[str(f) for f in files],
                message=f"Found {len(files)} audio files",
                count=len(files),
            )
        except Exception as e:
            return ControllerResult.fail(str(e))

    def get_metadata(self, filepath: str) -> ControllerResult[AudioMetadata]:
        """
        Get metadata from an audio file.

        Args:
            filepath: Path to audio file

        Returns:
            Result with audio metadata
        """
        error = self._validate_input_path(filepath)
        if error:
            return ControllerResult.fail(error)

        try:
            from bioamla.core.utils import get_wav_metadata

            metadata = get_wav_metadata(filepath)

            return ControllerResult.ok(
                data=AudioMetadata(
                    filepath=filepath,
                    duration_seconds=metadata.get("duration", 0),
                    sample_rate=metadata.get("sample_rate", 0),
                    channels=metadata.get("channels", 0),
                    bit_depth=metadata.get("bit_depth"),
                    format=metadata.get("format"),
                )
            )
        except Exception as e:
            return ControllerResult.fail(str(e))

    # =========================================================================
    # Signal Processing
    # =========================================================================

    def resample(
        self,
        input_path: str,
        output_path: str,
        target_rate: int,
    ) -> ControllerResult[ProcessedAudio]:
        """
        Resample an audio file to a different sample rate.

        Args:
            input_path: Input audio file path
            output_path: Output audio file path
            target_rate: Target sample rate in Hz

        Returns:
            Result with processed audio info
        """
        error = self._validate_input_path(input_path)
        if error:
            return ControllerResult.fail(error)

        error = self._validate_output_path(output_path)
        if error:
            return ControllerResult.fail(error)

        try:
            from bioamla.core.audio.signal import load_audio, resample_audio, save_audio

            audio, sr = load_audio(input_path)
            resampled = resample_audio(audio, sr, target_rate)
            save_audio(output_path, resampled, target_rate)

            duration = len(resampled) / target_rate

            return ControllerResult.ok(
                data=ProcessedAudio(
                    input_path=input_path,
                    output_path=output_path,
                    operation="resample",
                    sample_rate=target_rate,
                    duration_seconds=duration,
                ),
                message=f"Resampled {input_path} to {target_rate}Hz",
            )
        except Exception as e:
            return ControllerResult.fail(str(e))

    def resample_batch(
        self,
        input_dir: str,
        output_dir: str,
        target_rate: int,
        recursive: bool = True,
    ) -> ControllerResult[BatchResult]:
        """
        Resample multiple audio files.

        Args:
            input_dir: Input directory
            output_dir: Output directory
            target_rate: Target sample rate in Hz
            recursive: Search subdirectories

        Returns:
            Result with batch processing summary
        """
        error = self._validate_input_path(input_dir)
        if error:
            return ControllerResult.fail(error)

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        try:
            from bioamla.core.audio.signal import load_audio, resample_audio, save_audio

            files = self._get_audio_files(input_dir, recursive=recursive)
            if not files:
                return ControllerResult.fail(f"No audio files found in {input_dir}")

            processed = 0
            errors = []

            def process_file(filepath: Path) -> ProcessedAudio:
                nonlocal processed
                audio, sr = load_audio(str(filepath))
                resampled = resample_audio(audio, sr, target_rate)

                out_file = output_path / filepath.name
                save_audio(str(out_file), resampled, target_rate)

                processed += 1
                return ProcessedAudio(
                    input_path=str(filepath),
                    output_path=str(out_file),
                    operation="resample",
                    sample_rate=target_rate,
                    duration_seconds=len(resampled) / target_rate,
                )

            for filepath, _result, error in self._process_batch(files, process_file):
                if error:
                    errors.append(f"{filepath.name}: {error}")

            return ControllerResult.ok(
                data=BatchResult(
                    processed=processed,
                    failed=len(errors),
                    output_path=str(output_path),
                    errors=errors,
                ),
                message=f"Resampled {processed} files to {target_rate}Hz",
            )
        except Exception as e:
            return ControllerResult.fail(str(e))

    def normalize(
        self,
        input_path: str,
        output_path: str,
        target_db: float = -20.0,
        peak: bool = False,
    ) -> ControllerResult[ProcessedAudio]:
        """
        Normalize audio loudness.

        Args:
            input_path: Input audio file path
            output_path: Output audio file path
            target_db: Target loudness in dB
            peak: Use peak normalization instead of RMS

        Returns:
            Result with processed audio info
        """
        error = self._validate_input_path(input_path)
        if error:
            return ControllerResult.fail(error)

        error = self._validate_output_path(output_path)
        if error:
            return ControllerResult.fail(error)

        try:
            from bioamla.core.audio.signal import (
                load_audio,
                normalize_loudness,
                peak_normalize,
                save_audio,
            )

            audio, sr = load_audio(input_path)

            if peak:
                target_linear = 10 ** (target_db / 20)
                normalized = peak_normalize(audio, target_peak=min(target_linear, 0.99))
            else:
                normalized = normalize_loudness(audio, sr, target_db=target_db)

            save_audio(output_path, normalized, sr)
            duration = len(normalized) / sr

            return ControllerResult.ok(
                data=ProcessedAudio(
                    input_path=input_path,
                    output_path=output_path,
                    operation="normalize",
                    sample_rate=sr,
                    duration_seconds=duration,
                ),
                message=f"Normalized {input_path} to {target_db}dB",
            )
        except Exception as e:
            return ControllerResult.fail(str(e))

    def filter_audio(
        self,
        input_path: str,
        output_path: str,
        lowpass: Optional[float] = None,
        highpass: Optional[float] = None,
        bandpass: Optional[Tuple[float, float]] = None,
        order: int = 5,
    ) -> ControllerResult[ProcessedAudio]:
        """
        Apply frequency filter to audio.

        Args:
            input_path: Input audio file path
            output_path: Output audio file path
            lowpass: Lowpass cutoff frequency in Hz
            highpass: Highpass cutoff frequency in Hz
            bandpass: Tuple of (low, high) for bandpass filter
            order: Filter order

        Returns:
            Result with processed audio info
        """
        if not any([lowpass, highpass, bandpass]):
            return ControllerResult.fail("Must specify lowpass, highpass, or bandpass")

        error = self._validate_input_path(input_path)
        if error:
            return ControllerResult.fail(error)

        error = self._validate_output_path(output_path)
        if error:
            return ControllerResult.fail(error)

        try:
            from bioamla.core.audio.signal import (
                bandpass_filter,
                highpass_filter,
                load_audio,
                lowpass_filter,
                save_audio,
            )

            audio, sr = load_audio(input_path)

            if bandpass:
                filtered = bandpass_filter(audio, sr, bandpass[0], bandpass[1], order)
                filter_desc = f"bandpass {bandpass[0]}-{bandpass[1]}Hz"
            elif lowpass:
                filtered = lowpass_filter(audio, sr, lowpass, order)
                filter_desc = f"lowpass {lowpass}Hz"
            else:
                filtered = highpass_filter(audio, sr, highpass, order)
                filter_desc = f"highpass {highpass}Hz"

            save_audio(output_path, filtered, sr)
            duration = len(filtered) / sr

            return ControllerResult.ok(
                data=ProcessedAudio(
                    input_path=input_path,
                    output_path=output_path,
                    operation=f"filter ({filter_desc})",
                    sample_rate=sr,
                    duration_seconds=duration,
                ),
                message=f"Applied {filter_desc} to {input_path}",
            )
        except Exception as e:
            return ControllerResult.fail(str(e))

    def trim(
        self,
        input_path: str,
        output_path: str,
        start: Optional[float] = None,
        end: Optional[float] = None,
        trim_silence: bool = False,
        silence_threshold_db: float = -40.0,
    ) -> ControllerResult[ProcessedAudio]:
        """
        Trim audio by time or remove silence.

        Args:
            input_path: Input audio file path
            output_path: Output audio file path
            start: Start time in seconds
            end: End time in seconds
            trim_silence: Trim silence from start/end instead
            silence_threshold_db: Silence threshold in dB

        Returns:
            Result with processed audio info
        """
        if not trim_silence and start is None and end is None:
            return ControllerResult.fail("Must specify start/end or use trim_silence")

        error = self._validate_input_path(input_path)
        if error:
            return ControllerResult.fail(error)

        error = self._validate_output_path(output_path)
        if error:
            return ControllerResult.fail(error)

        try:
            from bioamla.core.audio.signal import (
                load_audio,
                save_audio,
                trim_audio,
            )
            from bioamla.core.audio.signal import (
                trim_silence as do_trim_silence,
            )

            audio, sr = load_audio(input_path)

            if trim_silence:
                trimmed = do_trim_silence(audio, sr, threshold_db=silence_threshold_db)
                operation = "trim silence"
            else:
                trimmed = trim_audio(audio, sr, start_time=start, end_time=end)
                operation = f"trim {start or 0}s-{end or 'end'}"

            save_audio(output_path, trimmed, sr)
            duration = len(trimmed) / sr

            return ControllerResult.ok(
                data=ProcessedAudio(
                    input_path=input_path,
                    output_path=output_path,
                    operation=operation,
                    sample_rate=sr,
                    duration_seconds=duration,
                ),
                message=f"Trimmed {input_path} ({operation})",
            )
        except Exception as e:
            return ControllerResult.fail(str(e))

    def denoise(
        self,
        input_path: str,
        output_path: str,
        strength: float = 1.0,
    ) -> ControllerResult[ProcessedAudio]:
        """
        Apply noise reduction to audio.

        Args:
            input_path: Input audio file path
            output_path: Output audio file path
            strength: Noise reduction strength (0-2, default: 1.0)

        Returns:
            Result with processed audio info
        """
        error = self._validate_input_path(input_path)
        if error:
            return ControllerResult.fail(error)

        error = self._validate_output_path(output_path)
        if error:
            return ControllerResult.fail(error)

        try:
            from bioamla.core.audio.signal import load_audio, save_audio, spectral_denoise

            audio, sr = load_audio(input_path)
            denoised = spectral_denoise(audio, sr, noise_reduce_factor=strength)
            save_audio(output_path, denoised, sr)

            duration = len(denoised) / sr

            return ControllerResult.ok(
                data=ProcessedAudio(
                    input_path=input_path,
                    output_path=output_path,
                    operation=f"denoise (strength={strength})",
                    sample_rate=sr,
                    duration_seconds=duration,
                ),
                message=f"Denoised {input_path}",
            )
        except Exception as e:
            return ControllerResult.fail(str(e))

    # =========================================================================
    # Analysis
    # =========================================================================

    def analyze(
        self,
        filepath: str,
        silence_threshold_db: float = -40.0,
    ) -> ControllerResult[AnalysisResult]:
        """
        Analyze an audio file.

        Args:
            filepath: Path to audio file
            silence_threshold_db: Silence detection threshold

        Returns:
            Result with analysis data
        """
        error = self._validate_input_path(filepath)
        if error:
            return ControllerResult.fail(error)

        try:
            from bioamla.core.audio import analyze_audio
            from bioamla.core.audio.signal import load_audio

            audio, sr = load_audio(filepath)
            analysis = analyze_audio(audio, sr, silence_threshold_db=silence_threshold_db)

            return ControllerResult.ok(
                data=AnalysisResult(
                    filepath=filepath,
                    duration_seconds=analysis.get("duration", 0),
                    sample_rate=sr,
                    channels=analysis.get("channels", 1),
                    rms_db=analysis.get("rms_db", 0),
                    peak_db=analysis.get("peak_db", 0),
                    silence_ratio=analysis.get("silence_ratio", 0),
                    frequency_stats=analysis.get("frequency_stats", {}),
                )
            )
        except Exception as e:
            return ControllerResult.fail(str(e))

    def analyze_batch(
        self,
        directory: str,
        output_csv: Optional[str] = None,
        recursive: bool = True,
    ) -> ControllerResult[BatchResult]:
        """
        Analyze multiple audio files.

        Args:
            directory: Directory containing audio files
            output_csv: Optional CSV output path
            recursive: Search subdirectories

        Returns:
            Result with batch processing summary
        """
        error = self._validate_input_path(directory)
        if error:
            return ControllerResult.fail(error)

        try:
            from bioamla.core.audio import analyze_audio
            from bioamla.core.audio.signal import load_audio
            from bioamla.core.files import TextFile

            files = self._get_audio_files(directory, recursive=recursive)
            if not files:
                return ControllerResult.fail(f"No audio files found in {directory}")

            results = []
            errors = []

            def process_file(filepath: Path) -> AnalysisResult:
                audio, sr = load_audio(str(filepath))
                analysis = analyze_audio(audio, sr)
                return AnalysisResult(
                    filepath=str(filepath),
                    duration_seconds=analysis.get("duration", 0),
                    sample_rate=sr,
                    channels=analysis.get("channels", 1),
                    rms_db=analysis.get("rms_db", 0),
                    peak_db=analysis.get("peak_db", 0),
                    silence_ratio=analysis.get("silence_ratio", 0),
                    frequency_stats=analysis.get("frequency_stats", {}),
                )

            for filepath, result, error in self._process_batch(files, process_file):
                if error:
                    errors.append(f"{filepath.name}: {error}")
                elif result:
                    results.append(result)

            # Write CSV if requested
            if output_csv and results:
                with TextFile(output_csv, mode="w", newline="") as f:
                    writer = csv.writer(f.handle)
                    writer.writerow(
                        [
                            "filepath",
                            "duration_s",
                            "sample_rate",
                            "channels",
                            "rms_db",
                            "peak_db",
                            "silence_ratio",
                        ]
                    )
                    for r in results:
                        writer.writerow(
                            [
                                r.filepath,
                                f"{r.duration_seconds:.2f}",
                                r.sample_rate,
                                r.channels,
                                f"{r.rms_db:.1f}",
                                f"{r.peak_db:.1f}",
                                f"{r.silence_ratio:.2f}",
                            ]
                        )

            return ControllerResult.ok(
                data=BatchResult(
                    processed=len(results),
                    failed=len(errors),
                    output_path=output_csv,
                    errors=errors,
                ),
                message=f"Analyzed {len(results)} files",
            )
        except Exception as e:
            return ControllerResult.fail(str(e))

    # =========================================================================
    # Generic Signal Processing
    # =========================================================================

    def process_batch(
        self,
        input_dir: str,
        output_dir: str,
        processor: Callable,
        operation_name: str = "process",
        recursive: bool = True,
        output_sample_rate: Optional[int] = None,
    ) -> ControllerResult[BatchResult]:
        """
        Apply a custom processor to multiple audio files.

        Args:
            input_dir: Input directory
            output_dir: Output directory
            processor: Function(audio, sr) -> processed_audio
            operation_name: Name of the operation for logging
            recursive: Search subdirectories
            output_sample_rate: Output sample rate (if different from input)

        Returns:
            Result with batch processing summary
        """
        error = self._validate_input_path(input_dir)
        if error:
            return ControllerResult.fail(error)

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        try:
            from bioamla.core.audio.signal import load_audio, save_audio

            files = self._get_audio_files(input_dir, recursive=recursive)
            if not files:
                return ControllerResult.fail(f"No audio files found in {input_dir}")

            processed = 0
            errors = []

            def process_file(filepath: Path):
                nonlocal processed
                audio, sr = load_audio(str(filepath))
                result = processor(audio, sr)

                out_sr = output_sample_rate or sr
                out_file = output_path / filepath.name
                save_audio(str(out_file), result, out_sr)

                processed += 1
                return out_file

            for filepath, _result, error in self._process_batch(files, process_file):
                if error:
                    errors.append(f"{filepath.name}: {error}")

            return ControllerResult.ok(
                data=BatchResult(
                    processed=processed,
                    failed=len(errors),
                    output_path=str(output_path),
                    errors=errors,
                ),
                message=f"{operation_name}: processed {processed} files",
            )
        except Exception as e:
            return ControllerResult.fail(str(e))
