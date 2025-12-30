# services/audio_transform.py
"""
Service for audio signal processing operations, both in-memory and file-based.

Uses OpenSoundscape adapters for audio processing operations.
"""

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from bioamla.models.audio import (
    AnalysisResult,
    AudioData,
    AudioMetadata,
    BatchResult,
    ProcessedAudio,
)
from bioamla.repository.protocol import FileRepositoryProtocol

from .base import BaseService, ServiceResult


class AudioTransformService(BaseService):
    """
    Service for audio signal processing operations.

    Provides both in-memory and file-based audio processing:

    In-memory operations (work on AudioData objects):
    - Filtering (lowpass, highpass, bandpass)
    - Normalization (peak, RMS/loudness)
    - Resampling
    - Trimming
    - Noise reduction
    - Gain adjustment
    - Channel operations (mono conversion)
    - Playback preparation

    File-based operations (single-file, require file_repository):
    - list_files: Discover audio files
    - get_metadata: Extract file metadata
    - resample_file: Resample single file
    - normalize_file: Normalize single file
    - filter_file: Filter single file
    - analyze_file: Analyze single file
    - segment_file: Segment single file
    - visualize_file: Visualize single file

    Analysis operations:
    - Get amplitude statistics
    - Get frequency statistics
    - Detect silence regions

    Note: Batch operations have been moved to BatchAudioTransformService.
    """

    def __init__(self, file_repository: FileRepositoryProtocol) -> None:
        """Initialize the audio transform service.

        Args:
            file_repository: File repository for file I/O operations (required).
        """
        super().__init__(file_repository=file_repository)
        if file_repository is None:
            raise ValueError("AudioTransformService requires a file_repository")

    # =========================================================================
    # Filtering Operations
    # =========================================================================

    def apply_lowpass(
        self,
        audio: AudioData,
        cutoff_hz: float,
        order: int = 5,
    ) -> ServiceResult[AudioData]:
        """
        Apply a lowpass filter using OpenSoundscape.

        Args:
            audio: Input AudioData
            cutoff_hz: Cutoff frequency in Hz
            order: Filter order

        Returns:
            ServiceResult with filtered AudioData
        """
        try:
            from bioamla.adapters.opensoundscape import AudioAdapter

            # Use OpenSoundscape adapter for filtering
            adapter = AudioAdapter.from_samples(audio.samples, audio.sample_rate)
            filtered_adapter = adapter.lowpass(cutoff_hz, order=order)
            filtered = filtered_adapter.to_samples()

            result_audio = AudioData(
                samples=filtered,
                sample_rate=audio.sample_rate,
                channels=audio.channels,
                source_path=audio.source_path,
                is_modified=True,
                metadata={
                    **audio.metadata,
                    "last_operation": f"lowpass_{cutoff_hz}Hz",
                },
            )

            return ServiceResult.ok(
                data=result_audio,
                message=f"Applied lowpass filter at {cutoff_hz}Hz",
            )

        except Exception as e:
            return ServiceResult.fail(f"Lowpass filter failed: {e}")

    def apply_highpass(
        self,
        audio: AudioData,
        cutoff_hz: float,
        order: int = 5,
    ) -> ServiceResult[AudioData]:
        """
        Apply a highpass filter using OpenSoundscape.

        Args:
            audio: Input AudioData
            cutoff_hz: Cutoff frequency in Hz
            order: Filter order

        Returns:
            ServiceResult with filtered AudioData
        """
        try:
            from bioamla.adapters.opensoundscape import AudioAdapter

            # Use OpenSoundscape adapter for filtering
            adapter = AudioAdapter.from_samples(audio.samples, audio.sample_rate)
            filtered_adapter = adapter.highpass(cutoff_hz, order=order)
            filtered = filtered_adapter.to_samples()

            result_audio = AudioData(
                samples=filtered,
                sample_rate=audio.sample_rate,
                channels=audio.channels,
                source_path=audio.source_path,
                is_modified=True,
                metadata={
                    **audio.metadata,
                    "last_operation": f"highpass_{cutoff_hz}Hz",
                },
            )

            return ServiceResult.ok(
                data=result_audio,
                message=f"Applied highpass filter at {cutoff_hz}Hz",
            )

        except Exception as e:
            return ServiceResult.fail(f"Highpass filter failed: {e}")

    def apply_bandpass(
        self,
        audio: AudioData,
        low_hz: float,
        high_hz: float,
        order: int = 5,
    ) -> ServiceResult[AudioData]:
        """
        Apply a bandpass filter using OpenSoundscape.

        Args:
            audio: Input AudioData
            low_hz: Lower cutoff frequency in Hz
            high_hz: Upper cutoff frequency in Hz
            order: Filter order

        Returns:
            ServiceResult with filtered AudioData
        """
        try:
            from bioamla.adapters.opensoundscape import AudioAdapter

            # Use OpenSoundscape adapter for filtering
            adapter = AudioAdapter.from_samples(audio.samples, audio.sample_rate)
            filtered_adapter = adapter.bandpass(low_hz, high_hz, order=order)
            filtered = filtered_adapter.to_samples()

            result_audio = AudioData(
                samples=filtered,
                sample_rate=audio.sample_rate,
                channels=audio.channels,
                source_path=audio.source_path,
                is_modified=True,
                metadata={
                    **audio.metadata,
                    "last_operation": f"bandpass_{low_hz}-{high_hz}Hz",
                },
            )

            return ServiceResult.ok(
                data=result_audio,
                message=f"Applied bandpass filter {low_hz}-{high_hz}Hz",
            )

        except Exception as e:
            return ServiceResult.fail(f"Bandpass filter failed: {e}")

    # =========================================================================
    # Normalization Operations
    # =========================================================================

    def normalize_peak(
        self,
        audio: AudioData,
        target_peak: float = 0.99,
    ) -> ServiceResult[AudioData]:
        """
        Normalize audio to a target peak amplitude using OpenSoundscape.

        Args:
            audio: Input AudioData
            target_peak: Target peak amplitude (0.0 to 1.0)

        Returns:
            ServiceResult with normalized AudioData
        """
        try:
            from bioamla.adapters.opensoundscape import AudioAdapter

            # Use OpenSoundscape adapter for normalization
            adapter = AudioAdapter.from_samples(audio.samples, audio.sample_rate)
            normalized_adapter = adapter.normalize(peak_level=target_peak)
            normalized = normalized_adapter.to_samples()

            result_audio = AudioData(
                samples=normalized,
                sample_rate=audio.sample_rate,
                channels=audio.channels,
                source_path=audio.source_path,
                is_modified=True,
                metadata={
                    **audio.metadata,
                    "last_operation": f"peak_normalize_{target_peak}",
                },
            )

            return ServiceResult.ok(
                data=result_audio,
                message=f"Peak normalized to {target_peak}",
            )

        except Exception as e:
            return ServiceResult.fail(f"Peak normalization failed: {e}")

    def normalize_loudness(
        self,
        audio: AudioData,
        target_db: float = -20.0,
    ) -> ServiceResult[AudioData]:
        """
        Normalize audio to a target loudness (RMS level).

        Args:
            audio: Input AudioData
            target_db: Target loudness in dBFS

        Returns:
            ServiceResult with normalized AudioData
        """
        try:
            from bioamla.core.signal import normalize_loudness

            normalized = normalize_loudness(audio.samples, audio.sample_rate, target_db=target_db)

            result_audio = AudioData(
                samples=normalized,
                sample_rate=audio.sample_rate,
                channels=audio.channels,
                source_path=audio.source_path,
                is_modified=True,
                metadata={
                    **audio.metadata,
                    "last_operation": f"loudness_normalize_{target_db}dB",
                },
            )

            return ServiceResult.ok(
                data=result_audio,
                message=f"Loudness normalized to {target_db} dBFS",
            )

        except Exception as e:
            return ServiceResult.fail(f"Loudness normalization failed: {e}")

    # =========================================================================
    # Resampling
    # =========================================================================

    def resample(
        self,
        audio: AudioData,
        target_sample_rate: int,
    ) -> ServiceResult[AudioData]:
        """
        Resample audio to a different sample rate using OpenSoundscape.

        Args:
            audio: Input AudioData
            target_sample_rate: Target sample rate in Hz

        Returns:
            ServiceResult with resampled AudioData
        """
        if target_sample_rate == audio.sample_rate:
            return ServiceResult.ok(
                data=audio,
                message="Already at target sample rate",
            )

        try:
            from bioamla.adapters.opensoundscape import AudioAdapter

            # Use OpenSoundscape adapter for resampling
            adapter = AudioAdapter.from_samples(audio.samples, audio.sample_rate)
            resampled_adapter = adapter.resample(target_sample_rate)
            resampled = resampled_adapter.to_samples()

            result_audio = AudioData(
                samples=resampled,
                sample_rate=target_sample_rate,
                channels=audio.channels,
                source_path=audio.source_path,
                is_modified=True,
                metadata={
                    **audio.metadata,
                    "original_sample_rate": audio.sample_rate,
                    "last_operation": f"resample_{target_sample_rate}Hz",
                },
            )

            return ServiceResult.ok(
                data=result_audio,
                message=f"Resampled from {audio.sample_rate}Hz to {target_sample_rate}Hz",
            )

        except Exception as e:
            return ServiceResult.fail(f"Resampling failed: {e}")

    # =========================================================================
    # Trimming Operations
    # =========================================================================

    def trim(
        self,
        audio: AudioData,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
    ) -> ServiceResult[AudioData]:
        """
        Trim audio to a time range using OpenSoundscape.

        Args:
            audio: Input AudioData
            start_time: Start time in seconds (default: 0)
            end_time: End time in seconds (default: end of audio)

        Returns:
            ServiceResult with trimmed AudioData
        """
        try:
            from bioamla.adapters.opensoundscape import AudioAdapter

            start = start_time or 0.0
            end = end_time or audio.duration

            # Use OpenSoundscape adapter for trimming
            adapter = AudioAdapter.from_samples(audio.samples, audio.sample_rate)
            trimmed_adapter = adapter.trim(start, end)
            trimmed = trimmed_adapter.to_samples()

            result_audio = AudioData(
                samples=trimmed,
                sample_rate=audio.sample_rate,
                channels=audio.channels,
                source_path=audio.source_path,
                is_modified=True,
                metadata={
                    **audio.metadata,
                    "trim_start": start,
                    "trim_end": end,
                    "last_operation": f"trim_{start}-{end}s",
                },
            )

            return ServiceResult.ok(
                data=result_audio,
                message=f"Trimmed to {start:.2f}s - {end:.2f}s",
            )

        except Exception as e:
            return ServiceResult.fail(f"Trim failed: {e}")

    def trim_silence(
        self,
        audio: AudioData,
        threshold_db: float = -40.0,
        min_silence_duration: float = 0.1,
    ) -> ServiceResult[AudioData]:
        """
        Trim silence from the beginning and end of audio.

        Args:
            audio: Input AudioData
            threshold_db: Silence threshold in dBFS
            min_silence_duration: Minimum silence duration to trim

        Returns:
            ServiceResult with trimmed AudioData
        """
        try:
            from bioamla.core.signal import trim_silence

            trimmed = trim_silence(
                audio.samples,
                audio.sample_rate,
                threshold_db=threshold_db,
            )

            result_audio = AudioData(
                samples=trimmed,
                sample_rate=audio.sample_rate,
                channels=audio.channels,
                source_path=audio.source_path,
                is_modified=True,
                metadata={
                    **audio.metadata,
                    "last_operation": "trim_silence",
                },
            )

            removed_duration = audio.duration - result_audio.duration

            return ServiceResult.ok(
                data=result_audio,
                message=f"Trimmed {removed_duration:.2f}s of silence",
                removed_duration=removed_duration,
            )

        except Exception as e:
            return ServiceResult.fail(f"Silence trimming failed: {e}")

    # =========================================================================
    # Noise Reduction
    # =========================================================================

    def denoise(
        self,
        audio: AudioData,
        strength: float = 1.0,
    ) -> ServiceResult[AudioData]:
        """
        Apply spectral noise reduction.

        Args:
            audio: Input AudioData
            strength: Noise reduction strength (0.0 to 2.0)

        Returns:
            ServiceResult with denoised AudioData
        """
        try:
            from bioamla.core.signal import spectral_denoise

            denoised = spectral_denoise(
                audio.samples,
                audio.sample_rate,
                noise_reduce_factor=strength,
            )

            result_audio = AudioData(
                samples=denoised,
                sample_rate=audio.sample_rate,
                channels=audio.channels,
                source_path=audio.source_path,
                is_modified=True,
                metadata={
                    **audio.metadata,
                    "last_operation": f"denoise_{strength}",
                },
            )

            return ServiceResult.ok(
                data=result_audio,
                message=f"Applied noise reduction (strength={strength})",
            )

        except Exception as e:
            return ServiceResult.fail(f"Noise reduction failed: {e}")

    # =========================================================================
    # Gain Operations
    # =========================================================================

    def apply_gain(
        self,
        audio: AudioData,
        gain_db: float,
    ) -> ServiceResult[AudioData]:
        """
        Apply gain (volume adjustment) to audio.

        Args:
            audio: Input AudioData
            gain_db: Gain in decibels (positive = louder, negative = quieter)

        Returns:
            ServiceResult with gain-adjusted AudioData
        """
        try:
            gain_linear = 10 ** (gain_db / 20)
            gained = audio.samples * gain_linear

            # Clip to prevent clipping
            gained = np.clip(gained, -1.0, 1.0)

            result_audio = AudioData(
                samples=gained,
                sample_rate=audio.sample_rate,
                channels=audio.channels,
                source_path=audio.source_path,
                is_modified=True,
                metadata={
                    **audio.metadata,
                    "last_operation": f"gain_{gain_db}dB",
                },
            )

            return ServiceResult.ok(
                data=result_audio,
                message=f"Applied {gain_db:+.1f} dB gain",
            )

        except Exception as e:
            return ServiceResult.fail(f"Gain adjustment failed: {e}")

    # =========================================================================
    # Channel Operations
    # =========================================================================

    def to_mono(self, audio: AudioData) -> ServiceResult[AudioData]:
        """
        Convert stereo audio to mono.

        Args:
            audio: Input AudioData

        Returns:
            ServiceResult with mono AudioData
        """
        try:
            if audio.samples.ndim == 1:
                # Already mono
                return ServiceResult.ok(
                    data=audio,
                    message="Already mono",
                )

            # Average channels
            mono = audio.samples.mean(axis=-1)

            result_audio = AudioData(
                samples=mono,
                sample_rate=audio.sample_rate,
                channels=1,
                source_path=audio.source_path,
                is_modified=True,
                metadata={
                    **audio.metadata,
                    "original_channels": audio.channels,
                    "last_operation": "to_mono",
                },
            )

            return ServiceResult.ok(
                data=result_audio,
                message="Converted to mono",
            )

        except Exception as e:
            return ServiceResult.fail(f"Mono conversion failed: {e}")

    # =========================================================================
    # Playback Preparation
    # =========================================================================

    def prepare_for_playback(
        self,
        audio: AudioData,
        target_sample_rate: int = 44100,
        normalize: bool = True,
    ) -> ServiceResult[AudioData]:
        """
        Prepare audio for playback.

        This applies common preprocessing for audio playback:
        - Resample to target rate
        - Convert to mono
        - Normalize to prevent clipping

        Args:
            audio: Input AudioData
            target_sample_rate: Target sample rate for playback
            normalize: Whether to normalize audio

        Returns:
            ServiceResult with playback-ready AudioData
        """
        result = audio

        # Resample if needed
        if result.sample_rate != target_sample_rate:
            resample_result = self.resample(result, target_sample_rate)
            if not resample_result.success:
                return resample_result
            result = resample_result.data

        # Convert to mono if needed
        if result.channels > 1:
            mono_result = self.to_mono(result)
            if not mono_result.success:
                return mono_result
            result = mono_result.data

        # Normalize if requested
        if normalize:
            norm_result = self.normalize_peak(result, target_peak=0.9)
            if not norm_result.success:
                return norm_result
            result = norm_result.data

        return ServiceResult.ok(
            data=result,
            message="Prepared for playback",
            sample_rate=target_sample_rate,
        )

    # =========================================================================
    # Analysis (Read-only operations)
    # =========================================================================

    def get_amplitude_stats(
        self,
        audio: AudioData,
    ) -> ServiceResult[dict]:
        """
        Get amplitude statistics for audio.

        Args:
            audio: Input AudioData

        Returns:
            ServiceResult with amplitude statistics dict
        """
        try:
            from bioamla.core.audio.audio import get_amplitude_stats

            stats = get_amplitude_stats(audio.samples)

            return ServiceResult.ok(
                data=stats.to_dict(),
                message="Calculated amplitude statistics",
            )

        except Exception as e:
            return ServiceResult.fail(f"Amplitude analysis failed: {e}")

    def get_frequency_stats(
        self,
        audio: AudioData,
    ) -> ServiceResult[dict]:
        """
        Get frequency statistics for audio.

        Args:
            audio: Input AudioData

        Returns:
            ServiceResult with frequency statistics dict
        """
        try:
            from bioamla.core.audio.audio import get_frequency_stats

            stats = get_frequency_stats(audio.samples, audio.sample_rate)

            return ServiceResult.ok(
                data=stats.to_dict(),
                message="Calculated frequency statistics",
            )

        except Exception as e:
            return ServiceResult.fail(f"Frequency analysis failed: {e}")

    def detect_silence(
        self,
        audio: AudioData,
        threshold_db: float = -40.0,
    ) -> ServiceResult[dict]:
        """
        Detect silent regions in audio.

        Args:
            audio: Input AudioData
            threshold_db: Silence threshold in dBFS

        Returns:
            ServiceResult with silence detection results
        """
        try:
            from bioamla.core.audio.audio import detect_silence

            info = detect_silence(
                audio.samples,
                audio.sample_rate,
                threshold_db=threshold_db,
            )

            return ServiceResult.ok(
                data=info.to_dict(),
                message="Detected silence regions",
            )

        except Exception as e:
            return ServiceResult.fail(f"Silence detection failed: {e}")

    # =========================================================================
    # Chaining Support
    # =========================================================================

    def chain(
        self,
        audio: AudioData,
        operations: List[Tuple[str, dict]],
    ) -> ServiceResult[AudioData]:
        """
        Apply a chain of operations to audio.

        Args:
            audio: Input AudioData
            operations: List of (operation_name, kwargs) tuples

        Returns:
            ServiceResult with processed AudioData

        Example:
            result = controller.chain(audio, [
                ("apply_bandpass", {"low_hz": 500, "high_hz": 8000}),
                ("normalize_loudness", {"target_db": -20}),
                ("resample", {"target_sample_rate": 16000}),
            ])
        """
        current = audio
        applied = []

        for op_name, kwargs in operations:
            method = getattr(self, op_name, None)
            if method is None:
                return ServiceResult.fail(f"Unknown operation: {op_name}")

            result = method(current, **kwargs)
            if not result.success:
                return ServiceResult.fail(
                    f"Chain failed at {op_name}: {result.error}",
                    warnings=[f"Successfully applied: {', '.join(applied)}"] if applied else [],
                )

            current = result.data
            applied.append(op_name)

        return ServiceResult.ok(
            data=current,
            message=f"Applied chain: {' -> '.join(applied)}",
        )

    # =========================================================================
    # File-Based Operations (for CLI/batch processing)
    # =========================================================================

    def list_files(
        self,
        directory: str,
        recursive: bool = True,
    ) -> ServiceResult[List[str]]:
        """
        List audio files in a directory using the file repository.

        Args:
            directory: Directory path to search
            recursive: Search subdirectories

        Returns:
            Result with list of audio file paths
        """
        if not self.file_repository.exists(directory):
            return ServiceResult.fail(f"Path does not exist: {directory}")

        if not self.file_repository.is_dir(directory):
            return ServiceResult.fail(f"Path is not a directory: {directory}")

        try:
            # Use repository to list audio files
            extensions = [".wav", ".mp3", ".flac", ".ogg", ".m4a"]
            files = []
            for ext in extensions:
                pattern = f"**/*{ext}" if recursive else f"*{ext}"
                files.extend(self.file_repository.list_files(directory, pattern, recursive))

            return ServiceResult.ok(
                data=[str(f) for f in files],
                message=f"Found {len(files)} audio files",
                count=len(files),
            )
        except Exception as e:
            return ServiceResult.fail(str(e))

    def get_metadata(self, filepath: str) -> ServiceResult[AudioMetadata]:
        """
        Get metadata from an audio file.

        Args:
            filepath: Path to audio file

        Returns:
            Result with audio metadata
        """
        error = self._validate_input_path(filepath)
        if error:
            return ServiceResult.fail(error)

        try:
            from bioamla.core.utils import get_wav_metadata

            metadata = get_wav_metadata(filepath)

            return ServiceResult.ok(
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
            return ServiceResult.fail(str(e))

    def resample_file(
        self,
        input_path: str,
        output_path: str,
        target_rate: int,
    ) -> ServiceResult[ProcessedAudio]:
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
            return ServiceResult.fail(error)

        error = self._validate_output_path(output_path)
        if error:
            return ServiceResult.fail(error)

        try:
            from bioamla.core.signal import load_audio, resample_audio, save_audio

            audio, sr = load_audio(input_path)
            resampled = resample_audio(audio, sr, target_rate)
            save_audio(output_path, resampled, target_rate)

            duration = len(resampled) / target_rate

            return ServiceResult.ok(
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
            return ServiceResult.fail(str(e))


    def normalize_file(
        self,
        input_path: str,
        output_path: str,
        target_db: float = -20.0,
        peak: bool = False,
    ) -> ServiceResult[ProcessedAudio]:
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
            return ServiceResult.fail(error)

        error = self._validate_output_path(output_path)
        if error:
            return ServiceResult.fail(error)

        try:
            from bioamla.core.signal import (
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

            return ServiceResult.ok(
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
            return ServiceResult.fail(str(e))

    def filter_file(
        self,
        input_path: str,
        output_path: str,
        lowpass: Optional[float] = None,
        highpass: Optional[float] = None,
        bandpass: Optional[Tuple[float, float]] = None,
        order: int = 5,
    ) -> ServiceResult[ProcessedAudio]:
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
            return ServiceResult.fail("Must specify lowpass, highpass, or bandpass")

        error = self._validate_input_path(input_path)
        if error:
            return ServiceResult.fail(error)

        error = self._validate_output_path(output_path)
        if error:
            return ServiceResult.fail(error)

        try:
            from bioamla.core.signal import (
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

            return ServiceResult.ok(
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
            return ServiceResult.fail(str(e))

    def trim_file(
        self,
        input_path: str,
        output_path: str,
        start: Optional[float] = None,
        end: Optional[float] = None,
        trim_silence: bool = False,
        silence_threshold_db: float = -40.0,
    ) -> ServiceResult[ProcessedAudio]:
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
            return ServiceResult.fail("Must specify start/end or use trim_silence")

        error = self._validate_input_path(input_path)
        if error:
            return ServiceResult.fail(error)

        error = self._validate_output_path(output_path)
        if error:
            return ServiceResult.fail(error)

        try:
            from bioamla.core.signal import (
                load_audio,
                save_audio,
                trim_audio,
            )
            from bioamla.core.signal import (
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

            return ServiceResult.ok(
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
            return ServiceResult.fail(str(e))

    def segment_file(
        self,
        input_path: str,
        output_dir: str,
        duration: float = 3.0,
        overlap: float = 0.0,
        format: str = "wav",
        prefix: Optional[str] = None,
    ) -> ServiceResult[BatchResult]:
        """
        Segment audio file into fixed-duration clips.

        Args:
            input_path: Input audio file path
            output_dir: Output directory for segments
            duration: Segment duration in seconds
            overlap: Overlap between segments in seconds
            format: Output format (wav, mp3, flac, ogg)
            prefix: Prefix for output filenames (default: input filename)

        Returns:
            Result with batch processing summary
        """

        error = self._validate_input_path(input_path)
        if error:
            return ServiceResult.fail(error)

        try:
            from bioamla.core.signal import load_audio, save_audio

            audio, sr = load_audio(input_path)

            # Calculate segment parameters
            segment_samples = int(duration * sr)
            overlap_samples = int(overlap * sr)
            step_samples = segment_samples - overlap_samples

            if step_samples <= 0:
                return ServiceResult.fail("Overlap must be less than duration")

            # Create output directory using repository
            self.file_repository.mkdir(output_dir, parents=True)
            output_path = Path(output_dir)

            # Determine prefix
            if prefix is None:
                prefix = Path(input_path).stem

            # Calculate hop duration for segment timing
            hop_duration = step_samples / sr
            total_duration = len(audio) / sr

            # Segment the audio
            segments_created = 0
            errors = []
            position = 0
            segment_infos = []

            while position + segment_samples <= len(audio):
                segment = audio[position : position + segment_samples]
                segment_file = output_path / f"{prefix}_{segments_created:04d}.{format}"

                # Calculate temporal bounds for this segment
                start_time = segments_created * hop_duration
                end_time = min(start_time + duration, total_duration)

                try:
                    save_audio(str(segment_file), segment, sr)

                    # Track segment info for metadata
                    from bioamla.models.batch import SegmentInfo
                    segment_infos.append(
                        SegmentInfo(
                            segment_path=segment_file,
                            segment_id=segments_created,
                            start_time=start_time,
                            end_time=end_time,
                            duration=duration,
                        )
                    )

                    segments_created += 1
                except Exception as e:
                    errors.append(f"{segment_file}: {e}")
                position += step_samples

            return ServiceResult.ok(
                data={
                    "batch_result": BatchResult(
                        processed=segments_created,
                        failed=len(errors),
                        output_path=str(output_path),
                        errors=errors,
                    ),
                    "segments": segment_infos,
                },
                message=f"Created {segments_created} segments in {output_dir}",
            )
        except Exception as e:
            return ServiceResult.fail(str(e))

    def denoise_file(
        self,
        input_path: str,
        output_path: str,
        strength: float = 1.0,
    ) -> ServiceResult[ProcessedAudio]:
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
            return ServiceResult.fail(error)

        error = self._validate_output_path(output_path)
        if error:
            return ServiceResult.fail(error)

        try:
            from bioamla.core.signal import load_audio, save_audio, spectral_denoise

            audio, sr = load_audio(input_path)
            denoised = spectral_denoise(audio, sr, noise_reduce_factor=strength)
            save_audio(output_path, denoised, sr)

            duration = len(denoised) / sr

            return ServiceResult.ok(
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
            return ServiceResult.fail(str(e))

    def analyze_file(
        self,
        filepath: str,
        silence_threshold_db: float = -40.0,
    ) -> ServiceResult[AnalysisResult]:
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
            return ServiceResult.fail(error)

        try:
            from bioamla.core.audio import analyze_audio
            from bioamla.core.signal import load_audio

            audio, sr = load_audio(filepath)
            analysis = analyze_audio(audio, sr, silence_threshold_db=silence_threshold_db)

            return ServiceResult.ok(
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
            return ServiceResult.fail(str(e))



    def visualize_file(
        self,
        input_path: str,
        output_path: str,
        viz_type: str = "mel",
        n_fft: int = 2048,
        hop_length: int = 512,
        n_mels: int = 128,
        n_mfcc: int = 20,
        cmap: str = "viridis",
        dpi: int = 100,
        show_legend: bool = True,
    ) -> ServiceResult[str]:
        """
        Generate audio visualization.

        Args:
            input_path: Input audio file path
            output_path: Output image file path
            viz_type: Visualization type (mel, stft, mfcc, waveform)
            n_fft: FFT window size
            hop_length: Hop length
            n_mels: Number of mel bands
            n_mfcc: Number of MFCCs
            cmap: Colormap name
            dpi: Output DPI
            show_legend: If True, show axes, title, and colorbar. If False, clean image only.

        Returns:
            Result with output path
        """
        error = self._validate_input_path(input_path)
        if error:
            return ServiceResult.fail(error)

        try:
            from bioamla.core.visualize import generate_spectrogram

            generate_spectrogram(
                audio_path=input_path,
                output_path=output_path,
                viz_type=viz_type,
                n_fft=n_fft,
                hop_length=hop_length,
                n_mels=n_mels,
                n_mfcc=n_mfcc,
                cmap=cmap,
                dpi=dpi,
                show_colorbar=show_legend,
            )

            return ServiceResult.ok(
                data=output_path,
                message=f"Generated {viz_type} visualization: {output_path}",
            )
        except Exception as e:
            return ServiceResult.fail(str(e))

