# controllers/audio_transform.py
"""
Audio Transform Controller
==========================

Controller for in-memory audio signal processing and transforms.

This controller handles:
- Signal processing operations (filter, normalize, resample, etc.)
- Audio transformations that operate purely in memory
- Playback preparation (format conversion, gain adjustment)
- Analysis operations that don't modify files

Design principle: AudioTransformController NEVER writes to permanent storage.
All file operations must go through AudioFileController. If a temporary file
is needed (e.g., for external tool integration), it delegates to
AudioFileController.create_temp_file().

Usage:
    from bioamla.controllers import AudioFileController, AudioTransformController

    file_ctrl = AudioFileController()
    transform_ctrl = AudioTransformController()

    # Load audio
    result = file_ctrl.open("input.wav")
    audio = result.data

    # Apply transforms (in memory)
    audio = transform_ctrl.apply_bandpass(audio, 500, 8000)
    audio = transform_ctrl.normalize(audio, target_db=-20)

    # Save through file controller (only way to persist)
    file_ctrl.save(audio, "output.wav")
"""
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from .audio_file import AudioData
from .base import BaseController, ControllerResult


@dataclass
class TransformResult:
    """Result of an audio transform operation."""

    audio: AudioData
    operation: str
    parameters: dict


class AudioTransformController(BaseController):
    """
    Controller for in-memory audio signal processing.

    All methods return new AudioData objects without modifying the originals.
    This controller never writes files - use AudioFileController for persistence.

    Transform operations:
    - Filtering (lowpass, highpass, bandpass)
    - Normalization (peak, RMS/loudness)
    - Resampling
    - Trimming
    - Noise reduction
    - Gain adjustment

    Playback operations:
    - Convert to playback format
    - Apply real-time gain
    - Mix channels

    Analysis operations:
    - Get amplitude statistics
    - Get frequency statistics
    - Detect silence regions
    """

    # =========================================================================
    # Filtering Operations
    # =========================================================================

    def apply_lowpass(
        self,
        audio: AudioData,
        cutoff_hz: float,
        order: int = 5,
    ) -> ControllerResult[AudioData]:
        """
        Apply a lowpass filter.

        Args:
            audio: Input AudioData
            cutoff_hz: Cutoff frequency in Hz
            order: Filter order

        Returns:
            ControllerResult with filtered AudioData
        """
        try:
            from bioamla.core.audio.signal import lowpass_filter

            filtered = lowpass_filter(
                audio.samples, audio.sample_rate, cutoff_hz, order
            )

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

            return ControllerResult.ok(
                data=result_audio,
                message=f"Applied lowpass filter at {cutoff_hz}Hz",
            )

        except Exception as e:
            return ControllerResult.fail(f"Lowpass filter failed: {e}")

    def apply_highpass(
        self,
        audio: AudioData,
        cutoff_hz: float,
        order: int = 5,
    ) -> ControllerResult[AudioData]:
        """
        Apply a highpass filter.

        Args:
            audio: Input AudioData
            cutoff_hz: Cutoff frequency in Hz
            order: Filter order

        Returns:
            ControllerResult with filtered AudioData
        """
        try:
            from bioamla.core.audio.signal import highpass_filter

            filtered = highpass_filter(
                audio.samples, audio.sample_rate, cutoff_hz, order
            )

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

            return ControllerResult.ok(
                data=result_audio,
                message=f"Applied highpass filter at {cutoff_hz}Hz",
            )

        except Exception as e:
            return ControllerResult.fail(f"Highpass filter failed: {e}")

    def apply_bandpass(
        self,
        audio: AudioData,
        low_hz: float,
        high_hz: float,
        order: int = 5,
    ) -> ControllerResult[AudioData]:
        """
        Apply a bandpass filter.

        Args:
            audio: Input AudioData
            low_hz: Lower cutoff frequency in Hz
            high_hz: Upper cutoff frequency in Hz
            order: Filter order

        Returns:
            ControllerResult with filtered AudioData
        """
        try:
            from bioamla.core.audio.signal import bandpass_filter

            filtered = bandpass_filter(
                audio.samples, audio.sample_rate, low_hz, high_hz, order
            )

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

            return ControllerResult.ok(
                data=result_audio,
                message=f"Applied bandpass filter {low_hz}-{high_hz}Hz",
            )

        except Exception as e:
            return ControllerResult.fail(f"Bandpass filter failed: {e}")

    # =========================================================================
    # Normalization Operations
    # =========================================================================

    def normalize_peak(
        self,
        audio: AudioData,
        target_peak: float = 0.99,
    ) -> ControllerResult[AudioData]:
        """
        Normalize audio to a target peak amplitude.

        Args:
            audio: Input AudioData
            target_peak: Target peak amplitude (0.0 to 1.0)

        Returns:
            ControllerResult with normalized AudioData
        """
        try:
            from bioamla.core.audio.signal import peak_normalize

            normalized = peak_normalize(audio.samples, target_peak=target_peak)

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

            return ControllerResult.ok(
                data=result_audio,
                message=f"Peak normalized to {target_peak}",
            )

        except Exception as e:
            return ControllerResult.fail(f"Peak normalization failed: {e}")

    def normalize_loudness(
        self,
        audio: AudioData,
        target_db: float = -20.0,
    ) -> ControllerResult[AudioData]:
        """
        Normalize audio to a target loudness (RMS level).

        Args:
            audio: Input AudioData
            target_db: Target loudness in dBFS

        Returns:
            ControllerResult with normalized AudioData
        """
        try:
            from bioamla.core.audio.signal import normalize_loudness

            normalized = normalize_loudness(
                audio.samples, audio.sample_rate, target_db=target_db
            )

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

            return ControllerResult.ok(
                data=result_audio,
                message=f"Loudness normalized to {target_db} dBFS",
            )

        except Exception as e:
            return ControllerResult.fail(f"Loudness normalization failed: {e}")

    # =========================================================================
    # Resampling
    # =========================================================================

    def resample(
        self,
        audio: AudioData,
        target_sample_rate: int,
    ) -> ControllerResult[AudioData]:
        """
        Resample audio to a different sample rate.

        Args:
            audio: Input AudioData
            target_sample_rate: Target sample rate in Hz

        Returns:
            ControllerResult with resampled AudioData
        """
        if target_sample_rate == audio.sample_rate:
            return ControllerResult.ok(
                data=audio,
                message="Already at target sample rate",
            )

        try:
            from bioamla.core.audio.signal import resample_audio

            resampled = resample_audio(
                audio.samples, audio.sample_rate, target_sample_rate
            )

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

            return ControllerResult.ok(
                data=result_audio,
                message=f"Resampled from {audio.sample_rate}Hz to {target_sample_rate}Hz",
            )

        except Exception as e:
            return ControllerResult.fail(f"Resampling failed: {e}")

    # =========================================================================
    # Trimming Operations
    # =========================================================================

    def trim(
        self,
        audio: AudioData,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
    ) -> ControllerResult[AudioData]:
        """
        Trim audio to a time range.

        Args:
            audio: Input AudioData
            start_time: Start time in seconds (default: 0)
            end_time: End time in seconds (default: end of audio)

        Returns:
            ControllerResult with trimmed AudioData
        """
        try:
            from bioamla.core.audio.signal import trim_audio

            trimmed = trim_audio(
                audio.samples,
                audio.sample_rate,
                start_time=start_time,
                end_time=end_time,
            )

            start = start_time or 0.0
            end = end_time or audio.duration

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

            return ControllerResult.ok(
                data=result_audio,
                message=f"Trimmed to {start:.2f}s - {end:.2f}s",
            )

        except Exception as e:
            return ControllerResult.fail(f"Trim failed: {e}")

    def trim_silence(
        self,
        audio: AudioData,
        threshold_db: float = -40.0,
        min_silence_duration: float = 0.1,
    ) -> ControllerResult[AudioData]:
        """
        Trim silence from the beginning and end of audio.

        Args:
            audio: Input AudioData
            threshold_db: Silence threshold in dBFS
            min_silence_duration: Minimum silence duration to trim

        Returns:
            ControllerResult with trimmed AudioData
        """
        try:
            from bioamla.core.audio.signal import trim_silence

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

            return ControllerResult.ok(
                data=result_audio,
                message=f"Trimmed {removed_duration:.2f}s of silence",
                removed_duration=removed_duration,
            )

        except Exception as e:
            return ControllerResult.fail(f"Silence trimming failed: {e}")

    # =========================================================================
    # Noise Reduction
    # =========================================================================

    def denoise(
        self,
        audio: AudioData,
        strength: float = 1.0,
    ) -> ControllerResult[AudioData]:
        """
        Apply spectral noise reduction.

        Args:
            audio: Input AudioData
            strength: Noise reduction strength (0.0 to 2.0)

        Returns:
            ControllerResult with denoised AudioData
        """
        try:
            from bioamla.core.audio.signal import spectral_denoise

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

            return ControllerResult.ok(
                data=result_audio,
                message=f"Applied noise reduction (strength={strength})",
            )

        except Exception as e:
            return ControllerResult.fail(f"Noise reduction failed: {e}")

    # =========================================================================
    # Gain Operations
    # =========================================================================

    def apply_gain(
        self,
        audio: AudioData,
        gain_db: float,
    ) -> ControllerResult[AudioData]:
        """
        Apply gain (volume adjustment) to audio.

        Args:
            audio: Input AudioData
            gain_db: Gain in decibels (positive = louder, negative = quieter)

        Returns:
            ControllerResult with gain-adjusted AudioData
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

            return ControllerResult.ok(
                data=result_audio,
                message=f"Applied {gain_db:+.1f} dB gain",
            )

        except Exception as e:
            return ControllerResult.fail(f"Gain adjustment failed: {e}")

    # =========================================================================
    # Channel Operations
    # =========================================================================

    def to_mono(self, audio: AudioData) -> ControllerResult[AudioData]:
        """
        Convert stereo audio to mono.

        Args:
            audio: Input AudioData

        Returns:
            ControllerResult with mono AudioData
        """
        try:
            if audio.samples.ndim == 1:
                # Already mono
                return ControllerResult.ok(
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

            return ControllerResult.ok(
                data=result_audio,
                message="Converted to mono",
            )

        except Exception as e:
            return ControllerResult.fail(f"Mono conversion failed: {e}")

    # =========================================================================
    # Playback Preparation
    # =========================================================================

    def prepare_for_playback(
        self,
        audio: AudioData,
        target_sample_rate: int = 44100,
        normalize: bool = True,
    ) -> ControllerResult[AudioData]:
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
            ControllerResult with playback-ready AudioData
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

        return ControllerResult.ok(
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
    ) -> ControllerResult[dict]:
        """
        Get amplitude statistics for audio.

        Args:
            audio: Input AudioData

        Returns:
            ControllerResult with amplitude statistics dict
        """
        try:
            from bioamla.core.audio.audio import get_amplitude_stats

            stats = get_amplitude_stats(audio.samples)

            return ControllerResult.ok(
                data=stats.to_dict(),
                message="Calculated amplitude statistics",
            )

        except Exception as e:
            return ControllerResult.fail(f"Amplitude analysis failed: {e}")

    def get_frequency_stats(
        self,
        audio: AudioData,
    ) -> ControllerResult[dict]:
        """
        Get frequency statistics for audio.

        Args:
            audio: Input AudioData

        Returns:
            ControllerResult with frequency statistics dict
        """
        try:
            from bioamla.core.audio.audio import get_frequency_stats

            stats = get_frequency_stats(audio.samples, audio.sample_rate)

            return ControllerResult.ok(
                data=stats.to_dict(),
                message="Calculated frequency statistics",
            )

        except Exception as e:
            return ControllerResult.fail(f"Frequency analysis failed: {e}")

    def detect_silence(
        self,
        audio: AudioData,
        threshold_db: float = -40.0,
    ) -> ControllerResult[dict]:
        """
        Detect silent regions in audio.

        Args:
            audio: Input AudioData
            threshold_db: Silence threshold in dBFS

        Returns:
            ControllerResult with silence detection results
        """
        try:
            from bioamla.core.audio.audio import detect_silence

            info = detect_silence(
                audio.samples,
                audio.sample_rate,
                threshold_db=threshold_db,
            )

            return ControllerResult.ok(
                data=info.to_dict(),
                message="Detected silence regions",
            )

        except Exception as e:
            return ControllerResult.fail(f"Silence detection failed: {e}")

    # =========================================================================
    # Chaining Support
    # =========================================================================

    def chain(
        self,
        audio: AudioData,
        operations: List[Tuple[str, dict]],
    ) -> ControllerResult[AudioData]:
        """
        Apply a chain of operations to audio.

        Args:
            audio: Input AudioData
            operations: List of (operation_name, kwargs) tuples

        Returns:
            ControllerResult with processed AudioData

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
                return ControllerResult.fail(f"Unknown operation: {op_name}")

            result = method(current, **kwargs)
            if not result.success:
                return ControllerResult.fail(
                    f"Chain failed at {op_name}: {result.error}",
                    warnings=[f"Successfully applied: {', '.join(applied)}"] if applied else [],
                )

            current = result.data
            applied.append(op_name)

        return ControllerResult.ok(
            data=current,
            message=f"Applied chain: {' -> '.join(applied)}",
        )
