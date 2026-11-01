"""Operations for ``batch audio``: thin adapters over :mod:`bioamla.audio.batch`.

See :mod:`bioamla.operations`. Every request extends :class:`~bioamla.batch.BatchRequest`
(a folder or a metadata CSV, and where results go); each operation turns it into a
:class:`~bioamla.batch.BatchConfig` and calls the library's batch function, which
handles both modes. All are ``job`` operations that report progress where the library
does, and return a :class:`~bioamla.batch.BatchResult`.
"""

from __future__ import annotations

from dataclasses import field
from typing import Literal

from bioamla.audio.batch import (
    batch_audio_info,
    batch_convert,
    batch_frame_features,
    batch_recording_metadata,
    batch_resample,
    batch_segment,
    batch_transform,
    batch_validate_wavs,
    batch_visualize,
    denoise_processor,
    filter_processor,
    normalize_processor,
    trim_processor,
)
from bioamla.audio.features import FrameFeatureConfig
from bioamla.batch import BatchRequest, BatchResult
from bioamla.common.config import config_dataclass
from bioamla.operations import OperationContext, operation


@operation("batch.audio.info", request=BatchRequest, kind="job")
def batch_audio_info_op(request: BatchRequest, context: OperationContext) -> BatchResult:
    """Read every file's header (duration, sample rate, channels, samples, format)."""
    return batch_audio_info(
        request.to_config(), on_progress=context.progress, operation="batch audio info"
    )


@operation("batch.audio.validate", request=BatchRequest, kind="job")
def batch_audio_validate_op(request: BatchRequest, context: OperationContext) -> BatchResult:
    """Check that every WAV is well-formed (header only); invalid files are listed in
    ``metadata["invalid_files"]``."""
    return batch_validate_wavs(
        request.to_config(), on_progress=context.progress, operation="batch audio validate"
    )


@operation("batch.audio.metadata", request=BatchRequest, kind="job")
def batch_audio_metadata_op(request: BatchRequest, context: OperationContext) -> BatchResult:
    """Read every WAV's recorder metadata (GUANO, AudioMoth, WAMD, bext)."""
    return batch_recording_metadata(
        request.to_config(), on_progress=context.progress, operation="batch audio metadata"
    )


@config_dataclass
class BatchFeaturesRequest(BatchRequest):
    """Per-frame feature tables for a batch.

    Attributes:
        config: Analysis settings.
        trajectory: Add the MFCC-PCA timbre trajectory.
        gate_db: Trajectory gate (dB below the loudest frame).
        file_format: Table format, ``csv`` or ``parquet``.
    """

    config: FrameFeatureConfig = field(default_factory=FrameFeatureConfig)
    trajectory: bool = False
    gate_db: float = 38.0
    file_format: Literal["csv", "parquet"] = "csv"


@operation("batch.audio.features", request=BatchFeaturesRequest, kind="job")
def batch_audio_features_op(
    request: BatchFeaturesRequest, context: OperationContext
) -> BatchResult:
    """Write a per-frame feature table for every file."""
    return batch_frame_features(
        request.to_config(),
        analysis=request.config,
        trajectory=request.trajectory,
        gate_db=request.gate_db,
        file_format=request.file_format,
        on_progress=context.progress,
        operation="batch audio features",
    )


@config_dataclass
class BatchConvertRequest(BatchRequest):
    """Convert every file's format, sample rate or channel count.

    Attributes:
        target_format: Output format.
        sample_rate: Target sample rate (Hz).
        channels: Target channel count (1 or 2).
        delete_original: Delete each source file after it converts.
    """

    target_format: Literal["wav", "mp3", "flac", "ogg"] = "wav"
    sample_rate: int | None = None
    channels: int | None = None
    delete_original: bool = False


@operation("batch.audio.convert", request=BatchConvertRequest, kind="job")
def batch_audio_convert_op(request: BatchConvertRequest, context: OperationContext) -> BatchResult:
    """Convert every file's format, sample rate or channel count."""
    return batch_convert(
        request.to_config(),
        target_format=request.target_format,
        sample_rate=request.sample_rate,
        channels=request.channels,
        delete_original=request.delete_original,
        on_progress=context.progress,
        operation="batch audio convert",
    )


@config_dataclass
class BatchResampleRequest(BatchRequest):
    """Resample every file.

    Attributes:
        sample_rate: Target sample rate (Hz).
    """

    sample_rate: int = 16000


@operation("batch.audio.resample", request=BatchResampleRequest, kind="job")
def batch_audio_resample_op(
    request: BatchResampleRequest, context: OperationContext
) -> BatchResult:
    """Resample every file."""
    return batch_resample(request.to_config(), request.sample_rate, on_progress=context.progress)


@config_dataclass
class BatchNormalizeRequest(BatchRequest):
    """Normalize every file's level.

    Attributes:
        target_db: Target RMS loudness (dB).
        peak: Peak-normalize instead.
    """

    target_db: float = -20.0
    peak: bool = False


@operation("batch.audio.normalize", request=BatchNormalizeRequest, kind="job")
def batch_audio_normalize_op(
    request: BatchNormalizeRequest, context: OperationContext
) -> BatchResult:
    """Normalize every file's level."""
    return batch_transform(
        request.to_config(),
        normalize_processor(target_db=request.target_db, peak=request.peak),
        on_progress=context.progress,
        operation="batch audio normalize",
    )


@config_dataclass
class BatchTrimRequest(BatchRequest):
    """Trim every file to a time range, or trim its silence.

    Attributes:
        start: Start time (s).
        end: End time (s).
        trim_silence: Trim leading and trailing silence instead of a range.
        silence_threshold_db: Silence threshold (dB), with ``trim_silence``.
    """

    start: float | None = None
    end: float | None = None
    trim_silence: bool = False
    silence_threshold_db: float = -40.0


@operation("batch.audio.trim", request=BatchTrimRequest, kind="job")
def batch_audio_trim_op(request: BatchTrimRequest, context: OperationContext) -> BatchResult:
    """Trim every file to a time range, or trim its silence."""
    processor = trim_processor(
        start=request.start,
        end=request.end,
        silence_threshold_db=request.silence_threshold_db if request.trim_silence else None,
    )
    return batch_transform(
        request.to_config(),
        processor,
        on_progress=context.progress,
        operation="batch audio trim",
    )


@config_dataclass
class BatchFilterRequest(BatchRequest):
    """Apply one frequency filter to every file (band-pass, else low-pass, else high-pass).

    Attributes:
        lowpass: Low-pass cutoff (Hz).
        highpass: High-pass cutoff (Hz).
        bandpass_low: Band-pass lower edge (Hz).
        bandpass_high: Band-pass upper edge (Hz).
        order: Filter order.
    """

    lowpass: float | None = None
    highpass: float | None = None
    bandpass_low: float | None = None
    bandpass_high: float | None = None
    order: int = 5


@operation("batch.audio.filter", request=BatchFilterRequest, kind="job")
def batch_audio_filter_op(request: BatchFilterRequest, context: OperationContext) -> BatchResult:
    """Apply a frequency filter to every file."""
    processor = filter_processor(
        lowpass=request.lowpass,
        highpass=request.highpass,
        bandpass_low=request.bandpass_low,
        bandpass_high=request.bandpass_high,
        order=request.order,
    )
    return batch_transform(
        request.to_config(),
        processor,
        on_progress=context.progress,
        operation="batch audio filter",
    )


@config_dataclass
class BatchDenoiseRequest(BatchRequest):
    """Spectral noise reduction on every file.

    Attributes:
        strength: Reduction strength, 0 to 2.
    """

    strength: float = 1.0


@operation("batch.audio.denoise", request=BatchDenoiseRequest, kind="job")
def batch_audio_denoise_op(request: BatchDenoiseRequest, context: OperationContext) -> BatchResult:
    """Reduce stationary noise in every file."""
    return batch_transform(
        request.to_config(),
        denoise_processor(strength=request.strength),
        on_progress=context.progress,
        operation="batch audio denoise",
    )


@config_dataclass
class BatchSegmentRequest(BatchRequest):
    """Cut every file into fixed-length clips.

    Attributes:
        duration: Clip length (s).
        overlap: Overlap between clips (s).
    """

    duration: float = 3.0
    overlap: float = 0.0


@operation("batch.audio.segment", request=BatchSegmentRequest, kind="job")
def batch_audio_segment_op(request: BatchSegmentRequest, context: OperationContext) -> BatchResult:
    """Cut every file into fixed-length clips."""
    return batch_segment(
        request.to_config(),
        duration=request.duration,
        overlap=request.overlap,
        on_progress=context.progress,
        operation="batch audio segment",
    )


@config_dataclass
class BatchVisualizeRequest(BatchRequest):
    """A spectrogram, MFCC or waveform image of every file (directory mode).

    Attributes:
        plot_type: ``mel``, ``stft``, ``mfcc`` or ``waveform``.
        verbose: Log each file at INFO level.
    """

    plot_type: Literal["mel", "stft", "mfcc", "waveform"] = "mel"
    verbose: bool = True


@operation("batch.audio.visualize", request=BatchVisualizeRequest, kind="job")
def batch_audio_visualize_op(
    request: BatchVisualizeRequest, context: OperationContext
) -> BatchResult:
    """Draw an image of every file in a folder."""
    return batch_visualize(
        request.to_config(),
        viz_type=request.plot_type,
        verbose=request.verbose,
        on_progress=context.progress,
    )
