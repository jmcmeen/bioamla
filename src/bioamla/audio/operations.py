"""Operations for the audio domain: thin adapters over :mod:`bioamla.audio`.

See :mod:`bioamla.operations`. Each operation maps its request onto the public
library calls a script would make, and returns their result. Reading a header or
listing a folder is ``quick``; anything that decodes audio is a ``job``. Playback is
local-only (``remote=False``): it makes sound where it runs.
"""

from __future__ import annotations

from dataclasses import field
from typing import Any, Literal

from bioamla.audio.convert import convert_audio_file
from bioamla.audio.discovery import list_audio_files
from bioamla.audio.features import FrameFeatureConfig, frame_features_file
from bioamla.audio.files import (
    AudioFileDescription,
    AudioFileList,
    ProcessedFile,
    RecordingMetadataReport,
    SegmentedFile,
    add_noise_audio_file,
    denoise_audio_file,
    describe_audio_file,
    filter_audio_file,
    gain_audio_file,
    normalize_audio_file,
    pitch_shift_audio_file,
    recording_metadata_report,
    resample_audio_file,
    time_stretch_audio_file,
    trim_audio_file,
)
from bioamla.common.config import config_dataclass
from bioamla.common.constants import (
    DEFAULT_CMAP,
    DEFAULT_DPI,
    DEFAULT_HOP_LENGTH,
    DEFAULT_N_FFT,
    DEFAULT_N_MELS,
    DEFAULT_N_MFCC,
)
from bioamla.operations import OperationContext, PathStr, operation

AudioFormat = Literal["wav", "mp3", "flac", "ogg"]


# --------------------------------------------------------------------------- #
# Reading
# --------------------------------------------------------------------------- #


@config_dataclass
class AudioInfoRequest:
    """A file's header information.

    Attributes:
        file: The audio file.
        metadata: Also read recorder metadata (GUANO, AudioMoth, WAMD, bext).
    """

    file: PathStr
    metadata: bool = False


@operation("audio.info", request=AudioInfoRequest)
def audio_info(request: AudioInfoRequest, context: OperationContext) -> AudioFileDescription:
    """Read a file's header (and optionally its recorder metadata) without decoding it."""
    return describe_audio_file(request.file, recording_metadata=request.metadata)


@config_dataclass
class AudioMetadataRequest:
    """A WAV file's recorder metadata.

    Attributes:
        file: The WAV file.
    """

    file: PathStr


@operation("audio.metadata", request=AudioMetadataRequest)
def audio_metadata(
    request: AudioMetadataRequest, context: OperationContext
) -> RecordingMetadataReport:
    """Read a WAV file's recorder metadata, flattened and per standard."""
    return recording_metadata_report(request.file)


@config_dataclass
class AudioListRequest:
    """The audio files under a folder.

    Attributes:
        path: The folder.
        recursive: Search subfolders.
    """

    path: PathStr
    recursive: bool = True


@operation("audio.list", request=AudioListRequest)
def audio_list(request: AudioListRequest, context: OperationContext) -> AudioFileList:
    """List the audio files under a folder."""
    files = list_audio_files(request.path, recursive=request.recursive)
    return AudioFileList(path=request.path, recursive=request.recursive, files=files)


@config_dataclass
class AudioPlayRequest:
    """Play a file on this machine's default output device.

    Attributes:
        file: The audio file.
        loop: Loop until stopped.
    """

    file: PathStr
    loop: bool = False


@operation("audio.play", request=AudioPlayRequest, remote=False)
def audio_play(request: AudioPlayRequest, context: OperationContext) -> Any:
    """Start playing a file; returns the player (``is_playing``, ``stop()``)."""
    from bioamla.audio.playback import play_audio

    return play_audio(request.file, loop=request.loop, block=False)


@config_dataclass
class AudioFeaturesRequest:
    """Frame-level descriptors and MFCCs of one file.

    Attributes:
        file: The audio file.
        output: Where to save the per-frame table (``.csv`` or ``.parquet``).
        config: Analysis settings.
        trajectory: Add the MFCC-PCA timbre trajectory (``pc1``..``pc3``).
        gate_db: Trajectory gate: frames this far below the loudest are left out.
    """

    file: PathStr
    output: PathStr | None = None
    config: FrameFeatureConfig = field(default_factory=FrameFeatureConfig)
    trajectory: bool = False
    gate_db: float = 38.0


@operation("audio.features", request=AudioFeaturesRequest, kind="job")
def audio_features(request: AudioFeaturesRequest, context: OperationContext) -> Any:
    """Compute frame-level descriptors and MFCCs of one file."""
    return frame_features_file(
        request.file,
        request.output,
        config=request.config,
        trajectory=request.trajectory,
        gate_db=request.gate_db,
    )


# --------------------------------------------------------------------------- #
# File in, file out
# --------------------------------------------------------------------------- #


@config_dataclass
class AudioConvertRequest:
    """Convert a file's format, sample rate or channel count.

    Attributes:
        file: The audio file.
        output: Where to save the result.
        sample_rate: Target sample rate (Hz).
        channels: Target channel count (1 or 2; default: keep the file's own).
        bit_depth: Bits per sample (WAV: 8, 16, 24, 32; FLAC: 8, 16, 24); by default the
            source's when the output holds it, else 16.
        format: Output format; by default, from ``output``'s extension.
    """

    file: PathStr
    output: PathStr
    sample_rate: int | None = None
    channels: int | None = None
    bit_depth: Literal[8, 16, 24, 32] | None = None
    format: AudioFormat | None = None


@operation("audio.convert", request=AudioConvertRequest, kind="job")
def audio_convert(request: AudioConvertRequest, context: OperationContext) -> ProcessedFile:
    """Convert a file's format, sample rate or channel count."""
    saved = convert_audio_file(
        request.file,
        request.output,
        target_format=request.format,
        target_sample_rate=request.sample_rate,
        target_channels=request.channels,
        target_bit_depth=request.bit_depth,
    )
    return ProcessedFile(file=request.file, output=str(saved))


@config_dataclass
class AudioSegmentRequest:
    """Cut a file into fixed-length clips, ``<prefix>_seg0000.<format>``, ...

    Attributes:
        file: The audio file.
        output_dir: The folder to write the clips to.
        duration: Clip length (s).
        overlap: Overlap between clips (s).
        format: Clip format.
        prefix: Clip name prefix (default: the file's name without extension).
        include_partial: Keep the last clip when it is shorter than ``duration``
            (left out by default).
    """

    file: PathStr
    output_dir: PathStr
    duration: float = 3.0
    overlap: float = 0.0
    format: AudioFormat = "wav"
    prefix: str | None = None
    include_partial: bool = False


@operation("audio.segment", request=AudioSegmentRequest, kind="job")
def audio_segment(request: AudioSegmentRequest, context: OperationContext) -> SegmentedFile:
    """Cut a file into fixed-length clips."""
    from bioamla.audio.batch import segment_audio_file

    segments = segment_audio_file(
        request.file,
        request.output_dir,
        duration=request.duration,
        overlap=request.overlap,
        prefix=request.prefix,
        format=request.format,
        include_partial=request.include_partial,
    )
    return SegmentedFile(file=request.file, output_dir=request.output_dir, segments=segments)


@config_dataclass
class AudioTrimRequest:
    """Keep part of a file.

    Attributes:
        file: The audio file.
        output: Where to save the result.
        start: Start time (s).
        end: End time (s); default the end of the file.
        duration: Length (s), instead of ``end``.
    """

    file: PathStr
    output: PathStr
    start: float = 0.0
    end: float | None = None
    duration: float | None = None


@operation("audio.trim", request=AudioTrimRequest, kind="job")
def audio_trim(request: AudioTrimRequest, context: OperationContext) -> ProcessedFile:
    """Keep part of a file."""
    return trim_audio_file(
        request.file,
        request.output,
        start=request.start,
        end=request.end,
        duration=request.duration,
    )


@config_dataclass
class AudioNormalizeRequest:
    """Normalize a file's level.

    Attributes:
        file: The audio file.
        output: Where to save the result.
        target_db: Target peak (``peak``) or loudness (``rms``) in dB.
        method: ``peak`` or ``rms``.
    """

    file: PathStr
    output: PathStr
    target_db: float = -3.0
    method: Literal["peak", "rms"] = "peak"


@operation("audio.normalize", request=AudioNormalizeRequest, kind="job")
def audio_normalize(request: AudioNormalizeRequest, context: OperationContext) -> ProcessedFile:
    """Normalize a file's level."""
    return normalize_audio_file(
        request.file, request.output, target_db=request.target_db, method=request.method
    )


@config_dataclass
class AudioResampleRequest:
    """Resample a file.

    Attributes:
        file: The audio file.
        output: Where to save the result.
        sample_rate: Target sample rate (Hz).
    """

    file: PathStr
    output: PathStr
    sample_rate: int


@operation("audio.resample", request=AudioResampleRequest, kind="job")
def audio_resample(request: AudioResampleRequest, context: OperationContext) -> ProcessedFile:
    """Resample a file."""
    return resample_audio_file(request.file, request.output, request.sample_rate)


@config_dataclass
class AudioFilterRequest:
    """Apply one frequency filter to a file (band-pass, else low-pass, else high-pass).

    Attributes:
        file: The audio file.
        output: Where to save the result.
        lowpass: Low-pass cutoff (Hz).
        highpass: High-pass cutoff (Hz).
        bandpass_low: Band-pass lower edge (Hz).
        bandpass_high: Band-pass upper edge (Hz).
        order: Filter order.
    """

    file: PathStr
    output: PathStr
    lowpass: float | None = None
    highpass: float | None = None
    bandpass_low: float | None = None
    bandpass_high: float | None = None
    order: int = 5


@operation("audio.filter", request=AudioFilterRequest, kind="job")
def audio_filter(request: AudioFilterRequest, context: OperationContext) -> ProcessedFile:
    """Apply a frequency filter to a file."""
    return filter_audio_file(
        request.file,
        request.output,
        lowpass=request.lowpass,
        highpass=request.highpass,
        bandpass_low=request.bandpass_low,
        bandpass_high=request.bandpass_high,
        order=request.order,
    )


@config_dataclass
class AudioDenoiseRequest:
    """Spectral noise reduction.

    Attributes:
        file: The audio file.
        output: Where to save the result.
        strength: Reduction strength, 0 to 2.
    """

    file: PathStr
    output: PathStr
    strength: float = 1.0


@operation("audio.denoise", request=AudioDenoiseRequest, kind="job")
def audio_denoise(request: AudioDenoiseRequest, context: OperationContext) -> ProcessedFile:
    """Reduce stationary noise in a file."""
    return denoise_audio_file(request.file, request.output, strength=request.strength)


@config_dataclass
class AudioPitchShiftRequest:
    """Shift pitch, keeping duration.

    Attributes:
        file: The audio file.
        output: Where to save the result.
        steps: Semitones (positive raises the pitch).
    """

    file: PathStr
    output: PathStr
    steps: float


@operation("audio.pitch-shift", request=AudioPitchShiftRequest, kind="job")
def audio_pitch_shift(request: AudioPitchShiftRequest, context: OperationContext) -> ProcessedFile:
    """Shift a file's pitch."""
    return pitch_shift_audio_file(request.file, request.output, request.steps)


@config_dataclass
class AudioTimeStretchRequest:
    """Stretch in time, keeping pitch.

    Attributes:
        file: The audio file.
        output: Where to save the result.
        rate: Stretch factor (>1 faster, <1 slower).
    """

    file: PathStr
    output: PathStr
    rate: float


@operation("audio.time-stretch", request=AudioTimeStretchRequest, kind="job")
def audio_time_stretch(
    request: AudioTimeStretchRequest, context: OperationContext
) -> ProcessedFile:
    """Stretch a file in time."""
    return time_stretch_audio_file(request.file, request.output, request.rate)


@config_dataclass
class AudioAddNoiseRequest:
    """Add white noise at a target signal-to-noise ratio.

    Attributes:
        file: The audio file.
        output: Where to save the result.
        snr_db: Target SNR (dB; lower is noisier).
        seed: Random seed, for repeatable noise.
    """

    file: PathStr
    output: PathStr
    snr_db: float
    seed: int | None = None


@operation("audio.add-noise", request=AudioAddNoiseRequest, kind="job")
def audio_add_noise(request: AudioAddNoiseRequest, context: OperationContext) -> ProcessedFile:
    """Add white noise to a file."""
    return add_noise_audio_file(request.file, request.output, request.snr_db, seed=request.seed)


@config_dataclass
class AudioGainRequest:
    """Apply a fixed gain.

    Attributes:
        file: The audio file.
        output: Where to save the result.
        gain_db: Gain (dB; positive amplifies).
    """

    file: PathStr
    output: PathStr
    gain_db: float


@operation("audio.gain", request=AudioGainRequest, kind="job")
def audio_gain(request: AudioGainRequest, context: OperationContext) -> ProcessedFile:
    """Apply a fixed gain to a file."""
    return gain_audio_file(request.file, request.output, request.gain_db)


# --------------------------------------------------------------------------- #
# Pictures and video
# --------------------------------------------------------------------------- #


@config_dataclass
class VisualizeBox:
    """A region to outline on a picture (see :class:`bioamla.viz.SpectrogramBox`).

    Attributes:
        start: Start (s).
        end: End (s).
        low_freq: Lower edge (Hz); leave both edges out for the full height.
        high_freq: Upper edge (Hz).
        label: Text at its top-left corner.
    """

    start: float
    end: float
    low_freq: float | None = None
    high_freq: float | None = None
    label: str = ""


@config_dataclass
class AudioVisualizeRequest:
    """A spectrogram, MFCC or waveform image of a file.

    Attributes:
        file: The audio file.
        output: The image file to write.
        viz_type: ``mel``, ``stft``, ``mfcc`` or ``waveform``.
        n_fft: FFT window size.
        hop_length: Hop length.
        n_mels: Mel bands.
        n_mfcc: MFCCs.
        cmap: Colormap name.
        dpi: Image DPI.
        legend: Show axes, title and colorbar.
        boxes: Regions to outline (seconds, Hz).
        annotations: An annotation file (any format bioamla reads) whose
            annotations are outlined too, labelled.
    """

    file: PathStr
    output: PathStr
    viz_type: Literal["mel", "stft", "mfcc", "waveform"] = "mel"
    n_fft: int = DEFAULT_N_FFT
    hop_length: int = DEFAULT_HOP_LENGTH
    n_mels: int = DEFAULT_N_MELS
    n_mfcc: int = DEFAULT_N_MFCC
    cmap: str = DEFAULT_CMAP
    dpi: int = DEFAULT_DPI
    legend: bool = True
    boxes: list[VisualizeBox] = field(default_factory=list)
    annotations: PathStr | None = None


@operation("audio.visualize", request=AudioVisualizeRequest, kind="job")
def audio_visualize(request: AudioVisualizeRequest, context: OperationContext) -> ProcessedFile:
    """Draw a spectrogram, MFCC or waveform image of a file, with boxes outlined."""
    from bioamla.viz import SpectrogramBox, annotation_boxes, generate_spectrogram

    boxes = [SpectrogramBox.of(b) for b in request.boxes]
    if request.annotations:
        boxes += annotation_boxes(request.annotations, recording=request.file)
    saved = generate_spectrogram(
        audio_path=request.file,
        output_path=request.output,
        viz_type=request.viz_type,
        n_fft=request.n_fft,
        hop_length=request.hop_length,
        n_mels=request.n_mels,
        n_mfcc=request.n_mfcc,
        cmap=request.cmap,
        dpi=request.dpi,
        show_colorbar=request.legend,
        boxes=boxes,
    )
    return ProcessedFile(file=request.file, output=str(saved), description=request.viz_type)


@config_dataclass
class AudioVideoRequest:
    """A spectrogram, waveform or timbre-trajectory video of a file, with its audio.

    Spectrogram options (``motion``, ``n_fft``, ``hop_length``, ``n_mels``, ``fmax``,
    ``db_range``, ``playhead``, ``waveform``) apply to ``mel``, ``stft`` and
    ``waveform``; ``trail``, ``orbit`` and ``spectrogram`` to ``trajectory``.

    Attributes:
        file: The audio file.
        output: The video file to write.
        viz_type: ``mel``, ``stft``, ``waveform`` or ``trajectory``.
        motion: ``scroll`` (picture moves) or ``page`` (playhead moves).
        start: Start (s).
        end: End (s); default the end of the file.
        fps: Frames per second.
        window: Seconds across the frame (0: the whole recording, page mode); for a
            trajectory, the spectrogram strip's span (must be positive).
        width: Frame width (px).
        height: Frame height (px).
        cmap: Colormap; by default the renderer's (``magma``, or ``plasma`` for a
            trajectory).
        n_fft: FFT window size.
        hop_length: Hop length.
        n_mels: Mel bands.
        fmax: Highest frequency shown (Hz); default Nyquist.
        db_range: Dynamic range (dB).
        playhead: Playhead position across the frame, 0 to 1.
        waveform: Scrolling waveform strip.
        axes: Frequency ticks and time readout.
        include_audio: Mux the audio track.
        trail: Trajectory: fade path older than this many seconds.
        orbit: Trajectory: turn the view slowly.
        spectrogram: Trajectory: mel spectrogram strip along the bottom.
    """

    file: PathStr
    output: PathStr
    viz_type: Literal["mel", "stft", "waveform", "trajectory"] = "mel"
    motion: Literal["scroll", "page"] = "scroll"
    start: float = 0.0
    end: float | None = None
    fps: int = 30
    window: float = 4.0
    width: int = 1280
    height: int = 720
    cmap: str | None = None
    n_fft: int = 2048
    hop_length: int = 512
    n_mels: int = 128
    fmax: float | None = None
    db_range: float = 80.0
    playhead: float = 0.5
    waveform: bool = True
    axes: bool = True
    include_audio: bool = True
    trail: float | None = None
    orbit: bool = True
    spectrogram: bool = True


@operation("audio.video", request=AudioVideoRequest, kind="job")
def audio_video(request: AudioVideoRequest, context: OperationContext) -> Any:
    """Render a spectrogram, waveform or trajectory video of a file."""
    style: dict[str, Any] = {} if request.cmap is None else {"cmap": request.cmap}
    common: dict[str, Any] = {
        **style,
        "audio_path": request.file,
        "output_path": request.output,
        "on_progress": context.progress,
        "start": request.start,
        "end": request.end,
        "width": request.width,
        "height": request.height,
        "fps": request.fps,
        "show_axes": request.axes,
        "include_audio": request.include_audio,
    }
    if request.viz_type == "trajectory":
        from bioamla.viz import generate_trajectory_video

        return generate_trajectory_video(
            **common,
            trail_seconds=request.trail,
            orbit=request.orbit,
            spectrogram=request.spectrogram,
            window_seconds=request.window,
        )
    from bioamla.viz import generate_spectrogram_video

    return generate_spectrogram_video(
        **common,
        viz_type=request.viz_type,
        motion=request.motion,
        n_fft=request.n_fft,
        hop_length=request.hop_length,
        n_mels=request.n_mels,
        fmax=request.fmax,
        db_range=request.db_range,
        window_seconds=request.window or None,
        playhead=request.playhead,
        waveform=request.waveform,
    )
