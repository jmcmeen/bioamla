"""Audio processing on files: load one, transform it, save the result.

These are the file-level calls behind the ``bioamla audio`` commands. Each takes an
input file and an output path, loads the audio (mono, as :func:`load_audio` does),
applies one step and writes the result, returning a :class:`ProcessedFile`.
:func:`describe_audio_file` and :func:`recording_metadata_report` read a file's
header and recorder metadata without decoding it.

Example:
    >>> from bioamla.audio import filter_audio_file, trim_audio_file
    >>> trim_audio_file("night.wav", "clip.wav", start=90, duration=30)
    >>> filter_audio_file("clip.wav", "clip-hp.wav", highpass=500)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from bioamla.audio.info import AudioInfo, get_audio_info
from bioamla.audio.io import load_audio, save_audio
from bioamla.exceptions import InvalidInputError


@dataclass
class ProcessedFile:
    """An audio file that was transformed and saved.

    Attributes:
        file: The input file.
        output: The file written.
        description: What was done, e.g. ``"highpass filter (500.0 Hz)"``.
    """

    file: str
    output: str
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        """``{"file", "output", "description"}``."""
        return {"file": self.file, "output": self.output, "description": self.description}


def _transform(
    input_path: str | Path, output_path: str | Path, fn: Any, description: str = ""
) -> ProcessedFile:
    """Load ``input_path``, save ``fn(audio, sr)`` (an array, or ``(array, rate)``)."""
    audio, sr = load_audio(str(input_path))
    out = fn(audio, sr)
    samples, rate = out if isinstance(out, tuple) else (out, sr)
    save_audio(str(output_path), samples, rate)
    return ProcessedFile(file=str(input_path), output=str(output_path), description=description)


def trim_audio_file(
    input_path: str | Path,
    output_path: str | Path,
    *,
    start: float = 0.0,
    end: float | None = None,
    duration: float | None = None,
) -> ProcessedFile:
    """Save the part of a file between ``start`` and ``end`` (or ``start + duration``).

    Args:
        input_path: The audio file.
        output_path: Where to save the trimmed audio.
        start: Start time (s).
        end: End time (s); ``None`` for the end of the file.
        duration: Length (s), instead of ``end``.

    Raises:
        InvalidInputError: If both ``end`` and ``duration`` are given, ``start`` is
            negative or ``duration`` isn't positive.
        ProcessingError: If ``end`` isn't after ``start``.
        AudioLoadError / AudioSaveError: On I/O failure.
    """
    from bioamla.audio.processing import trim_audio

    if end is not None and duration is not None:
        raise InvalidInputError("Cannot specify both end and duration")
    if start < 0:
        raise InvalidInputError(f"start must not be negative (got {start})")
    if duration is not None:
        if duration <= 0:
            raise InvalidInputError(f"duration must be positive (got {duration})")
        end = start + duration
    return _transform(
        input_path,
        output_path,
        lambda a, sr: trim_audio(a, sr, start_time=start if start != 0.0 else None, end_time=end),
    )


def normalize_audio_file(
    input_path: str | Path,
    output_path: str | Path,
    *,
    target_db: float = -3.0,
    method: str = "peak",
) -> ProcessedFile:
    """Normalize a file's level and save it.

    Args:
        input_path: The audio file.
        output_path: Where to save the result.
        target_db: Target level (dB): the peak for ``"peak"`` (capped just below full
            scale), the loudness for ``"rms"``.
        method: ``"peak"`` or ``"rms"``.

    Raises:
        InvalidInputError: If ``method`` is unknown.
        AudioLoadError / AudioSaveError: On I/O failure.
    """
    from bioamla.audio.processing import normalize_loudness, peak_normalize

    if method == "peak":
        target = min(10 ** (target_db / 20), 0.99)
        return _transform(
            input_path, output_path, lambda a, sr: peak_normalize(a, target_peak=target)
        )
    if method == "rms":
        return _transform(
            input_path, output_path, lambda a, sr: normalize_loudness(a, sr, target_db=target_db)
        )
    raise InvalidInputError(f"Unknown normalization method {method!r}; use 'peak' or 'rms'")


def resample_audio_file(
    input_path: str | Path, output_path: str | Path, sample_rate: int
) -> ProcessedFile:
    """Resample a file to ``sample_rate`` (Hz) and save it."""
    from bioamla.audio.processing import resample_audio

    return _transform(
        input_path, output_path, lambda a, sr: (resample_audio(a, sr, sample_rate), sample_rate)
    )


def filter_audio_file(
    input_path: str | Path,
    output_path: str | Path,
    *,
    lowpass: float | None = None,
    highpass: float | None = None,
    bandpass_low: float | None = None,
    bandpass_high: float | None = None,
    order: int = 5,
) -> ProcessedFile:
    """Apply one Butterworth filter to a file and save it.

    A band-pass (``bandpass_low`` and ``bandpass_high`` together) wins over a
    low-pass, which wins over a high-pass.

    Args:
        input_path: The audio file.
        output_path: Where to save the result.
        lowpass: Low-pass cutoff (Hz).
        highpass: High-pass cutoff (Hz).
        bandpass_low: Band-pass lower edge (Hz).
        bandpass_high: Band-pass upper edge (Hz).
        order: Filter order.

    Returns:
        The result; its ``description`` names the filter applied.

    Raises:
        InvalidInputError: If no filter is given, or only one band-pass edge.
        AudioLoadError / AudioSaveError: On I/O failure.
    """
    from bioamla.audio.processing import bandpass_filter, highpass_filter, lowpass_filter

    if (bandpass_low is not None) != (bandpass_high is not None):
        raise InvalidInputError("Both bandpass_low and bandpass_high must be specified together")
    if lowpass is None and highpass is None and bandpass_low is None:
        raise InvalidInputError("Must specify lowpass, highpass, or bandpass_low and bandpass_high")
    if bandpass_low is not None:
        return _transform(
            input_path,
            output_path,
            lambda a, sr: bandpass_filter(a, sr, bandpass_low, bandpass_high, order),
            f"bandpass filter ({bandpass_low}-{bandpass_high} Hz)",
        )
    if lowpass is not None:
        return _transform(
            input_path,
            output_path,
            lambda a, sr: lowpass_filter(a, sr, lowpass, order),
            f"lowpass filter ({lowpass} Hz)",
        )
    return _transform(
        input_path,
        output_path,
        lambda a, sr: highpass_filter(a, sr, highpass, order),
        f"highpass filter ({highpass} Hz)",
    )


def denoise_audio_file(
    input_path: str | Path, output_path: str | Path, *, strength: float = 1.0
) -> ProcessedFile:
    """Apply spectral noise reduction (``strength`` 0–2) to a file and save it."""
    from bioamla.audio.processing import spectral_denoise

    return _transform(
        input_path, output_path, lambda a, sr: spectral_denoise(a, sr, noise_reduce_factor=strength)
    )


def pitch_shift_audio_file(
    input_path: str | Path, output_path: str | Path, steps: float
) -> ProcessedFile:
    """Shift a file's pitch by ``steps`` semitones (keeping its length) and save it."""
    from bioamla.audio.processing import pitch_shift

    return _transform(input_path, output_path, lambda a, sr: pitch_shift(a, sr, steps))


def time_stretch_audio_file(
    input_path: str | Path, output_path: str | Path, rate: float
) -> ProcessedFile:
    """Stretch a file in time by ``rate`` (>1 faster; pitch kept) and save it."""
    from bioamla.audio.processing import time_stretch

    return _transform(input_path, output_path, lambda a, sr: time_stretch(a, rate))


def add_noise_audio_file(
    input_path: str | Path, output_path: str | Path, snr_db: float, *, seed: int | None = None
) -> ProcessedFile:
    """Add white noise at ``snr_db`` to a file and save it (``seed`` for repeatability)."""
    from bioamla.audio.processing import add_noise

    return _transform(input_path, output_path, lambda a, sr: add_noise(a, snr_db, seed=seed))


def gain_audio_file(
    input_path: str | Path, output_path: str | Path, gain_db: float
) -> ProcessedFile:
    """Apply a fixed gain (dB) to a file and save it."""
    from bioamla.audio.processing import apply_gain

    return _transform(input_path, output_path, lambda a, sr: apply_gain(a, gain_db))


@dataclass
class SegmentedFile:
    """An audio file cut into fixed-length clips.

    Attributes:
        file: The input file.
        output_dir: The folder the clips were written to.
        segments: The clips written, in order.
    """

    file: str
    output_dir: str
    segments: list[Any]

    @property
    def count(self) -> int:
        """The number of clips."""
        return len(self.segments)

    def to_dict(self) -> dict[str, Any]:
        """``{"file", "output_dir", "count", "segments"}``, each clip's path and times."""
        return {
            "file": self.file,
            "output_dir": self.output_dir,
            "count": self.count,
            "segments": [
                {
                    "path": str(seg.segment_path),
                    "start_time": seg.start_time,
                    "end_time": seg.end_time,
                }
                for seg in self.segments
            ],
        }


@dataclass
class AudioFileDescription:
    """A file's header information and, if asked for, its recorder metadata.

    Attributes:
        file: The audio file.
        info: Format, duration, sample rate, channels, ...
        recording_metadata: The recorder metadata (``None`` if not asked for, or the
            file isn't a WAV).
        metadata_requested: Whether recorder metadata was asked for.
    """

    file: str
    info: AudioInfo
    recording_metadata: Any = None
    metadata_requested: bool = False

    def to_dict(self) -> dict[str, Any]:
        """``{"file", "info"}``, plus ``"recording_metadata"`` when it was asked for:
        ``None`` for a non-WAV file, ``{}`` for a WAV without recorder metadata."""
        result: dict[str, Any] = {"file": self.file, "info": self.info}
        if self.metadata_requested:
            meta = self.recording_metadata
            result["recording_metadata"] = (
                None if meta is None else ({} if meta.is_empty else meta.to_dict())
            )
        return result


def describe_audio_file(
    path: str | Path, *, recording_metadata: bool = False
) -> AudioFileDescription:
    """Read a file's header (it isn't decoded) and optionally its recorder metadata.

    Args:
        path: The audio file.
        recording_metadata: Also read GUANO, AudioMoth, WAMD and bext metadata. A
            file that isn't a WAV has none (``recording_metadata`` stays ``None``).

    Raises:
        NotFoundError / AudioLoadError: If the file can't be read.
    """
    from bioamla.audio.metadata import get_recording_metadata

    info = get_audio_info(str(path))
    meta = None
    if recording_metadata:
        try:
            meta = get_recording_metadata(str(path))
        except InvalidInputError:  # not a WAV file
            meta = None
    return AudioFileDescription(
        file=str(path), info=info, recording_metadata=meta, metadata_requested=recording_metadata
    )


@dataclass
class RecordingMetadataReport:
    """A WAV file's recorder metadata, flattened and per standard.

    Attributes:
        file: The WAV file.
        summary: The flattened view (when, where, which device) datasets use.
        standards: Everything decoded, per standard (GUANO, INFO, bext, ...).
    """

    file: str
    summary: Any
    standards: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """``{"file", "summary", "standards"}``."""
        return {"file": self.file, "summary": self.summary.to_dict(), "standards": self.standards}


def recording_metadata_report(path: str | Path) -> RecordingMetadataReport:
    """Read a WAV file's recorder metadata, both flattened and per standard (header only).

    Raises:
        InvalidInputError: If the file isn't a WAV.
        NotFoundError: If it doesn't exist.
    """
    from bioamla.audio.metadata import dump_recording_metadata, get_recording_metadata

    standards = dump_recording_metadata(str(path))
    file = standards.pop("file", str(path))
    return RecordingMetadataReport(
        file=file, summary=get_recording_metadata(str(path)), standards=standards
    )


@dataclass
class AudioFileList:
    """The audio files found under a folder.

    Attributes:
        path: The folder searched.
        recursive: Whether subfolders were searched.
        files: The files found.
    """

    path: str
    recursive: bool
    files: list[Path]

    @property
    def count(self) -> int:
        """The number of files."""
        return len(self.files)

    def to_dict(self) -> dict[str, Any]:
        """``{"path", "recursive", "count", "files"}``."""
        return {
            "path": self.path,
            "recursive": self.recursive,
            "count": self.count,
            "files": self.files,
        }


__all__ = [
    "AudioFileDescription",
    "AudioFileList",
    "ProcessedFile",
    "RecordingMetadataReport",
    "SegmentedFile",
    "add_noise_audio_file",
    "denoise_audio_file",
    "describe_audio_file",
    "filter_audio_file",
    "gain_audio_file",
    "normalize_audio_file",
    "pitch_shift_audio_file",
    "recording_metadata_report",
    "resample_audio_file",
    "time_stretch_audio_file",
    "trim_audio_file",
]
