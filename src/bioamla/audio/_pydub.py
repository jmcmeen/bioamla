"""Low-level audio I/O backend using soundfile / pydub / ffprobe.

This private module is the raw decode/encode/metadata backend for
:mod:`bioamla.audio`. Higher-level modules (:mod:`bioamla.audio.io`,
:mod:`bioamla.audio.info`) wrap these functions to add exception translation
and dataclass return types; external callers should use the public
``bioamla.audio`` API rather than importing this directly.

Design notes:
- All audio loading returns mono float32 in [-1.0, 1.0]
- Uses soundfile for WAV/FLAC/OGG (fast native C) with pydub fallback
- Uses ffprobe for fast metadata extraction without full file decode
- Saving writes WAV and FLAC through soundfile (16-bit unless a bit depth is given),
  other formats through pydub at 16 bits
"""

import json
import logging
import shutil
import subprocess
from collections.abc import Iterator
from pathlib import Path

import numpy as np
from pydub import AudioSegment

from bioamla.exceptions import (
    AudioLoadError,
    AudioSaveError,
    DependencyError,
    InvalidInputError,
    NotFoundError,
)

logger = logging.getLogger(__name__)


def _audiosegment_to_numpy(
    segment: AudioSegment, keep_channels: bool = False
) -> tuple[np.ndarray, int]:
    """Convert pydub AudioSegment to numpy float32 array.

    Stereo/multi-channel audio is downmixed to mono by averaging channels, unless
    ``keep_channels``: then it is ``(n_samples, n_channels)``.

    Args:
        segment: AudioSegment to convert.
        keep_channels: Keep every channel instead of downmixing.

    Returns:
        Tuple of (audio_array, sample_rate), float32 in [-1.0, 1.0].

    Raises:
        AudioLoadError: If sample width is not 1, 2, 3, or 4 bytes.
    """
    array = np.array(segment.get_array_of_samples())
    sample_width = segment.sample_width

    if sample_width == 1:
        # 8-bit unsigned, range [0, 255], center at 128
        array = array.astype(np.uint8)
        audio = (array.astype(np.float32) - 128.0) / 128.0
    elif sample_width == 2:
        # 16-bit signed, range [-32768, 32767]
        array = array.astype(np.int16)
        audio = array.astype(np.float32) / 32768.0
    elif sample_width == 3:
        # 24-bit signed, range [-8388608, 8388607]
        array = array.astype(np.int32)
        audio = array.astype(np.float32) / 8388608.0
    elif sample_width == 4:
        # 32-bit: float samples stay as they are; integer PCM is scaled. Decided by
        # the array's type, never by the values (a near-silent int file is not float).
        if np.issubdtype(array.dtype, np.floating):
            audio = array.astype(np.float32)
        else:
            audio = array.astype(np.int32).astype(np.float32) / 2147483648.0
    else:
        raise AudioLoadError(
            f"Unsupported sample width: {sample_width} bytes. "
            f"Expected 1 (8-bit), 2 (16-bit), 3 (24-bit), or 4 (32-bit)."
        )

    # Downmix to mono if multi-channel
    if segment.channels > 1:
        audio = audio.reshape((-1, segment.channels))
        if keep_channels:
            return audio, segment.frame_rate
        audio = audio.mean(axis=1)
        logger.debug(
            f"Downmixed {segment.channels}-channel audio to mono "
            f"({sample_width * 8}-bit, {segment.frame_rate}Hz)"
        )
    else:
        logger.debug(f"Loaded mono audio ({sample_width * 8}-bit, {segment.frame_rate}Hz)")

    return audio, segment.frame_rate


def _numpy_to_audiosegment(
    audio: np.ndarray, sample_rate: int, channels: int | None = None
) -> AudioSegment:
    """Convert numpy float32 array to pydub AudioSegment.

    Args:
        audio: Audio data as numpy array (float32, range [-1.0, 1.0]).
        sample_rate: Sample rate in Hz.
        channels: Number of channels (auto-detected if None).

    Returns:
        AudioSegment object.
    """
    if channels is None:
        channels = 1 if audio.ndim == 1 else audio.shape[1]

    if audio.ndim == 2 and channels == 1:
        audio = audio.flatten()

    # Convert float32 [-1.0, 1.0] to int16
    audio_clipped = np.clip(audio, -1.0, 1.0)
    audio_int16 = (audio_clipped * 32767.0).astype(np.int16)

    return AudioSegment(
        audio_int16.tobytes(),
        frame_rate=sample_rate,
        sample_width=2,  # 16-bit
        channels=channels,
    )


def _load_via_pydub(filepath: str, keep_channels: bool = False) -> tuple[np.ndarray, int]:
    """Load audio file via pydub/ffmpeg.

    Args:
        filepath: Path to audio file.
        keep_channels: Keep every channel (``(n_samples, n_channels)``) instead of
            downmixing to mono.

    Returns:
        Tuple of (audio_array, sample_rate).

    Raises:
        NotFoundError: If file doesn't exist.
        AudioLoadError: If file cannot be loaded.
    """
    path = Path(filepath)
    if not path.exists():
        raise NotFoundError(f"Audio file not found: {filepath}")

    try:
        segment = AudioSegment.from_file(str(path))
        return _audiosegment_to_numpy(segment, keep_channels=keep_channels)
    except Exception as e:
        raise AudioLoadError(f"Error opening '{filepath}': {e}") from e


# Formats soundfile reads natively (and can seek in without decoding the rest).
_NATIVE_EXTENSIONS = {".wav", ".flac", ".ogg"}


def _to_mono(audio: np.ndarray) -> np.ndarray:
    return audio.mean(axis=1) if audio.ndim > 1 else audio


def load_audio_channels(filepath: str) -> tuple[np.ndarray, int]:
    """Load an audio file keeping its channels: 1-D for mono, ``(n_samples,
    n_channels)`` otherwise, float32 in [-1.0, 1.0]. soundfile (native C, ~10-30x
    faster) for WAV/FLAC/OGG, pydub (ffmpeg) for the rest or when soundfile fails.

    Raises:
        NotFoundError: If file doesn't exist.
        AudioLoadError: If file cannot be loaded.
    """
    path = Path(filepath)
    if not path.exists():
        raise NotFoundError(f"Audio file not found: {filepath}")
    if path.suffix.lower() in _NATIVE_EXTENSIONS:
        try:
            import soundfile as sf

            audio, sample_rate = sf.read(str(path), dtype="float32", always_2d=False)
            return audio, sample_rate
        except Exception as e:
            logger.debug(f"soundfile failed for {filepath}, using pydub fallback: {e}")
    return _load_via_pydub(str(path), keep_channels=True)


def load_audio(filepath: str) -> tuple[np.ndarray, int]:
    """Load an audio file as mono float32 in [-1.0, 1.0] (channels averaged), using
    the fastest backend (see :func:`load_audio_channels`).

    Raises:
        NotFoundError: If file doesn't exist.
        AudioLoadError: If file cannot be loaded.
    """
    audio, sample_rate = load_audio_channels(filepath)
    return _to_mono(audio).astype(np.float32, copy=False), sample_rate


def load_audio_window(
    filepath: str, start: float = 0.0, duration: float | None = None
) -> tuple[np.ndarray, int]:
    """Load part of an audio file without decoding the rest.

    WAV/FLAC/OGG seek straight to ``start`` and read only the requested frames
    via soundfile. Other formats, and native files soundfile can't read, are
    decoded by ffmpeg seeking on the input (``-ss`` before ``-i``), so only the
    window is decoded.

    Args:
        filepath: Path to audio file.
        start: Window start in seconds.
        duration: Window length in seconds; None reads to the end of the file.

    Returns:
        Tuple of (audio_array, sample_rate), mono float32 [-1.0, 1.0]. The array
        is shorter than ``duration`` when the window runs past the end, and
        empty when ``start`` is beyond it.

    Raises:
        NotFoundError: If file doesn't exist.
        DependencyError: If the format needs ffmpeg/ffprobe and they're missing.
        AudioLoadError: If the window cannot be decoded.
    """
    path = Path(filepath)
    if not path.exists():
        raise NotFoundError(f"Audio file not found: {filepath}")

    if path.suffix.lower() in _NATIVE_EXTENSIONS:
        try:
            import soundfile as sf

            with sf.SoundFile(str(path)) as f:
                sample_rate = f.samplerate
                f.seek(min(round(start * sample_rate), f.frames))
                frames = -1 if duration is None else round(duration * sample_rate)
                audio = f.read(frames, dtype="float32", always_2d=False)
            return _to_mono(audio), sample_rate
        except Exception as e:
            logger.debug(f"soundfile window read failed for {filepath}, using ffmpeg: {e}")

    return _ffmpeg_window(path, start, duration)


# Seconds decoded and discarded before an ffmpeg window, so lossy decoders have
# settled by the window's first sample (MP3 needs ~2 frames, AAC a little more).
_FFMPEG_PREROLL = 0.25


def _ffmpeg_window(path: Path, start: float, duration: float | None) -> tuple[np.ndarray, int]:
    """Decode ``[start, start + duration)`` of ``path`` with ffmpeg, mono float32."""
    if shutil.which("ffmpeg") is None:
        raise DependencyError(
            f"ffmpeg is required to read {path.suffix or 'this'} files but was not found on PATH."
        )
    meta = _get_metadata_ffprobe(str(path))
    if not meta or not meta.get("sample_rate") or not meta.get("channels"):
        raise AudioLoadError(f"Error opening '{path}': ffprobe could not read the stream format")
    sample_rate = int(meta["sample_rate"])
    channels = int(meta["channels"])

    # -ss before -i seeks the input, so ffmpeg skips (not decodes) what precedes it.
    # Lossy decoders (MP3, AAC) need a few frames to settle after a seek, so decode
    # a short pre-roll before the window and drop it.
    # Near the start there is nothing to seek past: decode from the beginning
    # without -ss, which also keeps ffmpeg's handling of the encoder's leading
    # padding identical to a full decode.
    preroll = start if start <= _FFMPEG_PREROLL else _FFMPEG_PREROLL
    seek = start - preroll
    cmd = ["ffmpeg", "-nostdin", "-v", "error"]
    if seek > 0:
        cmd += ["-ss", f"{seek:.6f}"]
    cmd += ["-i", str(path)]
    if duration is not None:
        cmd += ["-t", f"{duration + preroll:.6f}"]
    cmd += ["-map", "0:a:0", "-f", "f32le", "-acodec", "pcm_f32le", "-"]
    try:
        result = subprocess.run(cmd, capture_output=True, check=False)
    except OSError as e:
        raise AudioLoadError(f"Error opening '{path}': {e}") from e
    if result.returncode != 0:
        detail = result.stderr.decode(errors="replace").strip()
        raise AudioLoadError(f"Error opening '{path}': {detail}")

    audio = np.frombuffer(result.stdout, dtype=np.float32)
    audio = audio[: len(audio) - len(audio) % channels].reshape(-1, channels)
    audio = audio[round(preroll * sample_rate) :]
    if duration is not None:
        audio = audio[: round(duration * sample_rate)]
    return _to_mono(audio).astype(np.float32, copy=False), sample_rate


def iter_audio_blocks(
    filepath: str, block_duration: float, overlap: float = 0.0
) -> Iterator[tuple[np.ndarray, int, float]]:
    """Yield consecutive fixed-length blocks of an audio file.

    Only one block is held in memory at a time. WAV/FLAC/OGG are streamed with
    soundfile; other formats decode one window per block through ffmpeg,
    reading until the file runs out (header durations of VBR or truncated
    files aren't trusted).

    Args:
        filepath: Path to audio file.
        block_duration: Block length in seconds.
        overlap: Seconds each block shares with the previous one.

    Yields:
        Tuples of (audio_array, sample_rate, start_time) where audio is mono
        float32 and ``start_time`` is the block's offset in seconds. The last
        block may be shorter.

    Raises:
        NotFoundError: If file doesn't exist.
        InvalidInputError: If a block, or the step between blocks, is shorter
            than one sample at the file's rate.
        DependencyError: If the format needs ffmpeg/ffprobe and they're missing.
        AudioLoadError: If the file cannot be decoded.
    """
    path = Path(filepath)
    if not path.exists():
        raise NotFoundError(f"Audio file not found: {filepath}")

    if path.suffix.lower() in _NATIVE_EXTENSIONS:
        import soundfile as sf

        try:
            f = sf.SoundFile(str(path))
        except Exception as e:
            logger.debug(f"soundfile can't stream {filepath}, using ffmpeg: {e}")
        else:
            with f:
                sample_rate = f.samplerate
                block = round(block_duration * sample_rate)
                step = block - round(overlap * sample_rate)
                _check_block_samples(block, step, sample_rate)
                position = 0
                while position < f.frames:
                    f.seek(position)
                    audio = f.read(block, dtype="float32", always_2d=False)
                    yield _to_mono(audio), sample_rate, position / sample_rate
                    if position + block >= f.frames:
                        return
                    position += step
            return

    step = block_duration - overlap
    index = 0
    while True:
        start = index * step  # multiply, don't accumulate: no float drift
        audio, sample_rate = _ffmpeg_window(path, start, block_duration)
        if index == 0:
            _check_block_samples(
                round(block_duration * sample_rate), round(step * sample_rate), sample_rate
            )
        if len(audio) == 0:
            return
        yield audio, sample_rate, start
        if len(audio) < round(block_duration * sample_rate):
            return  # a short block is the end of the file
        index += 1


def _check_block_samples(block: int, step: int, sample_rate: int) -> None:
    if block < 1 or step < 1:
        raise InvalidInputError(
            "block_duration and block_duration - overlap must each be at least one "
            f"sample ({1 / sample_rate:.6g} s at {sample_rate} Hz)"
        )


#: Bit depth -> soundfile subtype, for the formats that store integer samples.
BIT_DEPTH_SUBTYPES = {
    "wav": {8: "PCM_U8", 16: "PCM_16", 24: "PCM_24", 32: "PCM_32"},
    "flac": {8: "PCM_S8", 16: "PCM_16", 24: "PCM_24"},
}

# Extension -> the format ``save_audio`` writes (pydub calls M4A/AAC "ipod").
_EXTENSION_FORMATS = {".wav": "wav", ".flac": "flac", ".ogg": "ogg", ".mp3": "mp3", ".m4a": "ipod"}


def output_format(path: str | Path, format: str | None = None) -> str:
    """The format :func:`save_audio` writes ``path`` in: ``format`` (any case, with or
    without a leading dot), else from the extension, else WAV."""
    if format:
        return format.lower().lstrip(".")
    return _EXTENSION_FORMATS.get(Path(path).suffix.lower(), "wav")


def check_bit_depth(format: str, bit_depth: int | None) -> None:
    """Raise unless ``format`` (as :func:`output_format` gives it) can hold ``bit_depth``
    (``None`` always passes).

    Raises:
        InvalidInputError: If it can't (lossy formats hold no bit depth).
    """
    if bit_depth is None or bit_depth in BIT_DEPTH_SUBTYPES.get(format, {}):
        return
    choices = "; ".join(
        f"{fmt.upper()}: {', '.join(map(str, bits))}" for fmt, bits in BIT_DEPTH_SUBTYPES.items()
    )
    raise InvalidInputError(
        f"A bit depth of {bit_depth} can't be written as {format.upper()} "
        f"({choices}; lossy formats have none)"
    )


def save_audio(
    filepath: str,
    audio: np.ndarray,
    sample_rate: int,
    format: str | None = None,
    bit_depth: int | None = None,
) -> None:
    """Save numpy audio array to file.

    WAV and FLAC are written by soundfile, at ``bit_depth`` bits (default 16); other
    formats by pydub (ffmpeg), at 16 bits.

    Args:
        filepath: Destination file path.
        audio: Audio data as numpy array (float32), 1-D or ``(n_samples, n_channels)``.
        sample_rate: Sample rate in Hz.
        format: Output format (auto-detected from extension if None).
        bit_depth: Bits per sample (WAV: 8, 16, 24, 32; FLAC: 8, 16, 24); default 16.

    Raises:
        InvalidInputError: If ``bit_depth`` doesn't apply to the format.
        AudioSaveError: If file cannot be saved.
    """
    path = Path(filepath)
    format_to_use = output_format(path, format)
    check_bit_depth(format_to_use, bit_depth)

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        if format_to_use in BIT_DEPTH_SUBTYPES:
            import soundfile as sf

            subtype = BIT_DEPTH_SUBTYPES[format_to_use][bit_depth or 16]
            sf.write(str(path), np.clip(audio, -1.0, 1.0), sample_rate,
                     subtype=subtype, format=format_to_use.upper())  # fmt: skip
            return
        channels = 1 if audio.ndim == 1 else audio.shape[1]
        segment = _numpy_to_audiosegment(audio, sample_rate, channels)
        # export() returns the open output file; close it rather than leak it.
        segment.export(str(path), format=format_to_use).close()
    except Exception as e:
        raise AudioSaveError(f"Failed to save audio to '{filepath}': {e}") from e


def _get_metadata_ffprobe(filepath: str) -> dict | None:
    """Extract audio metadata using ffprobe (header-only, no decoding).

    Args:
        filepath: Path to audio file.

    Returns:
        Dictionary with metadata, or None if ffprobe fails.
    """
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "quiet",
                "-print_format",
                "json",
                "-show_format",
                "-show_streams",
                str(filepath),
            ],
            capture_output=True,
            text=True,
            timeout=5.0,
        )

        if result.returncode != 0:
            return None

        data = json.loads(result.stdout)

        # Find first audio stream
        audio_stream = None
        for stream in data.get("streams", []):
            if stream.get("codec_type") == "audio":
                audio_stream = stream
                break

        if not audio_stream:
            return None

        sample_rate = int(audio_stream.get("sample_rate", 0))
        channels = int(audio_stream.get("channels", 0))
        duration = float(audio_stream.get("duration") or data.get("format", {}).get("duration", 0))
        samples = int(duration * sample_rate) if duration and sample_rate else 0
        codec = audio_stream.get("codec_name", "unknown")

        bit_depth = audio_stream.get("bits_per_sample") or audio_stream.get("bits_per_raw_sample")
        if bit_depth:
            bit_depth = int(bit_depth)

        # Map codec to subtype string
        codec_to_subtype = {
            "pcm_s16le": "PCM_16",
            "pcm_s16be": "PCM_16",
            "pcm_s24le": "PCM_24",
            "pcm_s24be": "PCM_24",
            "pcm_s32le": "PCM_32",
            "pcm_s32be": "PCM_32",
            "pcm_u8": "PCM_U8",
            "pcm_f32le": "FLOAT",
            "pcm_f64le": "DOUBLE",
            "flac": "FLAC",
            "mp3": "MP3",
            "mp3float": "MP3",
            "aac": "AAC",
            "vorbis": "VORBIS",
        }
        subtype = codec_to_subtype.get(codec, codec.upper())

        return {
            "duration": duration,
            "sample_rate": sample_rate,
            "channels": channels,
            "samples": samples,
            "format": Path(filepath).suffix.upper().lstrip("."),
            "codec": codec,
            "bit_depth": bit_depth,
            "subtype": subtype,
        }

    except (
        subprocess.TimeoutExpired,
        subprocess.SubprocessError,
        json.JSONDecodeError,
        KeyError,
        ValueError,
        FileNotFoundError,
    ) as e:
        logger.debug(f"ffprobe failed for {filepath}: {e}")
        return None


def get_audio_info(filepath: str) -> dict:
    """Get audio file metadata without loading full audio data.

    Uses ffprobe for fast header-only extraction. Falls back to
    full pydub decode if ffprobe fails.

    Args:
        filepath: Path to audio file.

    Returns:
        Dictionary with keys: duration, sample_rate, channels, samples,
        format, codec, bit_depth, subtype.

    Raises:
        NotFoundError: If file doesn't exist.
        AudioLoadError: If metadata cannot be extracted.
    """
    path = Path(filepath)
    if not path.exists():
        raise NotFoundError(f"Audio file not found: {filepath}")

    # Try fast ffprobe extraction first
    metadata = _get_metadata_ffprobe(str(path))
    if metadata is not None:
        logger.debug(f"Extracted metadata via ffprobe: {filepath}")
        return metadata

    # Fallback: load full file with pydub
    logger.debug(f"ffprobe failed, using pydub full decode: {filepath}")

    try:
        segment = AudioSegment.from_file(str(path))

        duration_sec = len(segment) / 1000.0
        sample_rate = segment.frame_rate
        channels = segment.channels
        samples = int(duration_sec * sample_rate)
        bit_depth = segment.sample_width * 8 if segment.sample_width else None

        subtype_map = {1: "PCM_U8", 2: "PCM_16", 3: "PCM_24", 4: "PCM_32"}
        subtype = subtype_map.get(segment.sample_width)

        return {
            "duration": duration_sec,
            "sample_rate": sample_rate,
            "channels": channels,
            "samples": samples,
            "format": path.suffix.upper().lstrip("."),
            "codec": None,
            "bit_depth": bit_depth,
            "subtype": subtype,
        }
    except Exception as e:
        raise AudioLoadError(f"Failed to get audio info from '{filepath}': {e}") from e
