"""PydubAudioAdapter - unified audio I/O using pydub/ffmpeg.

This module consolidates audio loading, saving, and metadata extraction
into a single adapter with simple function interfaces.

Design notes:
- All audio loading returns mono float32 in [-1.0, 1.0]
- Uses soundfile for WAV/FLAC/OGG (fast native C) with pydub fallback
- Uses ffprobe for fast metadata extraction without full file decode
- Saving always outputs 16-bit audio
"""

import json
import logging
import subprocess
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from pydub import AudioSegment

logger = logging.getLogger(__name__)


class PydubAudioAdapter:
    """Adapter for audio I/O operations using pydub/ffmpeg.

    Provides methods to load, save, and get metadata from audio files.
    All methods return numpy arrays (mono float32) for compatibility
    with the bioamla services layer.

    Example:
        >>> adapter = PydubAudioAdapter()
        >>> audio, sr = adapter.load("audio.m4a")
        >>> adapter.save("output.wav", audio, sr)
        >>> info = adapter.get_info("audio.m4a")
    """

    def load(self, filepath: str) -> Tuple[np.ndarray, int]:
        """Load audio file using the fastest available backend.

        Routes to optimal loader:
        - soundfile (native C) for WAV/FLAC/OGG: ~10-30x faster
        - pydub (ffmpeg subprocess) for M4A/MP3: slower but wider format support

        Args:
            filepath: Path to audio file.

        Returns:
            Tuple of (audio_array, sample_rate) where audio is mono float32 [-1.0, 1.0].

        Raises:
            FileNotFoundError: If file doesn't exist.
            Exception: If file cannot be loaded.
        """
        return load_audio(filepath)

    def save(
        self,
        filepath: str,
        audio: np.ndarray,
        sample_rate: int,
        format: Optional[str] = None,
    ) -> None:
        """Save numpy audio array to file.

        Args:
            filepath: Destination file path.
            audio: Audio data as numpy array (float32).
            sample_rate: Sample rate in Hz.
            format: Output format (auto-detected from extension if None).

        Raises:
            Exception: If file cannot be saved.
        """
        save_audio(filepath, audio, sample_rate, format)

    def get_info(self, filepath: str) -> dict:
        """Get audio file metadata without loading full audio data.

        Uses ffprobe for fast header-only extraction, falls back to
        pydub full decode if ffprobe fails.

        Args:
            filepath: Path to audio file.

        Returns:
            Dictionary with keys: duration, sample_rate, channels, samples,
            format, codec, bit_depth, subtype.

        Raises:
            FileNotFoundError: If file doesn't exist.
            Exception: If metadata cannot be extracted.
        """
        return get_audio_info(filepath)


def _audiosegment_to_numpy(segment: AudioSegment) -> Tuple[np.ndarray, int]:
    """Convert pydub AudioSegment to numpy float32 array.

    Always returns mono audio - stereo/multi-channel files are
    automatically downmixed by averaging channels.

    Args:
        segment: AudioSegment to convert.

    Returns:
        Tuple of (audio_array, sample_rate) where audio is mono float32 [-1.0, 1.0].

    Raises:
        ValueError: If sample width is not 1, 2, 3, or 4 bytes.
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
        # 32-bit: could be int32 or float32
        if array.dtype == np.float32 or (len(array) > 0 and np.max(np.abs(array)) <= 1.0):
            audio = array.astype(np.float32)
        else:
            array = array.astype(np.int32)
            audio = array.astype(np.float32) / 2147483648.0
    else:
        raise ValueError(
            f"Unsupported sample width: {sample_width} bytes. "
            f"Expected 1 (8-bit), 2 (16-bit), 3 (24-bit), or 4 (32-bit)."
        )

    # Downmix to mono if multi-channel
    if segment.channels > 1:
        audio = audio.reshape((-1, segment.channels))
        audio = audio.mean(axis=1)
        logger.debug(
            f"Downmixed {segment.channels}-channel audio to mono "
            f"({sample_width * 8}-bit, {segment.frame_rate}Hz)"
        )
    else:
        logger.debug(f"Loaded mono audio ({sample_width * 8}-bit, {segment.frame_rate}Hz)")

    return audio, segment.frame_rate


def _numpy_to_audiosegment(
    audio: np.ndarray, sample_rate: int, channels: Optional[int] = None
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


def _load_via_pydub(filepath: str) -> Tuple[np.ndarray, int]:
    """Load audio file via pydub/ffmpeg.

    Args:
        filepath: Path to audio file.

    Returns:
        Tuple of (audio_array, sample_rate).

    Raises:
        FileNotFoundError: If file doesn't exist.
        Exception: If file cannot be loaded.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {filepath}")

    try:
        segment = AudioSegment.from_file(str(path))
        return _audiosegment_to_numpy(segment)
    except Exception as e:
        raise Exception(f"Error opening '{filepath}': {e}")


def load_audio(filepath: str) -> Tuple[np.ndarray, int]:
    """Load audio file using the fastest available backend.

    Routes to optimal loader:
    - soundfile (native C via libsndfile) for WAV/FLAC/OGG: ~10-30x faster
    - pydub (ffmpeg subprocess) for M4A/MP3: slower but wider format support

    Args:
        filepath: Path to audio file.

    Returns:
        Tuple of (audio_array, sample_rate) where audio is mono float32 [-1.0, 1.0].

    Raises:
        FileNotFoundError: If file doesn't exist.
        Exception: If file cannot be loaded.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {filepath}")

    ext = path.suffix.lower()

    # Fast path: soundfile for native formats
    if ext in {".wav", ".flac", ".ogg"}:
        try:
            import soundfile as sf

            audio, sample_rate = sf.read(str(path), dtype="float32", always_2d=False)

            # Convert to mono if stereo/multichannel
            if audio.ndim > 1:
                audio = audio.mean(axis=1)

            logger.debug(
                f"Loaded {ext} file via soundfile (fast): {path.name} "
                f"({sample_rate}Hz, {len(audio)} samples)"
            )
            return audio, sample_rate

        except Exception as e:
            logger.debug(f"soundfile failed for {filepath}, using pydub fallback: {e}")

    # Slow path: pydub for M4A, MP3, or if soundfile failed
    logger.debug(f"Loading {ext} file via pydub (slow): {path.name}")
    return _load_via_pydub(str(path))


def save_audio(
    filepath: str,
    audio: np.ndarray,
    sample_rate: int,
    format: Optional[str] = None,
) -> None:
    """Save numpy audio array to file.

    Args:
        filepath: Destination file path.
        audio: Audio data as numpy array (float32).
        sample_rate: Sample rate in Hz.
        format: Output format (auto-detected from extension if None).

    Raises:
        Exception: If file cannot be saved.
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)

    format_to_use = format
    if format_to_use is None:
        ext = path.suffix.lower()
        format_map = {
            ".wav": "wav",
            ".flac": "flac",
            ".ogg": "ogg",
            ".mp3": "mp3",
            ".m4a": "ipod",  # pydub uses "ipod" for M4A/AAC
        }
        format_to_use = format_map.get(ext, "wav")

    try:
        channels = 1 if audio.ndim == 1 else audio.shape[1]
        segment = _numpy_to_audiosegment(audio, sample_rate, channels)
        segment.export(str(path), format=format_to_use)
    except Exception as e:
        raise Exception(f"Failed to save audio to '{filepath}': {e}")


def _get_metadata_ffprobe(filepath: str) -> Optional[dict]:
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
                "-v", "quiet",
                "-print_format", "json",
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
        duration = float(
            audio_stream.get("duration") or data.get("format", {}).get("duration", 0)
        )
        samples = int(duration * sample_rate) if duration and sample_rate else 0
        codec = audio_stream.get("codec_name", "unknown")

        bit_depth = audio_stream.get("bits_per_sample") or audio_stream.get(
            "bits_per_raw_sample"
        )
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
        FileNotFoundError: If file doesn't exist.
        Exception: If metadata cannot be extracted.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {filepath}")

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
        raise Exception(f"Failed to get audio info from '{filepath}': {e}")
