"""
Pydub-based audio file I/O utilities.

This module provides drop-in replacements for soundfile operations using pydub,
enabling support for M4A and other audio formats via ffmpeg.

IMPORTANT DESIGN NOTES:

1. MONO-ONLY OUTPUT:
   All functions that load audio (load_audio_pydub, audiosegment_to_numpy)
   automatically downmix stereo/multi-channel audio to mono by averaging channels.
   This is by design - the entire bioamla codebase expects mono audio for:
   - Machine learning models (all use in_channels=1)
   - Acoustic analysis (RMS, peak detection, frequency analysis)
   - Detection algorithms (energy, ribbit, peaks, accelerating)
   - Acoustic indices (ACI, ADI, AEI, BIO, NDSI)

2. BIT DEPTH HANDLING:
   The audio loading functions correctly handle 8-bit, 16-bit, 24-bit, and 32-bit
   audio by detecting the bit depth from AudioSegment.sample_width and applying
   the appropriate scaling factor to convert to float32 in range [-1.0, 1.0].

3. METADATA EXTRACTION:
   get_audio_info_pydub() uses ffprobe for fast header-only metadata extraction,
   falling back to full pydub decode only if ffprobe fails. This provides 10-100x
   speedup compared to always decoding the entire file.

4. SAVING AUDIO:
   save_audio_pydub() always saves as 16-bit audio regardless of input bit depth.
   This is acceptable for this application since:
   - 16-bit provides sufficient dynamic range for bioacoustic analysis
   - Most ML models downsample to 16kHz mono anyway
   - File size is reduced compared to 24-bit or 32-bit
"""

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from pydub import AudioSegment

from bioamla.core.logger import get_logger

logger = get_logger(__name__)


def audiosegment_to_numpy(segment: AudioSegment) -> Tuple[np.ndarray, int]:
    """
    Convert pydub AudioSegment to numpy float32 array.

    IMPORTANT: This function ALWAYS returns MONO audio. Stereo/multi-channel files are
    automatically downmixed to mono by averaging channels. This is by design - the entire
    bioamla codebase expects mono audio for all analysis, ML models, and detection algorithms.

    Args:
        segment: AudioSegment to convert

    Returns:
        Tuple of (audio_array, sample_rate) where audio_array is float32 [-1.0, 1.0]
        and ALWAYS 1D mono.

    Raises:
        ValueError: If sample width is not 1, 2, 3, or 4 bytes
    """
    # Get raw samples - pydub stores as array of integers
    array = np.array(segment.get_array_of_samples())

    # Determine bit depth from sample_width (bytes)
    sample_width = segment.sample_width  # 1=8bit, 2=16bit, 3=24bit, 4=32bit

    # Map sample width to numpy dtype and scaling factor
    if sample_width == 1:
        # 8-bit audio: unsigned int, range [0, 255], center at 128
        array = array.astype(np.uint8)
        # Convert to signed, then to float32 in range [-1.0, 1.0]
        audio = (array.astype(np.float32) - 128.0) / 128.0
    elif sample_width == 2:
        # 16-bit audio: signed int, range [-32768, 32767]
        array = array.astype(np.int16)
        audio = array.astype(np.float32) / 32768.0
    elif sample_width == 3:
        # 24-bit audio: signed int, range [-8388608, 8388607]
        array = array.astype(np.int32)
        audio = array.astype(np.float32) / 8388608.0
    elif sample_width == 4:
        # 32-bit audio: could be int32 or float32
        # Check if values are in float range [-1.0, 1.0]
        if array.dtype == np.float32 or (len(array) > 0 and np.max(np.abs(array)) <= 1.0):
            # Already float32 in correct range
            audio = array.astype(np.float32)
        else:
            # 32-bit signed int, range [-2147483648, 2147483647]
            array = array.astype(np.int32)
            audio = array.astype(np.float32) / 2147483648.0
    else:
        raise ValueError(
            f"Unsupported sample width: {sample_width} bytes. "
            f"Expected 1 (8-bit), 2 (16-bit), 3 (24-bit), or 4 (32-bit)."
        )

    # Handle multi-channel audio - ALWAYS downmix to mono
    if segment.channels > 1:
        audio = audio.reshape((-1, segment.channels))
        # Average all channels to create mono
        audio = audio.mean(axis=1)
        logger.debug(
            f"Downmixed {segment.channels}-channel audio to mono "
            f"({sample_width * 8}-bit, {segment.frame_rate}Hz)"
        )
    else:
        logger.debug(f"Loaded mono audio ({sample_width * 8}-bit, {segment.frame_rate}Hz)")

    return audio, segment.frame_rate


def numpy_to_audiosegment(
    audio: np.ndarray, sample_rate: int, channels: Optional[int] = None
) -> AudioSegment:
    """
    Convert numpy float32 array to pydub AudioSegment.

    Args:
        audio: Audio data as numpy array (float32, range [-1.0, 1.0])
        sample_rate: Sample rate in Hz
        channels: Number of channels (auto-detected if None)

    Returns:
        AudioSegment object
    """
    # Infer channels from shape if not provided
    if channels is None:
        channels = 1 if audio.ndim == 1 else audio.shape[1]

    # Ensure proper shape
    if audio.ndim == 2 and channels == 1:
        audio = audio.flatten()

    # Convert float32 [-1.0, 1.0] to int16
    # Clip to prevent overflow
    audio_clipped = np.clip(audio, -1.0, 1.0)
    audio_int16 = (audio_clipped * 32767.0).astype(np.int16)

    # Create AudioSegment
    segment = AudioSegment(
        audio_int16.tobytes(),
        frame_rate=sample_rate,
        sample_width=2,  # 16-bit = 2 bytes
        channels=channels,
    )
    return segment


def load_audio_pydub(filepath: str) -> Tuple[np.ndarray, int]:
    """
    Load audio file and return as numpy float32 array.

    Drop-in replacement for soundfile.read(filepath, dtype="float32").

    Args:
        filepath: Path to audio file (supports WAV, MP3, FLAC, OGG, M4A, etc.)

    Returns:
        Tuple of (audio_array, sample_rate)

    Raises:
        FileNotFoundError: If file doesn't exist
        Exception: If file cannot be loaded
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {filepath}")

    try:
        segment = AudioSegment.from_file(str(path))
        return audiosegment_to_numpy(segment)
    except Exception as e:
        raise Exception(f"Error opening '{filepath}': {e}")


def load_audio(filepath: str) -> Tuple[np.ndarray, int]:
    """
    Load audio file using the fastest available backend.

    This function intelligently routes to the optimal audio loader:
    - soundfile (native C via libsndfile) for WAV/FLAC/OGG: ~10-30x faster
    - pydub (ffmpeg subprocess) for M4A/MP3: slower but supports more formats

    Performance comparison for 4-minute WAV file:
    - soundfile: ~8ms (native C library)
    - pydub/ffmpeg: ~240ms (subprocess overhead)

    Args:
        filepath: Path to audio file

    Returns:
        Tuple of (audio_array, sample_rate) where audio is mono float32 in [-1.0, 1.0]

    Raises:
        FileNotFoundError: If file doesn't exist
        Exception: If file cannot be loaded
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {filepath}")

    ext = path.suffix.lower()

    # Fast path: Use soundfile for formats it supports natively
    if ext in {'.wav', '.flac', '.ogg'}:
        try:
            import soundfile as sf

            # Read audio file as float32
            audio, sample_rate = sf.read(str(path), dtype='float32', always_2d=False)

            # Convert to mono if stereo/multichannel
            if audio.ndim > 1:
                audio = audio.mean(axis=1)

            logger.debug(
                f"Loaded {ext} file via soundfile (fast): {path.name} "
                f"({sample_rate}Hz, {len(audio)} samples)"
            )
            return audio, sample_rate

        except Exception as e:
            # Fall back to pydub if soundfile fails
            logger.debug(f"soundfile failed for {filepath}, using pydub fallback: {e}")

    # Slow path: Use pydub for M4A, MP3, or if soundfile failed
    logger.debug(f"Loading {ext} file via pydub (slow): {path.name}")
    return load_audio_pydub(str(path))


def save_audio_pydub(
    filepath: str,
    audio: np.ndarray,
    sample_rate: int,
    format: Optional[str] = None,
) -> None:
    """
    Save numpy audio array to file.

    Drop-in replacement for soundfile.write(filepath, audio, sample_rate).

    Args:
        filepath: Destination file path
        audio: Audio data as numpy array (float32)
        sample_rate: Sample rate in Hz
        format: Output format (auto-detected from extension if None)

    Raises:
        Exception: If file cannot be saved
    """
    path = Path(filepath)

    # Ensure parent directory exists
    path.parent.mkdir(parents=True, exist_ok=True)

    # Determine format from extension if not specified
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
        # Infer channels
        channels = 1 if audio.ndim == 1 else audio.shape[1]

        # Convert to AudioSegment
        segment = numpy_to_audiosegment(audio, sample_rate, channels)

        # Export to file
        segment.export(str(path), format=format_to_use)
    except Exception as e:
        raise Exception(f"Failed to save audio to '{filepath}': {e}")


def get_audio_metadata_ffprobe(filepath: str) -> Optional[dict]:
    """
    Extract audio metadata using ffprobe (header-only, no decoding).

    This is much faster than loading the entire audio file, as it only
    reads file headers using ffprobe.

    Args:
        filepath: Path to audio file

    Returns:
        Dictionary with metadata, or None if ffprobe fails
        Keys: duration, sample_rate, channels, samples, codec, bit_depth, subtype, format

    Raises:
        None - returns None on any error (caller should fallback to pydub)
    """
    import json
    import subprocess

    try:
        # Use ffprobe to get audio metadata without decoding
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "quiet",  # Suppress ffmpeg warnings
                "-print_format",
                "json",  # Output as JSON
                "-show_format",  # Show container format
                "-show_streams",  # Show stream info
                str(filepath),
            ],
            capture_output=True,
            text=True,
            timeout=5.0,  # Timeout after 5 seconds
        )

        if result.returncode != 0:
            return None

        data = json.loads(result.stdout)

        # Find the first audio stream
        audio_stream = None
        for stream in data.get("streams", []):
            if stream.get("codec_type") == "audio":
                audio_stream = stream
                break

        if not audio_stream:
            return None

        # Extract metadata
        sample_rate = int(audio_stream.get("sample_rate", 0))
        channels = int(audio_stream.get("channels", 0))
        duration = float(
            audio_stream.get("duration") or data.get("format", {}).get("duration", 0)
        )

        # Calculate total samples
        samples = int(duration * sample_rate) if duration and sample_rate else 0

        # Get codec and bit depth
        codec = audio_stream.get("codec_name", "unknown")

        # Bit depth from bits_per_sample or bits_per_raw_sample
        bit_depth = audio_stream.get("bits_per_sample") or audio_stream.get(
            "bits_per_raw_sample"
        )
        if bit_depth:
            bit_depth = int(bit_depth)

        # Construct subtype string (similar to soundfile format)
        # e.g., "PCM_16", "PCM_24", "FLAC", "MP3"
        if codec in ["pcm_s16le", "pcm_s16be"]:
            subtype = "PCM_16"
        elif codec in ["pcm_s24le", "pcm_s24be"]:
            subtype = "PCM_24"
        elif codec in ["pcm_s32le", "pcm_s32be"]:
            subtype = "PCM_32"
        elif codec in ["pcm_u8"]:
            subtype = "PCM_U8"
        elif codec == "flac":
            subtype = "FLAC"
        elif codec in ["mp3", "mp3float"]:
            subtype = "MP3"
        elif codec == "aac":
            subtype = "AAC"
        elif codec == "vorbis":
            subtype = "VORBIS"
        else:
            subtype = codec.upper()

        # Get format from file extension
        format_name = Path(filepath).suffix.upper().lstrip(".")

        return {
            "duration": duration,
            "sample_rate": sample_rate,
            "channels": channels,
            "samples": samples,
            "format": format_name,
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
        # Any error - return None to trigger fallback
        logger.debug(f"ffprobe failed for {filepath}: {e}")
        return None


def get_audio_info_pydub(filepath: str) -> dict:
    """
    Get audio file metadata without loading full audio data (when possible).

    This function uses ffprobe for fast header-only metadata extraction.
    If ffprobe fails, it falls back to loading the full file with pydub.

    Drop-in replacement for soundfile.info() / soundfile.SoundFile().

    Args:
        filepath: Path to audio file

    Returns:
        Dictionary with keys:
        - duration: Duration in seconds (float)
        - sample_rate: Sample rate in Hz (int)
        - channels: Number of channels (int)
        - samples: Total number of samples (int)
        - format: File format extension (str, e.g., 'WAV', 'MP3')
        - codec: Audio codec (str, e.g., 'pcm_s16le', 'mp3') - may be None
        - bit_depth: Bit depth if available (int or None)
        - subtype: Subtype string if available (str or None, e.g., 'PCM_16', 'FLAC')

    Raises:
        FileNotFoundError: If file doesn't exist
        Exception: If metadata cannot be extracted by either method

    Note:
        - Attempts fast ffprobe extraction first (header-only, no decode)
        - Falls back to full pydub decode if ffprobe fails
        - Much faster than full decode for metadata-only queries (10-100x speedup)
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {filepath}")

    # Try fast ffprobe extraction first
    metadata = get_audio_metadata_ffprobe(str(path))

    if metadata is not None:
        logger.debug(f"Extracted metadata via ffprobe: {filepath}")
        return metadata

    # Fallback: load full file with pydub (slow but reliable)
    logger.debug(f"ffprobe failed, using pydub full decode: {filepath}")

    try:
        segment = AudioSegment.from_file(str(path))

        duration_sec = len(segment) / 1000.0  # pydub uses milliseconds
        sample_rate = segment.frame_rate
        channels = segment.channels
        samples = int(duration_sec * sample_rate)

        # Infer bit depth from sample_width
        # Note: This is the bit depth pydub uses internally after loading,
        # which may differ from the original file's bit depth
        bit_depth = segment.sample_width * 8 if segment.sample_width else None

        # Map sample_width to subtype
        if segment.sample_width == 1:
            subtype = "PCM_U8"  # 8-bit unsigned
        elif segment.sample_width == 2:
            subtype = "PCM_16"
        elif segment.sample_width == 3:
            subtype = "PCM_24"
        elif segment.sample_width == 4:
            subtype = "PCM_32"
        else:
            subtype = None

        return {
            "duration": duration_sec,
            "sample_rate": sample_rate,
            "channels": channels,
            "samples": samples,
            "format": path.suffix.upper().lstrip("."),
            "codec": None,  # Not available from pydub
            "bit_depth": bit_depth,
            "subtype": subtype,
        }
    except Exception as e:
        raise Exception(f"Failed to get audio info from '{filepath}': {e}")
