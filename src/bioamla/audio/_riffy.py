"""
riffy backend for WAV headers
=============================

Header-only metadata for ``.wav`` files via `riffy <https://github.com/jmcmeen/riffy>`_,
a pure-Python RIFF parser. It needs no FFmpeg, libsndfile, or torch, so it is the first
thing :func:`bioamla.audio.get_audio_info` tries for WAV input. riffy only understands
PCM; anything else raises :class:`AudioLoadError` and callers fall back to the pydub /
ffprobe path.

This is the only module that imports riffy. Its exceptions are mapped to bioamla's
hierarchy here and nowhere else.
"""

from __future__ import annotations

import logging
from pathlib import Path

from bioamla.exceptions import AudioLoadError, InvalidInputError, NotFoundError

logger = logging.getLogger(__name__)

WAV_SUFFIXES = frozenset({".wav", ".wave"})


def is_wav_path(filepath: str | Path) -> bool:
    """Return ``True`` when ``filepath`` has a WAV suffix (case-insensitive)."""
    return Path(filepath).suffix.lower() in WAV_SUFFIXES


def get_wav_info(filepath: str | Path) -> dict:
    """Read a WAV header with riffy.

    Returns the same dictionary shape as :func:`bioamla.audio._pydub.get_audio_info`
    (``duration``, ``sample_rate``, ``channels``, ``samples``, ``format``, ``codec``,
    ``bit_depth``, ``subtype``) so the two backends are interchangeable.

    Raises:
        NotFoundError: If the file does not exist.
        InvalidInputError: If the path does not have a WAV suffix.
        AudioLoadError: If riffy cannot parse the file, including non-PCM WAVs.
    """
    from riffy import WAVParser
    from riffy.exceptions import RiffyError

    path = Path(filepath)
    if not path.exists():
        raise NotFoundError(f"Audio file not found: {filepath}")
    if not is_wav_path(path):
        raise InvalidInputError(f"Not a WAV file: {filepath}")

    try:
        parser = WAVParser(path)
        info = parser.get_info()
    except RiffyError as e:
        raise AudioLoadError(f"Cannot read WAV header: {filepath}: {e}") from e

    fmt = info["format"]
    bits = int(fmt["bits_per_sample"])
    is_pcm = bool(fmt["is_pcm"])
    if is_pcm:
        subtype = "PCM_U8" if bits == 8 else f"PCM_{bits}"
    else:
        subtype = None

    logger.debug(f"Extracted metadata via riffy: {filepath}")
    return {
        "duration": float(info["duration_seconds"]),
        "sample_rate": int(fmt["sample_rate"]),
        "channels": int(fmt["channels"]),
        "samples": int(info["sample_count"]),
        "format": "WAV",
        "codec": "pcm" if is_pcm else None,
        "bit_depth": bits,
        "subtype": subtype,
    }


def _check_wav(filepath: str | Path) -> Path:
    path = Path(filepath)
    if not path.exists():
        raise NotFoundError(f"Audio file not found: {filepath}")
    if not is_wav_path(path):
        raise InvalidInputError(f"Not a WAV file: {filepath}")
    return path


def read_recorder_metadata(filepath: str | Path):  # -> riffy.RecordingMetadata
    """Parse the recorder-metadata standards embedded in a WAV file.

    Returns riffy's :class:`~riffy.RecordingMetadata`, whose ``guano``, ``info``,
    ``bext``, ``audiomoth`` and ``wamd`` attributes are each a typed view or ``None``.

    Raises:
        NotFoundError: If the file does not exist.
        InvalidInputError: If the path does not have a WAV suffix.
        AudioLoadError: If riffy cannot parse the file.
    """
    from riffy import read_metadata
    from riffy.exceptions import RiffyError

    path = _check_wav(filepath)
    try:
        return read_metadata(path)
    except RiffyError as e:
        raise AudioLoadError(f"Cannot read recorder metadata: {filepath}: {e}") from e


def dump_recorder_metadata(filepath: str | Path) -> dict:
    """Recorder metadata as a JSON-serializable dict, one entry per standard.

    Thin wrapper over :func:`riffy.dump_metadata` with bioamla's error mapping.
    """
    from riffy import dump_metadata
    from riffy.exceptions import RiffyError

    path = _check_wav(filepath)
    try:
        return dump_metadata(path)
    except RiffyError as e:
        raise AudioLoadError(f"Cannot read recorder metadata: {filepath}: {e}") from e


__all__ = [
    "WAV_SUFFIXES",
    "dump_recorder_metadata",
    "get_wav_info",
    "is_wav_path",
    "read_recorder_metadata",
]
