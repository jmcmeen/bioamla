"""
riffy backend for WAV headers
=============================

Header-only metadata for ``.wav`` files via `riffy <https://github.com/jmcmeen/riffy>`_,
a pure-Python RIFF parser. It needs no FFmpeg, libsndfile, or torch, so it is the first
thing :func:`bioamla.audio.get_audio_info` tries for WAV input. Files are parsed without
reading their audio (``read_audio=False``), so a multi-GB recording costs only its
headers. :func:`get_wav_info` handles integer PCM and IEEE float (plain or extensible);
compressed WAVs raise :class:`AudioLoadError` and callers fall back to the pydub /
ffprobe path, which can measure them.

This is the only module that imports riffy. Its exceptions are mapped to bioamla's
hierarchy here and nowhere else.
"""

from __future__ import annotations

import contextlib
import logging
import warnings
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from bioamla.exceptions import AudioLoadError, InvalidInputError, NotFoundError

logger = logging.getLogger(__name__)

WAV_SUFFIXES = frozenset({".wav", ".wave"})


@contextlib.contextmanager
def _riffy_warnings_to_log(filepath: str | Path) -> Iterator[None]:
    """Turn riffy's parse warnings (odd encodings, padded chunks) into debug logs.

    Field recorders write slightly off-spec chunks all the time; the parse still
    succeeds, and a warning on stderr for every file would drown batch output.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        yield
    for w in caught:
        logger.debug(f"riffy: {filepath}: {w.message}")


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
        AudioLoadError: If riffy cannot parse the file, or its audio is compressed.
    """
    from riffy import WAVParser
    from riffy.exceptions import RiffyError

    path = Path(filepath)
    if not path.exists():
        raise NotFoundError(f"Audio file not found: {filepath}")
    if not is_wav_path(path):
        raise InvalidInputError(f"Not a WAV file: {filepath}")

    try:
        with _riffy_warnings_to_log(path):
            parser = WAVParser(path, read_audio=False)
            info = parser.get_info()
    except RiffyError as e:
        raise AudioLoadError(f"Cannot read WAV header: {filepath}: {e}") from e

    fmt = info["format"]
    bits = int(fmt["bits_per_sample"])
    is_pcm = bool(fmt["is_pcm"])
    # Names match the ffprobe backend so the two stay interchangeable.
    if is_pcm:
        subtype = "PCM_U8" if bits == 8 else f"PCM_{bits}"
        codec = "pcm_u8" if bits == 8 else f"pcm_s{bits}le"
    elif fmt["is_float"]:
        subtype = "FLOAT" if bits == 32 else "DOUBLE"
        codec = f"pcm_f{bits}le"
    else:
        # riffy can't measure compressed audio; ffprobe can.
        raise AudioLoadError(f"Compressed WAV ({fmt['description']}): {filepath}")

    logger.debug(f"Extracted metadata via riffy: {filepath}")
    return {
        "duration": float(info["duration_seconds"]),
        "sample_rate": int(fmt["sample_rate"]),
        "channels": int(fmt["channels"]),
        "samples": int(info["sample_count"]),
        "format": "WAV",
        "codec": codec,
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
        with _riffy_warnings_to_log(path):
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
        with _riffy_warnings_to_log(path):
            return dump_metadata(path)
    except RiffyError as e:
        raise AudioLoadError(f"Cannot read recorder metadata: {filepath}: {e}") from e


# ---------------------------------------------------------------------------
# Location redaction (see datasets.redact)
# ---------------------------------------------------------------------------

# Words marking a GUANO field as positional. A field counts when one of the
# strong words appears ("Loc Position", "GPS Fix", "Site Coords"), or when the
# key is made only of weak words ("Lat", "Lon", "Elevation"). So "Long
# Description", "Mic Position" and "Clock Drift" are kept.
_STRONG_LOCATION_WORDS = frozenset(
    {"loc", "location", "gps", "latitude", "longitude", "coords", "coordinates", "geo"}
)
_WEAK_LOCATION_WORDS = frozenset(
    {"lat", "lon", "lng", "long", "position", "elevation", "altitude", "accuracy"}
)


def is_location_field(key: str) -> bool:
    """Whether a GUANO field name describes the recording location."""
    words = set("".join(c if c.isalnum() else " " for c in key.lower()).split())
    if words & _STRONG_LOCATION_WORDS:
        return True
    return bool(words) and words <= _WEAK_LOCATION_WORDS


def _redact_guano(guano, mode: str, decimals: int) -> bool:
    changed = False
    for namespace, key in list(guano.fields):
        if not is_location_field(key):
            continue
        position = guano.loc_position if (namespace, key) == ("", "Loc Position") else None
        if mode == "round" and position is not None:
            lat, lon = position
            guano.loc_position = (round(lat, decimals), round(lon, decimals))
        else:
            guano.remove(namespace, key)
        changed = True
    return changed


def _redact_wamd(wamd, mode: str, decimals: int) -> bool:
    from riffy.metadata.wamd import GPSFIRST_ID, GPSTRACK_ID

    changed = False
    if wamd.get_raw(GPSTRACK_ID) is not None:
        wamd.remove(GPSTRACK_ID)  # a track is many fixes; nothing to round it to
        changed = True
    if wamd.get_raw(GPSFIRST_ID) is not None:
        position = wamd.loc_position
        if mode == "round" and position is not None:
            lat, lon = position
            # The setter keeps the waypoint's datum (and altitude); drop the altitude.
            wamd.loc_position = (round(lat, decimals), round(lon, decimals))
            datum, r_lat, r_lon = (wamd.get_text(GPSFIRST_ID) or "").split(",")[:3]
            wamd.set_text(GPSFIRST_ID, f"{datum},{r_lat},{r_lon}")
        else:
            wamd.remove(GPSFIRST_ID)
        changed = True
    return changed


def _location_chunks(parser) -> Iterator[tuple[str, int, Any]]:
    """Yield ``(chunk_id, index, metadata)`` for every GUANO/WAMD-bearing chunk."""
    from riffy import GuanoMetadata, WamdMetadata
    from riffy.metadata.wamd import _looks_like_wamd

    for chunk_id in ("guan", "wamd", "junk", "JUNK"):
        for i, chunk in enumerate(parser.chunks.get(chunk_id, [])):
            payload = parser.read_chunk_data(chunk)
            if chunk_id == "guan":
                yield chunk_id, i, GuanoMetadata.from_bytes(payload)
            elif chunk_id == "wamd" or _looks_like_wamd(payload):
                yield chunk_id, i, WamdMetadata.from_bytes(payload, chunk_id=chunk_id)


def _has_location(chunk_id: str, meta: Any) -> bool:
    from riffy.metadata.wamd import GPSFIRST_ID, GPSTRACK_ID

    if chunk_id == "guan":
        return any(is_location_field(key) for _, key in meta.fields)
    return meta.get_raw(GPSFIRST_ID) is not None or meta.get_raw(GPSTRACK_ID) is not None


def _redact_parser(parser, mode: str, decimals: int) -> bool:
    """Redact the location in every GUANO, WAMD and WAMD-bearing junk chunk of ``parser``.

    Edits the parser's chunks in memory. Every occurrence is handled, not just
    the first one riffy's ``from_parser`` helpers read, so a duplicate chunk
    can't carry an unredacted position through.
    """
    from riffy import WAVChunk

    changed = False
    for chunk_id, i, meta in list(_location_chunks(parser)):
        if chunk_id == "guan":
            edited = _redact_guano(meta, mode, decimals)
            if edited and meta.version is None:
                meta.version = "1.0"  # the spec requires it; some writers leave it out
        else:
            edited = _redact_wamd(meta, mode, decimals)
        if edited:
            new = meta.to_chunk_bytes()
            old = parser.chunks[chunk_id][i]
            parser.chunks[chunk_id][i] = WAVChunk(
                id=chunk_id, size=len(new), data=new, offset=old.offset
            )
            changed = True
    return changed


def _metadata_errors() -> tuple[type[BaseException], ...]:
    import struct

    from riffy.exceptions import RiffyError

    # riffy's own errors, plus what its metadata codecs raise on malformed payloads.
    return (RiffyError, OSError, ValueError, IndexError, struct.error)


def _open_header_only(path: Path):
    from riffy import WAVParser

    return WAVParser(path, read_audio=False)


def wav_has_location(filepath: str | Path) -> bool:
    """Whether a WAV's GUANO or WAMD metadata carries a location (audio is not read)."""
    path = _check_wav(filepath)
    try:
        with _riffy_warnings_to_log(path):
            return any(
                _has_location(chunk_id, meta)
                for chunk_id, _, meta in _location_chunks(_open_header_only(path))
            )
    except _metadata_errors() as e:
        raise AudioLoadError(f"Cannot read recorder metadata: {filepath}: {e}") from e


def rewrite_wav_location(src: str | Path, dest: str | Path, mode: str, decimals: int = 2) -> bool:
    """Write ``src`` to ``dest`` with its GUANO/WAMD location rounded or stripped.

    The audio is streamed across by riffy, so memory use doesn't grow with file
    size, and any audio format works. RF64/BW64 sources stay RF64/BW64. riffy
    writes chunks in its own order (``fmt``, ``data``, then the rest by ID), so
    the copy's layout can differ from the source's; every chunk's content other
    than the redacted metadata is unchanged.

    Returns:
        True if ``src`` carried a location and ``dest`` was written.

    Raises:
        AudioLoadError: If ``src`` can't be parsed or the copy can't be written.
    """
    path = _check_wav(src)
    try:
        with _riffy_warnings_to_log(path):
            parser = _open_header_only(path)
            if not _redact_parser(parser, mode, decimals):
                return False
            out = Path(dest)
            out.parent.mkdir(parents=True, exist_ok=True)
            parser.write_wav(out, force_rf64=parser.is_rf64)
            return True
    except _metadata_errors() as e:
        raise AudioLoadError(f"Cannot rewrite recorder metadata: {src}: {e}") from e


__all__ = [
    "WAV_SUFFIXES",
    "is_location_field",
    "rewrite_wav_location",
    "wav_has_location",
    "dump_recorder_metadata",
    "get_wav_info",
    "is_wav_path",
    "read_recorder_metadata",
]
