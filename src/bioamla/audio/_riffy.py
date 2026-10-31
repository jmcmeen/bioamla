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

import contextlib
import logging
import warnings
from collections.abc import Iterator
from pathlib import Path

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
        with _riffy_warnings_to_log(path):
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
# Location redaction (streaming; see datasets.redact)
# ---------------------------------------------------------------------------
#
# riffy's WAVParser loads every chunk, audio included, into memory and only
# accepts PCM. To redact recorder positions in datasets of long or float
# recordings, these helpers walk the RIFF chunk list themselves, read only the
# metadata chunks, and stream everything else across. riffy still parses and
# re-serializes the GUANO and WAMD payloads. Once riffy can read and rewrite
# metadata without loading the audio, this should move there.

_MAX_U32 = 0xFFFFFFFF
_RF64_FORMS = (b"RF64", b"BW64")
_LOCATION_CHUNK_IDS = ("guan", "wamd", "junk", "JUNK")
_COPY_BLOCK = 1 << 20

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


class _Chunk:
    __slots__ = ("id", "offset", "size")

    def __init__(self, chunk_id: str, size: int, offset: int) -> None:
        self.id = chunk_id
        self.size = size  # real payload size (from ds64 for RF64 placeholders)
        self.offset = offset  # file offset of the payload


def _walk_chunks(f) -> tuple[bytes, list[_Chunk], bytes | None]:
    """Return ``(form, chunks, ds64_payload)`` without reading any payload but ds64."""
    import struct

    header = f.read(12)
    if len(header) < 12 or header[8:12] != b"WAVE" or header[:4] not in (b"RIFF", *_RF64_FORMS):
        raise AudioLoadError("not a RIFF/RF64 WAVE file")
    form = header[:4]
    chunks: list[_Chunk] = []
    ds64: bytes | None = None
    data_size64: int | None = None
    table: dict[str, list[int]] = {}
    while True:
        head = f.read(8)
        if len(head) < 8:
            break
        chunk_id = head[:4].decode("latin-1")
        (size,) = struct.unpack("<I", head[4:])
        offset = f.tell()
        if chunk_id == "ds64":
            ds64 = f.read(size)
            data_size64 = struct.unpack_from("<Q", ds64, 8)[0]
            count = struct.unpack_from("<I", ds64, 24)[0] if len(ds64) >= 28 else 0
            for i in range(count):
                entry = 28 + 12 * i
                cid = ds64[entry : entry + 4].decode("latin-1")
                table.setdefault(cid, []).append(struct.unpack_from("<Q", ds64, entry + 4)[0])
            f.seek(offset)
        if form in _RF64_FORMS and size == _MAX_U32:
            if chunk_id == "data" and data_size64 is not None:
                size = data_size64
            elif table.get(chunk_id):
                size = table[chunk_id].pop(0)
        chunks.append(_Chunk(chunk_id, size, offset))
        f.seek(offset + size + (size & 1))
    return form, chunks, ds64


def _redact_guano(payload: bytes, mode: str, decimals: int) -> bytes | None:
    from riffy import GuanoMetadata

    guano = GuanoMetadata.from_bytes(payload)
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
    return guano.to_chunk_bytes() if changed else None


def _redact_wamd(payload: bytes, chunk_id: str, mode: str, decimals: int) -> bytes | None:
    from riffy import WamdMetadata
    from riffy.metadata.wamd import GPSFIRST_ID, GPSTRACK_ID, _looks_like_wamd

    if chunk_id != "wamd" and not _looks_like_wamd(payload):
        return None  # ordinary junk padding
    wamd = WamdMetadata.from_bytes(payload, chunk_id=chunk_id)
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
    return wamd.to_chunk_bytes() if changed else None


def _redacted_payloads(f, chunks: list[_Chunk], mode: str, decimals: int) -> dict[int, bytes]:
    """Map chunk index -> replacement payload for chunks that carry a location."""
    out: dict[int, bytes] = {}
    for i, chunk in enumerate(chunks):
        if chunk.id not in _LOCATION_CHUNK_IDS:
            continue
        f.seek(chunk.offset)
        payload = f.read(chunk.size)
        if len(payload) < chunk.size:
            # A cut-off metadata chunk can't be parsed reliably, so it can't be
            # shown to be free of a location.
            raise AudioLoadError(f"{chunk.id!r} chunk is truncated")
        if chunk.id == "guan":
            new = _redact_guano(payload, mode, decimals)
        else:
            new = _redact_wamd(payload, chunk.id, mode, decimals)
        if new is not None:
            out[i] = new
    return out


def wav_has_location(filepath: str | Path) -> bool:
    """Whether a WAV's GUANO or WAMD metadata carries a location (audio is not read)."""
    from riffy.exceptions import RiffyError

    path = _check_wav(filepath)
    try:
        with _riffy_warnings_to_log(path), path.open("rb") as f:
            _, chunks, _ = _walk_chunks(f)
            return bool(_redacted_payloads(f, chunks, "strip", 0))
    except (RiffyError, OSError, ValueError, IndexError) as e:
        raise AudioLoadError(f"Cannot read recorder metadata: {filepath}: {e}") from e


def rewrite_wav_location(src: str | Path, dest: str | Path, mode: str, decimals: int = 2) -> bool:
    """Write ``src`` to ``dest`` with its GUANO/WAMD location rounded or stripped.

    Streams the audio across, so memory use doesn't grow with file size, and
    works for any audio format. Only the location-bearing metadata payloads
    change; every other chunk is copied byte for byte.

    Returns:
        True if ``src`` carried a location and ``dest`` was written.

    Raises:
        AudioLoadError: If ``src`` can't be parsed or the copy can't be written.
    """
    import struct

    from riffy.exceptions import RiffyError

    path = _check_wav(src)
    try:
        with _riffy_warnings_to_log(path), path.open("rb") as f:
            form, chunks, ds64 = _walk_chunks(f)
            replaced = _redacted_payloads(f, chunks, mode, decimals)
            if not replaced:
                return False

            def size_of(i: int) -> int:
                return len(replaced[i]) if i in replaced else chunks[i].size

            riff_size = 4 + sum(8 + size_of(i) + (size_of(i) & 1) for i in range(len(chunks)))
            if form not in _RF64_FORMS and riff_size > _MAX_U32:
                raise AudioLoadError("redacted file would exceed the 4 GB RIFF limit")

            out_path = Path(dest)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with out_path.open("wb") as out:
                header_size = _MAX_U32 if form in _RF64_FORMS else riff_size
                out.write(form + struct.pack("<I", header_size) + b"WAVE")
                for i, chunk in enumerate(chunks):
                    size = size_of(i)
                    if chunk.id == "ds64" and ds64 is not None:
                        payload = struct.pack("<Q", riff_size) + ds64[8:]
                        out.write(b"ds64" + struct.pack("<I", len(payload)) + payload)
                        continue
                    stored = _MAX_U32 if size > _MAX_U32 else size
                    if form in _RF64_FORMS and chunk.id == "data":
                        stored = _MAX_U32
                    out.write(chunk.id.encode("latin-1") + struct.pack("<I", stored))
                    if i in replaced:
                        out.write(replaced[i])
                    else:
                        f.seek(chunk.offset)
                        remaining = size
                        while remaining:
                            block = f.read(min(_COPY_BLOCK, remaining))
                            if not block:
                                raise AudioLoadError(f"{chunk.id!r} chunk is truncated")
                            out.write(block)
                            remaining -= len(block)
                    if size & 1:
                        out.write(b"\x00")
            return True
    except AudioLoadError:
        raise
    except (RiffyError, OSError, ValueError, IndexError, struct.error) as e:
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
