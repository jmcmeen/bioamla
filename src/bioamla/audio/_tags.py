"""Whether an audio file's metadata may hold a location a redacting push leaves in it.

A redacting dataset push rewrites the position in WAV recorder metadata (GUANO and
Wildlife Acoustics WAMD) but has no writer for anything else: other WAV chunks
(``LIST``/``INFO`` comments, ``bext``, ``iXML``, an ``id3`` chunk) and FLAC or MP3 tags.
It vouches for an audio file only when all of that can be read and none of it looks
like a location: a word such as ``latitude``, ``GPS``, ``location`` or ``elevation``,
or a value that looks like coordinates (a decimal pair, hemisphere letters, degrees,
UTM). Metadata it can't read (FLAC application blocks, which can carry a WAV's GUANO
chunk, embedded pictures, binary ID3 frames, compressed or unsynchronised tags, unknown
WAV chunks, APE or Lyrics3 tags) means it can't vouch. This is a check of what it can
read, not a proof: free text that names a place without these words gets through.

Only the metadata is read, with the standard library; the audio is skipped.
"""

from __future__ import annotations

import re
import struct
from pathlib import Path
from typing import BinaryIO

_LOCATION_WORDS = re.compile(
    r"latitude|longitude|\bgps|\bgeo(?:location|tag|tagged|position|coordinates|referenced?)?\b"
    r"|coordinate|\blocation|locality|loc position|\belevation\b|\baltitude\b|\butm\b"
    r"|\beasting\b|\bnorthing\b|\blat\b|\blon\b|\blng\b|\blat\s*/\s*long?\b",
    re.IGNORECASE,
)
_NUMBER = r"[-+]?\d{1,3}(?:\.\d+)?"
_COORDINATES = re.compile(
    "|".join(
        [
            # a decimal pair: "35.12, -82.57"
            r"[-+]?\d{1,3}\.\d{2,}\s*[,; ]\s*[-+]?\d{1,3}\.\d{2,}",
            # hemisphere letters: "N36.1234 W82.5678", "36.12N 82.57W"
            rf"\b[NS]\s?{_NUMBER}°?[\s,;]+[EW]\s?{_NUMBER}",
            rf"{_NUMBER}°?\s?[NS][\s,;]+{_NUMBER}°?\s?[EW]\b",
            # degrees: "36°07'24\"", "36.12° N" (not "15°C")
            r"\d{1,3}(?:\.\d+)?\s*°\s*(?:\d|[NSEW]\b)",
            # UTM: "17S 412345 3998765"
            r"\b\d{1,2}\s?[C-HJ-NP-X]\s+\d{6}\s*(?:m?E)?\s+\d{7}\b",
        ]
    ),
    re.IGNORECASE,
)

_FLAC_SAFE_BLOCKS = {0, 1, 3, 5}  # STREAMINFO, PADDING, SEEKTABLE, CUESHEET
_FLAC_VORBIS_COMMENT = 4
# ID3v2 frames holding text are the text frames (T***, including user text TXXX) and
# comments (COMM); the rest, such as APIC, GEOB or PRIV, are binary.
_ID3_TEXT_ENCODINGS = {0: "latin-1", 1: "utf-16", 2: "utf-16-be", 3: "utf-8"}
# Markers of tags at the end of an MP3 that aren't read: an appended ID3v2 tag's
# footer, APE, Lyrics3, and the extended ID3v1 block.
_MP3_TRAILING_TAGS = (b"3DI", b"APETAGEX", b"LYRICS", b"TAG+")

# WAV chunks: the push redacts GUANO and WAMD (WAMD may sit in a junk chunk); these
# others hold no text; these are read as text; anything else can't be vouched for.
_WAV_REDACTED = {b"guan", b"wamd", b"junk", b"JUNK"}
_WAV_NO_TEXT = {
    b"fmt ", b"data", b"fact", b"ds64", b"cue ", b"smpl", b"inst", b"PAD ", b"FLLR",
    b"PEAK", b"acid", b"umid", b"minf", b"elm1", b"regn", b"ovwf",
}  # fmt: skip
_WAV_TEXT = {b"LIST", b"bext", b"iXML", b"axml", b"cart", b"_PMX"}  # _PMX: XMP
_WAV_ID3 = {b"id3 ", b"ID3 "}
_MAX_TEXT_CHUNK = 1 << 20

TAGGED_SUFFIXES = (".flac", ".mp3")


def mentions_location(text: str) -> bool:
    """Whether text has a location word or something that looks like coordinates."""
    return bool(_LOCATION_WORDS.search(text) or _COORDINATES.search(text))


def looks_like_coordinates(text: str) -> bool:
    """Whether text has something that looks like coordinates (no words)."""
    return bool(_COORDINATES.search(text))


def _flac_may_hold_location(f: BinaryIO) -> bool:
    if f.read(4) != b"fLaC":
        return True
    while True:
        header = f.read(4)
        if len(header) < 4:
            return True
        last, kind = header[0] & 0x80, header[0] & 0x7F
        size = int.from_bytes(header[1:], "big")
        if kind == _FLAC_VORBIS_COMMENT:
            if _vorbis_comments_hold_location(f.read(size)):
                return True
        elif kind in _FLAC_SAFE_BLOCKS:
            f.seek(size, 1)
        else:  # APPLICATION (e.g. foreign WAV chunks), PICTURE, or unknown
            return True
        if last:
            return False


def _vorbis_comments_hold_location(block: bytes) -> bool:
    try:
        (vendor_length,) = struct.unpack_from("<I", block, 0)
        offset = 4 + vendor_length
        (count,) = struct.unpack_from("<I", block, offset)
        offset += 4
        for _ in range(count):
            (length,) = struct.unpack_from("<I", block, offset)
            offset += 4
            comment = block[offset : offset + length]
            if len(comment) < length:
                return True
            offset += length
            if mentions_location(comment.decode("utf-8", errors="replace")):
                return True
    except struct.error:
        return True
    return False


def _syncsafe(data: bytes) -> int:
    return (data[0] << 21) | (data[1] << 14) | (data[2] << 7) | data[3]


def _id3_text(frame_id: str, body: bytes) -> str | None:
    """The text of a text or comment frame, or None for a frame that isn't text."""
    if not body:
        return ""
    if not (frame_id.startswith("T") or frame_id in ("COMM", "COM")):
        return None
    encoding = _ID3_TEXT_ENCODINGS.get(body[0])
    if encoding is None:
        return None
    payload = body[1:]
    if frame_id in ("COMM", "COM"):
        payload = payload[3:]  # the language
    return payload.decode(encoding, errors="replace").replace("\x00", " ")


def _id3v2_may_hold_location(data: bytes) -> bool:
    """``data`` is a whole ID3v2 tag, header included."""
    if data[:3] != b"ID3" or len(data) < 10:
        return True
    version, flags = data[3], data[5]
    if version not in (2, 3, 4) or flags & 0x80 or (version == 2 and flags & 0x40):
        return True  # unsynchronised, or a compressed v2.2 tag
    end = 10 + _syncsafe(data[6:10])
    offset = 10
    if version != 2 and flags & 0x40:  # an extended header
        extended = _syncsafe(data[10:14]) if version == 4 else int.from_bytes(data[10:14], "big")
        offset += extended if version == 4 else extended + 4
    id_length, header_length = (3, 6) if version == 2 else (4, 10)
    while offset + header_length <= end:
        header = data[offset : offset + header_length]
        if header[0] == 0:  # padding
            return False
        frame_id = header[:id_length].decode("latin-1")
        if version == 2:
            frame_size = int.from_bytes(header[3:6], "big")
        elif version == 3:
            frame_size = int.from_bytes(header[4:8], "big")
        else:
            frame_size = _syncsafe(header[4:8])
        if version != 2 and header[9]:
            return True  # a compressed, encrypted, grouped or unsynchronised frame
        body = data[offset + header_length : offset + header_length + frame_size]
        if len(body) < frame_size:
            return True
        text = _id3_text(frame_id, body)
        if text is None or mentions_location(frame_id + " " + text):
            return True
        offset += header_length + frame_size
    return False


def _mp3_may_hold_location(f: BinaryIO) -> bool:
    head = f.read(10)
    if head[:3] == b"ID3":
        size = _syncsafe(head[6:10])
        if _id3v2_may_hold_location(head + f.read(size)):
            return True
        if head[5] & 0x10:  # a v2.4 footer follows the tag
            f.seek(10, 1)
        if f.read(3) == b"ID3":
            return True  # a second tag, which isn't read
    f.seek(0, 2)
    length = f.tell()
    f.seek(max(0, length - 512))
    end = f.read()
    if any(marker in end for marker in _MP3_TRAILING_TAGS):
        return True
    tail = end[-128:]
    if len(tail) == 128 and tail[:3] == b"TAG":  # ID3v1: title, artist, album, year, comment
        fields = (tail[3:33], tail[33:63], tail[63:93], tail[93:97], tail[97:127])
        if mentions_location(" ".join(field.decode("latin-1") for field in fields)):
            return True
    return False


def _wav_may_hold_location(f: BinaryIO) -> bool:
    """Every chunk but the redacted GUANO/WAMD ones and the audio is read or refused."""
    head = f.read(12)
    if len(head) < 12 or head[:4] not in (b"RIFF", b"RF64", b"BW64") or head[8:] != b"WAVE":
        return True
    data_size = None  # an RF64 data chunk's real size, from its ds64 chunk
    while True:
        header = f.read(8)
        if len(header) < 8:
            return False  # the end of the file
        chunk_id, size = header[:4], int.from_bytes(header[4:], "little")
        if chunk_id == b"data" and size == 0xFFFFFFFF:
            if data_size is None:
                return True
            size = data_size
        if chunk_id in _WAV_REDACTED or chunk_id in _WAV_NO_TEXT:
            if chunk_id == b"ds64":
                body = f.read(size)
                data_size = int.from_bytes(body[8:16], "little")
            else:
                f.seek(size, 1)
        elif size > _MAX_TEXT_CHUNK:
            return True
        elif chunk_id in _WAV_TEXT:
            text = f.read(size).decode("latin-1").replace("\x00", " ")
            if mentions_location(text):
                return True
        elif chunk_id in _WAV_ID3:
            if _id3v2_may_hold_location(f.read(size)):
                return True
        else:
            return True
        if size % 2:
            f.seek(1, 1)


def tags_may_hold_location(path: str | Path) -> bool:
    """Whether an audio file's metadata may hold a location a redacting push leaves in
    it (see the module docstring): for a WAV, any chunk but the GUANO and WAMD the push
    redacts; for FLAC and MP3, their tags.

    True for any other format, and for a file whose metadata can't be read: a
    redacting push can only vouch for what it has read.
    """
    from bioamla.audio._riffy import is_wav_path

    path = Path(path)
    suffix = path.suffix.lower()
    if is_wav_path(path):
        read = _wav_may_hold_location
    elif suffix == ".flac":
        read = _flac_may_hold_location
    elif suffix == ".mp3":
        read = _mp3_may_hold_location
    else:
        return True
    try:
        with path.open("rb") as f:
            return read(f)
    except (OSError, IndexError, ValueError):
        return True
