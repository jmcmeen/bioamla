"""Whether FLAC and MP3 tags may hold a location, read from hand-built headers."""

from __future__ import annotations

import struct

import pytest

from bioamla.audio._tags import mentions_location, tags_may_hold_location

FRAME = b"\xff\xfb\x90\x00" + bytes(413)  # one MPEG audio frame, enough to follow a tag


def _block(kind: int, body: bytes, last: bool = False) -> bytes:
    return bytes([(0x80 if last else 0) | kind]) + len(body).to_bytes(3, "big") + body


def flac(comments: list[str] | None = None, extra: list[tuple[int, bytes]] = ()) -> bytes:
    """A FLAC header: STREAMINFO, Vorbis comments (if given), then any ``extra`` blocks."""
    blocks = [(0, bytes(34))]
    if comments is not None:
        vendor = b"reference libFLAC"
        body = struct.pack("<I", len(vendor)) + vendor + struct.pack("<I", len(comments))
        for comment in comments:
            data = comment.encode()
            body += struct.pack("<I", len(data)) + data
        blocks.append((4, body))
    blocks += list(extra)
    out = b"fLaC"
    for i, (kind, body) in enumerate(blocks):
        out += _block(kind, body, last=i == len(blocks) - 1)
    return out + b"\xff\xf8" + bytes(64)


def wav(chunks: list[tuple[bytes, bytes]] = ()) -> bytes:
    """A 16-bit mono WAV with a little silence and then ``chunks``."""
    body = b"WAVE" + b"fmt " + struct.pack("<IHHIIHH", 16, 1, 1, 16000, 32000, 2, 16)
    body += b"data" + struct.pack("<I", 64) + bytes(64)
    for chunk_id, data in chunks:
        body += chunk_id + struct.pack("<I", len(data)) + data + (b"\x00" if len(data) % 2 else b"")
    return b"RIFF" + struct.pack("<I", len(body)) + body


def _syncsafe(n: int) -> bytes:
    return bytes([(n >> 21) & 0x7F, (n >> 14) & 0x7F, (n >> 7) & 0x7F, n & 0x7F])


def id3v2_mp3(frames: list[tuple[str, str]], version: int = 4, flags: int = 0) -> bytes:
    """An MP3 with an ID3v2 tag of text frames (UTF-8, or Latin-1 for v2.3)."""
    encoding = 3 if version == 4 else 0
    body = b""
    for frame_id, text in frames:
        data = bytes([encoding]) + text.encode("utf-8" if version == 4 else "latin-1")
        size = _syncsafe(len(data)) if version == 4 else len(data).to_bytes(4, "big")
        body += frame_id.encode() + size + b"\x00\x00" + data
    body += bytes(16)  # padding
    return b"ID3" + bytes([version, 0, flags]) + _syncsafe(len(body)) + body + FRAME


@pytest.mark.parametrize(
    "comments",
    [[], ["TITLE=Spring peepers", "ARTIST=Frog survey", "DATE=2024-04-01"]],
)
def test_flac_with_clean_tags_is_vouched_for(tmp_path, comments):
    path = tmp_path / "a.flac"
    path.write_bytes(flac(comments))
    assert not tags_may_hold_location(path)


@pytest.mark.parametrize(
    "comment",
    [
        "LOCATION=Roan Mountain",
        "GPS=36.1,-82.1",
        "LATITUDE=36.1234",
        "COMMENT=recorded at 36.1234, -82.1234",
        "DESCRIPTION=near the lat/lon of site 3",
    ],
)
def test_flac_with_a_location_tag_is_not(tmp_path, comment):
    path = tmp_path / "a.flac"
    path.write_bytes(flac([comment]))
    assert tags_may_hold_location(path)


@pytest.mark.parametrize("kind", [2, 6, 9])  # APPLICATION (foreign WAV chunks), PICTURE, unknown
def test_flac_blocks_it_cant_read_mean_it_cant_vouch(tmp_path, kind):
    path = tmp_path / "a.flac"
    path.write_bytes(flac(["TITLE=x"], extra=[(kind, b"riff" + b"guan" + bytes(8))]))
    assert tags_may_hold_location(path)


def test_a_truncated_or_foreign_flac_cant_be_vouched_for(tmp_path):
    path = tmp_path / "a.flac"
    path.write_bytes(flac(["TITLE=x"])[:30])
    assert tags_may_hold_location(path)
    path.write_bytes(b"OggS" + bytes(60))
    assert tags_may_hold_location(path)


@pytest.mark.parametrize("version", [3, 4])
def test_mp3_with_clean_tags_is_vouched_for(tmp_path, version):
    path = tmp_path / "a.mp3"
    path.write_bytes(id3v2_mp3([("TIT2", "Spring peepers"), ("TPE1", "Frog survey")], version))
    assert not tags_may_hold_location(path)
    path.write_bytes(FRAME * 3)  # no tag at all
    assert not tags_may_hold_location(path)


@pytest.mark.parametrize(
    "frame",
    [
        ("TXXX", "GPS\x0036.1,-82.1"),
        ("COMM", "eng\x00recorded at 36.1234, -82.1234"),
        ("TIT2", "Roan Mountain location 3"),
        ("APIC", "image/jpeg\x00\x03\x00not text"),  # binary frames can't be read
    ],
)
def test_mp3_with_a_location_or_binary_frame_is_not(tmp_path, frame):
    path = tmp_path / "a.mp3"
    path.write_bytes(id3v2_mp3([("TIT2", "x"), frame]))
    assert tags_may_hold_location(path)


def test_mp3_tags_it_cant_read_mean_it_cant_vouch(tmp_path):
    path = tmp_path / "a.mp3"
    path.write_bytes(id3v2_mp3([("TIT2", "x")], flags=0x80))  # unsynchronised
    assert tags_may_hold_location(path)
    v1 = (
        b"TAG"
        + b"Peepers".ljust(30, b"\0")
        + bytes(60)
        + b"2024"
        + b"lat 36.1".ljust(30, b"\0")
        + b"\x0c"
    )
    path.write_bytes(FRAME + v1)
    assert tags_may_hold_location(path)
    path.write_bytes(FRAME * 2 + b"APETAGEX" + bytes(24))
    assert tags_may_hold_location(path)


def test_other_formats_are_never_vouched_for(tmp_path):
    for name in ("a.ogg", "a.m4a", "a.opus", "a.aiff"):
        (tmp_path / name).write_bytes(bytes(64))
        assert tags_may_hold_location(tmp_path / name)


@pytest.mark.parametrize(
    "text",
    [
        "36.12, -82.57",  # two decimals is still about 1 km
        "N36.1234 W82.5678",
        "36.12N 82.57W",
        "36°07'24\"N 82°34'W",
        "17S 412345 3998765",
        "GPS fix ok",
        "elevation 1200 m",
        "lat/long in notes",
    ],
)
def test_coordinates_and_location_words_are_recognized(text):
    assert mentions_location(text)


@pytest.mark.parametrize(
    "text",
    ["George Smith", "Geothlypis trichas", "long call series", "geophony", "15°C, gain 2"],
)
def test_ordinary_tags_are_not_mistaken_for_locations(text):
    assert not mentions_location(text)


def test_wav_chunks_other_than_the_redacted_ones_are_read(tmp_path):
    path = tmp_path / "a.wav"
    for chunks, holds in [
        ([], False),
        ([(b"guan", b"GUANO|Version:1.0\nLoc Position:36.1 -82.1\n")], False),  # redacted
        ([(b"LIST", b"INFOICMT" + b"by AudioMoth at 4.2V")], False),
        ([(b"bext", b"Recorded at Roan, location B")], True),
        ([(b"id3 ", id3v2_mp3([("TXXX", "GPS\x0036.1,-82.1")])[: -len(FRAME)])], True),
        ([(b"id3 ", id3v2_mp3([("TIT2", "x")])[: -len(FRAME)])], False),
        ([(b"zzzz", b"anything")], True),  # unknown
    ]:
        path.write_bytes(wav(chunks))
        assert tags_may_hold_location(path) is holds, chunks


def test_mp3_tags_after_the_first_or_at_the_end_are_refused(tmp_path):
    path = tmp_path / "a.mp3"
    tag = id3v2_mp3([("TIT2", "x")])[: -len(FRAME)]
    path.write_bytes(tag + tag + FRAME)  # a second tag
    assert tags_may_hold_location(path)
    footer = b"3DI" + tag[3:10]
    v1 = b"TAG" + bytes(125)
    path.write_bytes(FRAME * 2 + tag + footer + v1)  # an appended tag before ID3v1
    assert tags_may_hold_location(path)
