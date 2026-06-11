"""Tests for bioamla.common.files."""

import pytest

from bioamla.common.files import (
    append_binary,
    append_text,
    create_directory,
    create_zip_file,
    directory_exists,
    download_file,
    ensure_directory,
    extract_zip_file,
    file_exists,
    get_extension_from_content_type,
    get_extension_from_url,
    get_files_by_extension,
    get_relative_path,
    prepare_output_path,
    read_binary,
    read_text,
    require_exists,
    sanitize_filename,
    write_binary,
    write_text,
    zip_directory,
)
from bioamla.exceptions import NotFoundError


class TestTextBinaryRoundTrip:
    def test_text_round_trip(self, tmp_path):
        p = tmp_path / "sub" / "a.txt"
        n = write_text(p, "hello")
        assert n == 5
        assert read_text(p) == "hello"

    def test_binary_round_trip(self, tmp_path):
        p = tmp_path / "sub" / "a.bin"
        n = write_binary(p, b"\x00\x01\x02")
        assert n == 3
        assert read_binary(p) == b"\x00\x01\x02"

    def test_append_text(self, tmp_path):
        p = tmp_path / "a.txt"
        write_text(p, "ab")
        append_text(p, "cd")
        assert read_text(p) == "abcd"

    def test_append_binary(self, tmp_path):
        p = tmp_path / "a.bin"
        write_binary(p, b"ab")
        append_binary(p, b"cd")
        assert read_binary(p) == b"abcd"


class TestDiscovery:
    def test_get_files_by_extension_recursive(self, tmp_path):
        (tmp_path / "a.wav").write_text("x")
        (tmp_path / "b.mp3").write_text("x")
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "c.wav").write_text("x")

        wavs = get_files_by_extension(tmp_path, [".wav"], recursive=True)
        assert len(wavs) == 2
        assert all(f.endswith(".wav") for f in wavs)
        # sorted
        assert wavs == sorted(wavs)

    def test_extension_normalization_no_dot(self, tmp_path):
        (tmp_path / "a.WAV").write_text("x")
        wavs = get_files_by_extension(tmp_path, ["wav"])
        assert len(wavs) == 1

    def test_non_recursive(self, tmp_path):
        (tmp_path / "a.wav").write_text("x")
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "c.wav").write_text("x")
        wavs = get_files_by_extension(tmp_path, [".wav"], recursive=False)
        assert len(wavs) == 1

    def test_no_extension_filter_returns_all(self, tmp_path):
        (tmp_path / "a.wav").write_text("x")
        (tmp_path / "b.txt").write_text("x")
        allf = get_files_by_extension(tmp_path, None)
        assert len(allf) == 2

    def test_missing_directory_returns_empty(self, tmp_path):
        assert get_files_by_extension(tmp_path / "nope", [".wav"]) == []

    def test_file_and_directory_exists(self, tmp_path):
        f = tmp_path / "a.txt"
        f.write_text("x")
        assert file_exists(f)
        assert not file_exists(tmp_path / "no.txt")
        assert directory_exists(tmp_path)
        assert not directory_exists(tmp_path / "nope")

    def test_create_and_ensure_directory(self, tmp_path):
        d = create_directory(tmp_path / "x" / "y")
        assert d.is_dir()
        d2 = ensure_directory(str(tmp_path / "z"))
        assert d2.is_dir()
        # ensure with existing Path
        d3 = ensure_directory(tmp_path / "z")
        assert d3.is_dir()


class TestValidators:
    def test_require_exists_ok(self, tmp_path):
        f = tmp_path / "a.txt"
        f.write_text("x")
        assert require_exists(f) == f

    def test_require_exists_raises(self, tmp_path):
        with pytest.raises(NotFoundError, match="does not exist"):
            require_exists(tmp_path / "nope")

    def test_prepare_output_path_creates_parent(self, tmp_path):
        p = prepare_output_path(tmp_path / "deep" / "out.txt")
        assert p.parent.is_dir()


class TestUrlHelpers:
    @pytest.mark.parametrize(
        "url,ext",
        [
            ("http://x/file.wav", ".wav"),
            ("http://x/file.m4a", ".m4a"),
            ("http://x/file.mp3", ".mp3"),
            ("http://x/file.ogg", ".ogg"),
            ("http://x/file.flac", ".flac"),
            ("http://x/nofile", ".mp3"),
        ],
    )
    def test_get_extension_from_url(self, url, ext):
        assert get_extension_from_url(url) == ext

    @pytest.mark.parametrize(
        "ct,ext",
        [
            ("audio/mpeg", ".mp3"),
            ("audio/wav", ".wav"),
            ("audio/x-flac; charset=utf-8", ".flac"),
            ("application/octet-stream", ""),
        ],
    )
    def test_get_extension_from_content_type(self, ct, ext):
        assert get_extension_from_content_type(ct) == ext


class TestDownloadFile:
    def test_download_file(self, tmp_path, monkeypatch):
        captured = {}

        def fake_urlretrieve(url, dest):
            captured["url"] = url
            captured["dest"] = dest
            from pathlib import Path

            Path(dest).write_text("data")
            return dest, None

        monkeypatch.setattr("bioamla.common.files.urlretrieve", fake_urlretrieve)
        out = download_file("http://x/file.wav", tmp_path / "sub" / "out.wav", show_progress=True)
        assert out.read_text() == "data"
        assert captured["url"] == "http://x/file.wav"

    def test_download_file_no_progress(self, tmp_path, monkeypatch):
        def fake_urlretrieve(url, dest):
            from pathlib import Path

            Path(dest).write_text("d")
            return dest, None

        monkeypatch.setattr("bioamla.common.files.urlretrieve", fake_urlretrieve)
        out = download_file("http://x/f.mp3", tmp_path / "o.mp3", show_progress=False)
        assert out.exists()


class TestPathUtilities:
    @pytest.mark.parametrize(
        "name,expected",
        [
            ("My Species Name", "my_species_name"),
            ("Test: File?", "test__file_"),
            ("", "unknown"),
            ("...", "unknown"),
        ],
    )
    def test_sanitize_filename(self, name, expected):
        assert sanitize_filename(name) == expected

    def test_get_relative_path(self, tmp_path):
        base = tmp_path
        f = tmp_path / "sub" / "a.txt"
        assert get_relative_path(f, base) == "sub/a.txt"

    def test_get_relative_path_outside_base(self, tmp_path):
        from pathlib import Path

        f = Path("/other/place/a.txt")
        assert get_relative_path(f, tmp_path) == "a.txt"


class TestZipHelpers:
    def test_create_and_extract_zip(self, tmp_path):
        f1 = tmp_path / "a.txt"
        f1.write_text("one")
        f2 = tmp_path / "b.txt"
        f2.write_text("two")

        zip_path = tmp_path / "out" / "arc.zip"
        created = create_zip_file([f1, f2], zip_path)
        assert created == str(zip_path)
        assert zip_path.exists()

        dest = tmp_path / "extracted"
        extracted = extract_zip_file(zip_path, dest)
        assert len(extracted) == 2
        assert (dest / "a.txt").read_text() == "one"

    def test_zip_directory(self, tmp_path):
        d = tmp_path / "data"
        d.mkdir()
        (d / "a.txt").write_text("x")
        sub = d / "sub"
        sub.mkdir()
        (sub / "b.txt").write_text("y")

        zip_path = tmp_path / "dir.zip"
        result = zip_directory(d, zip_path)
        assert result == str(zip_path)

        dest = tmp_path / "out"
        names = extract_zip_file(zip_path, dest)
        assert (dest / "a.txt").exists()
        assert (dest / "sub" / "b.txt").exists()
        assert len(names) == 2
