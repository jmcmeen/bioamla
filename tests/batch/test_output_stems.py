"""Tests for bioamla.batch.output_stems."""

from pathlib import Path

from bioamla.batch import output_stems


def test_plain_stems_mirror_the_tree(tmp_path) -> None:
    files = [str(tmp_path / "a.wav"), str(tmp_path / "sub" / "rec.2024.flac")]
    assert output_stems(files, tmp_path) == {
        files[0]: Path("a"),
        files[1]: Path("sub") / "rec.2024",
    }


def test_same_stem_different_extension_kept_apart(tmp_path) -> None:
    files = [str(tmp_path / "a.wav"), str(tmp_path / "a.mp3"), str(tmp_path / "b.wav")]
    stems = output_stems(files, tmp_path)
    assert stems == {files[0]: Path("a_wav"), files[1]: Path("a_mp3"), files[2]: Path("b")}


def test_file_outside_root_uses_its_name(tmp_path) -> None:
    outside = str(tmp_path.parent / "elsewhere.wav")
    assert output_stems([outside], tmp_path / "root") == {outside: Path("elsewhere")}
