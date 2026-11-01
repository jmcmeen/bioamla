"""CLI tests for `bioamla util` commands (download / unzip / zip)."""

from __future__ import annotations

from unittest.mock import patch

import pytest
from click.testing import CliRunner

from bioamla.cli.cli import cli


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


def test_util_group_help(runner: CliRunner) -> None:
    result = runner.invoke(cli, ["util", "--help"])
    assert result.exit_code == 0
    for sub in ["download", "unzip", "zip"]:
        assert sub in result.output


def test_util_download(runner: CliRunner, tmp_path) -> None:
    with patch("bioamla.common.files.download_file") as m:
        result = runner.invoke(cli, ["util", "download", "http://x/file.zip", str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert "Downloaded to" in result.output
    m.assert_called_once()


def test_util_download_error(runner: CliRunner, tmp_path) -> None:
    with patch("bioamla.common.files.download_file", side_effect=OSError("no net")):
        result = runner.invoke(cli, ["util", "download", "http://x/file.zip", str(tmp_path)])
    assert result.exit_code != 0


def test_util_unzip(runner: CliRunner, tmp_path) -> None:
    import zipfile

    with zipfile.ZipFile(tmp_path / "a.zip", "w") as zf:
        zf.writestr("x/a.txt", "a")
    result = runner.invoke(cli, ["util", "unzip", str(tmp_path / "a.zip"), str(tmp_path / "o")])
    assert result.exit_code == 0, result.output
    assert "Extracted to" in result.output
    assert (tmp_path / "o" / "x" / "a.txt").read_text() == "a"


def test_util_unzip_error(runner: CliRunner, tmp_path) -> None:
    (tmp_path / "a.zip").write_bytes(b"not a zip")
    result = runner.invoke(cli, ["util", "unzip", str(tmp_path / "a.zip"), str(tmp_path)])
    assert result.exit_code != 0


def test_util_zip_file(runner: CliRunner, tmp_path) -> None:
    src = tmp_path / "a.txt"
    src.write_text("x")
    with (
        patch("bioamla.common.files.create_zip_file") as mfile,
        patch("bioamla.common.files.zip_directory") as mdir,
    ):
        result = runner.invoke(cli, ["util", "zip", str(src), str(tmp_path / "out.zip")])
    assert result.exit_code == 0, result.output
    mfile.assert_called_once()
    mdir.assert_not_called()


def test_util_zip_dir(runner: CliRunner, tmp_path) -> None:
    src = tmp_path / "srcdir"
    src.mkdir()
    with (
        patch("bioamla.common.files.create_zip_file") as mfile,
        patch("bioamla.common.files.zip_directory") as mdir,
    ):
        result = runner.invoke(cli, ["util", "zip", str(src), str(tmp_path / "out.zip")])
    assert result.exit_code == 0, result.output
    mdir.assert_called_once()
    mfile.assert_not_called()


def test_util_zip_error(runner: CliRunner, tmp_path) -> None:
    src = tmp_path / "a.txt"
    src.write_text("x")
    with (
        patch("bioamla.common.files.zip_directory"),
        patch("bioamla.common.files.create_zip_file", side_effect=OSError("disk full")),
    ):
        result = runner.invoke(cli, ["util", "zip", str(src), str(tmp_path / "out.zip")])
    assert result.exit_code != 0
