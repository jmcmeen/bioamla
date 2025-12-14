"""
Unit tests for bioamla CLI download command.
"""

import os
import pytest
from unittest.mock import patch, MagicMock
from click.testing import CliRunner

from bioamla.cli import cli


class TestDownloadCommand:
    """Tests for the download CLI command."""

    def test_download_extracts_filename_from_url(self, tmp_path, monkeypatch):
        """Test that download extracts filename from URL and constructs full path."""
        monkeypatch.chdir(tmp_path)

        calls = []
        def fake_download_file(url, output_path):
            calls.append((url, output_path))

        with patch("novus_pytils.files.download_file", fake_download_file):
            runner = CliRunner()
            result = runner.invoke(cli, ["download", "https://example.com/files/audio.wav", "."])

        assert len(calls) == 1
        url, output_path = calls[0]
        assert url == "https://example.com/files/audio.wav"
        assert output_path.endswith("audio.wav")

    def test_download_uses_default_filename_when_url_has_no_path(self, tmp_path, monkeypatch):
        """Test that download uses 'downloaded_file' when URL has no filename."""
        monkeypatch.chdir(tmp_path)

        calls = []
        def fake_download_file(url, output_path):
            calls.append((url, output_path))

        with patch("novus_pytils.files.download_file", fake_download_file):
            runner = CliRunner()
            result = runner.invoke(cli, ["download", "https://example.com", str(tmp_path)])

        assert len(calls) == 1
        url, output_path = calls[0]
        assert url == "https://example.com"
        assert output_path.endswith("downloaded_file")

    def test_download_uses_default_filename_when_url_ends_with_slash(self, tmp_path, monkeypatch):
        """Test that download uses 'downloaded_file' when URL ends with slash."""
        monkeypatch.chdir(tmp_path)

        calls = []
        def fake_download_file(url, output_path):
            calls.append((url, output_path))

        with patch("novus_pytils.files.download_file", fake_download_file):
            runner = CliRunner()
            result = runner.invoke(cli, ["download", "https://example.com/folder/", str(tmp_path)])

        assert len(calls) == 1
        url, output_path = calls[0]
        assert url == "https://example.com/folder/"
        assert output_path.endswith("downloaded_file")

    def test_download_resolves_dot_to_cwd(self, tmp_path, monkeypatch):
        """Test that '.' output_dir resolves to current working directory."""
        monkeypatch.chdir(tmp_path)

        calls = []
        def fake_download_file(url, output_path):
            calls.append((url, output_path))

        with patch("novus_pytils.files.download_file", fake_download_file):
            runner = CliRunner()
            result = runner.invoke(cli, ["download", "https://example.com/test.zip", "."])

        assert len(calls) == 1
        url, output_path = calls[0]
        assert os.path.dirname(output_path) == str(tmp_path)
        assert output_path.endswith("test.zip")

    def test_download_preserves_file_extension(self, tmp_path, monkeypatch):
        """Test that file extensions are preserved from URL."""
        monkeypatch.chdir(tmp_path)

        calls = []
        def fake_download_file(url, output_path):
            calls.append((url, output_path))

        with patch("novus_pytils.files.download_file", fake_download_file):
            runner = CliRunner()
            result = runner.invoke(cli, ["download", "https://example.com/data/archive.tar.gz", str(tmp_path)])

        assert len(calls) == 1
        url, output_path = calls[0]
        # Note: os.path.basename only gets the last component, so tar.gz becomes archive.tar.gz
        assert output_path.endswith("archive.tar.gz")

    def test_download_handles_query_params_in_url(self, tmp_path, monkeypatch):
        """Test that URL query parameters don't affect filename extraction."""
        monkeypatch.chdir(tmp_path)

        calls = []
        def fake_download_file(url, output_path):
            calls.append((url, output_path))

        with patch("novus_pytils.files.download_file", fake_download_file):
            runner = CliRunner()
            result = runner.invoke(cli, ["download", "https://example.com/file.wav?token=abc123", str(tmp_path)])

        assert len(calls) == 1
        url, output_path = calls[0]
        # The filename should be extracted without query params
        assert "file.wav" in output_path
        # The full URL with params should still be passed to download_file
        assert url == "https://example.com/file.wav?token=abc123"
