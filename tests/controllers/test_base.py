# tests/controllers/test_base.py
"""
Tests for BaseController and ControllerResult.
"""

import pytest

from bioamla.controllers.base import BaseController, ControllerResult


class TestControllerResult:
    """Tests for ControllerResult dataclass."""

    def test_ok_creates_successful_result(self):
        """Test that ControllerResult.ok() creates a success result."""
        result = ControllerResult.ok(data={"key": "value"}, message="Success")

        assert result.success is True
        assert result.data == {"key": "value"}
        assert result.message == "Success"
        assert result.error is None

    def test_fail_creates_failed_result(self):
        """Test that ControllerResult.fail() creates a failure result."""
        result = ControllerResult.fail("Something went wrong")

        assert result.success is False
        assert result.error == "Something went wrong"
        assert result.data is None

    def test_to_dict_includes_all_fields(self):
        """Test that to_dict() includes all result fields."""
        result = ControllerResult.ok(
            data={"count": 5},
            message="Processed",
            warnings=["Warning 1"],
        )
        result_dict = result.to_dict()

        assert "success" in result_dict
        assert "data" in result_dict
        assert "message" in result_dict
        assert "error" in result_dict
        assert "warnings" in result_dict
        assert result_dict["success"] is True
        assert result_dict["data"] == {"count": 5}


class TestBaseController:
    """Tests for BaseController."""

    @pytest.fixture
    def controller(self):
        return BaseController()

    def test_validate_input_path_nonexistent_returns_error(self, controller):
        """Test that validating a nonexistent path returns an error message."""
        error = controller._validate_input_path("/nonexistent/path/file.wav")

        assert error is not None
        assert "does not exist" in error

    def test_validate_input_path_existing_returns_none(self, controller, tmp_audio_file):
        """Test that validating an existing path returns None (no error)."""
        error = controller._validate_input_path(tmp_audio_file)

        assert error is None

    def test_get_audio_files_finds_wav_files(self, controller, tmp_dir_with_audio_files):
        """Test that _get_audio_files finds audio files in a directory."""
        files = controller._get_audio_files(tmp_dir_with_audio_files)

        assert len(files) == 3
        assert all(f.suffix == ".wav" for f in files)
