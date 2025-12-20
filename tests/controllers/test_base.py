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

    def test_validate_output_path_creates_parents(self, controller, tmp_path):
        """Test that _validate_output_path creates parent directories."""
        nested_path = tmp_path / "nested" / "dir" / "output.wav"
        error = controller._validate_output_path(str(nested_path))

        assert error is None
        assert (tmp_path / "nested" / "dir").exists()

    def test_validate_output_path_no_overwrite_fails(self, controller, tmp_audio_file):
        """Test that _validate_output_path fails when overwrite is disallowed."""
        error = controller._validate_output_path(tmp_audio_file, allow_overwrite=False)

        assert error is not None
        assert "already exists" in error

    def test_validate_output_path_allows_overwrite(self, controller, tmp_audio_file):
        """Test that _validate_output_path succeeds when overwrite is allowed."""
        error = controller._validate_output_path(tmp_audio_file, allow_overwrite=True)

        assert error is None

    def test_get_audio_files_single_file(self, controller, tmp_audio_file):
        """Test that _get_audio_files handles single file path."""
        files = controller._get_audio_files(tmp_audio_file)

        assert len(files) == 1
        assert files[0].suffix == ".wav"


class TestProgressCallback:
    """Tests for progress callback functionality."""

    @pytest.fixture
    def controller(self):
        return BaseController()

    def test_set_progress_callback(self, controller):
        """Test that progress callback can be set."""
        callback_called = []

        def callback(progress):
            callback_called.append(progress)

        controller.set_progress_callback(callback)
        assert controller._progress_callback is not None

    def test_report_progress_calls_callback(self, controller):
        """Test that _report_progress calls the registered callback."""
        from bioamla.controllers.base import BatchProgress

        callback_called = []

        def callback(progress):
            callback_called.append(progress)

        controller.set_progress_callback(callback)
        progress = BatchProgress(total=10, completed=5)
        controller._report_progress(progress)

        assert len(callback_called) == 1
        assert callback_called[0].total == 10
        assert callback_called[0].completed == 5

    def test_report_progress_no_callback(self, controller):
        """Test that _report_progress handles no callback gracefully."""
        from bioamla.controllers.base import BatchProgress

        progress = BatchProgress(total=10)
        # Should not raise
        controller._report_progress(progress)


class TestBatchProgress:
    """Tests for BatchProgress dataclass."""

    def test_percent_calculation(self):
        """Test that percent is calculated correctly."""
        from bioamla.controllers.base import BatchProgress

        progress = BatchProgress(total=100, completed=25)
        assert progress.percent == 25.0

    def test_percent_zero_total(self):
        """Test percent with zero total."""
        from bioamla.controllers.base import BatchProgress

        progress = BatchProgress(total=0, completed=0)
        assert progress.percent == 0

    def test_remaining_calculation(self):
        """Test remaining items calculation."""
        from bioamla.controllers.base import BatchProgress

        progress = BatchProgress(total=100, completed=25)
        assert progress.remaining == 75


class TestRunTracking:
    """Tests for run tracking functionality."""

    @pytest.fixture
    def controller(self):
        return BaseController()

    def test_start_run(self, controller, mocker):
        """Test that _start_run creates a run."""
        mock_storage = mocker.MagicMock()
        mock_storage.create_run.return_value = "test_run_123"
        mocker.patch.object(controller, "_storage", mock_storage)

        run_id = controller._start_run(
            name="Test Run",
            action="test",
            input_path="/input",
            output_path="/output",
            parameters={"key": "value"},
        )

        assert run_id == "test_run_123"
        assert controller._current_run_id == "test_run_123"
        mock_storage.create_run.assert_called_once()

    def test_complete_run(self, controller, mocker):
        """Test that _complete_run marks a run as complete."""
        mock_storage = mocker.MagicMock()
        mock_storage.update_run.return_value = True
        mocker.patch.object(controller, "_storage", mock_storage)
        controller._current_run_id = "test_run_123"

        success = controller._complete_run(
            results={"count": 10},
            output_files=["/output/file.csv"],
        )

        assert success is True
        assert controller._current_run_id is None
        mock_storage.update_run.assert_called_once()

    def test_complete_run_no_current_run(self, controller):
        """Test _complete_run when no run is active."""
        controller._current_run_id = None
        success = controller._complete_run()

        assert success is False

    def test_fail_run(self, controller, mocker):
        """Test that _fail_run marks a run as failed."""
        mock_storage = mocker.MagicMock()
        mock_storage.fail_run.return_value = True
        mocker.patch.object(controller, "_storage", mock_storage)
        controller._current_run_id = "test_run_123"

        success = controller._fail_run("Something went wrong")

        assert success is True
        assert controller._current_run_id is None
        mock_storage.fail_run.assert_called_once_with("test_run_123", "Something went wrong")

    def test_fail_run_no_current_run(self, controller):
        """Test _fail_run when no run is active."""
        controller._current_run_id = None
        success = controller._fail_run("Error message")

        assert success is False


class TestToDictMixin:
    """Tests for ToDictMixin."""

    def test_to_dict_simple_dataclass(self):
        """Test to_dict with simple dataclass fields."""
        from dataclasses import dataclass

        from bioamla.controllers.base import ToDictMixin

        @dataclass
        class SimpleResult(ToDictMixin):
            name: str
            count: int
            value: float

        result = SimpleResult(name="test", count=5, value=3.14)
        d = result.to_dict()

        assert d["name"] == "test"
        assert d["count"] == 5
        assert d["value"] == 3.14

    def test_to_dict_with_path(self):
        """Test to_dict with Path objects."""
        from dataclasses import dataclass
        from pathlib import Path

        from bioamla.controllers.base import ToDictMixin

        @dataclass
        class PathResult(ToDictMixin):
            filepath: Path

        result = PathResult(filepath=Path("/test/file.wav"))
        d = result.to_dict()

        assert d["filepath"] == "/test/file.wav"

    def test_to_dict_with_nested_list(self):
        """Test to_dict with nested lists."""
        from dataclasses import dataclass, field
        from typing import List

        from bioamla.controllers.base import ToDictMixin

        @dataclass
        class ListResult(ToDictMixin):
            items: List[str] = field(default_factory=list)

        result = ListResult(items=["a", "b", "c"])
        d = result.to_dict()

        assert d["items"] == ["a", "b", "c"]


class TestControllerResultMetadata:
    """Tests for ControllerResult metadata handling."""

    def test_ok_with_metadata(self):
        """Test that ok() includes metadata."""
        result = ControllerResult.ok(
            data="test",
            message="Success",
            duration=1.5,
            sample_rate=16000,
        )

        assert result.metadata["duration"] == 1.5
        assert result.metadata["sample_rate"] == 16000

    def test_fail_with_metadata(self):
        """Test that fail() includes metadata."""
        result = ControllerResult.fail(
            "Error occurred",
            file_count=10,
            errors_count=3,
        )

        assert result.metadata["file_count"] == 10
        assert result.metadata["errors_count"] == 3

    def test_to_dict_includes_metadata(self):
        """Test that to_dict includes non-empty metadata."""
        result = ControllerResult.ok(data="test", extra_info="value")
        d = result.to_dict()

        assert "metadata" in d
        assert d["metadata"]["extra_info"] == "value"

    def test_to_dict_excludes_empty_metadata(self):
        """Test that to_dict excludes empty metadata."""
        result = ControllerResult.ok(data="test")
        d = result.to_dict()

        assert "metadata" not in d or d.get("metadata") == {}
