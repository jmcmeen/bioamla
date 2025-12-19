"""Unit tests for command logging."""

import json
import pytest
from datetime import datetime, timedelta
from pathlib import Path

from bioamla.core.command_log import (
    CommandLogger,
    CommandEntry,
    create_command_entry,
    LOG_FILENAME,
    LOGS_DIR,
)
from bioamla.core.project import create_project, PROJECT_MARKER


class TestCommandEntry:
    """Tests for CommandEntry dataclass."""

    def test_to_json_roundtrip(self):
        """Test JSON serialization and deserialization."""
        entry = CommandEntry(
            timestamp="2025-01-15T10:30:00",
            command="predict",
            args=["audio.wav"],
            kwargs={"model": "ast"},
            exit_code=0,
            duration_seconds=1.5,
            working_dir="/home/user",
        )

        json_str = entry.to_json()
        restored = CommandEntry.from_json(json_str)

        assert restored.command == entry.command
        assert restored.args == entry.args
        assert restored.exit_code == entry.exit_code
        assert restored.duration_seconds == entry.duration_seconds

    def test_to_json_with_optional_fields(self):
        """Test JSON with optional fields."""
        entry = CommandEntry(
            timestamp="2025-01-15T10:30:00",
            command="predict",
            args=["audio.wav"],
            kwargs={},
            exit_code=1,
            duration_seconds=0.5,
            working_dir="/test",
            project_root="/test/project",
            error_message="File not found",
        )

        json_str = entry.to_json()
        restored = CommandEntry.from_json(json_str)

        assert restored.project_root == "/test/project"
        assert restored.error_message == "File not found"

    def test_to_json_produces_valid_json(self):
        """Test that to_json produces valid JSON."""
        entry = CommandEntry(
            timestamp="2025-01-15T10:30:00",
            command="test",
            args=["a", "b"],
            kwargs={"key": "value"},
            exit_code=0,
            duration_seconds=1.0,
            working_dir="/tmp",
        )

        json_str = entry.to_json()
        parsed = json.loads(json_str)

        assert parsed["command"] == "test"
        assert parsed["args"] == ["a", "b"]


class TestCommandLogger:
    """Tests for CommandLogger class."""

    @pytest.fixture
    def project_with_logger(self, tmp_path):
        """Create a project and return logger."""
        create_project(tmp_path)
        return CommandLogger(tmp_path)

    def test_is_available_in_project(self, project_with_logger):
        """Test logger is available in project."""
        assert project_with_logger.is_available()

    def test_not_available_outside_project(self, tmp_path):
        """Test logger not available outside project."""
        # Create a directory that is not a project (no .bioamla marker)
        non_project_dir = tmp_path / "not_a_project"
        non_project_dir.mkdir()

        logger = CommandLogger(non_project_dir)
        assert not logger.is_available()

    def test_log_path_is_correct(self, project_with_logger, tmp_path):
        """Test log path is set correctly."""
        expected = tmp_path / PROJECT_MARKER / LOGS_DIR / LOG_FILENAME
        assert project_with_logger.log_path == expected

    def test_log_and_retrieve(self, project_with_logger):
        """Test logging and retrieving entries."""
        entry = CommandEntry(
            timestamp=datetime.now().isoformat(),
            command="test",
            args=["arg1"],
            kwargs={},
            exit_code=0,
            duration_seconds=0.1,
            working_dir="/test",
        )

        project_with_logger.log_command(entry)
        history = project_with_logger.get_history()

        assert len(history) == 1
        assert history[0].command == "test"

    def test_multiple_entries(self, project_with_logger):
        """Test logging multiple entries."""
        for i in range(5):
            entry = CommandEntry(
                timestamp=datetime.now().isoformat(),
                command=f"cmd{i}",
                args=[],
                kwargs={},
                exit_code=0,
                duration_seconds=0.1,
                working_dir="/test",
            )
            project_with_logger.log_command(entry)

        history = project_with_logger.get_history()
        assert len(history) == 5

    def test_history_limit(self, project_with_logger):
        """Test history limit parameter."""
        for i in range(10):
            entry = CommandEntry(
                timestamp=datetime.now().isoformat(),
                command=f"cmd{i}",
                args=[],
                kwargs={},
                exit_code=0,
                duration_seconds=0.1,
                working_dir="/test",
            )
            project_with_logger.log_command(entry)

        history = project_with_logger.get_history(limit=5)
        assert len(history) == 5

    def test_history_returns_newest_first(self, project_with_logger):
        """Test history returns entries in newest-first order."""
        for i in range(3):
            entry = CommandEntry(
                timestamp=datetime.now().isoformat(),
                command=f"cmd{i}",
                args=[],
                kwargs={},
                exit_code=0,
                duration_seconds=0.1,
                working_dir="/test",
            )
            project_with_logger.log_command(entry)

        history = project_with_logger.get_history()
        assert history[0].command == "cmd2"  # Newest first
        assert history[2].command == "cmd0"  # Oldest last

    def test_history_command_filter(self, project_with_logger):
        """Test filtering history by command name."""
        for cmd in ["predict", "train", "predict"]:
            entry = CommandEntry(
                timestamp=datetime.now().isoformat(),
                command=cmd,
                args=[],
                kwargs={},
                exit_code=0,
                duration_seconds=0.1,
                working_dir="/test",
            )
            project_with_logger.log_command(entry)

        history = project_with_logger.get_history(command_filter="predict")
        assert len(history) == 2
        assert all(e.command == "predict" for e in history)

    def test_search(self, project_with_logger):
        """Test search functionality."""
        for cmd, args in [
            ("predict", ["audio.wav"]),
            ("train", ["model"]),
            ("predict", ["video.mp4"]),
        ]:
            entry = CommandEntry(
                timestamp=datetime.now().isoformat(),
                command=cmd,
                args=args,
                kwargs={},
                exit_code=0,
                duration_seconds=0.1,
                working_dir="/test",
            )
            project_with_logger.log_command(entry)

        results = project_with_logger.search("predict")
        assert len(results) == 2

    def test_search_in_args(self, project_with_logger):
        """Test search matches in args."""
        entry = CommandEntry(
            timestamp=datetime.now().isoformat(),
            command="process",
            args=["special_file.wav"],
            kwargs={},
            exit_code=0,
            duration_seconds=0.1,
            working_dir="/test",
        )
        project_with_logger.log_command(entry)

        results = project_with_logger.search("special")
        assert len(results) == 1

    def test_search_case_insensitive(self, project_with_logger):
        """Test search is case insensitive."""
        entry = CommandEntry(
            timestamp=datetime.now().isoformat(),
            command="PREDICT",
            args=[],
            kwargs={},
            exit_code=0,
            duration_seconds=0.1,
            working_dir="/test",
        )
        project_with_logger.log_command(entry)

        results = project_with_logger.search("predict")
        assert len(results) == 1

    def test_clear(self, project_with_logger):
        """Test clearing history."""
        entry = CommandEntry(
            timestamp=datetime.now().isoformat(),
            command="test",
            args=[],
            kwargs={},
            exit_code=0,
            duration_seconds=0.1,
            working_dir="/test",
        )
        project_with_logger.log_command(entry)

        count = project_with_logger.clear()

        assert count == 1
        assert len(project_with_logger.get_history()) == 0

    def test_clear_empty(self, project_with_logger):
        """Test clearing empty history."""
        count = project_with_logger.clear()
        assert count == 0

    def test_get_stats(self, project_with_logger):
        """Test getting statistics."""
        # Log some commands
        for cmd, exit_code in [("predict", 0), ("train", 0), ("predict", 1)]:
            entry = CommandEntry(
                timestamp=datetime.now().isoformat(),
                command=cmd,
                args=[],
                kwargs={},
                exit_code=exit_code,
                duration_seconds=0.1,
                working_dir="/test",
            )
            project_with_logger.log_command(entry)

        stats = project_with_logger.get_stats()

        assert stats["total_commands"] == 3
        assert stats["successful_commands"] == 2
        assert stats["failed_commands"] == 1
        assert stats["command_counts"]["predict"] == 2
        assert stats["command_counts"]["train"] == 1

    def test_get_stats_empty(self, project_with_logger):
        """Test getting stats with no history."""
        stats = project_with_logger.get_stats()

        assert stats["total_commands"] == 0
        assert stats["successful_commands"] == 0
        assert stats["failed_commands"] == 0
        assert stats["command_counts"] == {}

    def test_handles_malformed_entries(self, project_with_logger):
        """Test that malformed entries are skipped."""
        # Write a valid entry
        entry = CommandEntry(
            timestamp=datetime.now().isoformat(),
            command="valid",
            args=[],
            kwargs={},
            exit_code=0,
            duration_seconds=0.1,
            working_dir="/test",
        )
        project_with_logger.log_command(entry)

        # Write a malformed line directly
        with open(project_with_logger.log_path, "a") as f:
            f.write("this is not valid json\n")

        # Write another valid entry
        entry2 = CommandEntry(
            timestamp=datetime.now().isoformat(),
            command="valid2",
            args=[],
            kwargs={},
            exit_code=0,
            duration_seconds=0.1,
            working_dir="/test",
        )
        project_with_logger.log_command(entry2)

        # Should get both valid entries, skipping the malformed one
        history = project_with_logger.get_history()
        assert len(history) == 2

    def test_log_creates_directory(self, tmp_path):
        """Test that logging creates the logs directory if needed."""
        create_project(tmp_path)

        # Remove the logs directory
        logs_dir = tmp_path / PROJECT_MARKER / LOGS_DIR
        if logs_dir.exists():
            import shutil

            shutil.rmtree(logs_dir)

        logger = CommandLogger(tmp_path)
        entry = CommandEntry(
            timestamp=datetime.now().isoformat(),
            command="test",
            args=[],
            kwargs={},
            exit_code=0,
            duration_seconds=0.1,
            working_dir="/test",
        )
        logger.log_command(entry)

        assert logs_dir.exists()
        assert logger.log_path.exists()


class TestCreateCommandEntry:
    """Tests for create_command_entry helper function."""

    def test_creates_entry_with_timestamp(self):
        """Test that entry has current timestamp."""
        entry = create_command_entry(command="test")

        assert entry.timestamp
        # Should be a valid ISO format timestamp
        datetime.fromisoformat(entry.timestamp)

    def test_creates_entry_with_defaults(self):
        """Test entry with default values."""
        entry = create_command_entry(command="test")

        assert entry.command == "test"
        assert entry.args == []
        assert entry.kwargs == {}
        assert entry.exit_code == 0
        assert entry.duration_seconds == 0.0

    def test_creates_entry_with_custom_values(self):
        """Test entry with custom values."""
        entry = create_command_entry(
            command="predict",
            args=["audio.wav"],
            kwargs={"model": "ast"},
            exit_code=1,
            duration_seconds=2.5,
            error_message="Test error",
        )

        assert entry.command == "predict"
        assert entry.args == ["audio.wav"]
        assert entry.kwargs == {"model": "ast"}
        assert entry.exit_code == 1
        assert entry.duration_seconds == 2.5
        assert entry.error_message == "Test error"
