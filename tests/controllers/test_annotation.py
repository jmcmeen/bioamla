# tests/controllers/test_annotation.py
"""
Tests for AnnotationController.
"""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from bioamla.controllers.annotation_controller import (
    AnnotationController,
    AnnotationResult,
)


class TestAnnotationController:
    """Tests for AnnotationController."""

    @pytest.fixture
    def controller(self):
        return AnnotationController()

    @pytest.fixture
    def mock_annotations(self, mocker):
        """Mock annotations for testing."""
        mock_ann = MagicMock()
        mock_ann.start_time = 0.0
        mock_ann.end_time = 1.0
        mock_ann.low_freq = 1000.0
        mock_ann.high_freq = 5000.0
        mock_ann.label = "test_label"
        mock_ann.channel = 1
        mock_ann.confidence = 0.9
        mock_ann.notes = ""
        mock_ann.duration = 1.0
        mock_ann.bandwidth = 4000.0
        mock_ann.custom_fields = {}
        mock_ann.to_dict.return_value = {
            "start_time": 0.0,
            "end_time": 1.0,
            "label": "test_label",
        }
        return [mock_ann]

    def test_import_raven_success(self, controller, tmp_path, mocker):
        """Test that importing a Raven selection table succeeds."""
        # Create a mock Raven file
        raven_file = tmp_path / "selections.txt"
        raven_file.write_text(
            "Selection\tView\tChannel\tBegin Time (s)\tEnd Time (s)\tLow Freq (Hz)\tHigh Freq (Hz)\n"
            "1\tSpectrogram\t1\t0.0\t1.0\t1000\t5000\n"
        )

        mock_load = mocker.patch(
            "bioamla.controllers.annotation_controller.load_raven_selection_table"
        )
        mock_load.return_value = []

        mock_summary = mocker.patch(
            "bioamla.controllers.annotation_controller.summarize_annotations"
        )
        mock_summary.return_value = {"total": 0}

        result = controller.import_raven(str(raven_file))

        assert result.success is True
        assert result.data is not None
        assert isinstance(result.data, AnnotationResult)

    def test_import_raven_nonexistent_fails(self, controller):
        """Test that importing nonexistent file fails."""
        result = controller.import_raven("/nonexistent/path/selections.txt")

        assert result.success is False
        assert "does not exist" in result.error

    def test_create_annotation_success(self, controller):
        """Test that creating an annotation succeeds."""
        result = controller.create_annotation(
            start_time=0.0,
            end_time=1.0,
            label="test_species",
            low_freq=1000.0,
            high_freq=5000.0,
        )

        assert result.success is True
        assert result.data is not None
        assert result.data.label == "test_species"


class TestAnnotationValidation:
    """Tests for annotation validation."""

    @pytest.fixture
    def controller(self):
        return AnnotationController()

    def test_create_annotation_invalid_times_fails(self, controller):
        """Test that invalid time range fails."""
        result = controller.create_annotation(
            start_time=1.0,
            end_time=0.5,  # end before start
            label="test",
        )

        assert result.success is False
        assert "end_time must be greater than start_time" in result.error

    def test_create_annotation_invalid_freqs_fails(self, controller):
        """Test that invalid frequency range fails."""
        result = controller.create_annotation(
            start_time=0.0,
            end_time=1.0,
            label="test",
            low_freq=5000.0,
            high_freq=1000.0,  # high < low
        )

        assert result.success is False
        assert "high_freq must be greater than low_freq" in result.error


class TestExportOperations:
    """Tests for export operations."""

    @pytest.fixture
    def controller(self):
        return AnnotationController()

    @pytest.fixture
    def mock_annotations(self):
        """Create mock annotations for export."""
        ann = MagicMock()
        ann.start_time = 0.0
        ann.end_time = 1.0
        ann.low_freq = 1000.0
        ann.high_freq = 5000.0
        ann.label = "test"
        ann.channel = 1
        ann.to_dict.return_value = {
            "start_time": 0.0,
            "end_time": 1.0,
            "label": "test",
        }
        return [ann]

    def test_export_csv_success(self, controller, mock_annotations, tmp_path, mocker):
        """Test that exporting to CSV succeeds."""
        output_path = str(tmp_path / "annotations.csv")

        mock_save = mocker.patch("bioamla.controllers.annotation_controller.save_csv_annotations")
        mock_save.return_value = output_path

        result = controller.export_csv(mock_annotations, output_path)

        assert result.success is True
        assert result.data == output_path

    def test_export_json_success(self, controller, mock_annotations, tmp_path):
        """Test that exporting to JSON succeeds."""
        output_path = str(tmp_path / "annotations.json")

        result = controller.export_json(mock_annotations, output_path)

        assert result.success is True
        assert Path(output_path).exists()
