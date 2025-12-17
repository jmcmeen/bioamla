"""
Unit tests for the ast evaluate CLI command.
"""

from unittest.mock import patch

import numpy as np
import pytest
from click.testing import CliRunner

from bioamla.cli import cli
from bioamla.evaluate import EvaluationResult


@pytest.fixture
def runner():
    """Create a CLI test runner."""
    return CliRunner()


@pytest.fixture
def mock_evaluation_result():
    """Create a mock evaluation result."""
    return EvaluationResult(
        accuracy=0.85,
        precision=0.80,
        recall=0.75,
        f1_score=0.77,
        confusion_matrix=np.array([[10, 2], [3, 15]]),
        class_labels=["cat", "dog"],
        per_class_metrics={
            "cat": {"precision": 0.77, "recall": 0.83, "f1_score": 0.80, "support": 12},
            "dog": {"precision": 0.88, "recall": 0.83, "f1_score": 0.86, "support": 18},
        },
        total_samples=30,
        correct_predictions=25,
    )


class TestEvaluateHelp:
    """Tests for evaluate help and options."""

    def test_evaluate_help(self, runner):
        """Test evaluate --help shows all options."""
        result = runner.invoke(cli, ["models", "evaluate", "ast", "--help"])

        assert result.exit_code == 0
        assert "PATH" in result.output
        assert "--model-path" in result.output
        assert "--ground-truth" in result.output
        assert "--output" in result.output
        assert "--format" in result.output

    def test_evaluate_requires_ground_truth(self, runner, temp_dir):
        """Test that evaluate requires --ground-truth option."""
        audio_dir = temp_dir / "audio"
        audio_dir.mkdir()

        result = runner.invoke(cli, [
            "models", "evaluate", "ast",
            str(audio_dir),
        ])

        assert result.exit_code != 0
        assert "ground-truth" in result.output.lower() or "required" in result.output.lower()


class TestEvaluateCommand:
    """Tests for the evaluate command."""

    def test_evaluate_missing_path(self, runner, temp_dir):
        """Test error handling for missing path."""
        gt_path = temp_dir / "labels.csv"
        gt_path.write_text("file_name,label\naudio.wav,cat\n")

        result = runner.invoke(cli, [
            "models", "evaluate", "ast",
            "/nonexistent/path",
            "--ground-truth", str(gt_path),
        ])

        assert result.exit_code == 1
        assert "Error" in result.output

    def test_evaluate_missing_ground_truth(self, runner, temp_dir):
        """Test error handling for missing ground truth file."""
        audio_dir = temp_dir / "audio"
        audio_dir.mkdir()

        result = runner.invoke(cli, [
            "models", "evaluate", "ast",
            str(audio_dir),
            "--ground-truth", "/nonexistent/labels.csv",
        ])

        assert result.exit_code == 1
        assert "Error" in result.output

    @patch('bioamla.evaluate.evaluate_directory')
    def test_evaluate_displays_report(self, mock_evaluate, runner, temp_dir, mock_evaluation_result):
        """Test that evaluate displays formatted report."""
        mock_evaluate.return_value = mock_evaluation_result

        audio_dir = temp_dir / "audio"
        audio_dir.mkdir()
        gt_path = temp_dir / "labels.csv"
        gt_path.write_text("file_name,label\naudio.wav,cat\n")

        result = runner.invoke(cli, [
            "models", "evaluate", "ast",
            str(audio_dir),
            "--ground-truth", str(gt_path),
        ])

        assert result.exit_code == 0
        assert "Accuracy" in result.output
        assert "0.85" in result.output

    @patch('bioamla.evaluate.evaluate_directory')
    def test_evaluate_quiet_mode(self, mock_evaluate, runner, temp_dir, mock_evaluation_result):
        """Test quiet mode output."""
        mock_evaluate.return_value = mock_evaluation_result

        audio_dir = temp_dir / "audio"
        audio_dir.mkdir()
        gt_path = temp_dir / "labels.csv"
        gt_path.write_text("file_name,label\naudio.wav,cat\n")

        result = runner.invoke(cli, [
            "models", "evaluate", "ast",
            str(audio_dir),
            "--ground-truth", str(gt_path),
            "--quiet",
        ])

        assert result.exit_code == 0
        assert "Accuracy: 0.85" in result.output
        # Should not have the full report format
        assert "Model Evaluation Report" not in result.output

    @patch('bioamla.evaluate.evaluate_directory')
    @patch('bioamla.evaluate.save_evaluation_results')
    def test_evaluate_saves_output(self, mock_save, mock_evaluate, runner, temp_dir, mock_evaluation_result):
        """Test that evaluate saves output when --output specified."""
        mock_evaluate.return_value = mock_evaluation_result
        mock_save.return_value = str(temp_dir / "results.json")

        audio_dir = temp_dir / "audio"
        audio_dir.mkdir()
        gt_path = temp_dir / "labels.csv"
        gt_path.write_text("file_name,label\naudio.wav,cat\n")
        output_path = temp_dir / "results.json"

        result = runner.invoke(cli, [
            "models", "evaluate", "ast",
            str(audio_dir),
            "--ground-truth", str(gt_path),
            "--output", str(output_path),
            "--format", "json",
        ])

        assert result.exit_code == 0
        mock_save.assert_called_once()
        assert "Results saved to" in result.output

    @patch('bioamla.evaluate.evaluate_directory')
    def test_evaluate_custom_columns(self, mock_evaluate, runner, temp_dir, mock_evaluation_result):
        """Test evaluate with custom column names."""
        mock_evaluate.return_value = mock_evaluation_result

        audio_dir = temp_dir / "audio"
        audio_dir.mkdir()
        gt_path = temp_dir / "labels.csv"
        gt_path.write_text("filename,class\naudio.wav,cat\n")

        result = runner.invoke(cli, [
            "models", "evaluate", "ast",
            str(audio_dir),
            "--ground-truth", str(gt_path),
            "--file-column", "filename",
            "--label-column", "class",
        ])

        assert result.exit_code == 0
        # Verify the custom columns were passed
        mock_evaluate.assert_called_once()
        call_args = mock_evaluate.call_args
        assert call_args.kwargs['gt_file_column'] == 'filename'
        assert call_args.kwargs['gt_label_column'] == 'class'

    @patch('bioamla.evaluate.evaluate_directory')
    def test_evaluate_custom_model(self, mock_evaluate, runner, temp_dir, mock_evaluation_result):
        """Test evaluate with custom model path."""
        mock_evaluate.return_value = mock_evaluation_result

        audio_dir = temp_dir / "audio"
        audio_dir.mkdir()
        gt_path = temp_dir / "labels.csv"
        gt_path.write_text("file_name,label\naudio.wav,cat\n")

        result = runner.invoke(cli, [
            "models", "evaluate", "ast",
            str(audio_dir),
            "--ground-truth", str(gt_path),
            "--model-path", "custom/model",
        ])

        assert result.exit_code == 0
        mock_evaluate.assert_called_once()
        assert mock_evaluate.call_args.kwargs['model_path'] == 'custom/model'

    @patch('bioamla.evaluate.evaluate_directory')
    def test_evaluate_fp16_option(self, mock_evaluate, runner, temp_dir, mock_evaluation_result):
        """Test evaluate with FP16 option."""
        mock_evaluate.return_value = mock_evaluation_result

        audio_dir = temp_dir / "audio"
        audio_dir.mkdir()
        gt_path = temp_dir / "labels.csv"
        gt_path.write_text("file_name,label\naudio.wav,cat\n")

        result = runner.invoke(cli, [
            "models", "evaluate", "ast",
            str(audio_dir),
            "--ground-truth", str(gt_path),
            "--fp16",
        ])

        assert result.exit_code == 0
        assert mock_evaluate.call_args.kwargs['use_fp16'] is True


class TestEvaluateOutputFormats:
    """Tests for different output formats."""

    @patch('bioamla.evaluate.evaluate_directory')
    @patch('bioamla.evaluate.save_evaluation_results')
    def test_json_format(self, mock_save, mock_evaluate, runner, temp_dir, mock_evaluation_result):
        """Test JSON output format."""
        mock_evaluate.return_value = mock_evaluation_result
        mock_save.return_value = str(temp_dir / "results.json")

        audio_dir = temp_dir / "audio"
        audio_dir.mkdir()
        gt_path = temp_dir / "labels.csv"
        gt_path.write_text("file_name,label\naudio.wav,cat\n")

        result = runner.invoke(cli, [
            "models", "evaluate", "ast",
            str(audio_dir),
            "--ground-truth", str(gt_path),
            "--output", str(temp_dir / "results.json"),
            "--format", "json",
        ])

        assert result.exit_code == 0
        mock_save.assert_called_with(mock_evaluation_result, str(temp_dir / "results.json"), format="json")

    @patch('bioamla.evaluate.evaluate_directory')
    @patch('bioamla.evaluate.save_evaluation_results')
    def test_csv_format(self, mock_save, mock_evaluate, runner, temp_dir, mock_evaluation_result):
        """Test CSV output format."""
        mock_evaluate.return_value = mock_evaluation_result
        mock_save.return_value = str(temp_dir / "results.csv")

        audio_dir = temp_dir / "audio"
        audio_dir.mkdir()
        gt_path = temp_dir / "labels.csv"
        gt_path.write_text("file_name,label\naudio.wav,cat\n")

        result = runner.invoke(cli, [
            "models", "evaluate", "ast",
            str(audio_dir),
            "--ground-truth", str(gt_path),
            "--output", str(temp_dir / "results.csv"),
            "--format", "csv",
        ])

        assert result.exit_code == 0
        mock_save.assert_called_with(mock_evaluation_result, str(temp_dir / "results.csv"), format="csv")

    @patch('bioamla.evaluate.evaluate_directory')
    @patch('bioamla.evaluate.save_evaluation_results')
    def test_txt_format(self, mock_save, mock_evaluate, runner, temp_dir, mock_evaluation_result):
        """Test text output format."""
        mock_evaluate.return_value = mock_evaluation_result
        mock_save.return_value = str(temp_dir / "results.txt")

        audio_dir = temp_dir / "audio"
        audio_dir.mkdir()
        gt_path = temp_dir / "labels.csv"
        gt_path.write_text("file_name,label\naudio.wav,cat\n")

        result = runner.invoke(cli, [
            "models", "evaluate", "ast",
            str(audio_dir),
            "--ground-truth", str(gt_path),
            "--output", str(temp_dir / "results.txt"),
            "--format", "txt",
        ])

        assert result.exit_code == 0
        mock_save.assert_called_with(mock_evaluation_result, str(temp_dir / "results.txt"), format="txt")
