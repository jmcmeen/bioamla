"""
Unit tests for bioamla.core.evaluate module.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from bioamla.core.evaluate import (
    EvaluationResult,
    compute_metrics,
    load_ground_truth,
    evaluate_predictions,
    format_metrics_report,
    save_evaluation_results,
)


class TestComputeMetrics:
    """Tests for compute_metrics function."""

    def test_perfect_predictions(self):
        """Test metrics with perfect predictions."""
        y_true = ["cat", "dog", "bird", "cat", "dog"]
        y_pred = ["cat", "dog", "bird", "cat", "dog"]

        result = compute_metrics(y_true, y_pred)

        assert result.accuracy == 1.0
        assert result.precision == 1.0
        assert result.recall == 1.0
        assert result.f1_score == 1.0
        assert result.total_samples == 5
        assert result.correct_predictions == 5

    def test_all_wrong_predictions(self):
        """Test metrics with all wrong predictions."""
        y_true = ["cat", "cat", "cat"]
        y_pred = ["dog", "dog", "dog"]

        result = compute_metrics(y_true, y_pred)

        assert result.accuracy == 0.0
        assert result.correct_predictions == 0

    def test_partial_correct_predictions(self):
        """Test metrics with some correct predictions."""
        y_true = ["cat", "dog", "bird", "cat"]
        y_pred = ["cat", "dog", "cat", "dog"]  # 2 correct, 2 wrong

        result = compute_metrics(y_true, y_pred)

        assert result.accuracy == 0.5
        assert result.correct_predictions == 2
        assert result.total_samples == 4

    def test_confusion_matrix_shape(self):
        """Test confusion matrix has correct shape."""
        y_true = ["a", "b", "c", "a", "b"]
        y_pred = ["a", "b", "c", "b", "a"]

        result = compute_metrics(y_true, y_pred)

        assert result.confusion_matrix.shape == (3, 3)
        assert len(result.class_labels) == 3

    def test_confusion_matrix_values(self):
        """Test confusion matrix values are correct."""
        y_true = ["cat", "cat", "dog", "dog"]
        y_pred = ["cat", "dog", "cat", "dog"]

        result = compute_metrics(y_true, y_pred)

        # With sorted labels: cat=0, dog=1
        # cat predicted as cat: 1, cat predicted as dog: 1
        # dog predicted as cat: 1, dog predicted as dog: 1
        assert result.confusion_matrix[0, 0] == 1  # cat -> cat
        assert result.confusion_matrix[0, 1] == 1  # cat -> dog
        assert result.confusion_matrix[1, 0] == 1  # dog -> cat
        assert result.confusion_matrix[1, 1] == 1  # dog -> dog

    def test_per_class_metrics(self):
        """Test per-class metrics are computed."""
        y_true = ["cat", "cat", "dog", "dog", "dog"]
        y_pred = ["cat", "cat", "dog", "dog", "cat"]

        result = compute_metrics(y_true, y_pred)

        assert "cat" in result.per_class_metrics
        assert "dog" in result.per_class_metrics
        assert "precision" in result.per_class_metrics["cat"]
        assert "recall" in result.per_class_metrics["cat"]
        assert "f1_score" in result.per_class_metrics["cat"]
        assert "support" in result.per_class_metrics["cat"]

    def test_custom_labels(self):
        """Test using custom label ordering."""
        y_true = ["a", "b"]
        y_pred = ["a", "b"]
        labels = ["b", "a", "c"]  # Include extra label

        result = compute_metrics(y_true, y_pred, labels=labels)

        assert result.class_labels == ["b", "a", "c"]
        assert result.confusion_matrix.shape == (3, 3)

    def test_raises_on_length_mismatch(self):
        """Test error when y_true and y_pred have different lengths."""
        y_true = ["a", "b", "c"]
        y_pred = ["a", "b"]

        with pytest.raises(ValueError, match="Length mismatch"):
            compute_metrics(y_true, y_pred)

    def test_raises_on_empty_data(self):
        """Test error on empty data."""
        with pytest.raises(ValueError, match="empty data"):
            compute_metrics([], [])


class TestLoadGroundTruth:
    """Tests for load_ground_truth function."""

    def test_loads_csv(self, temp_dir):
        """Test loading ground truth from CSV."""
        csv_path = temp_dir / "labels.csv"
        csv_path.write_text("file_name,label\naudio1.wav,cat\naudio2.wav,dog\n")

        result = load_ground_truth(str(csv_path))

        assert result["audio1.wav"] == "cat"
        assert result["audio2.wav"] == "dog"

    def test_custom_columns(self, temp_dir):
        """Test loading with custom column names."""
        csv_path = temp_dir / "labels.csv"
        csv_path.write_text("filename,class\ntest.wav,bird\n")

        result = load_ground_truth(str(csv_path), file_column="filename", label_column="class")

        assert result["test.wav"] == "bird"

    def test_extracts_filename_from_path(self, temp_dir):
        """Test that full paths are reduced to filenames."""
        csv_path = temp_dir / "labels.csv"
        csv_path.write_text("file_name,label\n/path/to/audio.wav,cat\n")

        result = load_ground_truth(str(csv_path))

        assert "audio.wav" in result

    def test_raises_on_missing_column(self, temp_dir):
        """Test error when column is missing."""
        csv_path = temp_dir / "labels.csv"
        csv_path.write_text("file_name,category\naudio.wav,cat\n")

        with pytest.raises(ValueError, match="Column 'label' not found"):
            load_ground_truth(str(csv_path))


class TestEvaluatePredictions:
    """Tests for evaluate_predictions function."""

    def test_evaluates_from_csv_files(self, temp_dir):
        """Test evaluating predictions from CSV files."""
        # Create predictions CSV
        pred_path = temp_dir / "predictions.csv"
        pred_path.write_text("filepath,start,stop,prediction\n"
                            "/audio/file1.wav,0,1,cat\n"
                            "/audio/file2.wav,0,1,dog\n")

        # Create ground truth CSV
        gt_path = temp_dir / "ground_truth.csv"
        gt_path.write_text("file_name,label\nfile1.wav,cat\nfile2.wav,dog\n")

        result = evaluate_predictions(str(pred_path), str(gt_path))

        assert result.accuracy == 1.0
        assert result.total_samples == 2

    def test_handles_partial_matches(self, temp_dir):
        """Test handling when only some files match."""
        pred_path = temp_dir / "predictions.csv"
        pred_path.write_text("filepath,prediction\nfile1.wav,cat\nfile2.wav,dog\nfile3.wav,bird\n")

        gt_path = temp_dir / "ground_truth.csv"
        gt_path.write_text("file_name,label\nfile1.wav,cat\nfile2.wav,cat\n")

        result = evaluate_predictions(str(pred_path), str(gt_path))

        assert result.total_samples == 2  # Only 2 files match

    def test_raises_on_no_matches(self, temp_dir):
        """Test error when no files match."""
        pred_path = temp_dir / "predictions.csv"
        pred_path.write_text("filepath,prediction\nfileA.wav,cat\n")

        gt_path = temp_dir / "ground_truth.csv"
        gt_path.write_text("file_name,label\nfileB.wav,dog\n")

        with pytest.raises(ValueError, match="No matching files"):
            evaluate_predictions(str(pred_path), str(gt_path))


class TestFormatMetricsReport:
    """Tests for format_metrics_report function."""

    def test_includes_overall_metrics(self):
        """Test report includes overall metrics."""
        result = EvaluationResult(
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

        report = format_metrics_report(result)

        assert "Accuracy:" in report
        assert "0.85" in report
        assert "Precision:" in report
        assert "Recall:" in report
        assert "F1 Score:" in report

    def test_includes_per_class_metrics(self):
        """Test report includes per-class metrics."""
        result = EvaluationResult(
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

        report = format_metrics_report(result, include_per_class=True)

        assert "cat" in report
        assert "dog" in report
        assert "Per-Class Metrics:" in report

    def test_includes_confusion_matrix(self):
        """Test report includes confusion matrix."""
        result = EvaluationResult(
            accuracy=0.85,
            precision=0.80,
            recall=0.75,
            f1_score=0.77,
            confusion_matrix=np.array([[10, 2], [3, 15]]),
            class_labels=["cat", "dog"],
            per_class_metrics={},
            total_samples=30,
            correct_predictions=25,
        )

        report = format_metrics_report(result)

        assert "Confusion Matrix:" in report


class TestSaveEvaluationResults:
    """Tests for save_evaluation_results function."""

    def test_saves_as_json(self, temp_dir):
        """Test saving results as JSON."""
        result = EvaluationResult(
            accuracy=0.85,
            precision=0.80,
            recall=0.75,
            f1_score=0.77,
            confusion_matrix=np.array([[10, 2], [3, 15]]),
            class_labels=["cat", "dog"],
            per_class_metrics={
                "cat": {"precision": 0.77, "recall": 0.83, "f1_score": 0.80, "support": 12},
            },
            total_samples=30,
            correct_predictions=25,
        )

        output_path = temp_dir / "results.json"
        save_evaluation_results(result, str(output_path), format="json")

        assert output_path.exists()
        with open(output_path) as f:
            data = json.load(f)
        assert data["accuracy"] == 0.85
        assert "confusion_matrix" in data

    def test_saves_as_csv(self, temp_dir):
        """Test saving results as CSV."""
        result = EvaluationResult(
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

        output_path = temp_dir / "results.csv"
        save_evaluation_results(result, str(output_path), format="csv")

        assert output_path.exists()
        content = output_path.read_text()
        assert "class" in content
        assert "precision" in content

    def test_saves_as_txt(self, temp_dir):
        """Test saving results as text report."""
        result = EvaluationResult(
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

        output_path = temp_dir / "results.txt"
        save_evaluation_results(result, str(output_path), format="txt")

        assert output_path.exists()
        content = output_path.read_text()
        assert "Model Evaluation Report" in content

    def test_creates_output_directory(self, temp_dir):
        """Test that output directory is created if it doesn't exist."""
        result = EvaluationResult(
            accuracy=0.85,
            precision=0.80,
            recall=0.75,
            f1_score=0.77,
            confusion_matrix=np.array([[1]]),
            class_labels=["cat"],
            per_class_metrics={},
            total_samples=1,
            correct_predictions=1,
        )

        output_path = temp_dir / "nested" / "dir" / "results.json"
        save_evaluation_results(result, str(output_path), format="json")

        assert output_path.exists()

    def test_raises_on_invalid_format(self, temp_dir):
        """Test error on invalid format."""
        result = EvaluationResult(
            accuracy=0.85,
            precision=0.80,
            recall=0.75,
            f1_score=0.77,
            confusion_matrix=np.array([[1]]),
            class_labels=["cat"],
            per_class_metrics={},
            total_samples=1,
            correct_predictions=1,
        )

        with pytest.raises(ValueError, match="Unsupported format"):
            save_evaluation_results(result, str(temp_dir / "out.xyz"), format="xyz")
