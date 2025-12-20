"""
Model Evaluation Module
=======================

This module provides functions for evaluating audio classification models,
computing metrics (accuracy, precision, recall, F1), and generating
confusion matrices.

Key features:
- Compute classification metrics from predictions and ground truth
- Load ground truth labels from CSV files
- Evaluate predictions from CSV output files
- Evaluate models directly on audio directories
- Generate formatted evaluation reports
- Save results in JSON, CSV, or text formats

Example:
    >>> from bioamla.evaluate import evaluate_directory, format_metrics_report
    >>> result = evaluate_directory(
    ...     audio_dir="./test_audio",
    ...     model_path="./my_model",
    ...     ground_truth_csv="./labels.csv"
    ... )
    >>> print(format_metrics_report(result))
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from bioamla.core.files import TextFile


@dataclass
class EvaluationResult:
    """
    Results from model evaluation.

    Contains overall metrics, per-class metrics, and confusion matrix
    for a classification evaluation.

    Attributes:
        accuracy: Overall accuracy (correct predictions / total samples).
        precision: Macro-averaged precision across all classes.
        recall: Macro-averaged recall across all classes.
        f1_score: Macro-averaged F1 score across all classes.
        confusion_matrix: NxN numpy array where rows are true labels
            and columns are predicted labels.
        class_labels: Ordered list of class label names.
        per_class_metrics: Dictionary mapping class names to dictionaries
            containing 'precision', 'recall', 'f1_score', and 'support'.
        total_samples: Total number of samples evaluated.
        correct_predictions: Number of correctly classified samples.
    """

    accuracy: float
    precision: float
    recall: float
    f1_score: float
    confusion_matrix: np.ndarray
    class_labels: List[str]
    per_class_metrics: Dict[str, Dict[str, float]]
    total_samples: int
    correct_predictions: int


def compute_metrics(
    y_true: List[str],
    y_pred: List[str],
    labels: Optional[List[str]] = None,
) -> EvaluationResult:
    """
    Compute classification metrics from predictions and ground truth.

    Calculates accuracy, precision, recall, F1 score (macro-averaged),
    per-class metrics, and confusion matrix.

    Args:
        y_true: List of ground truth labels.
        y_pred: List of predicted labels. Must have same length as y_true.
        labels: Optional list of all possible labels for ordering the
            confusion matrix. If None, labels are derived from the data.

    Returns:
        EvaluationResult containing all computed metrics.

    Raises:
        ValueError: If y_true and y_pred have different lengths.
        ValueError: If y_true or y_pred are empty.

    Example:
        >>> result = compute_metrics(
        ...     y_true=["cat", "dog", "cat"],
        ...     y_pred=["cat", "cat", "cat"]
        ... )
        >>> print(f"Accuracy: {result.accuracy:.2f}")
        Accuracy: 0.67
    """
    if len(y_true) != len(y_pred):
        raise ValueError(f"Length mismatch: y_true ({len(y_true)}) != y_pred ({len(y_pred)})")

    if len(y_true) == 0:
        raise ValueError("Cannot compute metrics on empty data")

    # Get unique labels
    if labels is None:
        labels = sorted(set(y_true) | set(y_pred))

    # Build label to index mapping
    label_to_idx = {label: idx for idx, label in enumerate(labels)}
    n_classes = len(labels)

    # Build confusion matrix
    confusion = np.zeros((n_classes, n_classes), dtype=np.int64)
    for true_label, pred_label in zip(y_true, y_pred):
        if true_label in label_to_idx and pred_label in label_to_idx:
            true_idx = label_to_idx[true_label]
            pred_idx = label_to_idx[pred_label]
            confusion[true_idx, pred_idx] += 1

    # Compute overall accuracy
    total_samples = len(y_true)
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    accuracy = correct / total_samples if total_samples > 0 else 0.0

    # Compute per-class metrics
    per_class_metrics = {}
    precisions = []
    recalls = []
    f1_scores = []

    for idx, label in enumerate(labels):
        tp = confusion[idx, idx]
        fp = confusion[:, idx].sum() - tp
        fn = confusion[idx, :].sum() - tp

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        per_class_metrics[label] = {
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "support": int(confusion[idx, :].sum()),
        }

        # Weight by support for macro average
        support = confusion[idx, :].sum()
        if support > 0:
            precisions.append(precision)
            recalls.append(recall)
            f1_scores.append(f1)

    # Compute macro-averaged metrics
    macro_precision = np.mean(precisions) if precisions else 0.0
    macro_recall = np.mean(recalls) if recalls else 0.0
    macro_f1 = np.mean(f1_scores) if f1_scores else 0.0

    return EvaluationResult(
        accuracy=accuracy,
        precision=macro_precision,
        recall=macro_recall,
        f1_score=macro_f1,
        confusion_matrix=confusion,
        class_labels=labels,
        per_class_metrics=per_class_metrics,
        total_samples=total_samples,
        correct_predictions=correct,
    )


def load_ground_truth(
    csv_path: str,
    file_column: str = "file_name",
    label_column: str = "label",
) -> Dict[str, str]:
    """
    Load ground truth labels from a CSV file.

    Reads a CSV file and extracts a mapping from filenames to labels.
    File paths in the CSV are normalized to just the filename (basename)
    for matching.

    Args:
        csv_path: Path to the CSV file containing ground truth labels.
        file_column: Name of the column containing file names or paths.
        label_column: Name of the column containing ground truth labels.

    Returns:
        Dictionary mapping filenames (without path) to their labels.

    Raises:
        ValueError: If the specified columns are not found in the CSV.
        FileNotFoundError: If the CSV file does not exist.
    """
    df = pd.read_csv(csv_path)

    if file_column not in df.columns:
        raise ValueError(f"Column '{file_column}' not found in CSV. Available: {list(df.columns)}")
    if label_column not in df.columns:
        raise ValueError(f"Column '{label_column}' not found in CSV. Available: {list(df.columns)}")

    ground_truth = {}
    for _, row in df.iterrows():
        # Extract just the filename (without path) for matching
        file_name = Path(row[file_column]).name
        ground_truth[file_name] = str(row[label_column])

    return ground_truth


def evaluate_predictions(
    predictions_csv: str,
    ground_truth_csv: str,
    pred_file_column: str = "filepath",
    pred_label_column: str = "prediction",
    gt_file_column: str = "file_name",
    gt_label_column: str = "label",
) -> EvaluationResult:
    """
    Evaluate predictions against ground truth from CSV files.

    Args:
        predictions_csv: Path to predictions CSV (from ast predict --batch)
        ground_truth_csv: Path to ground truth CSV
        pred_file_column: Column name for file paths in predictions
        pred_label_column: Column name for predictions
        gt_file_column: Column name for file names in ground truth
        gt_label_column: Column name for labels in ground truth

    Returns:
        EvaluationResult with computed metrics
    """
    # Load ground truth
    ground_truth = load_ground_truth(ground_truth_csv, gt_file_column, gt_label_column)

    # Load predictions
    pred_df = pd.read_csv(predictions_csv)

    if pred_file_column not in pred_df.columns:
        raise ValueError(f"Column '{pred_file_column}' not found in predictions CSV")
    if pred_label_column not in pred_df.columns:
        raise ValueError(f"Column '{pred_label_column}' not found in predictions CSV")

    # Match predictions to ground truth
    y_true = []
    y_pred = []

    for _, row in pred_df.iterrows():
        file_name = Path(row[pred_file_column]).name
        if file_name in ground_truth:
            y_true.append(ground_truth[file_name])
            y_pred.append(str(row[pred_label_column]))

    if not y_true:
        raise ValueError("No matching files found between predictions and ground truth")

    return compute_metrics(y_true, y_pred)


def evaluate_directory(
    audio_dir: str,
    model_path: str,
    ground_truth_csv: str,
    gt_file_column: str = "file_name",
    gt_label_column: str = "label",
    resample_freq: int = 16000,
    batch_size: int = 8,
    use_fp16: bool = False,
    verbose: bool = True,
) -> EvaluationResult:
    """
    Evaluate a model on a directory of audio files.

    Args:
        audio_dir: Path to directory containing audio files
        model_path: Path to the AST model
        ground_truth_csv: Path to CSV with ground truth labels
        gt_file_column: Column name for file names in ground truth
        gt_label_column: Column name for labels in ground truth
        resample_freq: Target sample rate for audio
        batch_size: Batch size for inference
        use_fp16: Use half-precision inference
        verbose: Print progress information

    Returns:
        EvaluationResult with computed metrics
    """
    import torch

    from bioamla.core.audio.torchaudio import load_waveform_tensor, resample_waveform_tensor
    from bioamla.detection.ast import (
        ast_predict,
        extract_features,
        get_cached_feature_extractor,
        load_pretrained_ast_model,
    )

    audio_dir = Path(audio_dir)
    if not audio_dir.exists():
        raise FileNotFoundError(f"Audio directory not found: {audio_dir}")

    # Load ground truth
    ground_truth = load_ground_truth(ground_truth_csv, gt_file_column, gt_label_column)

    if verbose:
        print(f"Loaded {len(ground_truth)} ground truth labels")

    # Find audio files that have ground truth
    audio_extensions = [".wav", ".mp3", ".flac", ".ogg", ".m4a"]
    audio_files = []
    for ext in audio_extensions:
        audio_files.extend(audio_dir.rglob(f"*{ext}"))

    # Filter to only files with ground truth
    matched_files = [(f, ground_truth[f.name]) for f in audio_files if f.name in ground_truth]

    if not matched_files:
        raise ValueError("No audio files found that match ground truth labels")

    if verbose:
        print(f"Found {len(matched_files)} audio files with ground truth labels")

    # Load model
    model = load_pretrained_ast_model(model_path)
    feature_extractor = get_cached_feature_extractor(model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if use_fp16 and device.type == "cuda":
        model = model.half()

    if verbose:
        print(f"Running inference on {device}...")

    # Run inference
    y_true = []
    y_pred = []

    for i, (audio_path, true_label) in enumerate(matched_files):
        try:
            waveform, orig_freq = load_waveform_tensor(str(audio_path))
            waveform = resample_waveform_tensor(waveform, orig_freq, resample_freq)
            input_values = extract_features(waveform, resample_freq, feature_extractor, device)

            if use_fp16 and device.type == "cuda":
                input_values = input_values.half()

            prediction = ast_predict(input_values, model)

            y_true.append(true_label)
            y_pred.append(prediction)

            if verbose and (i + 1) % 10 == 0:
                print(f"  Processed {i + 1}/{len(matched_files)} files")

        except Exception as e:
            if verbose:
                print(f"  Error processing {audio_path}: {e}")

    if not y_true:
        raise ValueError("No predictions were generated")

    if verbose:
        print(f"Completed inference on {len(y_true)} files")

    return compute_metrics(y_true, y_pred)


def format_metrics_report(result: EvaluationResult, include_per_class: bool = True) -> str:
    """
    Format evaluation results as a human-readable report.

    Args:
        result: EvaluationResult to format
        include_per_class: Include per-class metrics breakdown

    Returns:
        Formatted string report
    """
    lines = [
        "=" * 60,
        "Model Evaluation Report",
        "=" * 60,
        "",
        "Overall Metrics:",
        f"  Accuracy:  {result.accuracy:.4f} ({result.correct_predictions}/{result.total_samples})",
        f"  Precision: {result.precision:.4f} (macro-averaged)",
        f"  Recall:    {result.recall:.4f} (macro-averaged)",
        f"  F1 Score:  {result.f1_score:.4f} (macro-averaged)",
        "",
    ]

    if include_per_class and result.per_class_metrics:
        lines.extend(
            [
                "Per-Class Metrics:",
                "-" * 60,
                f"{'Class':<30} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>8}",
                "-" * 60,
            ]
        )

        for label in result.class_labels:
            metrics = result.per_class_metrics[label]
            lines.append(
                f"{label:<30} {metrics['precision']:>10.4f} {metrics['recall']:>10.4f} "
                f"{metrics['f1_score']:>10.4f} {metrics['support']:>8}"
            )

        lines.append("-" * 60)

    lines.extend(
        [
            "",
            "Confusion Matrix:",
            "-" * 60,
        ]
    )

    # Add column headers
    max_label_len = max(len(str(l)) for l in result.class_labels) if result.class_labels else 10
    header = " " * (max_label_len + 2) + "  ".join(f"{l[:8]:>8}" for l in result.class_labels)
    lines.append(header)

    # Add rows
    for i, label in enumerate(result.class_labels):
        row_values = "  ".join(f"{v:>8}" for v in result.confusion_matrix[i])
        lines.append(f"{label:<{max_label_len}}  {row_values}")

    lines.append("=" * 60)

    return "\n".join(lines)


def save_evaluation_results(
    result: EvaluationResult,
    output_path: str,
    format: str = "json",
) -> str:
    """
    Save evaluation results to a file.

    Args:
        result: EvaluationResult to save
        output_path: Path to save results
        format: Output format ('json', 'csv', or 'txt')

    Returns:
        Path to saved file
    """
    import json

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if format == "json":
        data = {
            "accuracy": result.accuracy,
            "precision": result.precision,
            "recall": result.recall,
            "f1_score": result.f1_score,
            "total_samples": result.total_samples,
            "correct_predictions": result.correct_predictions,
            "class_labels": result.class_labels,
            "per_class_metrics": result.per_class_metrics,
            "confusion_matrix": result.confusion_matrix.tolist(),
        }
        with TextFile(output_path, mode="w") as f:
            json.dump(data, f.handle, indent=2)

    elif format == "csv":
        # Save per-class metrics as CSV
        rows = []
        for label in result.class_labels:
            metrics = result.per_class_metrics[label]
            rows.append(
                {
                    "class": label,
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "f1_score": metrics["f1_score"],
                    "support": metrics["support"],
                }
            )
        df = pd.DataFrame(rows)
        df.to_csv(output_path, index=False)

    elif format == "txt":
        report = format_metrics_report(result)
        with TextFile(output_path, mode="w") as f:
            f.write(report)

    else:
        raise ValueError(f"Unsupported format: {format}. Use 'json', 'csv', or 'txt'")

    return str(output_path)
