"""
AST Service Logic (internal)
============================

Higher-level AST operations folded from the former ``ASTService``: single-file
prediction, directory evaluation (with metrics), single-file embedding
extraction, and model info. These are plain functions that return data and
raise :class:`~bioamla.exceptions.BioamlaError` subclasses on failure.

This module is internal to :mod:`bioamla.ml`; the curated public API re-exports
the pieces that callers (and the CLI) need.

Heavy deps (torch / transformers / torchaudio / pandas) are imported lazily for
fast startup.
"""

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import TYPE_CHECKING, Any

from bioamla.exceptions import (
    InvalidInputError,
    ModelError,
    NotFoundError,
)
from bioamla.ml.ast import mono_waveform

if TYPE_CHECKING:
    from bioamla.ml.inference import ASTPredictionResult


@dataclass
class EvaluationResult:
    """Result of an AST model evaluation over a labelled directory."""

    accuracy: float
    precision: float
    recall: float
    f1_score: float
    total_samples: int
    #: Rows are true labels, columns predicted labels, both in ``labels`` order.
    confusion_matrix: list[list[int]] | None = None
    #: Every label among the evaluated files' true and predicted labels, sorted.
    labels: list[str] | None = None
    #: label -> ``precision``, ``recall``, ``f1_score`` (floats) and ``support``
    #: (int, the number of evaluated files with that true label).
    per_class: dict[str, dict[str, float | int]] | None = None
    #: Files that matched the ground truth but couldn't be evaluated, with the error.
    skipped: list[str] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to a plain dict."""
        # Every field, leaving out optional ones that weren't computed.
        return {
            f.name: getattr(self, f.name) for f in fields(self) if getattr(self, f.name) is not None
        }


@dataclass
class TrainProgress:
    """Progress of a running :func:`~bioamla.ml.train_ast` call.

    Attributes:
        step: Optimizer steps completed.
        total_steps: Steps the run will take in total.
        epoch: Fractional epoch reached (e.g. ``1.5``).
        loss: Most recent logged training loss, if any yet.
        eval_metrics: Most recent evaluation metrics (``eval_accuracy``, ...).
    """

    step: int
    total_steps: int
    epoch: float
    loss: float | None = None
    eval_metrics: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to a plain dict."""
        return {
            "step": self.step,
            "total_steps": self.total_steps,
            "epoch": self.epoch,
            "loss": self.loss,
            "eval_metrics": dict(self.eval_metrics),
        }


@dataclass
class TrainResult:
    """Result of an AST model training run."""

    model_path: str
    epochs: int
    final_accuracy: float | None = None
    final_loss: float | None = None
    #: The run's identifier, also on its log records (``run_id`` field).
    run_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to a plain dict."""
        return {
            "model_path": self.model_path,
            "epochs": self.epochs,
            "final_accuracy": self.final_accuracy,
            "final_loss": self.final_loss,
            "run_id": self.run_id,
        }


def predict_file(
    filepath: str,
    model_path: str = "bioamla/scp-frogs",
    resample_freq: int = 16000,
) -> "ASTPredictionResult":
    """
    Run AST prediction on a single audio file (whole-file).

    Args:
        filepath: Path to the audio file.
        model_path: Path to model or HuggingFace identifier.
        resample_freq: Target sample rate.

    Returns:
        An :class:`~bioamla.ml.inference.ASTPredictionResult`.

    Raises:
        NotFoundError: If the file does not exist.
        ModelError: On inference failure.
    """
    if not Path(filepath).exists():
        raise NotFoundError(f"Audio file not found: {filepath}")

    from bioamla.ml.inference import ASTInference

    inference = ASTInference(model_path=model_path, sample_rate=resample_freq)
    return inference.predict(filepath)


def extract_embeddings_file(
    filepath: str,
    model_path: str,
    layer: str | None = None,
    sample_rate: int = 16000,
) -> dict[str, Any]:
    """
    Extract AST embeddings from a single file (CLS-token of the base model).

    Mirrors the old ``ASTService.extract_embeddings`` behavior: uses
    ``AutoModel`` and the CLS token of the last hidden state.

    Args:
        filepath: Path to the audio file.
        model_path: Path to model or HuggingFace identifier.
        layer: Unused placeholder kept for API compatibility.
        sample_rate: Target sample rate.

    Returns:
        A dict with ``filepath``, ``embeddings`` (numpy array), ``shape``, and
        ``dtype``.

    Raises:
        NotFoundError: If the file does not exist.
        ModelError: On model-load or inference failure.
    """
    if not Path(filepath).exists():
        raise NotFoundError(f"Audio file not found: {filepath}")

    import torch
    from transformers import ASTFeatureExtractor, AutoModel

    from bioamla.audio.torchaudio import load_waveform_tensor, resample_waveform_tensor

    try:
        waveform, orig_sr = load_waveform_tensor(filepath)
    except Exception as e:
        raise ModelError(f"Failed to load audio {filepath}: {e}") from e

    if orig_sr != sample_rate:
        waveform = resample_waveform_tensor(waveform, orig_sr, sample_rate)

    try:
        feature_extractor = ASTFeatureExtractor.from_pretrained(model_path)
    except OSError:
        feature_extractor = ASTFeatureExtractor()

    try:
        model = AutoModel.from_pretrained(model_path)
    except Exception as e:
        raise ModelError(f"Failed to load AST model from {model_path}: {e}") from e

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    inputs = feature_extractor(
        mono_waveform(waveform),
        sampling_rate=sample_rate,
        padding="max_length",
        return_tensors="pt",
    ).to(device)

    try:
        with torch.inference_mode():
            outputs = model(**inputs)
            embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
    except Exception as e:
        raise ModelError(f"AST embedding extraction failed: {e}") from e

    return {
        "filepath": filepath,
        "embeddings": embeddings,
        "shape": embeddings.shape,
        "dtype": str(embeddings.dtype),
    }


def get_model_info(model_path: str) -> dict[str, Any]:
    """
    Return lightweight info about an AST model from its config.

    Args:
        model_path: Path to model or HuggingFace identifier.

    Returns:
        A dict with ``path``, ``model_type``, ``num_classes``, ``classes`` (the
        first 10, for display), ``has_more_classes``, ``labels`` (all class labels,
        in output order) and ``hidden_size``.

    Raises:
        ModelError: On model-load or inference failure.
    """
    from bioamla.ml.embedding import get_ast_model_info

    return get_ast_model_info(model_path)


# =============================================================================
# Directory Evaluation
# =============================================================================


def evaluate_directory(
    audio_dir: str,
    model_path: str,
    ground_truth_csv: str,
    file_column: str = "file_name",
    label_column: str = "label",
    resample_freq: int = 16000,
    use_fp16: bool = False,
    batch_size: int = 8,
) -> EvaluationResult:
    """
    Evaluate an AST model over a directory of audio files with ground truth.

    Files are classified ``batch_size`` at a time (multi-channel audio is
    downmixed to mono). A file that can't be loaded, or that fails inference on
    its own after its batch fails, is listed in ``skipped`` and left out of the
    metrics.

    Args:
        audio_dir: Directory containing audio files.
        model_path: Path to model or HuggingFace identifier.
        ground_truth_csv: CSV with ground-truth labels.
        file_column: Column name for file names in the CSV.
        label_column: Column name for labels in the CSV.
        resample_freq: Target sample rate.
        use_fp16: Use half-precision inference.
        batch_size: Files per inference batch.

    Returns:
        An :class:`EvaluationResult`, with per-class metrics and the confusion
        matrix.

    Raises:
        NotFoundError: If a required path or matching audio is missing.
        InvalidInputError: If the CSV is malformed.
        ModelError: On model-load or inference failure.
    """
    if not Path(audio_dir).exists():
        raise NotFoundError(f"Audio directory not found: {audio_dir}")
    if not Path(ground_truth_csv).exists():
        raise NotFoundError(f"Ground truth file not found: {ground_truth_csv}")

    import torch

    from bioamla.audio.torchaudio import load_waveform_tensor, resample_waveform_tensor
    from bioamla.ml.ast import (
        ast_predict_batch,
        extract_features,
        get_cached_feature_extractor,
        load_pretrained_ast_model,
    )

    if batch_size < 1:
        raise InvalidInputError(f"batch_size must be at least 1, got {batch_size}")

    audio_path = Path(audio_dir)
    ground_truth = _load_ground_truth(ground_truth_csv, file_column, label_column)

    audio_extensions = [".wav", ".mp3", ".flac", ".ogg", ".m4a"]
    audio_files = []
    for ext in audio_extensions:
        audio_files.extend(audio_path.rglob(f"*{ext}"))

    matched_files = [(f, ground_truth[f.name]) for f in audio_files if f.name in ground_truth]
    if not matched_files:
        raise NotFoundError("No audio files found that match ground truth labels")

    model = load_pretrained_ast_model(model_path, use_fp16=use_fp16)
    feature_extractor = get_cached_feature_extractor(model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    y_true: list[str] = []
    y_pred: list[str] = []
    skipped: list[str] = []
    pending: list[tuple[Path, Any, str]] = []  # (file, features, true label)

    def _predict(features: list[Any]) -> list[str]:
        batch = torch.cat(features, dim=0)
        if use_fp16 and device.type == "cuda":
            batch = batch.half()
        predictions = ast_predict_batch(batch, model)
        if len(predictions) != len(features):
            raise ModelError(
                f"The model returned {len(predictions)} prediction(s) for {len(features)} input(s)"
            )
        return predictions

    def _flush() -> None:
        try:
            predictions = _predict([features for _, features, _ in pending])
        except ModelError:
            # Retry one file at a time, so one bad input (or a batch too big for
            # the device) costs only that file, not the run.
            predictions = []
            for audio_file, features, _ in pending:
                try:
                    predictions.extend(_predict([features]))
                except ModelError as e:
                    skipped.append(f"{audio_file}: {e}")
                    predictions.append(None)
        for (_, _, true_label), prediction in zip(pending, predictions, strict=True):
            if prediction is not None:
                y_true.append(true_label)
                y_pred.append(prediction)
        pending.clear()

    for audio_file, true_label in matched_files:
        try:
            waveform, orig_freq = load_waveform_tensor(str(audio_file))
            waveform = resample_waveform_tensor(waveform, orig_freq, resample_freq)
            features = extract_features(waveform, resample_freq, feature_extractor, device)
            if features.shape[0] != 1:
                raise ModelError(f"expected one input per file, got {features.shape[0]}")
        except Exception as e:  # noqa: BLE001 - an unreadable file is reported, not fatal
            skipped.append(f"{audio_file}: {e}")
            continue
        pending.append((audio_file, features, true_label))
        if len(pending) == batch_size:
            _flush()
    if pending:
        _flush()

    if not y_true:
        raise ModelError(
            f"No predictions were generated ({len(skipped)} file(s) skipped; first: "
            f"{skipped[0] if skipped else 'none'})"
        )

    metrics = _compute_metrics(y_true, y_pred)
    return EvaluationResult(
        accuracy=metrics["accuracy"],
        precision=metrics["precision"],
        recall=metrics["recall"],
        f1_score=metrics["f1_score"],
        total_samples=metrics["total_samples"],
        confusion_matrix=metrics["confusion_matrix"],
        labels=metrics["labels"],
        per_class=metrics["per_class"],
        skipped=skipped,
    )


def _load_ground_truth(csv_path: str, file_column: str, label_column: str) -> dict[str, str]:
    """Load ground-truth labels from a CSV (keyed by base file name)."""
    import pandas as pd

    df = pd.read_csv(csv_path)

    if file_column not in df.columns:
        raise InvalidInputError(
            f"Column '{file_column}' not found in CSV. Available: {list(df.columns)}"
        )
    if label_column not in df.columns:
        raise InvalidInputError(
            f"Column '{label_column}' not found in CSV. Available: {list(df.columns)}"
        )

    ground_truth = {}
    for _, row in df.iterrows():
        file_name = Path(row[file_column]).name
        ground_truth[file_name] = str(row[label_column])
    return ground_truth


def _compute_metrics(y_true: list[str], y_pred: list[str]) -> dict[str, Any]:
    """Compute accuracy plus macro precision/recall/F1 from label lists."""
    import numpy as np

    if len(y_true) != len(y_pred):
        raise InvalidInputError(
            f"Length mismatch: y_true ({len(y_true)}) != y_pred ({len(y_pred)})"
        )

    labels = sorted(set(y_true) | set(y_pred))
    label_to_idx = {label: idx for idx, label in enumerate(labels)}
    n_classes = len(labels)

    confusion = np.zeros((n_classes, n_classes), dtype=np.int64)
    for true_label, pred_label in zip(y_true, y_pred, strict=False):
        if true_label in label_to_idx and pred_label in label_to_idx:
            confusion[label_to_idx[true_label], label_to_idx[pred_label]] += 1

    total_samples = len(y_true)
    correct = sum(1 for t, p in zip(y_true, y_pred, strict=False) if t == p)
    accuracy = correct / total_samples if total_samples > 0 else 0.0

    precisions = []
    recalls = []
    f1_scores = []
    per_class: dict[str, dict[str, float]] = {}
    for idx in range(n_classes):
        tp = confusion[idx, idx]
        fp = confusion[:, idx].sum() - tp
        fn = confusion[idx, :].sum() - tp

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        per_class[labels[idx]] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "support": int(confusion[idx, :].sum()),
        }

        if confusion[idx, :].sum() > 0:
            precisions.append(precision)
            recalls.append(recall)
            f1_scores.append(f1)

    return {
        "accuracy": accuracy,
        "precision": float(np.mean(precisions)) if precisions else 0.0,
        "recall": float(np.mean(recalls)) if recalls else 0.0,
        "f1_score": float(np.mean(f1_scores)) if f1_scores else 0.0,
        "total_samples": total_samples,
        "correct_predictions": correct,
        "labels": labels,
        "per_class": per_class,
        "confusion_matrix": confusion.tolist(),
    }
