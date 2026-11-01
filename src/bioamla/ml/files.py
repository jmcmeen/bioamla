"""AST models on files: classify, evaluate and embed, saving the results.

These are the file-level calls behind ``bioamla models ast predict|evaluate|embed``.

Example:
    >>> from bioamla.ml import classify_file
    >>> classify_file("rec.wav", model_path="bioamla/scp-frogs", segment_duration=3).count
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from bioamla.exceptions import ModelError


def _write_output(path: Path, write: Any) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        write(path)
    except OSError as e:
        raise ModelError(f"Failed to write {path}: {e}") from e


@dataclass
class FilePredictions:
    """AST predictions for one file, whole or per segment.

    Attributes:
        file: The audio file.
        predictions: The predictions (``start_time``, ``end_time``,
            ``predicted_label``, ``confidence``), those below the threshold left out.
        segmented: Whether the file was classified per segment.
        output: The CSV they were saved to, if any.
    """

    file: str
    predictions: list[Any] = field(default_factory=list)
    segmented: bool = False
    output: str | None = None

    @property
    def count(self) -> int:
        """The number of predictions."""
        return len(self.predictions)

    def to_dict(self) -> dict[str, Any]:
        """``{"file", "count", "predictions": [{start_time, end_time, label, confidence}],
        "output"}``."""
        return {
            "file": self.file,
            "count": self.count,
            "predictions": [
                {
                    "start_time": p.start_time,
                    "end_time": p.end_time,
                    "label": p.predicted_label,
                    "confidence": p.confidence,
                }
                for p in self.predictions
            ],
            "output": self.output,
        }


def classify_file(
    file: str | Path,
    *,
    model_path: str = "bioamla/scp-frogs",
    segment_duration: int = 0,
    overlap: int = 0,
    min_confidence: float = 0.0,
    resample_freq: int = 16000,
    output: str | Path | None = None,
) -> FilePredictions:
    """Classify one audio file with an AST model, whole or in segments.

    Args:
        file: The audio file.
        model_path: The model (Hugging Face ID or local path).
        segment_duration: Classify N-second segments (0: the whole file).
        overlap: Overlap between segments (s).
        min_confidence: Leave out predictions below this.
        resample_freq: The model's sample rate.
        output: A CSV to save ``filepath, start, stop, prediction, confidence`` to.

    Raises:
        ModelError: If the model can't be loaded or run, or the CSV can't be written.
    """
    from bioamla.ml.inference import ASTInference

    inference = ASTInference(model_path=model_path, sample_rate=resample_freq)
    if segment_duration > 0:
        results = inference.predict_segments(
            str(file), clip_length=segment_duration, overlap=overlap
        )
    else:
        results = [inference.predict(str(file))]
    kept = [r for r in results if r.confidence >= min_confidence]
    if output:

        def _write(path: Path) -> None:
            with path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["filepath", "start", "stop", "prediction", "confidence"])
                writer.writerows(
                    [r.filepath, r.start_time, r.end_time, r.predicted_label, f"{r.confidence:.6f}"]
                    for r in kept
                )

        _write_output(Path(output), _write)
    return FilePredictions(
        file=str(file),
        predictions=kept,
        segmented=segment_duration > 0,
        output=str(output) if output else None,
    )


def evaluation_text(result: Any) -> str:
    """An evaluation as plain text: overall metrics, a per-class table, the confusion
    matrix (rows true, columns predicted) and skipped files."""
    lines = [
        f"accuracy: {result.accuracy}",
        f"precision: {result.precision}",
        f"recall: {result.recall}",
        f"f1_score: {result.f1_score}",
        f"total_samples: {result.total_samples}",
    ]
    if result.per_class:
        width = max(len("label"), *(len(label) for label in result.per_class))
        lines += ["", f"{'label':<{width}}  precision  recall  f1     support"]
        for label, m in result.per_class.items():
            lines.append(
                f"{label:<{width}}  {m['precision']:<9.3f}  {m['recall']:<6.3f}  "
                f"{m['f1_score']:<5.3f}  {m['support']}"
            )
    if result.confusion_matrix and result.labels:
        width = max(len(label) for label in result.labels)
        lines += ["", "confusion matrix (rows: true, columns: predicted)"]
        lines.append(" " * width + "  " + "  ".join(result.labels))
        for label, row in zip(result.labels, result.confusion_matrix, strict=True):
            cells = "  ".join(f"{n:>{len(col)}}" for n, col in zip(row, result.labels, strict=True))
            lines.append(f"{label:<{width}}  {cells}")
    if result.skipped:
        lines += ["", "skipped:", *(f"  {s}" for s in result.skipped)]
    return "\n".join(lines) + "\n"


def evaluation_csv_rows(result: Any) -> list[dict[str, Any]]:
    """An evaluation as CSV rows: one overall row (with accuracy), then one per class."""
    rows: list[dict[str, Any]] = [
        {
            "label": "(overall)",
            "accuracy": result.accuracy,
            "precision": result.precision,
            "recall": result.recall,
            "f1_score": result.f1_score,
            "support": result.total_samples,
        }
    ]
    for label, m in (result.per_class or {}).items():
        rows.append({"label": label, "accuracy": "", **m})
    return rows


@dataclass
class EvaluationReport:
    """An AST model evaluated against ground truth.

    Attributes:
        path: The audio folder.
        metrics: The evaluation (accuracy, precision, recall, F1, per class, ...).
        output: The report written, if any.
    """

    path: str
    metrics: Any
    output: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """``{"path", "metrics", "output"}``."""
        return {"path": self.path, "metrics": self.metrics.to_dict(), "output": self.output}


def evaluate_model(
    path: str | Path,
    *,
    model_path: str = "bioamla/scp-frogs",
    ground_truth: str | Path,
    file_column: str = "file_name",
    label_column: str = "label",
    resample_freq: int = 16000,
    batch_size: int = 8,
    fp16: bool = False,
    output: str | Path | None = None,
    output_format: str = "txt",
) -> EvaluationReport:
    """Evaluate an AST model on a folder of audio against a ground-truth CSV.

    Args:
        path: The audio folder.
        model_path: The model (Hugging Face ID or local path).
        ground_truth: CSV of file names and their true labels.
        file_column: Its file-name column.
        label_column: Its label column.
        resample_freq: The model's sample rate.
        batch_size: Inference batch size.
        fp16: Half-precision inference.
        output: Where to save the report.
        output_format: ``txt``, ``json`` or ``csv``.

    Raises:
        ModelError: If the model can't be run, or the report can't be written.
    """
    from bioamla.ml.ast_service import evaluate_directory

    result = evaluate_directory(
        audio_dir=str(path),
        model_path=model_path,
        ground_truth_csv=str(ground_truth),
        file_column=file_column,
        label_column=label_column,
        resample_freq=resample_freq,
        use_fp16=fp16,
        batch_size=batch_size,
    )
    if output:
        out = Path(output)
        if output_format == "json":
            _write_output(out, lambda p: p.write_text(json.dumps(result.to_dict(), indent=2)))
        elif output_format == "csv":
            rows = evaluation_csv_rows(result)

            def _csv(p: Path) -> None:
                with p.open("w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
                    writer.writeheader()
                    writer.writerows(rows)

            _write_output(out, _csv)
        else:
            _write_output(out, lambda p: p.write_text(evaluation_text(result), encoding="utf-8"))
    return EvaluationReport(path=str(path), metrics=result, output=str(output) if output else None)


@dataclass
class SavedEmbeddings:
    """An audio file's AST embeddings, saved as ``.npy``.

    Attributes:
        file: The audio file.
        output: The ``.npy`` written (``.npy`` is added to a name without it).
        shape: The array's shape.
        dtype: Its dtype.
    """

    file: str
    output: str
    shape: list[int]
    dtype: str

    def to_dict(self) -> dict[str, Any]:
        """``{"file", "output", "shape", "dtype"}``."""
        return {"file": self.file, "output": self.output, "shape": self.shape, "dtype": self.dtype}


def embed_file(
    file: str | Path,
    output: str | Path,
    *,
    model_path: str,
    layer: str | None = None,
    sample_rate: int = 16000,
) -> SavedEmbeddings:
    """Extract an audio file's AST embeddings and save them as ``.npy``.

    Raises:
        ModelError: If the model can't be run, or the file can't be written.
    """
    import numpy as np

    from bioamla.ml.ast_service import extract_embeddings_file

    embeddings = extract_embeddings_file(
        filepath=str(file), model_path=model_path, layer=layer, sample_rate=sample_rate
    )["embeddings"]
    out = str(output) if str(output).endswith(".npy") else f"{output}.npy"  # as np.save does
    _write_output(Path(out), lambda p: np.save(p, embeddings))
    return SavedEmbeddings(
        file=str(file),
        output=out,
        shape=list(embeddings.shape),
        dtype=str(embeddings.dtype),
    )


__all__ = [
    "EvaluationReport",
    "FilePredictions",
    "SavedEmbeddings",
    "embed_file",
    "evaluate_model",
    "evaluation_csv_rows",
    "evaluation_text",
    "classify_file",
]
