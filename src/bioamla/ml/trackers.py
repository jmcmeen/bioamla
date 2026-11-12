"""Experiment-tracker adapters for :func:`bioamla.ml.train_ast`.

Pass one (or a list) as ``train_ast(tracker=...)``. Each satisfies
:class:`bioamla.protocols.ExperimentTracker`; write your own to report to any other
system. MLflow and TensorBoard ship with the ``ml`` extra; :class:`JsonlTracker`
needs nothing.
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import IO, Any

from bioamla.protocols import RunStatus

logger = logging.getLogger(__name__)

_MLFLOW_STATUS = {"finished": "FINISHED", "cancelled": "KILLED", "failed": "FAILED"}


def flatten_params(params: dict[str, Any], prefix: str = "") -> dict[str, str]:
    """Flatten nested params into ``a.b`` keys with string values, for flat-param backends."""
    flat: dict[str, str] = {}
    for key, value in params.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            flat.update(flatten_params(value, f"{name}."))
        elif isinstance(value, (list, tuple)):
            flat[name] = json.dumps(list(value), default=str)
        else:
            flat[name] = str(value)
    return flat


def _mlflow_param_limit() -> int:
    """MLflow's maximum param value length (6000 today; 500 in older versions)."""
    try:
        from mlflow.utils.validation import MAX_PARAM_VAL_LENGTH
    except ImportError:
        return 500
    return MAX_PARAM_VAL_LENGTH


class MLflowTracker:
    """Report a training run to MLflow.

    The run ends with the real outcome: ``KILLED`` for a cancelled run and
    ``FAILED`` for a failed one, where transformers' own MLflow callback would mark
    any run it started ``FINISHED``. If an MLflow run is already active when
    training starts (e.g. a sweep's parent run), each training run gets its own
    nested run inside it, so runs don't collide on MLflow's write-once params.
    Param values longer than MLflow accepts are truncated.

    Args:
        tracking_uri: MLflow tracking URI; default: MLflow's own configuration.
        experiment: Experiment name; default: MLflow's active experiment.
    """

    def __init__(self, tracking_uri: str | None = None, experiment: str | None = None) -> None:
        self.tracking_uri = tracking_uri
        self.experiment = experiment
        self._mlflow: Any = None
        self._in_run = False

    def start_run(self, name: str | None, params: dict[str, Any]) -> None:
        import mlflow

        self._mlflow = mlflow
        if self.tracking_uri:
            mlflow.set_tracking_uri(self.tracking_uri)
        if self.experiment:
            mlflow.set_experiment(self.experiment)
        mlflow.start_run(run_name=name, nested=mlflow.active_run() is not None)
        self._in_run = True
        limit = _mlflow_param_limit()
        mlflow.log_params(
            {
                k: v if len(v) <= limit else v[: limit - 3] + "..."
                for k, v in flatten_params(params).items()
            }
        )

    def log_metrics(self, step: int, metrics: dict[str, float]) -> None:
        if self._mlflow is not None:
            self._mlflow.log_metrics(metrics, step=step)

    def log_artifact(self, path: str) -> None:
        if self._mlflow is not None:
            self._mlflow.log_artifact(path)

    def end_run(self, status: RunStatus) -> None:
        if self._mlflow is not None and self._in_run:
            self._mlflow.end_run(status=_MLFLOW_STATUS[status])
            self._in_run = False


class TensorBoardTracker:
    """Write a training run to a TensorBoard log directory.

    Metrics become scalars; params, artifact paths and the final status are
    written as text.

    Args:
        log_dir: Directory for the event files.
    """

    def __init__(self, log_dir: str) -> None:
        self.log_dir = log_dir
        self._writer: Any = None
        self._step = 0

    def start_run(self, name: str | None, params: dict[str, Any]) -> None:
        from torch.utils.tensorboard import SummaryWriter

        self._writer = SummaryWriter(log_dir=self.log_dir)
        self._writer.add_text("params", json.dumps(params, indent=2, default=str))

    def log_metrics(self, step: int, metrics: dict[str, float]) -> None:
        if self._writer is None:
            return
        self._step = step
        for key, value in metrics.items():
            self._writer.add_scalar(key, value, step)

    def log_artifact(self, path: str) -> None:
        if self._writer is not None:
            self._writer.add_text("artifacts", path, self._step)

    def end_run(self, status: RunStatus) -> None:
        if self._writer is not None:
            self._writer.add_text("status", status, self._step)
            self._writer.close()
            self._writer = None


def _jsonable(value: Any) -> Any:
    """``value`` as JSON-native data: numpy scalars as numbers, NaN and infinities as
    None (strict JSON has no token for them), anything else unknown as text."""
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if hasattr(value, "item") and callable(value.item) and not isinstance(value, (str, bytes)):
        try:
            value = value.item()  # a numpy or torch scalar
        except (TypeError, ValueError):
            return str(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if value is None or isinstance(value, (bool, int, str)):
        return value
    return str(value)


class JsonlTracker:
    """Append a training run to a JSON Lines file, one object per call.

    The lines are ``{"event": "start", "name", "params"}``, then
    ``{"event": "metrics", "step", "metrics"}`` each time the trainer logs,
    ``{"event": "artifact", "path"}`` for each file the run writes, and
    ``{"event": "end", "status"}`` last. Each line is flushed as it's written, so a
    cancelled or failed run still leaves what it measured. Values are strict JSON
    (numpy scalars become numbers, NaN and infinities ``null``), so a script can read
    the file with nothing but a JSON parser.

    The file is created when the tracker is made, so a path that exists or can't be
    written is refused before training. :func:`~bioamla.ml.train_ast` reports to its
    trackers only once preprocessing is done; use the tracker as a context manager
    and a run that fails or is cancelled before that still ends with its status::

        with JsonlTracker("runs/metrics.jsonl") as tracker:
            train_ast(..., tracker=tracker)

    A tracker reused for several runs (a sweep) appends each, starting with its own
    ``start`` line.

    Args:
        path: The file to write. It must not exist yet, so two runs can't interleave
            their lines in one file; its folder is created if needed.

    Raises:
        InvalidInputError: If ``path`` exists or can't be created.
    """

    def __init__(self, path: str | Path) -> None:
        from bioamla.exceptions import InvalidInputError

        self.path = Path(path)
        if self.path.exists():
            raise InvalidInputError(f"Metrics file {self.path} already exists")
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._file: IO[str] | None = self.path.open("x", encoding="utf-8")  # atomic
        except OSError as e:  # including a run that created it a moment ago
            raise InvalidInputError(f"Can't create metrics file {self.path}: {e}") from e

    def _write(self, record: dict[str, Any]) -> None:
        if self._file is None:  # a later run of a reused tracker
            self._file = self.path.open("a", encoding="utf-8")
        self._file.write(json.dumps(_jsonable(record), allow_nan=False) + "\n")
        self._file.flush()

    def start_run(self, name: str | None, params: dict[str, Any]) -> None:
        self._write({"event": "start", "name": name, "params": params})

    def log_metrics(self, step: int, metrics: dict[str, float]) -> None:
        self._write({"event": "metrics", "step": step, "metrics": metrics})

    def log_artifact(self, path: str) -> None:
        self._write({"event": "artifact", "path": path})

    def end_run(self, status: RunStatus) -> None:
        if self._file is not None:
            self._write({"event": "end", "status": status})
            self.close()

    def close(self) -> None:
        """Close the file (without an ``end`` line)."""
        if self._file is not None:
            self._file.close()
            self._file = None

    def __enter__(self) -> JsonlTracker:
        return self

    def __exit__(self, exc_type: type[BaseException] | None, *_: object) -> None:
        """End a run the trainer didn't (one that failed or was cancelled before
        training started), then close."""
        from bioamla.exceptions import TrainingCancelled

        if exc_type is not None:
            cancelled = issubclass(exc_type, TrainingCancelled)
            self.end_run("cancelled" if cancelled else "failed")
        self.close()


__all__ = ["JsonlTracker", "MLflowTracker", "TensorBoardTracker", "flatten_params"]
