"""Experiment-tracker adapters for :func:`bioamla.ml.train_ast`.

Pass one (or a list) as ``train_ast(tracker=...)``. Each satisfies
:class:`bioamla.protocols.ExperimentTracker`; write your own to report to any other
system. MLflow and TensorBoard ship with the ``ml`` extra.
"""

from __future__ import annotations

import json
import logging
from typing import Any

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


__all__ = ["MLflowTracker", "TensorBoardTracker", "flatten_params"]
