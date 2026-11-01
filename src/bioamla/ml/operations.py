"""Operations for ``models``: thin adapters over :mod:`bioamla.ml`.

See :mod:`bioamla.operations`. Anything that loads a model is a ``job``; training
reports progress (steps) and stops cleanly when cancelled.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Literal

from pydantic import Field

from bioamla.common.config import config_dataclass
from bioamla.exceptions import InvalidInputError
from bioamla.operations import OperationContext, PathStr, operation


@config_dataclass
class InitConfigRequest:
    """A documented training config file to edit.

    Attributes:
        output: The file to write.
        force: Overwrite an existing file.
    """

    output: PathStr = "ast_training.toml"
    force: bool = False


@operation("models.ast.init-config", request=InitConfigRequest)
def init_config_op(request: InitConfigRequest, context: OperationContext) -> dict[str, Any]:
    """Write a documented AST training config file."""
    from bioamla.ml.train_config import write_train_config

    return {"output": str(write_train_config(request.output, force=request.force))}


@config_dataclass
class PredictRequest:
    """Classify one audio file with an AST model.

    Attributes:
        file: The audio file.
        model_path: The model (Hugging Face ID or local path).
        segment_duration: Classify N-second segments (0: the whole file).
        overlap: Overlap between segments (s).
        min_confidence: Leave out predictions below this.
        output: A CSV to save the predictions to.
        resample_freq: The model's sample rate.
        top_k: Also rank each prediction's ``top_k`` most likely labels, with the
            margin between the first two (0: don't).
    """

    file: PathStr
    model_path: str = "bioamla/scp-frogs"
    segment_duration: int = 0
    overlap: int = 0
    min_confidence: float = 0.0
    output: PathStr | None = None
    resample_freq: int = 16000
    top_k: int = Field(default=0, ge=0)


@operation("models.ast.predict", request=PredictRequest, kind="job")
def predict_op(request: PredictRequest, context: OperationContext) -> Any:
    """Classify one audio file with an AST model."""
    from bioamla.ml.files import classify_file

    return classify_file(
        request.file,
        model_path=_model(request.model_path, context),
        segment_duration=request.segment_duration,
        overlap=request.overlap,
        min_confidence=request.min_confidence,
        resample_freq=request.resample_freq,
        output=request.output,
        top_k=request.top_k,
    )


@config_dataclass
class TrainRequest:
    """Fine-tune an AST model.

    Attributes:
        train_dataset: A Hugging Face dataset ID, a metadata CSV, or a folder with one
            subfolder per class.
        config_path: A training config file (``models.ast.init-config``).
        training_dir: Where training writes its outputs and the model.
        options: Training options set explicitly; anything else comes from the config
            file, then the defaults (see :data:`bioamla.ml.AST_TRAIN_DEFAULTS`).
    """

    train_dataset: str
    config_path: PathStr | None = None
    training_dir: PathStr = "."
    options: dict[str, Any] | None = None


@operation("models.ast.train", request=TrainRequest, kind="job")
def train_op(request: TrainRequest, context: OperationContext) -> Any:
    """Fine-tune an AST model.

    Every option that names a local file or folder (the base model, a local MLflow
    store) goes through the path check, including those that come from the config
    file. ``push_to_hub`` and a remote ``mlflow_tracking_uri`` are honoured with the
    credentials of the machine that runs the training.
    """
    from bioamla.ml.train_config import resolve_train_options, train_ast_from_options

    options = dict(request.options or {})
    if "training_dir" in options:
        raise InvalidInputError("Give training_dir as its own field, not in options")
    options["training_dir"] = request.training_dir
    resolved = resolve_train_options(options, request.config_path)
    resolved["base_model"] = _model(str(resolved["base_model"]), context)
    resolved["mlflow_tracking_uri"] = _tracking_uri(resolved["mlflow_tracking_uri"], context)

    def progress(p: Any) -> None:
        if context.on_progress is not None:
            context.on_progress(p.step, p.total_steps)

    return train_ast_from_options(
        _model(request.train_dataset, context),
        resolved,
        on_progress=progress,
        should_stop=context.cancelled,
    )


def _tracking_uri(uri: str | None, context: OperationContext) -> str | None:
    """An MLflow tracking URI, with a local store (a path or ``file:`` URI) through the
    path check; a server URI (``http://...``) is left as it is."""
    if not uri:
        return uri
    from urllib.parse import urlparse

    parsed = urlparse(uri)
    if parsed.scheme == "file":
        return context.resolve_path(parsed.path)
    if not parsed.scheme or len(parsed.scheme) == 1:  # a plain path (or C:\...)
        return context.resolve_path(uri)
    return uri


@config_dataclass
class EvaluateRequest:
    """Evaluate an AST model on a folder of audio.

    Attributes:
        path: The audio folder.
        ground_truth: CSV of file names and true labels.
        model_path: The model (Hugging Face ID or local path).
        output: Where to save the report.
        output_format: ``txt``, ``json`` or ``csv``.
        file_column: The ground truth's file-name column.
        label_column: Its label column.
        resample_freq: The model's sample rate.
        batch_size: Inference batch size.
        fp16: Half-precision inference.
    """

    path: PathStr
    ground_truth: PathStr
    model_path: str = "bioamla/scp-frogs"
    output: PathStr | None = None
    output_format: Literal["json", "csv", "txt"] = "txt"
    file_column: str = "file_name"
    label_column: str = "label"
    resample_freq: int = 16000
    batch_size: int = 8
    fp16: bool = False


@operation("models.ast.evaluate", request=EvaluateRequest, kind="job")
def evaluate_op(request: EvaluateRequest, context: OperationContext) -> Any:
    """Evaluate an AST model against ground truth."""
    from bioamla.ml.files import evaluate_model

    return evaluate_model(
        request.path,
        model_path=_model(request.model_path, context),
        ground_truth=request.ground_truth,
        file_column=request.file_column,
        label_column=request.label_column,
        resample_freq=request.resample_freq,
        batch_size=request.batch_size,
        fp16=request.fp16,
        output=request.output,
        output_format=request.output_format,
    )


@config_dataclass
class EmbedRequest:
    """AST embeddings of one audio file, saved as ``.npy``.

    Attributes:
        file: The audio file.
        model_path: The model (Hugging Face ID or local path).
        output: The ``.npy`` to write.
        layer: The layer to take embeddings from.
        sample_rate: The model's sample rate.
    """

    file: PathStr
    model_path: str
    output: PathStr
    layer: str | None = None
    sample_rate: int = 16000


@operation("models.ast.embed", request=EmbedRequest, kind="job")
def embed_op(request: EmbedRequest, context: OperationContext) -> Any:
    """Extract one audio file's AST embeddings."""
    from bioamla.ml.files import embed_file

    return embed_file(
        request.file,
        request.output,
        model_path=_model(request.model_path, context),
        layer=request.layer,
        sample_rate=request.sample_rate,
    )


@config_dataclass
class InfoRequest:
    """An AST model's type, classes and size.

    Attributes:
        model_path: The model (Hugging Face ID or local path).
    """

    model_path: str


@operation("models.ast.info", request=InfoRequest, kind="job")
def info_op(request: InfoRequest, context: OperationContext) -> dict[str, Any]:
    """Describe an AST model."""
    from bioamla.ml.ast_service import get_model_info

    return get_model_info(_model(request.model_path, context))


_HF_ID = re.compile(r"[A-Za-z0-9][\w.-]*/[\w.-]+(:[\w.-]+)?")


def _model(name: str, context: OperationContext) -> str:
    """A model or dataset named by local path or Hugging Face ID.

    The name always goes through the path check first, and a name the check refuses
    is refused (a relative ``org/name`` only fails it when it's a link out of the
    allowed folders). A path that's allowed and exists is used; otherwise a name
    shaped like an ID (``org/name``) is taken as one, and anything else is returned
    as checked (so the library reports it missing).
    """
    resolved = context.resolve_path(name)
    if resolved is not None and Path(resolved).exists():
        return resolved
    return name if _HF_ID.fullmatch(name) else resolved
