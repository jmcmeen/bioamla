"""AST fine-tuning as a parameter-driven library function.

:func:`train_ast` fine-tunes an Audio Spectrogram Transformer on a custom
dataset and returns a :class:`~bioamla.ml.ast_service.TrainResult`. It ingests
data through parameters (no config-file reading here — the CLI layer maps flags
or a TOML config onto these parameters). All heavy dependencies (torch,
transformers, datasets, evaluate) are imported inside the function so importing
this module stays light.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from bioamla.exceptions import TrainingCancelled, TrainingError
from bioamla.ml.ast_service import TrainProgress, TrainResult

if TYPE_CHECKING:
    from collections.abc import Callable

    from bioamla.datasets.augmentation import WaveformAugmentationConfig

logger = logging.getLogger(__name__)

DEFAULT_BASE_MODEL = "MIT/ast-finetuned-audioset-10-10-0.4593"
SAMPLING_RATE = 16000

# Map metadata.csv split values (and common aliases) onto HF split keys.
_SPLIT_ALIAS = {
    "train": "train",
    "training": "train",
    "test": "test",
    "testing": "test",
    "eval": "test",
    "evaluation": "test",
    "val": "validation",
    "valid": "validation",
    "validation": "validation",
    "dev": "validation",
}


def _dataframe_to_ast_dataset(df, label_column, sampling_rate=SAMPLING_RATE):
    """Build an HF ``Dataset`` (or ``DatasetDict`` when a usable ``split`` column exists).

    When every row carries a recognized split value, a ``DatasetDict`` is returned
    so the trainer consumes that fixed split instead of re-splitting at train time
    (the partition ``--mode column`` artifact). If the ``split`` column is absent
    or any row's split is empty/unrecognized, a flat ``Dataset`` is returned and
    the caller re-splits as before — rows are never silently dropped.

    Returns:
        ``(dataset, used_fixed_split)``.
    """
    from datasets import Audio, Dataset, DatasetDict

    cols = ["audio", label_column]

    def _flat():
        ds = Dataset.from_pandas(df[cols], preserve_index=False)
        return ds.cast_column("audio", Audio(sampling_rate=sampling_rate))

    if "split" not in df.columns:
        return _flat(), False

    normalized = df["split"].astype(str).str.strip().str.lower().map(_SPLIT_ALIAS)
    if normalized.isna().any():
        return _flat(), False

    splits: dict = {}
    for split_key, group in df.groupby(normalized):
        ds = Dataset.from_pandas(group[cols], preserve_index=False)
        splits[split_key] = ds.cast_column("audio", Audio(sampling_rate=sampling_rate))

    # The trainer evaluates on the "test" split; if a fixed split provides only
    # validation, use it as the eval set so the split is still respected.
    if "test" not in splits and "validation" in splits:
        splits["test"] = splits.pop("validation")

    return DatasetDict(splits), True


def _validate_augmentation(aug: WaveformAugmentationConfig) -> None:
    """Raise :class:`TrainingError` on inverted min/max ranges for *enabled* layers.

    Ranges for disabled transforms are ignored — a layer that isn't added can't
    misconfigure training.
    """
    checks = [
        (aug.add_noise, "noise SNR", aug.noise_min_snr, aug.noise_max_snr),
        (aug.gain or aug.gain_transition, "gain dB", aug.gain_min_db, aug.gain_max_db),
        (
            aug.clipping_distortion,
            "clipping percentile",
            aug.clipping_min_percentile,
            aug.clipping_max_percentile,
        ),
        (aug.time_stretch, "time-stretch rate", aug.time_stretch_min, aug.time_stretch_max),
        (aug.pitch_shift, "pitch-shift semitones", aug.pitch_shift_min, aug.pitch_shift_max),
    ]
    for enabled, name, lo, hi in checks:
        if enabled and lo > hi:
            raise TrainingError(f"Augmentation {name} min ({lo}) must be <= max ({hi})")


def _load_train_dataset(train_dataset: str, split: str, category_label_column: str):
    """Resolve ``train_dataset`` (CSV / audiofolder dir / HF id) into a dataset.

    Returns ``(dataset, category_label_column)`` — the label column may be
    rewritten (e.g. ``"label"`` for an audiofolder, or the detected CSV column).
    """
    from datasets import load_dataset

    train_path = Path(train_dataset)

    if train_path.exists():
        if train_path.is_file() and train_path.suffix.lower() == ".csv":
            return _load_csv_dataset(train_path, category_label_column)
        if train_path.is_dir():
            # A dataset directory carrying a metadata.csv (from `dataset
            # extract-clips`/`partition` or `catalogs hf pull-dataset`) is loaded
            # via that CSV, so a populated `split` column and the reorganized
            # `train/val/test/<label>/` layout from `partition --mode subdirs` are
            # both honored. Otherwise fall back to the AudioFolder convention.
            metadata_csv = train_path / "metadata.csv"
            if metadata_csv.exists():
                return _load_csv_dataset(metadata_csv, category_label_column)
            return _load_directory_dataset(train_path, train_dataset, split, category_label_column)
        raise TrainingError(
            f"Local path {train_dataset} exists but is neither a CSV file nor a directory."
        )

    if ":" in train_dataset:
        # HuggingFace dataset with config (e.g., 'samuelstevens/BirdSet:HSN')
        dataset_name, config_name = train_dataset.rsplit(":", 1)
        logger.info(f"Loading from HuggingFace Hub: {dataset_name} (config: {config_name})")
        return load_dataset(dataset_name, config_name, split=split), category_label_column
    if "BirdSet" in train_dataset:
        logger.info(
            "BirdSet detected. Use format 'samuelstevens/BirdSet:HSN' to specify subset "
            "(available: HSN, NBP, NES, PER); defaulting to HSN."
        )
        return load_dataset(train_dataset, "HSN", split=split), category_label_column

    logger.info(f"Loading from HuggingFace Hub: {train_dataset}")
    return load_dataset(train_dataset, split=split), category_label_column


def _load_csv_dataset(train_path: Path, category_label_column: str):
    """Load a metadata-CSV dataset, resolving file/label columns and paths."""
    import pandas as pd

    logger.info(f"Loading from metadata CSV: {train_path}")
    df = pd.read_csv(train_path)

    file_col = None
    for col in ["file", "filepath", "path", "audio", "filename", "file_path", "file_name"]:
        if col in df.columns:
            file_col = col
            break
    if file_col is None:
        raise TrainingError(
            "CSV must have a file column (tried: file, filepath, path, audio, filename, "
            f"file_path, file_name). Found columns: {list(df.columns)}"
        )

    label_col = None
    for col in ["label", "category", "class", "species", category_label_column]:
        if col in df.columns:
            label_col = col
            break
    if label_col is None:
        raise TrainingError(
            "CSV must have a label column (tried: label, category, class, species, "
            f"{category_label_column}). Found columns: {list(df.columns)}"
        )

    category_label_column = label_col
    csv_dir = train_path.parent

    def resolve_path(p):
        p = Path(p)
        if not p.is_absolute():
            p = csv_dir / p
        return str(p)

    df["audio"] = df[file_col].apply(resolve_path)
    df[category_label_column] = df[label_col]

    dataset, used_fixed_split = _dataframe_to_ast_dataset(df, category_label_column)
    if used_fixed_split:
        counts = ", ".join(f"{k}={len(dataset[k])}" for k in dataset)
        logger.info(f"Using fixed split from CSV 'split' column: {counts}")
    else:
        logger.info(f"Loaded {len(dataset)} samples from CSV")
    return dataset, category_label_column


def _load_directory_dataset(
    train_path: Path, train_dataset: str, split: str, category_label_column: str
):
    """Load an audiofolder-style directory (class-named subdirectories)."""
    from datasets import load_dataset

    subdirs = [d for d in train_path.iterdir() if d.is_dir()]
    audio_extensions = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}

    if subdirs:
        has_audio_in_subdirs = any(
            any(f.suffix.lower() in audio_extensions for f in subdir.iterdir() if f.is_file())
            for subdir in subdirs
        )
        if has_audio_in_subdirs:
            logger.info(f"Loading from audiofolder directory: {train_dataset}")
            logger.info(f"Found class directories: {[d.name for d in subdirs]}")
            dataset = load_dataset("audiofolder", data_dir=train_dataset, split=split)
            return dataset, "label"  # audiofolder uses 'label'
        raise TrainingError(
            f"Directory {train_dataset} has subdirectories but they don't contain audio files. "
            f"Expected structure: {train_dataset}/class_name/*.wav"
        )

    csv_files = list(train_path.glob("*.csv"))
    if csv_files:
        raise TrainingError(
            f"Directory contains CSV files. Please specify the CSV directly: "
            f"--train-dataset {csv_files[0]}"
        )
    raise TrainingError(
        f"Directory {train_dataset} must either have class subdirectories (e.g., bird/, frog/) "
        f"or contain a metadata CSV file."
    )


def _cleanup_step(what: str, step: Callable[..., object], *args: Any) -> None:
    """Run one post-training cleanup step, logging (not raising) any failure."""
    try:
        step(*args)
    except Exception as e:  # noqa: BLE001 - cleanup must not mask the training outcome
        logger.warning(f"Could not {what} after training: {e}")


def _free_trainer(trainer: Any) -> None:
    """Release a trainer's accelerator memory and dataloaders (and their workers)."""
    if getattr(trainer, "accelerator", None) is not None:
        trainer.accelerator.free_memory()
    for attr in ("_train_dataloader", "_eval_dataloader"):
        if hasattr(trainer, attr):
            delattr(trainer, attr)


def _start_mlflow_run(report_to: str | list[str], run_name: str | None) -> Any:
    """Start an MLflow run for training when MLflow reporting is on.

    transformers' MLflow callback ends a run it started itself as FINISHED,
    even when training was cancelled. Starting the run here makes the callback
    log into it instead, so ``train_ast`` can end it with the real outcome.

    Returns:
        The ``mlflow`` module when this call started the run (the caller ends
        it), or None when MLflow is off, missing, or a run is already active.
    """
    names = report_to if isinstance(report_to, list) else [report_to]
    if "mlflow" not in names and "all" not in names:
        return None
    try:
        import mlflow
    except ImportError:
        return None
    try:
        if mlflow.active_run() is not None:
            return None
        mlflow.start_run(run_name=run_name)
    except Exception as e:  # noqa: BLE001 - fall back to transformers' own run handling
        logger.warning(f"Could not start an MLflow run: {e}")
        return None
    return mlflow


# Timing counters transformers adds to evaluation metrics; not model metrics.
_EVAL_BOOKKEEPING = {"eval_runtime", "eval_samples_per_second", "eval_steps_per_second"}


def _progress_callback(
    on_progress: Callable[[TrainProgress], None] | None,
    should_stop: Callable[[], bool] | None,
) -> Any:
    """Build the ``TrainerCallback`` behind ``train_ast``'s progress and cancel hooks."""
    from transformers import TrainerCallback

    class _ProgressCallback(TrainerCallback):
        def __init__(self) -> None:
            self.cancelled = False
            self.loss: float | None = None
            self.eval_metrics: dict[str, float] = {}

        def _report(self, state) -> None:
            if on_progress is not None and state.is_world_process_zero:
                on_progress(
                    TrainProgress(
                        step=state.global_step,
                        total_steps=state.max_steps,
                        epoch=float(state.epoch or 0.0),
                        loss=self.loss,
                        eval_metrics=dict(self.eval_metrics),
                    )
                )

        def on_log(self, args, state, control, logs=None, **kwargs):
            # Loss is logged after on_step_end, so report again with the fresh value.
            if logs and "loss" in logs:
                self.loss = float(logs["loss"])
                self._report(state)

        def on_evaluate(self, args, state, control, metrics=None, **kwargs):
            if metrics:
                self.eval_metrics = {
                    k: float(v)
                    for k, v in metrics.items()
                    if k.startswith("eval_")
                    and k not in _EVAL_BOOKKEEPING
                    and isinstance(v, (int, float))
                }
            self._report(state)

        def _stop_now(self, control) -> None:
            # transformers' flow callback (which runs first) may have scheduled an
            # evaluation and checkpoint for this step or epoch end; a cancelled run
            # skips both.
            control.should_training_stop = True
            control.should_evaluate = False
            control.should_save = False

        def on_step_end(self, args, state, control, **kwargs):
            self._report(state)
            if (
                not self.cancelled
                and should_stop is not None
                and state.global_step < state.max_steps  # a finished run is kept
                and should_stop()
            ):
                self.cancelled = True
            if self.cancelled:
                self._stop_now(control)
            return control

        def on_epoch_end(self, args, state, control, **kwargs):
            if self.cancelled:
                self._stop_now(control)
            return control

    return _ProgressCallback()


def train_ast(
    *,
    train_dataset: str,
    training_dir: str = ".",
    base_model: str = DEFAULT_BASE_MODEL,
    split: str = "train",
    category_label_column: str = "category",
    learning_rate: float = 5.0e-5,
    num_train_epochs: int = 1,
    per_device_train_batch_size: int = 8,
    eval_strategy: str = "epoch",
    save_strategy: str = "epoch",
    eval_steps: int = 1,
    save_steps: int = 1,
    load_best_model_at_end: bool = True,
    metric_for_best_model: str = "accuracy",
    logging_strategy: str = "steps",
    logging_steps: int = 100,
    report_to: str | list[str] = "tensorboard",
    fp16: bool = False,
    bf16: bool = False,
    gradient_accumulation_steps: int = 1,
    dataloader_num_workers: int = 4,
    torch_compile: bool = False,
    finetune_mode: str = "full",
    push_to_hub: bool = False,
    mlflow_tracking_uri: str | None = None,
    mlflow_experiment_name: str | None = None,
    mlflow_run_name: str | None = None,
    augmentation: WaveformAugmentationConfig | None = None,
    augment_multiplier: int = 1,
    on_progress: Callable[[TrainProgress], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> TrainResult:
    """Fine-tune an AST model on a custom dataset.

    The remaining keyword arguments mirror ``transformers.TrainingArguments``
    (learning rate, batch size, eval/save strategy, fp16/bf16, etc.).

    Args:
        train_dataset: A HuggingFace dataset id (``"bioamla/scp-frogs"`` or
            ``"samuelstevens/BirdSet:HSN"``), a local metadata CSV (with file and
            label columns), or a directory of class-named subdirectories.
        training_dir: Output root; the best model is written to
            ``{training_dir}/best_model``, runs to ``/runs``, logs to ``/logs``.
        base_model: Pretrained AST checkpoint to fine-tune.
        split: Split to use for single-split HuggingFace datasets.
        category_label_column: Label column name for HF/CSV datasets.
        augmentation: On-the-fly training augmentation; ``None`` disables it.
        augment_multiplier: Repeat the training split N times (with augmentation)
            to enlarge it; ``1`` means no duplication.
        on_progress: Called with a :class:`~bioamla.ml.TrainProgress` after each
            optimizer step and each evaluation.
        should_stop: Polled between the preprocessing phases (loading the
            dataset, mapping labels, filtering, normalization statistics, loading
            the model and metrics) and after each optimizer step except the
            last. When it returns True the run stops there, without the pending
            evaluation or checkpoint and without saving ``best_model``, shuts
            down its workers, and raises
            :class:`~bioamla.exceptions.TrainingCancelled`. A phase already
            under way (e.g. a download) finishes first. The hooks assume a
            single training process, which is how bioamla runs ``train_ast``.

    Returns:
        A :class:`~bioamla.ml.ast_service.TrainResult` with the best-model path,
        epoch count, and final eval accuracy/loss when available.

    Raises:
        TrainingError: On an unusable dataset/params or an empty training set.
        TrainingCancelled: If ``should_stop`` returned True.
    """
    import os

    import evaluate
    import numpy as np
    import torch
    from transformers import (
        ASTConfig,
        ASTFeatureExtractor,
        ASTForAudioClassification,
        Trainer,
        TrainingArguments,
    )

    from bioamla.datasets.augmentation import (
        create_augmentation_pipeline,
        describe_augmentation_pipeline,
    )
    from datasets import Audio, Dataset, DatasetDict

    if augmentation is not None:
        _validate_augmentation(augmentation)

    output_dir = training_dir + "/runs"
    best_model_path = training_dir + "/best_model"

    # TensorBoard log location. transformers deprecated the ``logging_dir``
    # TrainingArguments kwarg (removed in v5.2) in favor of this env var, which
    # its TensorBoardCallback reads at init — set it instead of passing the kwarg.
    os.environ["TENSORBOARD_LOGGING_DIR"] = training_dir + "/logs"

    if mlflow_tracking_uri or mlflow_experiment_name:
        try:
            import mlflow

            if mlflow_tracking_uri:
                mlflow.set_tracking_uri(mlflow_tracking_uri)
                logger.info(f"MLflow tracking URI: {mlflow_tracking_uri}")
            if mlflow_experiment_name:
                mlflow.set_experiment(mlflow_experiment_name)
                logger.info(f"MLflow experiment: {mlflow_experiment_name}")
            if isinstance(report_to, str) and "mlflow" not in report_to:
                report_to = f"{report_to},mlflow" if report_to else "mlflow"
            logger.info(f"MLflow integration enabled, reporting to: {report_to}")
        except ImportError:
            logger.warning(
                "MLflow not installed. Install with 'pip install mlflow' to enable tracking."
            )

    if isinstance(report_to, str) and "," in report_to:
        report_to = [r.strip() for r in report_to.split(",")]

    def _check_stop(phase: str) -> None:
        if should_stop is not None and should_stop():
            raise TrainingCancelled(f"Training cancelled before {phase}")

    _check_stop("loading the dataset")
    dataset, category_label_column = _load_train_dataset(
        train_dataset, split, category_label_column
    )

    if isinstance(dataset, Dataset):
        class_names = sorted(set(dataset[category_label_column]))
    elif isinstance(dataset, DatasetDict):
        first_split_name = list(dataset.keys())[0]
        class_names = sorted(set(dataset[first_split_name][category_label_column]))
    else:
        raise TrainingError("Dataset must be a Dataset or DatasetDict instance")

    label_to_id = {name: idx for idx, name in enumerate(class_names)}
    num_labels = len(class_names)

    _check_stop("mapping labels")

    def convert_labels(example):
        example["labels"] = label_to_id[example[category_label_column]]
        return example

    dataset = dataset.map(
        convert_labels,
        remove_columns=[category_label_column],
        writer_batch_size=100,
        # None = run in-process; num_proc=1 still spawns a one-worker Pool (fork).
        num_proc=None,
    )

    dataset = dataset.cast_column("audio", Audio(sampling_rate=SAMPLING_RATE))

    _check_stop("loading the feature extractor")
    pretrained_model = base_model
    feature_extractor = ASTFeatureExtractor.from_pretrained(pretrained_model)
    model_input_name = feature_extractor.model_input_names[0]
    sampling_rate = feature_extractor.sampling_rate

    def preprocess_audio(batch):
        wavs = [audio["array"] for audio in batch["input_values"]]
        inputs = feature_extractor(wavs, sampling_rate=sampling_rate, return_tensors="pt")
        return {model_input_name: inputs.get(model_input_name), "labels": list(batch["labels"])}

    label2id = {name: idx for idx, name in enumerate(class_names)}

    test_size = 0.2
    if isinstance(dataset, Dataset):
        try:
            dataset = dataset.train_test_split(
                test_size=test_size, shuffle=True, seed=0, stratify_by_column="labels"
            )
        except ValueError as e:
            logger.warning(f"Stratified split failed ({e}). Using regular split.")
            dataset = dataset.train_test_split(test_size=test_size, shuffle=True, seed=0)
    elif isinstance(dataset, DatasetDict) and "test" not in dataset:
        train_data = dataset["train"]
        try:
            dataset = train_data.train_test_split(
                test_size=test_size, shuffle=True, seed=0, stratify_by_column="labels"
            )
        except ValueError as e:
            logger.warning(f"Stratified split failed ({e}). Using regular split.")
            dataset = train_data.train_test_split(test_size=test_size, shuffle=True, seed=0)

    augment = augmentation is not None

    # Multiply training dataset if requested.
    if (
        augment
        and augment_multiplier > 1
        and isinstance(dataset, DatasetDict)
        and "train" in dataset
    ):
        from datasets import concatenate_datasets

        original_train = dataset["train"]
        logger.info(
            f"Multiplying training dataset by {augment_multiplier}x "
            f"(original: {len(original_train)} samples)"
        )
        dataset["train"] = concatenate_datasets([original_train] * augment_multiplier)
        logger.info(f"New training dataset size: {len(dataset['train'])} samples")

    if augment:
        audio_augmentations = create_augmentation_pipeline(augmentation)
        descriptions = describe_augmentation_pipeline(audio_augmentations)
        logger.info(
            "Audio augmentations enabled: %d transform(s) "
            "(pipeline p=%s, shuffle=%s, multiplier=%dx)",
            len(descriptions),
            augmentation.pipeline_probability,
            augmentation.shuffle,
            augment_multiplier,
        )
        for description in descriptions:
            logger.info("  - %s", description)
    else:
        audio_augmentations = None
        logger.info("Audio augmentations disabled")

    def preprocess_audio_with_transforms(batch):
        if audio_augmentations is not None:
            wavs = [
                audio_augmentations(audio["array"], sample_rate=sampling_rate)
                for audio in batch["input_values"]
            ]
        else:
            wavs = [audio["array"] for audio in batch["input_values"]]
        inputs = feature_extractor(wavs, sampling_rate=sampling_rate, return_tensors="pt")
        return {model_input_name: inputs.get(model_input_name), "labels": list(batch["labels"])}

    dataset = dataset.cast_column("audio", Audio(sampling_rate=feature_extractor.sampling_rate))
    dataset = dataset.rename_column("audio", "input_values")

    def is_valid_audio(example):
        try:
            _ = example["input_values"]["array"]
            return True
        except (RuntimeError, Exception):
            return False

    _check_stop("filtering corrupted audio")
    logger.info("Filtering out corrupted audio files...")
    original_sizes = {}
    filter_num_proc = min(dataloader_num_workers, 4) if dataloader_num_workers > 1 else None
    if isinstance(dataset, DatasetDict):
        for split_name in dataset.keys():
            original_sizes[split_name] = len(dataset[split_name])
        dataset = dataset.filter(is_valid_audio, num_proc=filter_num_proc)
        for split_name in dataset.keys():
            filtered_count = original_sizes[split_name] - len(dataset[split_name])
            if filtered_count > 0:
                logger.info(f"  Removed {filtered_count} corrupted files from {split_name} split")
    else:
        original_size = len(dataset)
        dataset = dataset.filter(is_valid_audio, num_proc=filter_num_proc)
        filtered_count = original_size - len(dataset)
        if filtered_count > 0:
            logger.info(f"  Removed {filtered_count} corrupted files")

    feature_extractor.do_normalize = False

    if isinstance(dataset, DatasetDict) and "train" in dataset:
        train_dataset_for_norm = dataset["train"]
        if len(train_dataset_for_norm) == 0:
            raise TrainingError("No valid audio samples found in training dataset after filtering")

        def compute_stats(batch):
            wavs = [audio["array"] for audio in batch["input_values"]]
            inputs = feature_extractor(wavs, sampling_rate=sampling_rate, return_tensors="pt")
            values = inputs.get(model_input_name)
            means = [float(torch.mean(values[i])) for i in range(values.shape[0])]
            stds = [float(torch.std(values[i])) for i in range(values.shape[0])]
            return {"_mean": means, "_std": stds}

        _check_stop("computing normalization statistics")
        logger.info("Calculating dataset normalization statistics...")
        norm_num_proc = min(dataloader_num_workers, 4) if dataloader_num_workers > 1 else None
        stats_dataset = train_dataset_for_norm.map(
            compute_stats,
            batched=True,
            batch_size=32,
            num_proc=norm_num_proc,
            remove_columns=train_dataset_for_norm.column_names,
        )
        all_means = stats_dataset["_mean"]
        all_stds = stats_dataset["_std"]

        if not all_means:
            raise TrainingError("No valid audio samples found in training dataset")

        feature_extractor.mean = float(np.mean(all_means))
        feature_extractor.std = float(np.mean(all_stds))
    else:
        raise TrainingError("Expected DatasetDict with 'train' split")

    feature_extractor.do_normalize = True
    logger.info(f"Calculated mean and std: {feature_extractor.mean} {feature_extractor.std}")

    if isinstance(dataset, DatasetDict):
        if "train" in dataset:
            dataset["train"].set_transform(
                preprocess_audio_with_transforms, output_all_columns=False
            )
        if "test" in dataset:
            dataset["test"].set_transform(preprocess_audio, output_all_columns=False)
    else:
        raise TrainingError("Expected DatasetDict for transform application")

    _check_stop("loading the model")
    config = ASTConfig.from_pretrained(pretrained_model)
    config.num_labels = num_labels
    config.label2id = label2id
    config.id2label = {v: k for k, v in label2id.items()}

    # ``ignore_mismatched_sizes`` lets the classifier head be resized from the base
    # model's label count to ours (the head is reinitialized; the encoder weights are
    # loaded via transformers' AST checkpoint key-conversion — needs transformers
    # >= 5.10.2). No explicit init_weights() call: it's a no-op here and only obscures
    # that the pretrained encoder is in fact loaded.
    model = ASTForAudioClassification.from_pretrained(
        pretrained_model, config=config, ignore_mismatched_sizes=True
    )

    if finetune_mode == "feature-extraction":
        logger.info("Feature extraction mode: freezing base model, training classifier head only")
        for param in model.audio_spectrogram_transformer.parameters():
            param.requires_grad = False
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(
            f"  Trainable parameters: {trainable_params:,} / {total_params:,} "
            f"({100 * trainable_params / total_params:.2f}%)"
        )
    else:
        logger.info("Full finetune mode: training all model layers")

    training_args = TrainingArguments(
        output_dir=output_dir,
        report_to=report_to,
        learning_rate=learning_rate,
        push_to_hub=push_to_hub,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        eval_strategy=eval_strategy,
        save_strategy=save_strategy,
        eval_steps=eval_steps,
        save_steps=save_steps,
        load_best_model_at_end=load_best_model_at_end,
        metric_for_best_model=metric_for_best_model,
        logging_strategy=logging_strategy,
        logging_steps=logging_steps,
        fp16=fp16,
        bf16=bf16,
        gradient_accumulation_steps=gradient_accumulation_steps,
        dataloader_num_workers=dataloader_num_workers,
        dataloader_persistent_workers=False,  # Prevent hang on exit
        torch_compile=torch_compile,
        run_name=mlflow_run_name,
    )

    _check_stop("loading metrics")
    accuracy = evaluate.load("accuracy")
    recall = evaluate.load("recall")
    precision = evaluate.load("precision")
    f1 = evaluate.load("f1")

    average = "macro" if config.num_labels > 2 else "binary"

    def compute_metrics(eval_pred) -> dict[str, float]:
        logits = eval_pred.predictions
        predictions = np.argmax(logits, axis=1)

        accuracy_result = accuracy.compute(predictions=predictions, references=eval_pred.label_ids)
        metrics: dict[str, float] = accuracy_result if accuracy_result is not None else {}

        precision_result = precision.compute(
            predictions=predictions, references=eval_pred.label_ids, average=average
        )
        if precision_result is not None:
            metrics.update(precision_result)

        recall_result = recall.compute(
            predictions=predictions, references=eval_pred.label_ids, average=average
        )
        if recall_result is not None:
            metrics.update(recall_result)

        f1_result = f1.compute(
            predictions=predictions, references=eval_pred.label_ids, average=average
        )
        if f1_result is not None:
            metrics.update(f1_result)

        return metrics

    if isinstance(dataset, DatasetDict):
        train_data = dataset.get("train")
        eval_data = dataset.get("test")
    else:
        raise TrainingError("Expected DatasetDict for trainer setup")

    _check_stop("training")
    progress = (
        _progress_callback(on_progress, should_stop)
        if on_progress is not None or should_stop is not None
        else None
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=eval_data,
        compute_metrics=compute_metrics,
        callbacks=[progress] if progress is not None else None,
    )

    import multiprocessing

    # Child processes that exist before training aren't ours to terminate.
    children_before = set(multiprocessing.active_children())
    mlflow_run = _start_mlflow_run(report_to, mlflow_run_name)

    final_accuracy = None
    final_loss = None
    cancelled = False
    run_status = "FAILED"  # MLflow status unless training and saving complete
    try:
        trainer.train()
        cancelled = progress is not None and progress.cancelled

        if cancelled:
            run_status = "KILLED"
        else:
            Path(best_model_path).mkdir(parents=True, exist_ok=True)
            trainer.save_model(best_model_path)

            # Pull final eval metrics from the trainer's log history (no extra eval pass).
            for entry in reversed(trainer.state.log_history):
                if final_accuracy is None and "eval_accuracy" in entry:
                    final_accuracy = float(entry["eval_accuracy"])
                if final_loss is None and "eval_loss" in entry:
                    final_loss = float(entry["eval_loss"])
                if final_accuracy is not None and final_loss is not None:
                    break
            run_status = "FINISHED"
    finally:
        # Cleanup to prevent hanging from dataloader workers, whether the run
        # finished, was cancelled, or failed. Each step is guarded so one
        # failure neither skips the rest nor replaces the training error.
        def _end_mlflow_run() -> None:
            if mlflow_run is not None:
                mlflow_run.end_run(status=run_status)

        def _stop_workers() -> None:
            for child in set(multiprocessing.active_children()) - children_before:
                child.terminate()
                child.join(timeout=1)

        _cleanup_step("free trainer resources", _free_trainer, trainer)
        del trainer
        del model
        if torch.cuda.is_available():
            _cleanup_step("empty the CUDA cache", torch.cuda.empty_cache)
        _cleanup_step("end the MLflow run", _end_mlflow_run)
        import gc

        _cleanup_step("collect garbage", gc.collect)
        _cleanup_step("stop dataloader workers", _stop_workers)

    if cancelled:
        raise TrainingCancelled("Training cancelled; no model was saved")

    return TrainResult(
        model_path=best_model_path,
        epochs=num_train_epochs,
        final_accuracy=final_accuracy,
        final_loss=final_loss,
    )
