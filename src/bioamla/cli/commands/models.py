"""ML model operations - AST models.

Command structure:
    bioamla models {architecture} {command}

Examples:
    bioamla models ast predict audio.wav --model-path my_model
    bioamla models ast train --train-dataset bioamla/scp-frogs
    bioamla models ast train --train-dataset ./metadata.csv
    bioamla models ast train --train-dataset ./audio_by_class/
"""

import click

from bioamla.exceptions import BioamlaError

# Map ``ast train`` flag names onto config-file (section, key) locations, using the
# template schema in ``bioamla.common.config.DEFAULT_CONFIG`` as the guidance shape
# (e.g. [training].epochs -> --num-train-epochs). Flags set on the command line
# override these; these override built-in defaults.
_TRAIN_CONFIG_MAP = {
    "base_model": ("models", "default_ast_model"),
    "learning_rate": ("training", "learning_rate"),
    "num_train_epochs": ("training", "epochs"),
    "per_device_train_batch_size": ("training", "batch_size"),
    "eval_strategy": ("training", "eval_strategy"),
    "save_strategy": ("training", "save_strategy"),
    "eval_steps": ("training", "eval_steps"),
    "save_steps": ("training", "save_steps"),
    "logging_steps": ("training", "logging_steps"),
}


def _apply_train_config(ctx: click.Context, config_path: str | None, values: dict) -> dict:
    """Overlay TOML config onto flag values, honoring CLI-over-file-over-default.

    For each mapped flag, the config value is used only when the flag was left at
    its default (not passed on the command line) and the config provides it.
    """
    if not config_path:
        return values

    from click.core import ParameterSource

    from bioamla.common.config import load_toml

    cfg = load_toml(config_path)
    resolved = dict(values)
    for name, (section, key) in _TRAIN_CONFIG_MAP.items():
        from_default = ctx.get_parameter_source(name) == ParameterSource.DEFAULT
        cfg_value = cfg.get(section, {}).get(key)
        if from_default and cfg_value is not None:
            resolved[name] = cfg_value
    return resolved


@click.group()
def models() -> None:
    """ML model operations - AST models."""
    pass


# =============================================================================
# AST Subgroup (Audio Spectrogram Transformer)
# =============================================================================


@models.group()
def ast() -> None:
    """Audio Spectrogram Transformer (AST) model operations."""
    pass


@ast.command("predict")
@click.argument("file", type=click.Path(exists=True))
@click.option("--model-path", default="bioamla/scp-frogs", help="AST model to use for inference")
@click.option("--resample-freq", default=16000, type=int, help="Resampling frequency")
def ast_predict(
    file: str,
    model_path: str,
    resample_freq: int,
) -> None:
    """Perform AST prediction on a single audio file.

    Example:
        bioamla models ast predict audio.wav --model-path my_model
    """
    from bioamla.ml import predict_file

    try:
        prediction = predict_file(filepath=file, model_path=model_path, resample_freq=resample_freq)
    except BioamlaError as e:
        raise click.ClickException(str(e)) from e

    click.echo(f"{prediction.predicted_label} ({prediction.confidence:.4f})")


@ast.command("annotate")
@click.argument("file", type=click.Path(exists=True))
@click.option("--output", "-o", required=True, help="Output annotation file")
@click.option("--model-path", default="bioamla/scp-frogs", help="AST model to use for inference")
@click.option("--segment-duration", default=3, type=int, help="Segment duration (seconds)")
@click.option("--overlap", default=0, type=int, help="Overlap between segments (seconds)")
@click.option("--resample-freq", default=16000, type=int, help="Resampling frequency")
@click.option(
    "--format",
    "fmt",
    type=click.Choice(["csv", "raven", "bioamla"]),
    default="csv",
    help="Annotation output format",
)
@click.option("--exclude", multiple=True, help="Predicted label(s) to drop (repeatable)")
@click.option(
    "--min-confidence", default=0.0, type=float, help="Drop predictions below this confidence"
)
def ast_annotate(
    file: str,
    output: str,
    model_path: str,
    segment_duration: int,
    overlap: int,
    resample_freq: int,
    fmt: str,
    exclude: tuple[str, ...],
    min_confidence: float,
) -> None:
    """Run segmented AST inference and write an editable annotation file.

    Seeds the manual annotation/review step: each segment's prediction becomes an
    annotation you can correct, then feed to ``bioamla dataset extract-clips``.

    Example:
        bioamla models ast annotate soundscape.wav -o soundscape.csv --exclude background
    """
    from pathlib import Path

    from bioamla.datasets import predictions_to_annotations
    from bioamla.datasets._io import save_annotations
    from bioamla.ml import load_pretrained_ast_model, segmented_wave_file_inference

    try:
        model = load_pretrained_ast_model(model_path)
        df = segmented_wave_file_inference(file, model, resample_freq, segment_duration, overlap)
        annotations = predictions_to_annotations(
            df.to_dict("records"), min_confidence=min_confidence, exclude_labels=exclude
        )
        save_annotations(annotations, Path(output), fmt)
    except BioamlaError as e:
        raise click.ClickException(str(e)) from e

    click.echo(f"Wrote {len(annotations)} annotations to {output}")


@ast.command("train")
@click.pass_context
@click.option(
    "--config",
    "config_path",
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    help="TOML config file (e.g. from 'bioamla config init'). Explicit flags override "
    "its values; its values override defaults. Reads [training] and [models].",
)
@click.option("--training-dir", default=".", help="Directory to save training outputs")
@click.option(
    "--base-model",
    default="MIT/ast-finetuned-audioset-10-10-0.4593",
    help="Base model to fine-tune",
)
@click.option(
    "--train-dataset",
    required=True,
    help="Training data source: HuggingFace dataset (e.g. 'bioamla/scp-frogs'), "
    "local metadata CSV (with 'file' and 'label' columns), "
    "or directory with class subdirectories (e.g. ./data/bird/, ./data/frog/)",
)
@click.option("--split", default="train", help="Dataset split to use (for HuggingFace datasets)")
@click.option("--category-id-column", default="target", help="Column name for category IDs")
@click.option("--category-label-column", default="category", help="Column name for category labels")
@click.option("--report-to", default="tensorboard", help="Where to report metrics")
@click.option("--learning-rate", default=5.0e-5, type=float, help="Learning rate for training")
@click.option(
    "--push-to-hub/--no-push-to-hub", default=False, help="Whether to push model to HuggingFace Hub"
)
@click.option("--num-train-epochs", default=1, type=int, help="Number of training epochs")
@click.option(
    "--per-device-train-batch-size", default=8, type=int, help="Training batch size per device"
)
@click.option("--eval-strategy", default="epoch", help="Evaluation strategy")
@click.option("--save-strategy", default="epoch", help="Model save strategy")
@click.option("--eval-steps", default=1, type=int, help="Number of steps between evaluations")
@click.option("--save-steps", default=1, type=int, help="Number of steps between saves")
@click.option(
    "--load-best-model-at-end/--no-load-best-model-at-end",
    default=True,
    help="Load best model at end of training",
)
@click.option(
    "--metric-for-best-model", default="accuracy", help="Metric to use for best model selection"
)
@click.option("--logging-strategy", default="steps", help="Logging strategy")
@click.option("--logging-steps", default=100, type=int, help="Number of steps between logging")
@click.option(
    "--fp16/--no-fp16", default=False, help="Use FP16 mixed precision training (for NVIDIA GPUs)"
)
@click.option(
    "--bf16/--no-bf16", default=False, help="Use BF16 mixed precision training (for Ampere+ GPUs)"
)
@click.option(
    "--gradient-accumulation-steps",
    default=1,
    type=int,
    help="Number of gradient accumulation steps",
)
@click.option("--dataloader-num-workers", default=4, type=int, help="Number of dataloader workers")
@click.option(
    "--torch-compile/--no-torch-compile",
    default=False,
    help="Use torch.compile for faster training (PyTorch 2.0+)",
)
@click.option(
    "--finetune-mode",
    type=click.Choice(["full", "feature-extraction"]),
    default="full",
    help="Training mode: full (all layers) or feature-extraction (freeze base, train classifier only)",
)
@click.option(
    "--mlflow-tracking-uri",
    default=None,
    help="MLflow tracking server URI (e.g., http://localhost:5000)",
)
@click.option("--mlflow-experiment-name", default=None, help="MLflow experiment name")
@click.option("--mlflow-run-name", default=None, help="MLflow run name")
@click.option(
    "--augment/--no-augment", default=True, help="Enable audio augmentations during training"
)
@click.option(
    "--augment-multiplier",
    default=1,
    type=int,
    help="Create N augmented copies of each training sample (1=no copies, 2=double dataset, etc.)",
)
@click.option(
    "--augment-probability",
    default=0.8,
    type=click.FloatRange(0.0, 1.0),
    help="Probability of applying augmentation (0-1)",
)
@click.option("--min-snr-db", default=10.0, type=float, help="Minimum SNR for Gaussian noise (dB)")
@click.option("--max-snr-db", default=20.0, type=float, help="Maximum SNR for Gaussian noise (dB)")
@click.option("--min-gain-db", default=-6.0, type=float, help="Minimum gain adjustment (dB)")
@click.option("--max-gain-db", default=6.0, type=float, help="Maximum gain adjustment (dB)")
@click.option(
    "--clipping-probability",
    default=0.5,
    type=click.FloatRange(0.0, 1.0),
    help="Probability of clipping distortion (0-1)",
)
@click.option("--min-percentile-threshold", default=0, type=int, help="Min percentile for clipping")
@click.option(
    "--max-percentile-threshold", default=30, type=int, help="Max percentile for clipping"
)
@click.option("--min-time-stretch", default=0.8, type=float, help="Minimum time stretch rate")
@click.option("--max-time-stretch", default=1.2, type=float, help="Maximum time stretch rate")
@click.option("--min-pitch-shift", default=-4, type=int, help="Minimum pitch shift (semitones)")
@click.option("--max-pitch-shift", default=4, type=int, help="Maximum pitch shift (semitones)")
def ast_train(
    ctx: click.Context,
    config_path: str | None,
    training_dir: str,
    base_model: str,
    train_dataset: str,
    split: str,
    category_id_column: str,
    category_label_column: str,
    report_to: str,
    learning_rate: float,
    push_to_hub: bool,
    num_train_epochs: int,
    per_device_train_batch_size: int,
    eval_strategy: str,
    save_strategy: str,
    eval_steps: int,
    save_steps: int,
    load_best_model_at_end: bool,
    metric_for_best_model: str,
    logging_strategy: str,
    logging_steps: int,
    fp16: bool,
    bf16: bool,
    gradient_accumulation_steps: int,
    dataloader_num_workers: int,
    torch_compile: bool,
    finetune_mode: str,
    mlflow_tracking_uri: str,
    mlflow_experiment_name: str,
    mlflow_run_name: str,
    augment: bool,
    augment_multiplier: int,
    augment_probability: float,
    min_snr_db: float,
    max_snr_db: float,
    min_gain_db: float,
    max_gain_db: float,
    clipping_probability: float,
    min_percentile_threshold: int,
    max_percentile_threshold: int,
    min_time_stretch: float,
    max_time_stretch: float,
    min_pitch_shift: int,
    max_pitch_shift: int,
) -> None:
    """Fine-tune an AST model on a custom dataset.

    The --train-dataset option accepts three formats:

    \b
    1. HuggingFace dataset: bioamla/scp-frogs or samuelstevens/BirdSet:HSN
    2. Metadata CSV: ./data/metadata.csv (must have file and label columns)
    3. Directory with class subdirs: ./data/ containing bird/, frog/, etc.

    Examples:
        bioamla models ast train --train-dataset bioamla/scp-frogs
        bioamla models ast train --train-dataset ./metadata.csv
        bioamla models ast train --train-dataset ./audio_by_class/
    """
    from bioamla.cli.logging_setup import configure_cli_logging
    from bioamla.datasets.augmentation import AugmentationConfig
    from bioamla.ml import train_ast

    # Surface the library's INFO progress messages on the console.
    configure_cli_logging()

    # Overlay a TOML config (flags win over file, file wins over defaults).
    try:
        overrides = _apply_train_config(
            ctx,
            config_path,
            {
                "base_model": base_model,
                "learning_rate": learning_rate,
                "num_train_epochs": num_train_epochs,
                "per_device_train_batch_size": per_device_train_batch_size,
                "eval_strategy": eval_strategy,
                "save_strategy": save_strategy,
                "eval_steps": eval_steps,
                "save_steps": save_steps,
                "logging_steps": logging_steps,
            },
        )
    except BioamlaError as e:
        raise click.ClickException(str(e)) from e
    base_model = overrides["base_model"]
    learning_rate = overrides["learning_rate"]
    num_train_epochs = overrides["num_train_epochs"]
    per_device_train_batch_size = overrides["per_device_train_batch_size"]
    eval_strategy = overrides["eval_strategy"]
    save_strategy = overrides["save_strategy"]
    eval_steps = overrides["eval_steps"]
    save_steps = overrides["save_steps"]
    logging_steps = overrides["logging_steps"]

    # Map augmentation flags onto the shared AugmentationConfig (None disables it).
    # Per-transform probabilities default to 0.5 (audiomentations' default); the
    # whole Compose is gated by --augment-probability and shuffled per sample.
    aug_config = (
        AugmentationConfig(
            add_noise=True,
            noise_min_snr=min_snr_db,
            noise_max_snr=max_snr_db,
            noise_probability=0.5,
            time_stretch=True,
            time_stretch_min=min_time_stretch,
            time_stretch_max=max_time_stretch,
            time_stretch_probability=0.5,
            pitch_shift=True,
            pitch_shift_min=min_pitch_shift,
            pitch_shift_max=max_pitch_shift,
            pitch_shift_probability=0.5,
            gain=True,
            gain_min_db=min_gain_db,
            gain_max_db=max_gain_db,
            gain_probability=0.5,
            gain_transition=True,
            gain_transition_probability=0.5,
            clipping_distortion=True,
            clipping_min_percentile=min_percentile_threshold,
            clipping_max_percentile=max_percentile_threshold,
            clipping_probability=clipping_probability,
            pipeline_probability=augment_probability,
            shuffle=True,
        )
        if augment
        else None
    )

    try:
        result = train_ast(
            train_dataset=train_dataset,
            training_dir=training_dir,
            base_model=base_model,
            split=split,
            category_label_column=category_label_column,
            learning_rate=learning_rate,
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
            report_to=report_to,
            fp16=fp16,
            bf16=bf16,
            gradient_accumulation_steps=gradient_accumulation_steps,
            dataloader_num_workers=dataloader_num_workers,
            torch_compile=torch_compile,
            finetune_mode=finetune_mode,
            push_to_hub=push_to_hub,
            mlflow_tracking_uri=mlflow_tracking_uri,
            mlflow_experiment_name=mlflow_experiment_name,
            mlflow_run_name=mlflow_run_name,
            augmentation=aug_config,
            augment_multiplier=augment_multiplier,
        )
    except BioamlaError as e:
        raise click.ClickException(str(e)) from e

    click.echo(f"Training complete. Best model saved to: {result.model_path}")
    if result.final_accuracy is not None:
        click.echo(f"Final eval accuracy: {result.final_accuracy:.4f}")
    if result.final_loss is not None:
        click.echo(f"Final eval loss: {result.final_loss:.4f}")


@ast.command("evaluate")
@click.argument("path", type=click.Path(exists=True))
@click.option("--model-path", default="bioamla/scp-frogs", help="AST model to use for evaluation")
@click.option(
    "--ground-truth", "-g", required=True, help="Path to CSV file with ground truth labels"
)
@click.option("--output", "-o", default=None, help="Output file for evaluation results")
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["json", "csv", "txt"]),
    default="txt",
    help="Output format",
)
@click.option(
    "--file-column", default="file_name", help="Column name for file names in ground truth CSV"
)
@click.option("--label-column", default="label", help="Column name for labels in ground truth CSV")
@click.option("--resample-freq", default=16000, type=int, help="Resampling frequency")
@click.option("--batch-size", default=8, type=int, help="Batch size for inference")
@click.option("--fp16/--no-fp16", default=False, help="Use half-precision inference")
@click.option("--quiet", is_flag=True, help="Only output metrics, suppress progress")
def ast_evaluate(
    path: str,
    model_path: str,
    ground_truth: str,
    output: str,
    output_format: str,
    file_column: str,
    label_column: str,
    resample_freq: int,
    batch_size: int,
    fp16: bool,
    quiet: bool,
) -> None:
    """Evaluate an AST model on a directory of audio files.

    Example:
        bioamla models ast evaluate ./audio_dir --model-path my_model -g labels.csv
    """
    import json as json_lib
    from pathlib import Path as PathLib

    from bioamla.ml import evaluate_directory

    try:
        eval_result = evaluate_directory(
            audio_dir=path,
            model_path=model_path,
            ground_truth_csv=ground_truth,
            file_column=file_column,
            label_column=label_column,
            resample_freq=resample_freq,
            use_fp16=fp16,
        )
    except BioamlaError as e:
        raise click.ClickException(str(e)) from e

    if not quiet:
        click.echo("\nEvaluation Results:")
        click.echo("-" * 40)
    click.echo(f"Accuracy: {eval_result.accuracy:.4f}")
    click.echo(f"Precision: {eval_result.precision:.4f}")
    click.echo(f"Recall: {eval_result.recall:.4f}")
    click.echo(f"F1 Score: {eval_result.f1_score:.4f}")
    click.echo(f"Total Samples: {eval_result.total_samples}")

    if output:
        out_path = PathLib(output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if output_format == "json":
            out_path.write_text(json_lib.dumps(eval_result.to_dict(), indent=2), encoding="utf-8")
        else:
            out_path.write_text(str(eval_result.to_dict()), encoding="utf-8")
        click.echo(f"Results saved to: {output}")


@ast.command("embed")
@click.argument("file", type=click.Path(exists=True))
@click.option("--model-path", required=True, help="Path to AST model or HuggingFace identifier")
@click.option("--output", "-o", required=True, help="Output file (.npy)")
@click.option("--layer", default=None, help="Layer to extract embeddings from")
@click.option("--sample-rate", default=16000, type=int, help="Target sample rate")
def ast_embed(file: str, model_path: str, output: str, layer: str, sample_rate: int) -> None:
    """Extract embeddings from audio using AST model.

    Example:
        bioamla models ast embed audio.wav --model-path my_model -o embeddings.npy
    """
    from pathlib import Path

    import numpy as np

    from bioamla.ml import extract_embeddings_file

    click.echo(f"Loading AST model from {model_path}...")

    try:
        result = extract_embeddings_file(
            filepath=file,
            model_path=model_path,
            layer=layer,
            sample_rate=sample_rate,
        )
    except BioamlaError as e:
        raise click.ClickException(str(e)) from e

    embeddings = result["embeddings"]
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    np.save(output, embeddings)
    click.echo(f"Embeddings saved to {output} (shape: {embeddings.shape})")


@ast.command("info")
@click.argument("model_path")
def ast_info(model_path: str) -> None:
    """Display information about an AST model.

    Example:
        bioamla models ast info bioamla/scp-frogs
    """
    from bioamla.ml import get_model_info

    try:
        info = get_model_info(model_path)
    except BioamlaError as e:
        raise click.ClickException(str(e)) from e

    click.echo(f"Model: {info['path']}")
    click.echo(f"Type: {info['model_type']}")
    click.echo(f"Classes: {info['num_classes']}")
    if info.get("classes"):
        labels = ", ".join(str(c) for c in info["classes"])
        if info.get("has_more_classes"):
            labels += f"... (+{info['num_classes'] - 10} more)"
        click.echo(f"Labels: {labels}")
