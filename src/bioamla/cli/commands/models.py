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


@ast.command("init-config")
@click.option("--output", "-o", default="ast_training.toml", help="Output file path")
@click.option("--force", "-f", is_flag=True, help="Overwrite an existing file")
def ast_init_config(output: str, force: bool) -> None:
    """Write a documented AST training-config file for use with ``train --config``.

    The file holds the [models], [training], and [augmentation] settings that
    ``ast train`` reads; edit it, then pass it via ``--config``. Command-line
    flags still override any value in the file.
    """
    from bioamla.cli.console import print_success
    from bioamla.cli.operations import run_operation
    from bioamla.cli.output import emit_result

    result = run_operation("models.ast.init-config", output=output, force=force)
    if emit_result(result):
        return
    print_success(f"Created training config: {result['output']}")


@ast.command("predict")
@click.argument("file", type=click.Path(exists=True))
@click.option("--model-path", default="bioamla/scp-frogs", help="AST model to use for inference")
@click.option(
    "--segment-seconds",
    "segment_duration",
    default=0,
    type=int,
    help="Split into N-second segments and classify each (0 = classify the whole file)",
)
@click.option("--overlap", default=0, type=int, help="Overlap between segments (seconds)")
@click.option(
    "--min-confidence", default=0.0, type=float, help="Drop predictions below this confidence"
)
@click.option(
    "--output",
    "-o",
    default=None,
    help="Write predictions to this CSV (filepath,start,stop,prediction,confidence). "
    "Default: print to stdout.",
)
@click.option("--resample-freq", default=16000, type=int, help="Resampling frequency")
@click.option(
    "--top-k",
    default=0,
    type=click.IntRange(min=0),
    help="Also rank the K most likely labels of each prediction, with the margin between "
    "the first two (adds `top` and `margin`; CSV: margin and top columns)",
)
def ast_predict(
    file: str,
    model_path: str,
    segment_duration: int,
    overlap: int,
    min_confidence: float,
    output: str | None,
    resample_freq: int,
    top_k: int,
) -> None:
    """Run AST prediction on a single audio file — whole file or in segments.

    With ``--segment-seconds`` the file is split into fixed-length (optionally
    overlapping) segments and each is classified, yielding one prediction per
    segment; otherwise the whole file gets a single prediction.

    Examples:
        bioamla models ast predict audio.wav --model-path my_model
        bioamla models ast predict rec.wav --segment-seconds 3 --overlap 1 -o rec.csv
        bioamla models ast predict rec.wav --segment-seconds 3 --top-k 3
    """
    from bioamla.cli.console import echo, print_success
    from bioamla.cli.operations import run_operation
    from bioamla.cli.output import emit_file_result, json_mode

    found = run_operation(
        "models.ast.predict",
        file=file,
        model_path=model_path,
        segment_duration=segment_duration,
        overlap=overlap,
        min_confidence=min_confidence,
        output=output,
        resample_freq=resample_freq,
        top_k=top_k,
    )
    if json_mode():
        emit_file_result(file, count=found.count, predictions=found.to_dict()["predictions"])
        return
    if output:
        print_success(f"Wrote {found.count} prediction(s) to {output}")
        return

    for r in found.predictions:
        span = f"{r.start_time:.2f}-{r.end_time:.2f}s  " if found.segmented else ""
        line = f"{span}{r.predicted_label} ({r.confidence:.4f})"
        if r.top_k_labels is not None:
            ranked = ", ".join(f"{t['label']} {t['confidence']:.4f}" for t in r.top())
            margin = "" if r.margin is None else f"  margin {r.margin:.4f}"
            line += f"  [{ranked}]{margin}"
        echo(line)


@ast.command("train")
@click.pass_context
@click.option(
    "--config",
    "config_path",
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    help="TOML config file (e.g. from 'bioamla models ast init-config'). Explicit flags "
    "override its values; its values override defaults. Reads [models], [training] and "
    "[augmentation].",
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
@click.option(
    "--label-column", "category_label_column", default="category", help="Column name for labels"
)
@click.option(
    "--report-to",
    default="tensorboard",
    help="Where to report metrics: tensorboard (default, bundled), mlflow, none, ... "
    "(comma-separated).",
)
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
    "--add-noise/--no-add-noise",
    default=False,
    help="Augmentation layer: add Gaussian noise (off by default)",
)
@click.option(
    "--time-stretch/--no-time-stretch",
    default=False,
    help="Augmentation layer: time-stretch (off by default)",
)
@click.option(
    "--pitch-shift/--no-pitch-shift",
    default=False,
    help="Augmentation layer: pitch-shift (off by default)",
)
@click.option(
    "--gain/--no-gain",
    default=False,
    help="Augmentation layer: random gain (off by default)",
)
@click.option(
    "--gain-transition/--no-gain-transition",
    default=False,
    help="Augmentation layer: smooth gain ramp (off by default)",
)
@click.option(
    "--clipping-distortion/--no-clipping-distortion",
    default=False,
    help="Augmentation layer: clipping distortion (off by default)",
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
    help="Probability the whole augmentation pipeline is applied to a sample (0-1)",
)
@click.option("--min-snr-db", default=10.0, type=float, help="Minimum SNR for Gaussian noise (dB)")
@click.option("--max-snr-db", default=20.0, type=float, help="Maximum SNR for Gaussian noise (dB)")
@click.option(
    "--noise-probability",
    default=0.5,
    type=click.FloatRange(0.0, 1.0),
    help="Per-sample probability of the noise layer (0-1)",
)
@click.option("--min-gain-db", default=-6.0, type=float, help="Minimum gain adjustment (dB)")
@click.option("--max-gain-db", default=6.0, type=float, help="Maximum gain adjustment (dB)")
@click.option(
    "--gain-probability",
    default=0.5,
    type=click.FloatRange(0.0, 1.0),
    help="Per-sample probability of the gain layer (0-1)",
)
@click.option(
    "--gain-transition-probability",
    default=0.5,
    type=click.FloatRange(0.0, 1.0),
    help="Per-sample probability of the gain-transition layer (0-1)",
)
@click.option(
    "--clipping-probability",
    default=0.5,
    type=click.FloatRange(0.0, 1.0),
    help="Per-sample probability of the clipping-distortion layer (0-1)",
)
@click.option("--min-percentile-threshold", default=0, type=int, help="Min percentile for clipping")
@click.option(
    "--max-percentile-threshold", default=30, type=int, help="Max percentile for clipping"
)
@click.option("--min-time-stretch", default=0.8, type=float, help="Minimum time stretch rate")
@click.option("--max-time-stretch", default=1.2, type=float, help="Maximum time stretch rate")
@click.option(
    "--time-stretch-probability",
    default=0.5,
    type=click.FloatRange(0.0, 1.0),
    help="Per-sample probability of the time-stretch layer (0-1)",
)
@click.option("--min-pitch-shift", default=-4, type=int, help="Minimum pitch shift (semitones)")
@click.option("--max-pitch-shift", default=4, type=int, help="Maximum pitch shift (semitones)")
@click.option(
    "--pitch-shift-probability",
    default=0.5,
    type=click.FloatRange(0.0, 1.0),
    help="Per-sample probability of the pitch-shift layer (0-1)",
)
def ast_train(
    ctx: click.Context,
    config_path: str | None,
    training_dir: str,
    base_model: str,
    train_dataset: str,
    split: str,
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
    add_noise: bool,
    time_stretch: bool,
    pitch_shift: bool,
    gain: bool,
    gain_transition: bool,
    clipping_distortion: bool,
    augment_multiplier: int,
    augment_probability: float,
    min_snr_db: float,
    max_snr_db: float,
    noise_probability: float,
    min_gain_db: float,
    max_gain_db: float,
    gain_probability: float,
    gain_transition_probability: float,
    clipping_probability: float,
    min_percentile_threshold: int,
    max_percentile_threshold: int,
    min_time_stretch: float,
    max_time_stretch: float,
    time_stretch_probability: float,
    min_pitch_shift: int,
    max_pitch_shift: int,
    pitch_shift_probability: float,
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
    from click.core import ParameterSource

    from bioamla.cli.console import print_kv, print_success
    from bioamla.cli.logging_setup import configure_cli_logging
    from bioamla.cli.operations import run_operation

    # Surface the library's INFO progress messages (and warnings) on the console.
    configure_cli_logging()

    # Only the flags given on the command line are passed: they win over the config
    # file, which wins over the library's defaults.
    not_options = {"config_path", "train_dataset", "training_dir"}
    options = {
        name: value
        for name, value in ctx.params.items()
        if name not in not_options and ctx.get_parameter_source(name) != ParameterSource.DEFAULT
    }
    result = run_operation(
        "models.ast.train",
        train_dataset=train_dataset,
        config_path=config_path,
        training_dir=training_dir,
        options=options,
    )

    from bioamla.cli.output import emit_result

    if emit_result(result):
        return
    print_success(f"Training complete. Best model saved to: {result.model_path}")
    if result.final_accuracy is not None:
        print_kv("Final eval accuracy", f"{result.final_accuracy:.4f}")
    if result.final_loss is not None:
        print_kv("Final eval loss", f"{result.final_loss:.4f}")


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
@click.option("--quiet", "-q", is_flag=True, help="Only output metrics, suppress progress")
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

    \b
    Examples:
        # Accuracy/precision/recall against a ground-truth CSV
        bioamla models ast evaluate ./audio_dir --model-path my_model -g labels.csv
        # Write a JSON report
        bioamla models ast evaluate ./audio_dir -g labels.csv \\
            --format json -o eval.json
    """
    from bioamla.cli.console import (
        echo,
        print_header,
        print_kv,
        print_success,
        print_table,
        print_warning,
    )
    from bioamla.cli.operations import run_operation
    from bioamla.cli.output import command_options, emit_result, json_mode

    report = run_operation(
        "models.ast.evaluate",
        path=path,
        ground_truth=ground_truth,
        model_path=model_path,
        output=output,
        output_format=output_format,
        file_column=file_column,
        label_column=label_column,
        resample_freq=resample_freq,
        batch_size=batch_size,
        fp16=fp16,
    )
    eval_result = report.metrics

    if json_mode():
        emit_result(
            {
                "path": path,
                "params": command_options(exclude=("output", "output_format")),
                "metrics": eval_result.to_dict(),
                "output": output,
            }
        )
        return

    if not quiet:
        print_header("\nEvaluation Results:")
        echo("-" * 40)
    print_kv("Accuracy", f"{eval_result.accuracy:.4f}")
    print_kv("Precision", f"{eval_result.precision:.4f}")
    print_kv("Recall", f"{eval_result.recall:.4f}")
    print_kv("F1 Score", f"{eval_result.f1_score:.4f}")
    print_kv("Total Samples", eval_result.total_samples)
    if eval_result.per_class:
        print_table(
            "Per class",
            ["Label", "Precision", "Recall", "F1", "Support"],
            [
                [
                    label,
                    f"{m['precision']:.3f}",
                    f"{m['recall']:.3f}",
                    f"{m['f1_score']:.3f}",
                    m["support"],
                ]
                for label, m in eval_result.per_class.items()
            ],
        )
    if eval_result.skipped:
        print_warning(f"Skipped {len(eval_result.skipped)} file(s):")
        for skipped in eval_result.skipped[:10]:
            echo(f"  {skipped}")
        if len(eval_result.skipped) > 10:
            echo(f"  ... and {len(eval_result.skipped) - 10} more (all in --format txt/json)")
    if output:
        print_success(f"Results saved to: {output}")


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
    from bioamla.cli.console import print_info, print_success
    from bioamla.cli.operations import run_operation
    from bioamla.cli.output import emit_file_result

    print_info(f"Loading AST model from {model_path}...")
    saved = run_operation(
        "models.ast.embed",
        file=file,
        model_path=model_path,
        output=output,
        layer=layer,
        sample_rate=sample_rate,
    )
    # The vectors stay in the .npy file; the result says where and what shape.
    if emit_file_result(file, shape=saved.shape, dtype=saved.dtype):
        return
    print_success(f"Embeddings saved to {output} (shape: {tuple(saved.shape)})")


@ast.command("info")
@click.argument("model_path")
def ast_info(model_path: str) -> None:
    """Display information about an AST model.

    Example:
        bioamla models ast info bioamla/scp-frogs
    """
    from bioamla.cli.console import print_kv
    from bioamla.cli.operations import run_operation
    from bioamla.cli.output import emit_result, json_mode

    info = run_operation("models.ast.info", model_path=model_path)
    if json_mode():
        emit_result(
            {
                "model": info["path"],
                "model_type": info["model_type"],
                "num_classes": info["num_classes"],
                "labels": info["labels"],
                "hidden_size": info["hidden_size"],
            }
        )
        return

    print_kv("Model", info["path"])
    print_kv("Type", info["model_type"])
    print_kv("Classes", info["num_classes"])
    if info.get("classes"):
        labels = ", ".join(str(c) for c in info["classes"])
        if info.get("has_more_classes"):
            labels += f"... (+{info['num_classes'] - 10} more)"
        print_kv("Labels", labels)
