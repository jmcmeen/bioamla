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

# Map ``ast train`` flag names onto config-file (section, key) locations
# (e.g. [training].epochs -> --num-train-epochs). The ``ast init-config`` template
# (bioamla.ml.train_config) documents this same schema. Flags set on the command
# line override these; these override built-in defaults.
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
    "report_to": ("training", "report_to"),
    # Augmentation lives in its own [augmentation] section. Every transform is
    # off by default; a key here (or its CLI flag) opts that layer in. Ranges and
    # probabilities apply only to the transforms that are enabled.
    "add_noise": ("augmentation", "add_noise"),
    "time_stretch": ("augmentation", "time_stretch"),
    "pitch_shift": ("augmentation", "pitch_shift"),
    "gain": ("augmentation", "gain"),
    "gain_transition": ("augmentation", "gain_transition"),
    "clipping_distortion": ("augmentation", "clipping_distortion"),
    "augment_multiplier": ("augmentation", "multiplier"),
    "augment_probability": ("augmentation", "probability"),
    "min_snr_db": ("augmentation", "min_snr_db"),
    "max_snr_db": ("augmentation", "max_snr_db"),
    "noise_probability": ("augmentation", "noise_probability"),
    "min_gain_db": ("augmentation", "min_gain_db"),
    "max_gain_db": ("augmentation", "max_gain_db"),
    "gain_probability": ("augmentation", "gain_probability"),
    "gain_transition_probability": ("augmentation", "gain_transition_probability"),
    "clipping_probability": ("augmentation", "clipping_probability"),
    "min_percentile_threshold": ("augmentation", "min_percentile_threshold"),
    "max_percentile_threshold": ("augmentation", "max_percentile_threshold"),
    "min_time_stretch": ("augmentation", "min_time_stretch"),
    "max_time_stretch": ("augmentation", "max_time_stretch"),
    "time_stretch_probability": ("augmentation", "time_stretch_probability"),
    "min_pitch_shift": ("augmentation", "min_pitch_shift"),
    "max_pitch_shift": ("augmentation", "max_pitch_shift"),
    "pitch_shift_probability": ("augmentation", "pitch_shift_probability"),
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
    from bioamla.exceptions import InvalidInputError
    from bioamla.ml import write_train_config

    try:
        path = write_train_config(output, force=force)
    except InvalidInputError as e:
        click.echo(str(e), err=True)
        if "already exists" in str(e):
            click.echo("Use --force to overwrite.")
        raise SystemExit(1) from e
    except BioamlaError as e:
        raise click.ClickException(str(e)) from e

    print_success(f"Created training config: {path}")


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
def ast_predict(
    file: str,
    model_path: str,
    segment_duration: int,
    overlap: int,
    min_confidence: float,
    output: str | None,
    resample_freq: int,
) -> None:
    """Run AST prediction on a single audio file — whole file or in segments.

    With ``--segment-seconds`` the file is split into fixed-length (optionally
    overlapping) segments and each is classified, yielding one prediction per
    segment; otherwise the whole file gets a single prediction.

    Examples:
        bioamla models ast predict audio.wav --model-path my_model
        bioamla models ast predict rec.wav --segment-seconds 3 --overlap 1 -o rec.csv
    """
    import csv
    from pathlib import Path

    from bioamla.cli.console import echo, print_success
    from bioamla.ml import ASTInference

    try:
        inference = ASTInference(model_path=model_path, sample_rate=resample_freq)
        if segment_duration > 0:
            results = inference.predict_segments(
                file, clip_length=segment_duration, overlap=overlap
            )
        else:
            results = [inference.predict(file)]
    except BioamlaError as e:
        raise click.ClickException(str(e)) from e

    results = [r for r in results if r.confidence >= min_confidence]

    if output:
        out_path = Path(output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["filepath", "start", "stop", "prediction", "confidence"])
            for r in results:
                writer.writerow(
                    [r.filepath, r.start_time, r.end_time, r.predicted_label, f"{r.confidence:.6f}"]
                )
        print_success(f"Wrote {len(results)} prediction(s) to {output}")
        return

    if segment_duration > 0:
        for r in results:
            echo(f"{r.start_time:.2f}-{r.end_time:.2f}s  {r.predicted_label} ({r.confidence:.4f})")
    else:
        for r in results:
            echo(f"{r.predicted_label} ({r.confidence:.4f})")


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
@click.option(
    "--id-column",
    "category_id_column",
    default="target",
    help="Column name for category/target IDs",
)
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
    from bioamla.cli.console import print_kv, print_success
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
                "report_to": report_to,
                "add_noise": add_noise,
                "time_stretch": time_stretch,
                "pitch_shift": pitch_shift,
                "gain": gain,
                "gain_transition": gain_transition,
                "clipping_distortion": clipping_distortion,
                "augment_multiplier": augment_multiplier,
                "augment_probability": augment_probability,
                "min_snr_db": min_snr_db,
                "max_snr_db": max_snr_db,
                "noise_probability": noise_probability,
                "min_gain_db": min_gain_db,
                "max_gain_db": max_gain_db,
                "gain_probability": gain_probability,
                "gain_transition_probability": gain_transition_probability,
                "clipping_probability": clipping_probability,
                "min_percentile_threshold": min_percentile_threshold,
                "max_percentile_threshold": max_percentile_threshold,
                "min_time_stretch": min_time_stretch,
                "max_time_stretch": max_time_stretch,
                "time_stretch_probability": time_stretch_probability,
                "min_pitch_shift": min_pitch_shift,
                "max_pitch_shift": max_pitch_shift,
                "pitch_shift_probability": pitch_shift_probability,
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
    report_to = overrides["report_to"]

    # Each augmentation layer is opt-in (off by default); enable only the ones the
    # user turned on via flag or [augmentation] TOML. If none are enabled, pass
    # None so training skips the pipeline entirely. Each enabled layer has its own
    # per-sample probability; the whole Compose is additionally gated by
    # --augment-probability and shuffled per sample. Settings come from `overrides`
    # so the TOML can drive them (CLI flag still wins).
    layers = (
        "add_noise",
        "time_stretch",
        "pitch_shift",
        "gain",
        "gain_transition",
        "clipping_distortion",
    )
    augment_multiplier = overrides["augment_multiplier"]
    aug_config = (
        AugmentationConfig(
            add_noise=overrides["add_noise"],
            noise_min_snr=overrides["min_snr_db"],
            noise_max_snr=overrides["max_snr_db"],
            noise_probability=overrides["noise_probability"],
            time_stretch=overrides["time_stretch"],
            time_stretch_min=overrides["min_time_stretch"],
            time_stretch_max=overrides["max_time_stretch"],
            time_stretch_probability=overrides["time_stretch_probability"],
            pitch_shift=overrides["pitch_shift"],
            pitch_shift_min=overrides["min_pitch_shift"],
            pitch_shift_max=overrides["max_pitch_shift"],
            pitch_shift_probability=overrides["pitch_shift_probability"],
            gain=overrides["gain"],
            gain_min_db=overrides["min_gain_db"],
            gain_max_db=overrides["max_gain_db"],
            gain_probability=overrides["gain_probability"],
            gain_transition=overrides["gain_transition"],
            gain_transition_probability=overrides["gain_transition_probability"],
            clipping_distortion=overrides["clipping_distortion"],
            clipping_min_percentile=overrides["min_percentile_threshold"],
            clipping_max_percentile=overrides["max_percentile_threshold"],
            clipping_probability=overrides["clipping_probability"],
            pipeline_probability=overrides["augment_probability"],
            shuffle=True,
        )
        if any(overrides[layer] for layer in layers)
        else None
    )

    # A multiplier only does useful work alongside augmentation — without it, the
    # train split is just duplicated verbatim. Warn rather than silently no-op.
    if aug_config is None and augment_multiplier > 1:
        click.echo(
            f"Warning: --augment-multiplier {augment_multiplier} ignored — no "
            "augmentation layers enabled (the copies would be identical).",
            err=True,
        )
        augment_multiplier = 1

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
    import json as json_lib
    from pathlib import Path as PathLib

    from bioamla.cli.console import echo, print_header, print_kv, print_success
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
        print_header("\nEvaluation Results:")
        echo("-" * 40)
    print_kv("Accuracy", f"{eval_result.accuracy:.4f}")
    print_kv("Precision", f"{eval_result.precision:.4f}")
    print_kv("Recall", f"{eval_result.recall:.4f}")
    print_kv("F1 Score", f"{eval_result.f1_score:.4f}")
    print_kv("Total Samples", eval_result.total_samples)

    if output:
        out_path = PathLib(output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if output_format == "json":
            out_path.write_text(json_lib.dumps(eval_result.to_dict(), indent=2), encoding="utf-8")
        else:
            out_path.write_text(str(eval_result.to_dict()), encoding="utf-8")
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
    from pathlib import Path

    import numpy as np

    from bioamla.cli.console import print_info, print_success
    from bioamla.ml import extract_embeddings_file

    print_info(f"Loading AST model from {model_path}...")

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
    print_success(f"Embeddings saved to {output} (shape: {embeddings.shape})")


@ast.command("info")
@click.argument("model_path")
def ast_info(model_path: str) -> None:
    """Display information about an AST model.

    Example:
        bioamla models ast info bioamla/scp-frogs
    """
    from bioamla.cli.console import print_kv
    from bioamla.ml import get_model_info

    try:
        info = get_model_info(model_path)
    except BioamlaError as e:
        raise click.ClickException(str(e)) from e

    print_kv("Model", info["path"])
    print_kv("Type", info["model_type"])
    print_kv("Classes", info["num_classes"])
    if info.get("classes"):
        labels = ", ".join(str(c) for c in info["classes"])
        if info.get("has_more_classes"):
            labels += f"... (+{info['num_classes'] - 10} more)"
        print_kv("Labels", labels)
