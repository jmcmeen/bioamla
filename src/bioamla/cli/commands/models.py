"""ML model operations - AST and CNN models.

Command structure:
    bioamla models {architecture} {command}

Examples:
    bioamla models ast predict audio.wav --model-path my_model
    bioamla models ast train --train-dataset bioamla/scp-frogs
    bioamla models ast train --train-dataset ./metadata.csv
    bioamla models ast train --train-dataset ./audio_by_class/
    bioamla models cnn predict audio.wav --model-path model.pt
    bioamla models cnn train --train-csv data.csv --output-dir ./model
"""

from typing import Dict

import click


@click.group()
def models() -> None:
    """ML model operations - AST and CNN models."""
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
    from bioamla.cli.service_helpers import handle_result, services

    result = services.inference.predict(filepath=file, model_path=model_path)
    predictions = handle_result(result)

    for pred in predictions:
        click.echo(f"{pred.predicted_label} ({pred.confidence:.4f})")


@ast.command("train")
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
@click.option("--max-percentile-threshold", default=30, type=int, help="Max percentile for clipping")
@click.option("--min-time-stretch", default=0.8, type=float, help="Minimum time stretch rate")
@click.option("--max-time-stretch", default=1.2, type=float, help="Maximum time stretch rate")
@click.option("--min-pitch-shift", default=-4, type=int, help="Minimum pitch shift (semitones)")
@click.option("--max-pitch-shift", default=4, type=int, help="Maximum pitch shift (semitones)")
def ast_train(
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
    import evaluate
    import numpy as np
    import torch
    from audiomentations import (
        AddGaussianSNR,
        ClippingDistortion,
        Compose,
        Gain,
        GainTransition,
        PitchShift,
        TimeStretch,
    )
    from datasets import Audio, Dataset, DatasetDict, load_dataset
    from transformers import (
        ASTConfig,
        ASTFeatureExtractor,
        ASTForAudioClassification,
        Trainer,
        TrainingArguments,
    )

    from bioamla.cli.service_helpers import services

    # Validate min/max ranges
    if min_snr_db > max_snr_db:
        raise click.BadParameter(f"--min-snr-db ({min_snr_db}) must be <= --max-snr-db ({max_snr_db})")
    if min_gain_db > max_gain_db:
        raise click.BadParameter(f"--min-gain-db ({min_gain_db}) must be <= --max-gain-db ({max_gain_db})")
    if min_percentile_threshold > max_percentile_threshold:
        raise click.BadParameter(
            f"--min-percentile-threshold ({min_percentile_threshold}) must be <= --max-percentile-threshold ({max_percentile_threshold})"
        )
    if min_time_stretch > max_time_stretch:
        raise click.BadParameter(
            f"--min-time-stretch ({min_time_stretch}) must be <= --max-time-stretch ({max_time_stretch})"
        )
    if min_pitch_shift > max_pitch_shift:
        raise click.BadParameter(
            f"--min-pitch-shift ({min_pitch_shift}) must be <= --max-pitch-shift ({max_pitch_shift})"
        )

    output_dir = training_dir + "/runs"
    logging_dir = training_dir + "/logs"
    best_model_path = training_dir + "/best_model"

    if mlflow_tracking_uri or mlflow_experiment_name:
        try:
            import mlflow

            if mlflow_tracking_uri:
                mlflow.set_tracking_uri(mlflow_tracking_uri)
                print(f"MLflow tracking URI: {mlflow_tracking_uri}")
            if mlflow_experiment_name:
                mlflow.set_experiment(mlflow_experiment_name)
                print(f"MLflow experiment: {mlflow_experiment_name}")
            if "mlflow" not in report_to:
                report_to = f"{report_to},mlflow" if report_to else "mlflow"
            print(f"MLflow integration enabled, reporting to: {report_to}")
        except ImportError:
            print(
                "Warning: MLflow not installed. Install with 'pip install mlflow' to enable MLflow tracking."
            )

    if report_to and "," in report_to:
        report_to = [r.strip() for r in report_to.split(",")]

    # Determine dataset source type and load accordingly
    from pathlib import Path
    train_path = Path(train_dataset)

    if train_path.exists():
        # Local file or directory
        if train_path.is_file() and train_path.suffix.lower() == ".csv":
            # Metadata CSV file
            click.echo(f"Loading from metadata CSV: {train_dataset}")
            import pandas as pd

            # Read CSV and determine structure
            df = pd.read_csv(train_path)

            # Look for common column names for file path
            file_col = None
            for col in ["file", "filepath", "path", "audio", "filename", "file_path", "file_name"]:
                if col in df.columns:
                    file_col = col
                    break
            if file_col is None:
                raise click.BadParameter(
                    f"CSV must have a file column (tried: file, filepath, path, audio, filename, file_path). "
                    f"Found columns: {list(df.columns)}"
                )

            # Look for label column
            label_col = None
            for col in ["label", "category", "class", "species", category_label_column]:
                if col in df.columns:
                    label_col = col
                    break
            if label_col is None:
                raise click.BadParameter(
                    f"CSV must have a label column (tried: label, category, class, species, {category_label_column}). "
                    f"Found columns: {list(df.columns)}"
                )

            # Update category_label_column to the found column
            category_label_column = label_col

            # Resolve file paths relative to CSV location
            csv_dir = train_path.parent
            def resolve_path(p):
                p = Path(p)
                if not p.is_absolute():
                    p = csv_dir / p
                return str(p)

            df["audio"] = df[file_col].apply(resolve_path)
            df[category_label_column] = df[label_col]

            # Create HuggingFace dataset from DataFrame
            dataset = Dataset.from_pandas(df[["audio", category_label_column]])
            dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
            click.echo(f"Loaded {len(dataset)} samples from CSV")

        elif train_path.is_dir():
            # Directory - check for audiofolder structure (subdirs as classes)
            subdirs = [d for d in train_path.iterdir() if d.is_dir()]
            audio_extensions = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}

            if subdirs:
                # Check if subdirs contain audio files (audiofolder format)
                has_audio_in_subdirs = any(
                    any(f.suffix.lower() in audio_extensions for f in subdir.iterdir() if f.is_file())
                    for subdir in subdirs
                )
                if has_audio_in_subdirs:
                    click.echo(f"Loading from audiofolder directory: {train_dataset}")
                    click.echo(f"Found class directories: {[d.name for d in subdirs]}")
                    dataset = load_dataset("audiofolder", data_dir=train_dataset, split=split)
                    # audiofolder uses 'label' column
                    category_label_column = "label"
                else:
                    raise click.BadParameter(
                        f"Directory {train_dataset} has subdirectories but they don't contain audio files. "
                        f"Expected structure: {train_dataset}/class_name/*.wav"
                    )
            else:
                # Flat directory - look for a metadata CSV inside
                csv_files = list(train_path.glob("*.csv"))
                if csv_files:
                    click.echo(f"Found metadata CSV in directory: {csv_files[0]}")
                    # Recursively call with the CSV path
                    train_dataset = str(csv_files[0])
                    train_path = Path(train_dataset)
                    # Re-process as CSV (this is a bit hacky but avoids code duplication)
                    raise click.BadParameter(
                        f"Directory contains CSV files. Please specify the CSV directly: --train-dataset {csv_files[0]}"
                    )
                else:
                    raise click.BadParameter(
                        f"Directory {train_dataset} must either have class subdirectories (e.g., bird/, frog/) "
                        f"or contain a metadata CSV file."
                    )
        else:
            raise click.BadParameter(
                f"Local path {train_dataset} exists but is neither a CSV file nor a directory."
            )
    elif ":" in train_dataset:
        # HuggingFace dataset with config (e.g., 'samuelstevens/BirdSet:HSN')
        dataset_name, config_name = train_dataset.rsplit(":", 1)
        click.echo(f"Loading from HuggingFace Hub: {dataset_name} (config: {config_name})")
        dataset = load_dataset(dataset_name, config_name, split=split)
    elif "BirdSet" in train_dataset:
        click.echo(
            "Note: BirdSet detected. Use format 'samuelstevens/BirdSet:HSN' to specify subset."
        )
        click.echo("Available subsets: HSN, NBP, NES, PER")
        dataset = load_dataset(train_dataset, "HSN", split=split)
    else:
        # Assume HuggingFace dataset name
        click.echo(f"Loading from HuggingFace Hub: {train_dataset}")
        dataset = load_dataset(train_dataset, split=split)

    if isinstance(dataset, Dataset):
        class_names = sorted(set(dataset[category_label_column]))
    elif isinstance(dataset, DatasetDict):
        first_split_name = list(dataset.keys())[0]
        first_split = dataset[first_split_name]
        class_names = sorted(set(first_split[category_label_column]))
    else:
        raise TypeError("Dataset must be a Dataset or DatasetDict instance")

    label_to_id = {name: idx for idx, name in enumerate(class_names)}
    num_labels = len(class_names)

    def convert_labels(example):
        example["labels"] = label_to_id[example[category_label_column]]
        return example

    dataset = dataset.map(
        convert_labels,
        remove_columns=[category_label_column],
        writer_batch_size=100,
        num_proc=1,
    )

    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

    pretrained_model = base_model
    feature_extractor = ASTFeatureExtractor.from_pretrained(pretrained_model)
    model_input_name = feature_extractor.model_input_names[0]
    SAMPLING_RATE = feature_extractor.sampling_rate

    def preprocess_audio(batch):
        wavs = [audio["array"] for audio in batch["input_values"]]
        inputs = feature_extractor(wavs, sampling_rate=SAMPLING_RATE, return_tensors="pt")
        return {model_input_name: inputs.get(model_input_name), "labels": list(batch["labels"])}

    label2id = {name: idx for idx, name in enumerate(class_names)}

    test_size = 0.2
    if isinstance(dataset, Dataset):
        try:
            dataset = dataset.train_test_split(
                test_size=test_size, shuffle=True, seed=0, stratify_by_column="labels"
            )
        except ValueError as e:
            print(f"Warning: Stratified split failed ({e}). Using regular split.")
            dataset = dataset.train_test_split(test_size=test_size, shuffle=True, seed=0)
    elif isinstance(dataset, DatasetDict) and "test" not in dataset:
        train_data = dataset["train"]
        try:
            dataset = train_data.train_test_split(
                test_size=test_size, shuffle=True, seed=0, stratify_by_column="labels"
            )
        except ValueError as e:
            print(f"Warning: Stratified split failed ({e}). Using regular split.")
            dataset = train_data.train_test_split(test_size=test_size, shuffle=True, seed=0)

    # Multiply training dataset if augment_multiplier > 1
    if augment and augment_multiplier > 1 and isinstance(dataset, DatasetDict) and "train" in dataset:
        from datasets import concatenate_datasets

        original_train = dataset["train"]
        original_size = len(original_train)
        print(f"Multiplying training dataset by {augment_multiplier}x (original: {original_size} samples)")

        # Create copies of the training set
        train_copies = [original_train]
        for i in range(augment_multiplier - 1):
            train_copies.append(original_train)

        dataset["train"] = concatenate_datasets(train_copies)
        print(f"New training dataset size: {len(dataset['train'])} samples ({augment_multiplier}x augmentation)")

    if augment:
        audio_augmentations = Compose(
            [
                AddGaussianSNR(min_snr_db=min_snr_db, max_snr_db=max_snr_db),
                Gain(min_gain_db=min_gain_db, max_gain_db=max_gain_db),
                GainTransition(
                    min_gain_db=min_gain_db,
                    max_gain_db=max_gain_db,
                    min_duration=0.01,
                    max_duration=0.3,
                    duration_unit="fraction",
                ),
                ClippingDistortion(
                    min_percentile_threshold=min_percentile_threshold,
                    max_percentile_threshold=max_percentile_threshold,
                    p=clipping_probability,
                ),
                TimeStretch(min_rate=min_time_stretch, max_rate=max_time_stretch),
                PitchShift(min_semitones=min_pitch_shift, max_semitones=max_pitch_shift),
            ],
            p=augment_probability,
            shuffle=True,
        )
        print(f"Audio augmentations enabled (p={augment_probability})")
    else:
        audio_augmentations = None
        print("Audio augmentations disabled")

    def preprocess_audio_with_transforms(batch):
        if audio_augmentations is not None:
            wavs = [
                audio_augmentations(audio["array"], sample_rate=SAMPLING_RATE)
                for audio in batch["input_values"]
            ]
        else:
            wavs = [audio["array"] for audio in batch["input_values"]]
        inputs = feature_extractor(wavs, sampling_rate=SAMPLING_RATE, return_tensors="pt")
        return {model_input_name: inputs.get(model_input_name), "labels": list(batch["labels"])}

    dataset = dataset.cast_column("audio", Audio(sampling_rate=feature_extractor.sampling_rate))
    dataset = dataset.rename_column("audio", "input_values")

    def is_valid_audio(example):
        try:
            _ = example["input_values"]["array"]
            return True
        except (RuntimeError, Exception):
            return False

    print("Filtering out corrupted audio files...")
    original_sizes = {}
    filter_num_proc = min(dataloader_num_workers, 4) if dataloader_num_workers > 1 else 1
    if isinstance(dataset, DatasetDict):
        for split_name in dataset.keys():
            original_sizes[split_name] = len(dataset[split_name])
        dataset = dataset.filter(is_valid_audio, num_proc=filter_num_proc)
        for split_name in dataset.keys():
            filtered_count = original_sizes[split_name] - len(dataset[split_name])
            if filtered_count > 0:
                print(f"  Removed {filtered_count} corrupted files from {split_name} split")
    else:
        original_size = len(dataset)
        dataset = dataset.filter(is_valid_audio, num_proc=filter_num_proc)
        filtered_count = original_size - len(dataset)
        if filtered_count > 0:
            print(f"  Removed {filtered_count} corrupted files")

    feature_extractor.do_normalize = False

    if isinstance(dataset, DatasetDict) and "train" in dataset:
        train_dataset_for_norm = dataset["train"]
        if len(train_dataset_for_norm) == 0:
            raise ValueError("No valid audio samples found in training dataset after filtering")

        def compute_stats(batch):
            wavs = [audio["array"] for audio in batch["input_values"]]
            inputs = feature_extractor(wavs, sampling_rate=SAMPLING_RATE, return_tensors="pt")
            values = inputs.get(model_input_name)
            means = [float(torch.mean(values[i])) for i in range(values.shape[0])]
            stds = [float(torch.std(values[i])) for i in range(values.shape[0])]
            return {"_mean": means, "_std": stds}

        print("Calculating dataset normalization statistics...")
        norm_num_proc = min(dataloader_num_workers, 4) if dataloader_num_workers > 1 else 1
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
            raise ValueError("No valid audio samples found in training dataset")

        feature_extractor.mean = float(np.mean(all_means))
        feature_extractor.std = float(np.mean(all_stds))
    else:
        raise ValueError("Expected DatasetDict with 'train' split")

    feature_extractor.do_normalize = True

    print("Calculated mean and std:", feature_extractor.mean, feature_extractor.std)

    if isinstance(dataset, DatasetDict):
        if "train" in dataset:
            dataset["train"].set_transform(
                preprocess_audio_with_transforms, output_all_columns=False
            )
        if "test" in dataset:
            dataset["test"].set_transform(preprocess_audio, output_all_columns=False)
    else:
        raise ValueError("Expected DatasetDict for transform application")

    config = ASTConfig.from_pretrained(pretrained_model)
    config.num_labels = num_labels
    config.label2id = label2id
    config.id2label = {v: k for k, v in label2id.items()}

    model = ASTForAudioClassification.from_pretrained(
        pretrained_model, config=config, ignore_mismatched_sizes=True
    )
    model.init_weights()

    if finetune_mode == "feature-extraction":
        print("Feature extraction mode: freezing base model, only training classifier head")
        for param in model.audio_spectrogram_transformer.parameters():
            param.requires_grad = False
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        print(
            f"  Trainable parameters: {trainable_params:,} / {total_params:,} ({100 * trainable_params / total_params:.2f}%)"
        )
    else:
        print("Full finetune mode: training all model layers")

    training_args = TrainingArguments(
        output_dir=output_dir,
        logging_dir=logging_dir,
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
        torch_compile=torch_compile,
        run_name=mlflow_run_name,
    )

    accuracy = evaluate.load("accuracy")
    recall = evaluate.load("recall")
    precision = evaluate.load("precision")
    f1 = evaluate.load("f1")

    AVERAGE = "macro" if config.num_labels > 2 else "binary"

    def compute_metrics(eval_pred) -> Dict[str, float]:
        logits = eval_pred.predictions
        predictions = np.argmax(logits, axis=1)

        accuracy_result = accuracy.compute(predictions=predictions, references=eval_pred.label_ids)
        metrics: Dict[str, float] = accuracy_result if accuracy_result is not None else {}

        precision_result = precision.compute(
            predictions=predictions, references=eval_pred.label_ids, average=AVERAGE
        )
        if precision_result is not None:
            metrics.update(precision_result)

        recall_result = recall.compute(
            predictions=predictions, references=eval_pred.label_ids, average=AVERAGE
        )
        if recall_result is not None:
            metrics.update(recall_result)

        f1_result = f1.compute(
            predictions=predictions, references=eval_pred.label_ids, average=AVERAGE
        )
        if f1_result is not None:
            metrics.update(f1_result)

        return metrics

    if isinstance(dataset, DatasetDict):
        train_data = dataset.get("train")
        eval_data = dataset.get("test")
    else:
        raise ValueError("Expected DatasetDict for trainer setup")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=eval_data,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    services.file.ensure_directory(best_model_path)
    trainer.save_model(best_model_path)


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
    from pathlib import Path as PathLib

    from bioamla.cli.service_helpers import handle_result, services

    path_obj = PathLib(path)
    if not path_obj.exists():
        click.echo(f"Error: Path not found: {path}")
        raise SystemExit(1)

    gt_path = PathLib(ground_truth)
    if not gt_path.exists():
        click.echo(f"Error: Ground truth file not found: {ground_truth}")
        raise SystemExit(1)

    result = services.ast.evaluate(
        audio_dir=path,
        model_path=model_path,
        ground_truth_csv=ground_truth,
        file_column=file_column,
        label_column=label_column,
        resample_freq=resample_freq,
        batch_size=batch_size,
        fp16=fp16,
    )
    eval_result = handle_result(result)

    if not quiet:
        click.echo("\nEvaluation Results:")
        click.echo("-" * 40)
    click.echo(f"Accuracy: {eval_result.accuracy:.4f}")
    click.echo(f"Precision: {eval_result.precision:.4f}")
    click.echo(f"Recall: {eval_result.recall:.4f}")
    click.echo(f"F1 Score: {eval_result.f1_score:.4f}")
    click.echo(f"Total Samples: {eval_result.total_samples}")

    if output:
        if output_format == "json":
            services.file.write_json(output, eval_result.to_dict())
        else:
            services.file.write_text(output, str(eval_result.to_dict()))
        click.echo(f"Results saved to: {output}")


@ast.command("embed")
@click.argument("file", type=click.Path(exists=True))
@click.option("--model-path", required=True, help="Path to AST model or HuggingFace identifier")
@click.option("--output", "-o", required=True, help="Output file (.npy)")
@click.option("--layer", default=None, help="Layer to extract embeddings from")
@click.option("--sample-rate", default=16000, type=int, help="Target sample rate")
def ast_embed(
    file: str, model_path: str, output: str, layer: str, sample_rate: int
) -> None:
    """Extract embeddings from audio using AST model.

    Example:
        bioamla models ast embed audio.wav --model-path my_model -o embeddings.npy
    """
    from bioamla.cli.service_helpers import handle_result, services

    click.echo(f"Loading AST model from {model_path}...")

    result = services.ast.extract_embeddings(
        filepath=file,
        model_path=model_path,
        layer=layer,
        sample_rate=sample_rate,
    )
    embeddings = handle_result(result)

    services.file.write_npy(output, embeddings)
    click.echo(f"Embeddings saved to {output} (shape: {embeddings.shape})")


@ast.command("info")
@click.argument("model_path")
def ast_info(model_path: str) -> None:
    """Display information about an AST model.

    Example:
        bioamla models ast info bioamla/scp-frogs
    """
    from bioamla.cli.service_helpers import handle_result, services

    result = services.ast.get_model_info(model_path)
    info = handle_result(result)

    click.echo(f"Model: {info['path']}")
    click.echo(f"Backend: {info['backend']}")
    click.echo(f"Classes: {info['num_classes']}")
    if info.get("classes"):
        labels = ", ".join(info["classes"])
        if info.get("has_more_classes"):
            labels += f"... (+{info['num_classes'] - 10} more)"
        click.echo(f"Labels: {labels}")


# =============================================================================
# CNN Subgroup (OpenSoundscape CNN via adapter)
# =============================================================================


@models.group()
def cnn() -> None:
    """CNN model operations (OpenSoundscape backend)."""
    pass


@cnn.command("predict")
@click.argument("file", type=click.Path(exists=True))
@click.option("--model-path", "-m", required=True, help="Path to trained CNN model (.pt)")
@click.option("--min-confidence", default=0.0, type=float, help="Minimum confidence threshold")
@click.option("--top-k", default=1, type=int, help="Number of top predictions per segment")
@click.option("--batch-size", default=1, type=int, help="Batch size for inference")
@click.option("--num-workers", default=0, type=int, help="Number of data loader workers")
def cnn_predict(
    file: str,
    model_path: str,
    min_confidence: float,
    top_k: int,
    batch_size: int,
    num_workers: int,
) -> None:
    """Run CNN prediction on a single audio file.

    Example:
        bioamla models cnn predict audio.wav --model-path model.pt
    """
    from bioamla.cli.service_helpers import handle_result, services

    result = services.cnn.predict(
        filepath=file,
        model_path=model_path,
        min_confidence=min_confidence,
        top_k=top_k,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    predictions = handle_result(result)

    if not predictions:
        click.echo("No predictions above threshold.")
        return

    for pred in predictions:
        click.echo(
            f"[{pred.start_time:.1f}s - {pred.end_time:.1f}s] "
            f"{pred.label} ({pred.confidence:.4f})"
        )


@cnn.command("train")
@click.option("--train-csv", required=True, help="Training CSV (file column + class columns with 0/1)")
@click.option("--output-dir", "-o", required=True, help="Output directory for model")
@click.option("--validation-csv", default=None, help="Validation CSV (same format)")
@click.option("--classes", "-c", required=True, multiple=True, help="Class names (repeat for each class)")
@click.option(
    "--architecture",
    "-a",
    default="resnet18",
    type=click.Choice([
        "resnet18", "resnet34", "resnet50", "resnet101", "resnet152",
        "efficientnet_b0", "efficientnet_b1", "efficientnet_b2",
        "densenet121", "densenet161",
    ]),
    help="Model architecture",
)
@click.option("--epochs", default=10, type=int, help="Number of training epochs")
@click.option("--batch-size", default=32, type=int, help="Batch size for training")
@click.option("--learning-rate", default=None, type=float, help="Learning rate (uses default if not specified)")
@click.option("--sample-duration", default=3.0, type=float, help="Audio clip duration in seconds")
@click.option("--sample-rate", default=16000, type=int, help="Target sample rate")
@click.option("--freeze-backbone/--no-freeze-backbone", default=False, help="Freeze backbone for transfer learning")
@click.option("--num-workers", default=0, type=int, help="Number of data loader workers")
def cnn_train(
    train_csv: str,
    output_dir: str,
    validation_csv: str,
    classes: tuple,
    architecture: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    sample_duration: float,
    sample_rate: int,
    freeze_backbone: bool,
    num_workers: int,
) -> None:
    """Train a CNN model on audio data.

    The training CSV should have file paths as the first column (used as index)
    and class columns with 0/1 values indicating presence/absence.

    Example:
        bioamla models cnn train --train-csv train.csv --output-dir ./model \\
            --classes bird --classes frog --architecture resnet18 --epochs 20
    """
    from bioamla.cli.service_helpers import handle_result, services

    class_list = list(classes)

    click.echo(f"Creating {architecture} model with {len(class_list)} classes...")

    # First create the model
    create_result = services.cnn.create_model(
        classes=class_list,
        architecture=architecture,
        sample_duration=sample_duration,
        sample_rate=sample_rate,
    )
    handle_result(create_result)

    click.echo(f"Training for {epochs} epochs...")

    # Then train
    result = services.cnn.train(
        train_csv=train_csv,
        output_dir=output_dir,
        validation_csv=validation_csv,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        freeze_backbone=freeze_backbone,
        num_workers=num_workers,
    )
    train_result = handle_result(result)

    click.echo("\nTraining complete!")
    click.echo(f"Model saved to: {train_result.model_path}")
    click.echo(f"Architecture: {train_result.architecture}")
    click.echo(f"Classes: {train_result.num_classes}")


@cnn.command("info")
@click.argument("model_path", type=click.Path(exists=True))
def cnn_info(model_path: str) -> None:
    """Display information about a CNN model.

    Example:
        bioamla models cnn info model.pt
    """
    from bioamla.cli.service_helpers import handle_result, services

    result = services.cnn.get_model_info(model_path)
    info = handle_result(result)

    click.echo(f"Model: {info['path']}")
    click.echo(f"Architecture: {info['architecture']}")
    click.echo(f"Classes: {info['num_classes']}")
    click.echo(f"Sample Duration: {info['sample_duration']}s")

    if info.get("classes"):
        labels = ", ".join(info["classes"])
        if info.get("has_more_classes"):
            labels += f"... (+{info['num_classes'] - 10} more)"
        click.echo(f"Labels: {labels}")


@cnn.command("architectures")
def cnn_architectures() -> None:
    """List available CNN architectures.

    Example:
        bioamla models cnn architectures
    """
    from bioamla.cli.service_helpers import handle_result, services

    result = services.cnn.list_architectures()
    architectures = handle_result(result)

    click.echo("Available CNN architectures:")
    click.echo("-" * 30)
    for arch in architectures:
        click.echo(f"  {arch}")


@cnn.command("embed")
@click.argument("file", type=click.Path(exists=True))
@click.option("--model-path", "-m", required=True, help="Path to trained CNN model (.pt)")
@click.option("--output", "-o", required=True, help="Output file (.npy)")
@click.option("--layer", default=None, help="Layer to extract embeddings from")
@click.option("--batch-size", default=1, type=int, help="Batch size for inference")
@click.option("--num-workers", default=0, type=int, help="Number of data loader workers")
def cnn_embed(
    file: str,
    model_path: str,
    output: str,
    layer: str,
    batch_size: int,
    num_workers: int,
) -> None:
    """Extract embeddings from audio using a CNN model.

    Example:
        bioamla models cnn embed audio.wav --model-path model.pt --output embeddings.npy
    """
    from bioamla.cli.service_helpers import handle_result, services

    click.echo(f"Loading CNN model from {model_path}...")

    result = services.cnn.extract_embeddings(
        filepath=file,
        model_path=model_path,
        batch_size=batch_size,
        num_workers=num_workers,
        target_layer=layer,
    )
    embeddings = handle_result(result)

    services.file.write_npy(output, embeddings)
    click.echo(f"Embeddings saved to {output} (shape: {embeddings.shape})")
