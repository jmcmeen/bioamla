"""ML model operations - AST (Audio Spectrogram Transformer) only."""

from typing import Dict

import click


@click.group()
def models() -> None:
    """ML model operations - AST (Audio Spectrogram Transformer) only."""
    pass


# Subgroups for models
@models.group()
def predict() -> None:
    """Run AST inference."""
    pass


@models.group()
def train() -> None:
    """Train AST models."""
    pass


@models.group()
def evaluate() -> None:
    """Evaluate AST models."""
    pass


# =============================================================================
# AST Commands (Audio Spectrogram Transformer)
# =============================================================================


@predict.command("ast")
@click.argument("file", type=click.Path(exists=True))
@click.option("--model-path", default="bioamla/scp-frogs", help="AST model to use for inference")
@click.option("--resample-freq", default=16000, type=int, help="Resampling frequency")
def ast_predict(
    file: str,
    model_path: str,
    resample_freq: int,
) -> None:
    """
    Perform AST prediction on a single audio file.

    Example:
        bioamla models predict ast audio.wav --model-path my_model
    """
    from bioamla.cli.service_helpers import handle_result, services

    result = services.inference.predict(filepath=file, model_path=model_path)
    predictions = handle_result(result)

    for pred in predictions:
        click.echo(f"{pred.predicted_label} ({pred.confidence:.4f})")


@train.command("ast")
@click.option("--training-dir", default=".", help="Directory to save training outputs")
@click.option(
    "--base-model",
    default="MIT/ast-finetuned-audioset-10-10-0.4593",
    help="Base model to fine-tune",
)
@click.option(
    "--train-dataset", default="bioamla/scp-frogs", help="Training dataset from HuggingFace Hub"
)
@click.option("--split", default="train", help="Dataset split to use")
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
) -> None:
    """Fine-tune an AST model on a custom dataset."""
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

    if ":" in train_dataset:
        dataset_name, config_name = train_dataset.rsplit(":", 1)
        dataset = load_dataset(dataset_name, config_name, split=split)
    elif "BirdSet" in train_dataset:
        click.echo(
            "Note: BirdSet detected. Use format 'samuelstevens/BirdSet:HSN' to specify subset."
        )
        click.echo("Available subsets: HSN, NBP, NES, PER")
        dataset = load_dataset(train_dataset, "HSN", split=split)
    else:
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

    audio_augmentations = Compose(
        [
            AddGaussianSNR(min_snr_db=10, max_snr_db=20),
            Gain(min_gain_db=-6, max_gain_db=6),
            GainTransition(
                min_gain_db=-6,
                max_gain_db=6,
                min_duration=0.01,
                max_duration=0.3,
                duration_unit="fraction",
            ),
            ClippingDistortion(min_percentile_threshold=0, max_percentile_threshold=30, p=0.5),
            TimeStretch(min_rate=0.8, max_rate=1.2),
            PitchShift(min_semitones=-4, max_semitones=4),
        ],
        p=0.8,
        shuffle=True,
    )

    def preprocess_audio_with_transforms(batch):
        wavs = [
            audio_augmentations(audio["array"], sample_rate=SAMPLING_RATE)
            for audio in batch["input_values"]
        ]
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


@evaluate.command("ast")
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
    """Evaluate an AST model on a directory of audio files."""
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


@models.command("embed")
@click.argument("file", type=click.Path(exists=True))
@click.option("--model-path", required=True, help="Path to AST model or HuggingFace identifier")
@click.option("--output", "-o", required=True, help="Output file (.npy)")
@click.option("--layer", default=None, help="Layer to extract embeddings from")
@click.option("--sample-rate", default=16000, type=int, help="Target sample rate")
def models_embed(
    file: str, model_path: str, output: str, layer: str, sample_rate: int
) -> None:
    """Extract embeddings from audio using AST model (single file)."""

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


@models.command("info")
@click.argument("model_path")
def models_info(model_path: str) -> None:
    """Display information about an AST model."""
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
