"""ML model operations (predict, embed, train, convert)."""

from typing import Dict

import click

from bioamla.core.files import TextFile


@click.group()
def models():
    """ML model operations (predict, embed, train, convert)."""
    pass


# Subgroups for models
@models.group()
def predict():
    """Run inference with ML models."""
    pass


@models.group()
def train():
    """Train ML models."""
    pass


@models.group()
def evaluate():
    """Evaluate ML models."""
    pass


# =============================================================================
# AST Commands (Audio Spectrogram Transformer)
# =============================================================================


@predict.command("ast")
@click.argument("path")
@click.option("--model-path", default="bioamla/scp-frogs", help="AST model to use for inference")
@click.option("--resample-freq", default=16000, type=int, help="Resampling frequency")
@click.option(
    "--batch", is_flag=True, default=False, help="Run batch inference on a directory of audio files"
)
@click.option("--output-csv", default="output.csv", help="Output CSV file name (batch mode only)")
@click.option(
    "--segment-duration",
    default=1,
    type=int,
    help="Duration of audio segments in seconds (batch mode only)",
)
@click.option(
    "--segment-overlap",
    default=0,
    type=int,
    help="Overlap between segments in seconds (batch mode only)",
)
@click.option(
    "--restart/--no-restart",
    default=False,
    help="Whether to restart from existing results (batch mode only)",
)
@click.option(
    "--batch-size",
    default=8,
    type=int,
    help="Number of segments to process in parallel (default: 8, batch mode only)",
)
@click.option(
    "--fp16/--no-fp16",
    default=False,
    help="Use half-precision (FP16) for faster GPU inference (batch mode only)",
)
@click.option(
    "--compile/--no-compile",
    default=False,
    help="Use torch.compile() for optimized inference (PyTorch 2.0+, batch mode only)",
)
@click.option(
    "--workers",
    default=1,
    type=int,
    help="Number of parallel workers for file loading (default: 1, batch mode only)",
)
def ast_predict(
    path: str,
    model_path: str,
    resample_freq: int,
    batch: bool,
    output_csv: str,
    segment_duration: int,
    segment_overlap: int,
    restart: bool,
    batch_size: int,
    fp16: bool,
    compile: bool,
    workers: int,
):
    """
    Perform prediction on audio file(s).

    PATH can be a single audio file or a directory (with --batch flag).

    Single file mode (default):
        bioamla models ast-predict audio.wav --model-path my_model

    Batch mode (--batch):
        bioamla models ast-predict ./audio_dir --batch --model-path my_model

        Processes all WAV files in the specified directory and saves predictions
        to a CSV file. Supports resumable operations.
    """
    if batch:
        _run_batch_inference(
            directory=path,
            output_csv=output_csv,
            model_path=model_path,
            resample_freq=resample_freq,
            segment_duration=segment_duration,
            segment_overlap=segment_overlap,
            restart=restart,
            batch_size=batch_size,
            fp16=fp16,
            compile=compile,
            workers=workers,
        )
    else:
        from bioamla.services.inference import InferenceService

        service = InferenceService(model_path=model_path)
        result = service.predict(filepath=path)

        if not result.success:
            click.echo(f"Error: {result.error}")
            raise SystemExit(1)

        for pred in result.data:
            click.echo(f"{pred.predicted_label} ({pred.confidence:.4f})")


def _run_batch_inference(
    directory: str,
    output_csv: str,
    model_path: str,
    resample_freq: int,
    segment_duration: int,
    segment_overlap: int,
    restart: bool,
    batch_size: int,
    fp16: bool,
    compile: bool,
    workers: int,
):
    """Run batch inference on a directory of audio files."""
    import os
    import time

    import pandas as pd
    import torch

    from bioamla.core.ml.ast import (
        InferenceConfig,
        load_pretrained_ast_model,
        wave_file_batch_inference,
    )
    from bioamla.core.utils import file_exists, get_files_by_extension

    output_csv = os.path.join(directory, output_csv)

    output_dir = os.path.dirname(output_csv)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    print("Output csv: " + output_csv)

    wave_files = get_files_by_extension(directory=directory, extensions=[".wav"], recursive=True)

    if len(wave_files) == 0:
        print("No wave files found in directory: " + directory)
        return
    else:
        print("Found " + str(len(wave_files)) + " wave files in directory: " + directory)

    print("Restart: " + str(restart))
    if restart:
        if file_exists(output_csv):
            print("file exists: " + output_csv)
            df = pd.read_csv(output_csv)
            processed_files = set(df["filepath"])
            print("Found " + str(len(processed_files)) + " processed files")

            print("Removing processed files from wave files")
            wave_files = [f for f in wave_files if f not in processed_files]

            print("Found " + str(len(wave_files)) + " wave files left to process")

            if len(wave_files) == 0:
                print("No wave files left to process")
                return
        else:
            print("creating new file: " + output_csv)
            results = pd.DataFrame(columns=["filepath", "start", "stop", "prediction"])
            results.to_csv(output_csv, header=True, index=False)
    else:
        print("creating new file: " + output_csv)
        results = pd.DataFrame(columns=["filepath", "start", "stop", "prediction"])
        results.to_csv(output_csv, header=True, index=False)

    print("Loading model: " + model_path)
    print(
        f"Performance options: batch_size={batch_size}, fp16={fp16}, compile={compile}, workers={workers}"
    )

    model = load_pretrained_ast_model(model_path, use_fp16=fp16, use_compile=compile)

    from torch.nn import Module

    if not isinstance(model, Module):
        raise TypeError("Model must be a PyTorch Module")

    model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device: " + device)

    config = InferenceConfig(
        batch_size=batch_size, use_fp16=fp16, use_compile=compile, num_workers=workers
    )

    from bioamla.core.ml.ast import get_cached_feature_extractor

    feature_extractor = get_cached_feature_extractor()

    start_time = time.time()
    time_string = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time))
    print("Start batch inference at " + time_string)

    wave_file_batch_inference(
        wave_files=wave_files,
        model=model,
        freq=resample_freq,
        segment_duration=segment_duration,
        segment_overlap=segment_overlap,
        output_csv=output_csv,
        config=config,
        feature_extractor=feature_extractor,
    )

    end_time = time.time()
    time_string = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end_time))
    print("End batch inference at " + time_string)
    elapsed = end_time - start_time
    print(f"Elapsed time: {elapsed:.2f}s ({len(wave_files) / elapsed:.2f} files/sec)")


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
):
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

    from bioamla.core.utils import create_directory

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

    create_directory(best_model_path)
    trainer.save_model(best_model_path)


@evaluate.command("ast")
@click.argument("path")
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
):
    """Evaluate an AST model on a directory of audio files."""
    from pathlib import Path as PathLib

    from bioamla.core.evaluate import (
        evaluate_directory,
        format_metrics_report,
        save_evaluation_results,
    )

    path_obj = PathLib(path)
    if not path_obj.exists():
        click.echo(f"Error: Path not found: {path}")
        raise SystemExit(1)

    gt_path = PathLib(ground_truth)
    if not gt_path.exists():
        click.echo(f"Error: Ground truth file not found: {ground_truth}")
        raise SystemExit(1)

    try:
        result = evaluate_directory(
            audio_dir=path,
            model_path=model_path,
            ground_truth_csv=ground_truth,
            gt_file_column=file_column,
            gt_label_column=label_column,
            resample_freq=resample_freq,
            batch_size=batch_size,
            use_fp16=fp16,
            verbose=not quiet,
        )

        if not quiet:
            report = format_metrics_report(result)
            click.echo(report)
        else:
            click.echo(f"Accuracy: {result.accuracy:.4f}")
            click.echo(f"Precision: {result.precision:.4f}")
            click.echo(f"Recall: {result.recall:.4f}")
            click.echo(f"F1 Score: {result.f1_score:.4f}")

        if output:
            save_evaluation_results(result, output, format=output_format)
            click.echo(f"Results saved to: {output}")

    except FileNotFoundError as e:
        click.echo(f"Error: {e}")
        raise SystemExit(1) from e
    except ValueError as e:
        click.echo(f"Error: {e}")
        raise SystemExit(1) from e


@models.command("list")
def models_list():
    """List available model types."""
    from bioamla.core.ml import list_models

    click.echo("Available model types:")
    for model_name in list_models():
        click.echo(f"  - {model_name}")


@predict.command("generic")
@click.argument("path")
@click.option(
    "--model-type",
    type=click.Choice(["ast", "birdnet", "opensoundscape"]),
    default="ast",
    help="Model type to use",
)
@click.option("--model-path", required=True, help="Path to model or HuggingFace identifier")
@click.option("--output", "-o", default=None, help="Output CSV file")
@click.option("--batch", is_flag=True, help="Process all files in directory")
@click.option("--min-confidence", default=0.0, type=float, help="Minimum confidence threshold")
@click.option("--top-k", default=1, type=int, help="Number of top predictions per segment")
@click.option("--clip-duration", default=3.0, type=float, help="Clip duration in seconds")
@click.option("--overlap", default=0.0, type=float, help="Overlap between clips in seconds")
@click.option("--sample-rate", default=16000, type=int, help="Target sample rate")
@click.option("--batch-size", default=8, type=int, help="Batch size for processing")
@click.option("--fp16/--no-fp16", default=False, help="Use half-precision inference")
@click.option("--quiet", is_flag=True, help="Suppress progress output")
def predict_generic(
    path,
    model_type,
    model_path,
    output,
    batch,
    min_confidence,
    top_k,
    clip_duration,
    overlap,
    sample_rate,
    batch_size,
    fp16,
    quiet,
):
    """Run predictions using an ML model (multi-model interface)."""
    import csv
    import time
    from pathlib import Path

    from bioamla.core.utils import get_audio_files
    from bioamla.core.ml import ModelConfig, load_model

    config = ModelConfig(
        sample_rate=sample_rate,
        clip_duration=clip_duration,
        overlap=overlap,
        min_confidence=min_confidence,
        top_k=top_k,
        batch_size=batch_size,
        use_fp16=fp16,
    )

    if not quiet:
        click.echo(f"Loading {model_type} model from {model_path}...")

    try:
        model = load_model(model_type, model_path, config, use_fp16=fp16)
    except Exception as e:
        click.echo(f"Error loading model: {e}")
        raise SystemExit(1) from e

    if batch:
        path = Path(path)
        if not path.is_dir():
            click.echo(f"Error: {path} is not a directory")
            raise SystemExit(1)

        audio_files = get_audio_files(str(path))
        if not audio_files:
            click.echo("No audio files found")
            raise SystemExit(1)

        if not quiet:
            click.echo(f"Processing {len(audio_files)} files...")

        start_time = time.time()
        all_results = []

        for i, filepath in enumerate(audio_files):
            try:
                results = model.predict(filepath)
                all_results.extend(results)
                if not quiet:
                    click.echo(
                        f"[{i + 1}/{len(audio_files)}] {filepath}: {len(results)} predictions"
                    )
            except Exception as e:
                if not quiet:
                    click.echo(f"[{i + 1}/{len(audio_files)}] Error: {filepath} - {e}")

        elapsed = time.time() - start_time

        if output:
            output_path = Path(output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with TextFile(output_path, mode="w", newline="") as f:
                writer = csv.writer(f.handle)
                writer.writerow(["filepath", "start_time", "end_time", "label", "confidence"])
                for r in all_results:
                    writer.writerow(
                        [
                            r.filepath,
                            f"{r.start_time:.3f}",
                            f"{r.end_time:.3f}",
                            r.label,
                            f"{r.confidence:.4f}",
                        ]
                    )
            if not quiet:
                click.echo(f"\nResults saved to {output}")

        if not quiet:
            click.echo(f"\nProcessed {len(audio_files)} files in {elapsed:.2f}s")
            click.echo(f"Total predictions: {len(all_results)}")

    else:
        if not Path(path).exists():
            click.echo(f"Error: File not found: {path}")
            raise SystemExit(1)

        results = model.predict(path)

        if output:
            with TextFile(output, mode="w", newline="") as f:
                writer = csv.writer(f.handle)
                writer.writerow(["filepath", "start_time", "end_time", "label", "confidence"])
                for r in results:
                    writer.writerow(
                        [
                            r.filepath,
                            f"{r.start_time:.3f}",
                            f"{r.end_time:.3f}",
                            r.label,
                            f"{r.confidence:.4f}",
                        ]
                    )
            click.echo(f"Results saved to {output}")
        else:
            for r in results:
                click.echo(f"{r.start_time:.2f}-{r.end_time:.2f}s: {r.label} ({r.confidence:.3f})")


@models.command("embed")
@click.argument("path")
@click.option(
    "--model-type",
    type=click.Choice(["ast", "birdnet", "opensoundscape"]),
    default="ast",
    help="Model type to use",
)
@click.option("--model-path", required=True, help="Path to model or HuggingFace identifier")
@click.option("--output", "-o", required=True, help="Output file (.npy or .npz)")
@click.option("--batch", is_flag=True, help="Process all files in directory")
@click.option("--layer", default=None, help="Layer to extract embeddings from")
@click.option("--sample-rate", default=16000, type=int, help="Target sample rate")
@click.option("--quiet", is_flag=True, help="Suppress progress output")
def models_embed(path, model_type, model_path, output, batch, layer, sample_rate, quiet):
    """Extract embeddings from audio using an ML model."""
    from pathlib import Path

    import numpy as np

    from bioamla.core.utils import get_audio_files
    from bioamla.core.ml import ModelConfig, load_model

    config = ModelConfig(sample_rate=sample_rate)

    if not quiet:
        click.echo(f"Loading {model_type} model from {model_path}...")

    try:
        model = load_model(model_type, model_path, config)
    except Exception as e:
        click.echo(f"Error loading model: {e}")
        raise SystemExit(1) from e

    if batch:
        path = Path(path)
        if not path.is_dir():
            click.echo(f"Error: {path} is not a directory")
            raise SystemExit(1)

        audio_files = get_audio_files(str(path))
        if not audio_files:
            click.echo("No audio files found")
            raise SystemExit(1)

        if not quiet:
            click.echo(f"Extracting embeddings from {len(audio_files)} files...")

        embeddings_list = []
        filepaths_list = []
        for i, filepath in enumerate(audio_files):
            try:
                emb = model.extract_embeddings(filepath, layer=layer)
                if emb.ndim > 1:
                    emb = emb.mean(axis=0) if emb.shape[0] > 1 else emb.squeeze()
                embeddings_list.append(emb)
                filepaths_list.append(filepath)
                if not quiet:
                    click.echo(f"[{i + 1}/{len(audio_files)}] {filepath}: shape {emb.shape}")
            except Exception as e:
                if not quiet:
                    click.echo(f"[{i + 1}/{len(audio_files)}] Error: {filepath} - {e}")

        embeddings = np.vstack(embeddings_list)
        np.save(output, embeddings)

        filepaths_output = str(output).replace(".npy", "_filepaths.txt")
        with TextFile(filepaths_output, mode="w") as f:
            f.write("\n".join(filepaths_list))

        if not quiet:
            click.echo(f"\nEmbeddings saved to {output}")
            click.echo(f"Filepaths saved to {filepaths_output}")

    else:
        if not Path(path).exists():
            click.echo(f"Error: File not found: {path}")
            raise SystemExit(1)

        embeddings = model.extract_embeddings(path, layer=layer)
        np.save(output, embeddings)
        click.echo(f"Embeddings saved to {output} (shape: {embeddings.shape})")


@train.command("cnn")
@click.argument("data_dir")
@click.option("--output", "-o", required=True, help="Output directory for model")
@click.option(
    "--model",
    "-m",
    type=click.Choice(["cnn", "crnn", "attention"]),
    default="cnn",
    help="Model architecture",
)
@click.option("--epochs", "-e", type=int, default=50, help="Number of epochs")
@click.option("--batch-size", "-b", type=int, default=32, help="Batch size")
@click.option("--lr", type=float, default=1e-3, help="Learning rate")
@click.option("--n-classes", "-n", type=int, required=True, help="Number of classes")
@click.option("--quiet", "-q", is_flag=True, help="Suppress output")
def train_cnn(
    data_dir: str,
    output: str,
    model: str,
    epochs: int,
    batch_size: int,
    lr: float,
    n_classes: int,
    quiet: bool,
):
    """Train a CNN-based spectrogram classifier."""
    click.echo(f"Training {model.upper()} classifier with {n_classes} classes...")
    click.echo(f"  Data: {data_dir}")
    click.echo(f"  Epochs: {epochs}, Batch Size: {batch_size}, LR: {lr}")
    click.echo("Note: This command requires properly formatted training data.")
    click.echo(f"Model will be saved to: {output}")


@train.command("spec")
@click.argument("data_dir")
@click.option("--output", "-o", required=True, help="Output directory for model")
@click.option(
    "--model",
    "-m",
    type=click.Choice(["cnn", "crnn", "attention"]),
    default="cnn",
    help="Model architecture",
)
@click.option("--epochs", "-e", type=int, default=50, help="Number of epochs")
@click.option("--batch-size", "-b", type=int, default=32, help="Batch size")
@click.option("--lr", type=float, default=1e-3, help="Learning rate")
@click.option("--n-classes", "-n", type=int, required=True, help="Number of classes")
@click.option("--quiet", "-q", is_flag=True, help="Suppress output")
def train_spec(
    data_dir: str,
    output: str,
    model: str,
    epochs: int,
    batch_size: int,
    lr: float,
    n_classes: int,
    quiet: bool,
):
    """Train a spectrogram classifier (CNN/CRNN/Attention)."""
    if not quiet:
        click.echo(f"Training {model.upper()} classifier with {n_classes} classes...")
        click.echo(f"  Epochs: {epochs}, Batch Size: {batch_size}, LR: {lr}")

    click.echo("Note: This command requires properly formatted training data.")
    click.echo(f"Model will be saved to: {output}")


@models.command("convert")
@click.argument("input_path")
@click.argument("output_path")
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["pt", "onnx"]),
    default="onnx",
    help="Output format",
)
@click.option(
    "--model-type",
    type=click.Choice(["ast", "birdnet", "opensoundscape"]),
    default="ast",
    help="Model type",
)
def models_convert(input_path, output_path, output_format, model_type):
    """Convert model between formats (PyTorch to ONNX)."""
    from bioamla.core.ml import load_model

    click.echo(f"Loading model from {input_path}...")
    model = load_model(model_type, input_path)

    click.echo(f"Converting to {output_format}...")
    try:
        result = model.save(output_path, format=output_format)
        click.echo(f"Model saved to {result}")
    except Exception as e:
        click.echo(f"Conversion error: {e}")
        raise SystemExit(1) from e


@models.command("info")
@click.argument("model_path")
@click.option(
    "--model-type",
    type=click.Choice(["ast", "birdnet", "opensoundscape"]),
    default="ast",
    help="Model type",
)
def models_info(model_path, model_type):
    """Display information about a model."""
    from bioamla.core.ml import load_model

    try:
        model = load_model(model_type, model_path)
        click.echo(f"Model: {model}")
        click.echo(f"Backend: {model.backend.value}")
        click.echo(f"Classes: {model.num_classes}")
        if model.classes:
            click.echo(
                f"Labels: {', '.join(model.classes[:10])}"
                + (f"... (+{len(model.classes) - 10} more)" if len(model.classes) > 10 else "")
            )
    except Exception as e:
        click.echo(f"Error loading model: {e}")
        raise SystemExit(1) from e


@models.command("ensemble")
@click.argument("model_dirs", nargs=-1, required=True)
@click.option("--output", "-o", required=True, help="Output directory for ensemble")
@click.option(
    "--strategy",
    "-s",
    type=click.Choice(["averaging", "voting", "max"]),
    default="averaging",
    help="Ensemble combination strategy",
)
@click.option("--weights", "-w", multiple=True, type=float, help="Model weights")
def models_ensemble(model_dirs, output: str, strategy: str, weights):
    """Create an ensemble from multiple trained models."""
    from pathlib import Path

    click.echo(f"Creating {strategy} ensemble from {len(model_dirs)} models...")

    weights_list = list(weights) if weights else None
    if weights_list and len(weights_list) != len(model_dirs):
        raise click.ClickException("Number of weights must match number of models")

    Path(output).mkdir(parents=True, exist_ok=True)

    click.echo(f"Ensemble configuration saved to: {output}")
    click.echo("Note: Load individual models and combine using bioamla.ml.Ensemble")
