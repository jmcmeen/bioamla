from typing import Dict

import click


@click.group()
def cli():
    """Bioamla CLI - Bioacoustics and Machine Learning Applications"""
    pass


# =============================================================================
# Top-level utility commands
# =============================================================================

@cli.command()
def devices():
    """Display comprehensive device information including CUDA and GPU details."""
    from bioamla.diagnostics import get_device_info
    device_info = get_device_info()

    click.echo("Devices:")
    click.echo(f'CUDA available: {device_info["cuda_available"]}')
    click.echo(f'Current device: {device_info["current_device"]}')
    click.echo(f'Device count: {device_info["device_count"]}')

    for device in device_info['devices']:
        click.echo(f'  - Index: {device["index"]}, Name: {device["name"]}')


@cli.command()
def version():
    """Display the current version of the bioamla package."""
    from bioamla.diagnostics import get_bioamla_version
    click.echo(f"bioamla v{get_bioamla_version()}")


@cli.command()
@click.argument('url', required=True)
@click.argument('output_dir', required=False, default='.')
def download(url: str, output_dir: str):
    """Download a file from the specified URL to the target directory."""
    import os
    from urllib.parse import urlparse

    from novus_pytils.files import download_file

    if output_dir == '.':
        output_dir = os.getcwd()

    parsed_url = urlparse(url)
    filename = os.path.basename(parsed_url.path)
    if not filename:
        filename = "downloaded_file"

    output_path = os.path.join(output_dir, filename)
    download_file(url, output_path)


@cli.command()
@click.argument('file_path')
@click.argument('output_path', required=False, default='.')
def unzip(file_path: str, output_path: str):
    """Extract a ZIP archive to the specified output directory."""
    from novus_pytils.compression import extract_zip_file
    if output_path == '.':
        import os
        output_path = os.getcwd()

    extract_zip_file(file_path, output_path)


@cli.command('zip')
@click.argument('source_path')
@click.argument('output_file')
def zip_cmd(source_path: str, output_file: str):
    """Create a ZIP archive from a file or directory."""
    import os

    from novus_pytils.compression import create_zip_file, zip_directory

    if os.path.isdir(source_path):
        zip_directory(source_path, output_file)
    else:
        create_zip_file([source_path], output_file)

    click.echo(f"Created {output_file}")


@cli.command()
@click.option('--models', is_flag=True, help='Purge cached models')
@click.option('--datasets', is_flag=True, help='Purge cached datasets')
@click.option('--all', 'purge_all', is_flag=True, help='Purge all cached data (models and datasets)')
@click.option('--yes', '-y', is_flag=True, help='Skip confirmation prompt')
def purge(models: bool, datasets: bool, purge_all: bool, yes: bool):
    """Purge cached HuggingFace Hub data from local storage."""
    import shutil
    from pathlib import Path

    from huggingface_hub import scan_cache_dir

    if not models and not datasets and not purge_all:
        click.echo("Please specify what to purge: --models, --datasets, or --all")
        click.echo("Run 'bioamla purge --help' for more information.")
        return

    if purge_all:
        models = True
        datasets = True

    cache_info = scan_cache_dir()

    models_to_delete = []
    datasets_to_delete = []

    for repo in cache_info.repos:
        if repo.repo_type == "model" and models:
            models_to_delete.append(repo)
        elif repo.repo_type == "dataset" and datasets:
            datasets_to_delete.append(repo)

    models_size = sum(repo.size_on_disk for repo in models_to_delete)
    datasets_size = sum(repo.size_on_disk for repo in datasets_to_delete)
    total_size = models_size + datasets_size

    if not models_to_delete and not datasets_to_delete:
        click.echo("No cached data found to purge.")
        return

    click.echo("The following cached data will be purged:")
    click.echo()

    if models_to_delete:
        click.echo(f"Models ({len(models_to_delete)} items, {_format_size(models_size)}):")
        for repo in models_to_delete:
            click.echo(f"  - {repo.repo_id} ({_format_size(repo.size_on_disk)})")
        click.echo()

    if datasets_to_delete:
        click.echo(f"Datasets ({len(datasets_to_delete)} items, {_format_size(datasets_size)}):")
        for repo in datasets_to_delete:
            click.echo(f"  - {repo.repo_id} ({_format_size(repo.size_on_disk)})")
        click.echo()

    click.echo(f"Total space to be freed: {_format_size(total_size)}")
    click.echo()

    if not yes:
        if not click.confirm("Are you sure you want to delete this cached data?"):
            click.echo("Aborted.")
            return

    deleted_count = 0
    freed_space = 0

    from huggingface_hub import constants
    cache_path = Path(constants.HF_HUB_CACHE)

    for repo in models_to_delete + datasets_to_delete:
        try:
            for revision in repo.revisions:
                shutil.rmtree(revision.snapshot_path, ignore_errors=True)
            repo_path = cache_path / f"{repo.repo_type}s--{repo.repo_id.replace('/', '--')}"
            if repo_path.exists():
                shutil.rmtree(repo_path, ignore_errors=True)
            deleted_count += 1
            freed_space += repo.size_on_disk
        except Exception as e:
            click.echo(f"Warning: Failed to delete {repo.repo_id}: {e}")

    click.echo(f"Successfully purged {deleted_count} items, freed {_format_size(freed_space)}.")


def _format_size(size_bytes: int) -> str:
    """Format bytes into human-readable size."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} PB"


@cli.command()
@click.argument('directory', required=True)
def explore(directory: str):
    """
    Launch interactive TUI dashboard for exploring audio datasets.

    Browse audio files, view metadata, play audio, and generate spectrograms
    in an interactive terminal interface.

    \b
    Keyboard shortcuts:
      ↑/↓, j/k    Navigate file list
      Enter       View file details
      p           Play selected audio
      s           Generate spectrogram
      r           Refresh file list
      /           Search files
      ?           Show help
      q           Quit
    """
    import os

    if not os.path.isdir(directory):
        click.echo(f"Error: '{directory}' is not a valid directory", err=True)
        raise SystemExit(1)

    from bioamla.tui import run_explorer
    run_explorer(directory)


# =============================================================================
# AST Command Group
# =============================================================================

@cli.group()
def ast():
    """Audio Spectrogram Transformer model commands."""
    pass


@ast.command('predict')
@click.argument('path')
@click.option('--model-path', default='bioamla/scp-frogs', help='AST model to use for inference')
@click.option('--resample-freq', default=16000, type=int, help='Resampling frequency')
@click.option('--batch', is_flag=True, default=False, help='Run batch inference on a directory of audio files')
@click.option('--output-csv', default='output.csv', help='Output CSV file name (batch mode only)')
@click.option('--clip-seconds', default=1, type=int, help='Duration of audio clips in seconds (batch mode only)')
@click.option('--overlap-seconds', default=0, type=int, help='Overlap between clips in seconds (batch mode only)')
@click.option('--restart/--no-restart', default=False, help='Whether to restart from existing results (batch mode only)')
@click.option('--batch-size', default=8, type=int, help='Number of segments to process in parallel (default: 8, batch mode only)')
@click.option('--fp16/--no-fp16', default=False, help='Use half-precision (FP16) for faster GPU inference (batch mode only)')
@click.option('--compile/--no-compile', default=False, help='Use torch.compile() for optimized inference (PyTorch 2.0+, batch mode only)')
@click.option('--workers', default=1, type=int, help='Number of parallel workers for file loading (default: 1, batch mode only)')
def ast_predict(
    path: str,
    model_path: str,
    resample_freq: int,
    batch: bool,
    output_csv: str,
    clip_seconds: int,
    overlap_seconds: int,
    restart: bool,
    batch_size: int,
    fp16: bool,
    compile: bool,
    workers: int
):
    """
    Perform prediction on audio file(s).

    PATH can be a single audio file or a directory (with --batch flag).

    Single file mode (default):
        bioamla ast predict audio.wav --model-path my_model

    Batch mode (--batch):
        bioamla ast predict ./audio_dir --batch --model-path my_model

        Processes all WAV files in the specified directory and saves predictions
        to a CSV file. Supports resumable operations.

        Performance options (batch mode only):
            --batch-size: Process multiple segments in one forward pass (GPU optimization)
            --fp16: Use half-precision inference for ~2x speedup on modern GPUs
            --compile: Use torch.compile() for optimized model execution
            --workers: Parallel file loading for I/O-bound workloads
    """
    if batch:
        _run_batch_inference(
            directory=path,
            output_csv=output_csv,
            model_path=model_path,
            resample_freq=resample_freq,
            clip_seconds=clip_seconds,
            overlap_seconds=overlap_seconds,
            restart=restart,
            batch_size=batch_size,
            fp16=fp16,
            compile=compile,
            workers=workers
        )
    else:
        from bioamla.ast import wav_ast_inference
        prediction = wav_ast_inference(path, model_path, resample_freq)
        click.echo(f"{prediction}")


def _run_batch_inference(
    directory: str,
    output_csv: str,
    model_path: str,
    resample_freq: int,
    clip_seconds: int,
    overlap_seconds: int,
    restart: bool,
    batch_size: int,
    fp16: bool,
    compile: bool,
    workers: int
):
    """Run batch inference on a directory of audio files."""
    import os
    import time

    import pandas as pd
    import torch
    from novus_pytils.files import file_exists, get_files_by_extension

    from bioamla.ast import (
        InferenceConfig,
        load_pretrained_ast_model,
        wave_file_batch_inference,
    )

    output_csv = os.path.join(directory, output_csv)
    print("Output csv: " + output_csv)

    wave_files = get_files_by_extension(directory=directory, extensions=['.wav'], recursive=True)

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
            processed_files = set(df['filepath'])
            print("Found " + str(len(processed_files)) + " processed files")

            print("Removing processed files from wave files")
            wave_files = [f for f in wave_files if f not in processed_files]

            print("Found " + str(len(wave_files)) + " wave files left to process")

            if len(wave_files) == 0:
                print("No wave files left to process")
                return
        else:
            print("creating new file: " + output_csv)
            results = pd.DataFrame(columns=['filepath', 'start', 'stop', 'prediction'])
            results.to_csv(output_csv, header=True, index=False)
    else:
        print("creating new file: " + output_csv)
        results = pd.DataFrame(columns=['filepath', 'start', 'stop', 'prediction'])
        results.to_csv(output_csv, header=True, index=False)

    print("Loading model: " + model_path)
    print(f"Performance options: batch_size={batch_size}, fp16={fp16}, compile={compile}, workers={workers}")

    model = load_pretrained_ast_model(model_path, use_fp16=fp16, use_compile=compile)

    from torch.nn import Module
    if not isinstance(model, Module):
        raise TypeError("Model must be a PyTorch Module")

    model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device: " + device)

    config = InferenceConfig(
        batch_size=batch_size,
        use_fp16=fp16,
        use_compile=compile,
        num_workers=workers
    )

    start_time = time.time()
    time_string = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time))
    print("Start batch inference at " + time_string)

    wave_file_batch_inference(
        wave_files=wave_files,
        model=model,
        freq=resample_freq,
        clip_seconds=clip_seconds,
        overlap_seconds=overlap_seconds,
        output_csv=output_csv,
        config=config
    )

    end_time = time.time()
    time_string = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end_time))
    print("End batch inference at " + time_string)
    elapsed = end_time - start_time
    print(f"Elapsed time: {elapsed:.2f}s ({len(wave_files)/elapsed:.2f} files/sec)")


@ast.command('train')
@click.option('--training-dir', default='.', help='Directory to save training outputs')
@click.option('--base-model', default='MIT/ast-finetuned-audioset-10-10-0.4593', help='Base model to fine-tune')
@click.option('--train-dataset', default='bioamla/scp-frogs', help='Training dataset from HuggingFace Hub')
@click.option('--split', default='train', help='Dataset split to use')
@click.option('--category-id-column', default='target', help='Column name for category IDs')
@click.option('--category-label-column', default='category', help='Column name for category labels')
@click.option('--report-to', default='tensorboard', help='Where to report metrics')
@click.option('--learning-rate', default=5.0e-5, type=float, help='Learning rate for training')
@click.option('--push-to-hub/--no-push-to-hub', default=False, help='Whether to push model to HuggingFace Hub')
@click.option('--num-train-epochs', default=1, type=int, help='Number of training epochs')
@click.option('--per-device-train-batch-size', default=8, type=int, help='Training batch size per device')
@click.option('--eval-strategy', default='epoch', help='Evaluation strategy')
@click.option('--save-strategy', default='epoch', help='Model save strategy')
@click.option('--eval-steps', default=1, type=int, help='Number of steps between evaluations')
@click.option('--save-steps', default=1, type=int, help='Number of steps between saves')
@click.option('--load-best-model-at-end/--no-load-best-model-at-end', default=True, help='Load best model at end of training')
@click.option('--metric-for-best-model', default='accuracy', help='Metric to use for best model selection')
@click.option('--logging-strategy', default='steps', help='Logging strategy')
@click.option('--logging-steps', default=100, type=int, help='Number of steps between logging')
@click.option('--fp16/--no-fp16', default=False, help='Use FP16 mixed precision training (for NVIDIA GPUs)')
@click.option('--bf16/--no-bf16', default=False, help='Use BF16 mixed precision training (for Ampere+ GPUs)')
@click.option('--gradient-accumulation-steps', default=1, type=int, help='Number of gradient accumulation steps')
@click.option('--dataloader-num-workers', default=4, type=int, help='Number of dataloader workers')
@click.option('--torch-compile/--no-torch-compile', default=False, help='Use torch.compile for faster training (PyTorch 2.0+)')
@click.option('--finetune-mode', type=click.Choice(['full', 'feature-extraction']), default='full', help='Training mode: full (all layers) or feature-extraction (freeze base, train classifier only)')
@click.option('--mlflow-tracking-uri', default=None, help='MLflow tracking server URI (e.g., http://localhost:5000)')
@click.option('--mlflow-experiment-name', default=None, help='MLflow experiment name')
@click.option('--mlflow-run-name', default=None, help='MLflow run name')
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
    mlflow_run_name: str
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
    from novus_pytils.files import create_directory
    from transformers import (
        ASTConfig,
        ASTFeatureExtractor,
        ASTForAudioClassification,
        Trainer,
        TrainingArguments,
    )

    output_dir = training_dir + "/runs"
    logging_dir = training_dir + "/logs"
    best_model_path = training_dir + "/best_model"

    mlflow_run = None
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
            print("Warning: MLflow not installed. Install with 'pip install mlflow' to enable MLflow tracking.")

    dataset = load_dataset(train_dataset, split=split)

    if isinstance(dataset, Dataset):
        class_names = sorted(list(set(dataset[category_label_column])))
    elif isinstance(dataset, DatasetDict):
        first_split_name = list(dataset.keys())[0]
        first_split = dataset[first_split_name]
        class_names = sorted(list(set(first_split[category_label_column])))
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
    id2label = {idx: name for idx, name in enumerate(class_names)}

    test_size = 0.2
    if isinstance(dataset, Dataset):
        try:
            dataset = dataset.train_test_split(
                test_size=test_size, shuffle=True, seed=0, stratify_by_column="labels")
        except ValueError as e:
            print(f"Warning: Stratified split failed ({e}). Using regular split.")
            dataset = dataset.train_test_split(
                test_size=test_size, shuffle=True, seed=0)
    elif isinstance(dataset, DatasetDict) and "test" not in dataset:
        train_data = dataset["train"]
        try:
            dataset = train_data.train_test_split(
                test_size=test_size, shuffle=True, seed=0, stratify_by_column="labels")
        except ValueError as e:
            print(f"Warning: Stratified split failed ({e}). Using regular split.")
            dataset = train_data.train_test_split(
                test_size=test_size, shuffle=True, seed=0)

    audio_augmentations = Compose([
        AddGaussianSNR(min_snr_db=10, max_snr_db=20),
        Gain(min_gain_db=-6, max_gain_db=6),
        GainTransition(min_gain_db=-6, max_gain_db=6, min_duration=0.01, max_duration=0.3, duration_unit="fraction"),
        ClippingDistortion(min_percentile_threshold=0, max_percentile_threshold=30, p=0.5),
        TimeStretch(min_rate=0.8, max_rate=1.2),
        PitchShift(min_semitones=-4, max_semitones=4),
    ], p=0.8, shuffle=True)

    def preprocess_audio_with_transforms(batch):
        wavs = [audio_augmentations(audio["array"], sample_rate=SAMPLING_RATE) for audio in batch["input_values"]]
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
            dataset["train"].set_transform(preprocess_audio_with_transforms, output_all_columns=False)
        if "test" in dataset:
            dataset["test"].set_transform(preprocess_audio, output_all_columns=False)
    else:
        raise ValueError("Expected DatasetDict for transform application")

    config = ASTConfig.from_pretrained(pretrained_model)
    config.num_labels = num_labels
    config.label2id = label2id
    config.id2label = {v: k for k, v in label2id.items()}

    model = ASTForAudioClassification.from_pretrained(pretrained_model, config=config, ignore_mismatched_sizes=True)
    model.init_weights()

    if finetune_mode == "feature-extraction":
        print("Feature extraction mode: freezing base model, only training classifier head")
        for param in model.audio_spectrogram_transformer.parameters():
            param.requires_grad = False
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"  Trainable parameters: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)")
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

        precision_result = precision.compute(predictions=predictions, references=eval_pred.label_ids, average=AVERAGE)
        if precision_result is not None:
            metrics.update(precision_result)

        recall_result = recall.compute(predictions=predictions, references=eval_pred.label_ids, average=AVERAGE)
        if recall_result is not None:
            metrics.update(recall_result)

        f1_result = f1.compute(predictions=predictions, references=eval_pred.label_ids, average=AVERAGE)
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


@ast.command('evaluate')
@click.argument('path')
@click.option('--model-path', default='bioamla/scp-frogs', help='AST model to use for evaluation')
@click.option('--ground-truth', '-g', required=True, help='Path to CSV file with ground truth labels')
@click.option('--output', '-o', default=None, help='Output file for evaluation results')
@click.option('--format', 'output_format', type=click.Choice(['json', 'csv', 'txt']), default='txt', help='Output format')
@click.option('--file-column', default='file_name', help='Column name for file names in ground truth CSV')
@click.option('--label-column', default='label', help='Column name for labels in ground truth CSV')
@click.option('--resample-freq', default=16000, type=int, help='Resampling frequency')
@click.option('--batch-size', default=8, type=int, help='Batch size for inference')
@click.option('--fp16/--no-fp16', default=False, help='Use half-precision inference')
@click.option('--quiet', is_flag=True, help='Only output metrics, suppress progress')
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
    """Evaluate an AST model on a directory of audio files.

    PATH is a directory containing audio files to evaluate.

    Example:
        bioamla ast evaluate ./test_audio --model bioamla/scp-frogs --ground-truth labels.csv
    """
    from pathlib import Path as PathLib

    from bioamla.evaluate import (
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

        # Display results
        if not quiet:
            report = format_metrics_report(result)
            click.echo(report)
        else:
            click.echo(f"Accuracy: {result.accuracy:.4f}")
            click.echo(f"Precision: {result.precision:.4f}")
            click.echo(f"Recall: {result.recall:.4f}")
            click.echo(f"F1 Score: {result.f1_score:.4f}")

        # Save results if output specified
        if output:
            save_evaluation_results(result, output, format=output_format)
            click.echo(f"Results saved to: {output}")

    except FileNotFoundError as e:
        click.echo(f"Error: {e}")
        raise SystemExit(1)
    except ValueError as e:
        click.echo(f"Error: {e}")
        raise SystemExit(1)


# =============================================================================
# Audio Command Group
# =============================================================================

@cli.group()
def audio():
    """Audio file utilities."""
    pass


@audio.command('list')
@click.argument('filepath', required=False, default='.')
def audio_list(filepath: str):
    """List audio files in a directory."""
    from novus_pytils.audio import get_audio_files
    try:
        if filepath == '.':
            import os
            filepath = os.getcwd()
        audio_files = get_audio_files(filepath)
        if audio_files:
            for file in audio_files:
                click.echo(file)
        else:
            click.echo("No audio files found in the specified directory.")
    except Exception as e:
        click.echo(f"An error occurred: {e}")


@audio.command('info')
@click.argument('filepath')
def audio_info(filepath: str):
    """Display metadata from an audio file."""
    from novus_pytils.audio.wave import get_wav_metadata
    metadata = get_wav_metadata(filepath)
    click.echo(f"{metadata}")


@audio.command('convert')
@click.argument('path')
@click.argument('target_format')
@click.option('--output', '-o', default=None, help='Output file or directory')
@click.option('--batch', is_flag=True, help='Process all files in directory')
@click.option('--dataset', is_flag=True, help='Convert dataset with metadata.csv (updates metadata)')
@click.option('--metadata-filename', default='metadata.csv', help='Name of metadata CSV file (for --dataset mode)')
@click.option('--keep-original', is_flag=True, help='Keep original files after conversion')
@click.option('--recursive/--no-recursive', default=True, help='Search subdirectories (batch mode)')
@click.option('--quiet', is_flag=True, help='Suppress progress output')
def audio_convert(
    path: str,
    target_format: str,
    output: str,
    batch: bool,
    dataset: bool,
    metadata_filename: str,
    keep_original: bool,
    recursive: bool,
    quiet: bool
):
    """Convert audio files to a different format.

    Single file: bioamla audio convert input.mp3 wav -o output.wav

    Batch mode: bioamla audio convert ./audio_dir wav --batch -o ./converted

    Dataset mode: bioamla audio convert ./dataset wav --dataset (updates metadata.csv)
    """
    from pathlib import Path

    if dataset:
        # Legacy dataset mode with metadata.csv
        from bioamla.datasets import convert_filetype

        stats = convert_filetype(
            dataset_path=path,
            target_format=target_format,
            metadata_filename=metadata_filename,
            keep_original=keep_original,
            verbose=not quiet
        )

        if quiet:
            click.echo(f"Converted {stats['files_converted']} files to {target_format}")

    elif batch:
        # Batch convert directory
        from bioamla.datasets import batch_convert_audio

        if output is None:
            output = str(Path(path)) + f"_{target_format}"

        try:
            stats = batch_convert_audio(
                input_dir=path,
                output_dir=output,
                target_format=target_format,
                recursive=recursive,
                keep_original=keep_original,
                verbose=not quiet
            )

            if quiet:
                click.echo(f"Converted {stats['files_converted']} files to {output}")
        except FileNotFoundError as e:
            click.echo(f"Error: {e}")
            raise SystemExit(1)
        except ValueError as e:
            click.echo(f"Error: {e}")
            raise SystemExit(1)

    else:
        # Single file conversion
        from bioamla.datasets import convert_audio_file

        input_path = Path(path)
        if not input_path.exists():
            click.echo(f"Error: File not found: {path}")
            raise SystemExit(1)

        if output is None:
            output = str(input_path.with_suffix(f".{target_format.lstrip('.')}"))

        try:
            result = convert_audio_file(str(input_path), output, target_format)
            if not quiet:
                click.echo(f"Converted: {result}")
        except ValueError as e:
            click.echo(f"Error: {e}")
            raise SystemExit(1)


@audio.command('filter')
@click.argument('path')
@click.option('--output', '-o', default=None, help='Output file or directory')
@click.option('--batch', is_flag=True, help='Process all files in directory')
@click.option('--bandpass', default=None, help='Bandpass filter range (e.g., "1000-8000")')
@click.option('--lowpass', default=None, type=float, help='Lowpass cutoff frequency in Hz')
@click.option('--highpass', default=None, type=float, help='Highpass cutoff frequency in Hz')
@click.option('--order', default=5, type=int, help='Filter order (default: 5)')
@click.option('--quiet', is_flag=True, help='Suppress progress output')
def audio_filter(path, output, batch, bandpass, lowpass, highpass, order, quiet):
    """Apply frequency filter to audio files."""
    from bioamla.signal import (
        bandpass_filter,
        highpass_filter,
        lowpass_filter,
    )

    if not any([bandpass, lowpass, highpass]):
        click.echo("Error: Must specify --bandpass, --lowpass, or --highpass")
        raise SystemExit(1)

    def processor(audio, sr):
        if bandpass:
            parts = bandpass.split('-')
            low, high = float(parts[0]), float(parts[1])
            return bandpass_filter(audio, sr, low, high, order)
        elif lowpass:
            return lowpass_filter(audio, sr, lowpass, order)
        elif highpass:
            return highpass_filter(audio, sr, highpass, order)
        return audio

    _run_signal_processing(path, output, batch, processor, quiet, "filter")


@audio.command('denoise')
@click.argument('path')
@click.option('--output', '-o', default=None, help='Output file or directory')
@click.option('--batch', is_flag=True, help='Process all files in directory')
@click.option('--method', type=click.Choice(['spectral']), default='spectral', help='Denoising method')
@click.option('--strength', default=1.0, type=float, help='Noise reduction strength (0-2, default: 1.0)')
@click.option('--quiet', is_flag=True, help='Suppress progress output')
def audio_denoise(path, output, batch, method, strength, quiet):
    """Apply noise reduction to audio files."""
    from bioamla.signal import spectral_denoise

    def processor(audio, sr):
        return spectral_denoise(audio, sr, noise_reduce_factor=strength)

    _run_signal_processing(path, output, batch, processor, quiet, "denoise")


@audio.command('segment')
@click.argument('path')
@click.option('--output', '-o', required=True, help='Output directory for segments')
@click.option('--silence-threshold', default=-40, type=float, help='Silence threshold in dB (default: -40)')
@click.option('--min-silence', default=0.3, type=float, help='Min silence duration in seconds (default: 0.3)')
@click.option('--min-segment', default=0.5, type=float, help='Min segment duration in seconds (default: 0.5)')
@click.option('--quiet', is_flag=True, help='Suppress progress output')
def audio_segment(path, output, silence_threshold, min_silence, min_segment, quiet):
    """Split audio on silence into separate files."""
    from pathlib import Path

    from bioamla.signal import load_audio, save_audio, split_audio_on_silence

    path = Path(path)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)

    if not path.exists():
        click.echo(f"Error: File not found: {path}")
        raise SystemExit(1)

    audio, sr = load_audio(str(path))
    chunks = split_audio_on_silence(
        audio, sr,
        silence_threshold_db=silence_threshold,
        min_silence_duration=min_silence,
        min_segment_duration=min_segment
    )

    if not chunks:
        click.echo("No segments found")
        return

    stem = path.stem
    for i, (chunk, start, end) in enumerate(chunks):
        out_path = output / f"{stem}_seg{i+1:03d}_{start:.2f}-{end:.2f}s.wav"
        save_audio(str(out_path), chunk, sr)
        if not quiet:
            click.echo(f"  Created: {out_path}")

    click.echo(f"Created {len(chunks)} segments in {output}")


@audio.command('detect-events')
@click.argument('path')
@click.option('--output', '-o', required=True, help='Output CSV file for events')
@click.option('--quiet', is_flag=True, help='Suppress progress output')
def audio_detect_events(path, output, quiet):
    """Detect onset events in audio and save to CSV."""
    import csv
    from pathlib import Path

    from bioamla.signal import detect_onsets, load_audio

    path = Path(path)
    if not path.exists():
        click.echo(f"Error: File not found: {path}")
        raise SystemExit(1)

    audio, sr = load_audio(str(path))
    events = detect_onsets(audio, sr)

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    with open(output, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['time', 'strength'])
        for event in events:
            writer.writerow([f"{event.time:.4f}", f"{event.strength:.4f}"])

    if not quiet:
        click.echo(f"Detected {len(events)} events, saved to {output}")


@audio.command('normalize')
@click.argument('path')
@click.option('--output', '-o', default=None, help='Output file or directory')
@click.option('--batch', is_flag=True, help='Process all files in directory')
@click.option('--target-db', default=-20, type=float, help='Target loudness in dB (default: -20)')
@click.option('--peak', is_flag=True, help='Use peak normalization instead of RMS')
@click.option('--quiet', is_flag=True, help='Suppress progress output')
def audio_normalize(path, output, batch, target_db, peak, quiet):
    """Normalize audio loudness."""
    from bioamla.signal import normalize_loudness, peak_normalize

    def processor(audio, sr):
        if peak:
            target_linear = 10 ** (target_db / 20)
            return peak_normalize(audio, target_peak=min(target_linear, 0.99))
        return normalize_loudness(audio, sr, target_db=target_db)

    _run_signal_processing(path, output, batch, processor, quiet, "normalize")


@audio.command('resample')
@click.argument('path')
@click.option('--output', '-o', default=None, help='Output file or directory')
@click.option('--batch', is_flag=True, help='Process all files in directory')
@click.option('--rate', required=True, type=int, help='Target sample rate in Hz')
@click.option('--quiet', is_flag=True, help='Suppress progress output')
def audio_resample(path, output, batch, rate, quiet):
    """Resample audio to a different sample rate."""
    from bioamla.signal import resample_audio

    def processor(audio, sr):
        return resample_audio(audio, sr, rate)

    _run_signal_processing(path, output, batch, processor, quiet, "resample", output_sr=rate)


@audio.command('trim')
@click.argument('path')
@click.option('--output', '-o', default=None, help='Output file or directory')
@click.option('--batch', is_flag=True, help='Process all files in directory')
@click.option('--start', default=None, type=float, help='Start time in seconds')
@click.option('--end', default=None, type=float, help='End time in seconds')
@click.option('--silence', is_flag=True, help='Trim silence from start/end instead')
@click.option('--threshold', default=-40, type=float, help='Silence threshold in dB (for --silence)')
@click.option('--quiet', is_flag=True, help='Suppress progress output')
def audio_trim(path, output, batch, start, end, silence, threshold, quiet):
    """Trim audio by time or remove silence."""
    from bioamla.signal import trim_audio, trim_silence

    if not silence and start is None and end is None:
        click.echo("Error: Must specify --start/--end or use --silence")
        raise SystemExit(1)

    def processor(audio, sr):
        if silence:
            return trim_silence(audio, sr, threshold_db=threshold)
        return trim_audio(audio, sr, start_time=start, end_time=end)

    _run_signal_processing(path, output, batch, processor, quiet, "trim")


def _run_signal_processing(path, output, batch, processor, quiet, operation, output_sr=None):
    """Helper to run signal processing on file or directory."""
    from pathlib import Path

    from bioamla.signal import batch_process, load_audio, save_audio

    path = Path(path)

    if batch:
        if output is None:
            output = str(path) + f"_{operation}"

        try:
            stats = batch_process(
                str(path), output, processor,
                sample_rate=output_sr, verbose=not quiet
            )
            if quiet:
                click.echo(f"Processed {stats['files_processed']} files to {stats['output_dir']}")
        except FileNotFoundError as e:
            click.echo(f"Error: {e}")
            raise SystemExit(1)
    else:
        if not path.exists():
            click.echo(f"Error: File not found: {path}")
            raise SystemExit(1)

        if output is None:
            output = str(path.with_stem(path.stem + f"_{operation}"))

        try:
            audio, sr = load_audio(str(path))
            processed = processor(audio, sr)
            if output_sr:
                sr = output_sr
            save_audio(output, processed, sr)
            if not quiet:
                click.echo(f"Saved: {output}")
        except Exception as e:
            click.echo(f"Error: {e}")
            raise SystemExit(1)


# =============================================================================
# Visualize Command
# =============================================================================

@cli.command()
@click.argument('path')
@click.option('--output', '-o', default=None, help='Output file path (single file) or directory (batch with --batch)')
@click.option('--batch', is_flag=True, default=False, help='Process all audio files in a directory')
@click.option('--type', 'viz_type', type=click.Choice(['mel', 'mfcc', 'waveform']), default='mel', help='Type of visualization')
@click.option('--sample-rate', default=16000, type=int, help='Target sample rate for processing')
@click.option('--n-mels', default=128, type=int, help='Number of mel bands (mel spectrogram only)')
@click.option('--n-mfcc', default=40, type=int, help='Number of MFCCs (mfcc only)')
@click.option('--cmap', default='magma', help='Colormap for spectrogram visualizations')
@click.option('--recursive/--no-recursive', default=True, help='Search subdirectories (batch mode only)')
@click.option('--quiet', is_flag=True, help='Suppress progress output')
def visualize(
    path: str,
    output: str,
    batch: bool,
    viz_type: str,
    sample_rate: int,
    n_mels: int,
    n_mfcc: int,
    cmap: str,
    recursive: bool,
    quiet: bool
):
    """
    Generate spectrogram visualizations from audio files.

    PATH can be a single audio file or a directory (with --batch flag).

    Single file mode (default):
        bioamla visualize audio.wav --output spec.png

    Batch mode (--batch):
        bioamla visualize ./audio_dir --batch --output ./specs

    Visualization types:
        mel: Mel spectrogram (default)
        mfcc: Mel-frequency cepstral coefficients
        waveform: Time-domain waveform plot
    """
    import os

    from bioamla.visualize import batch_generate_spectrograms, generate_spectrogram

    if batch:
        # Batch mode: process directory
        if output is None:
            output = os.path.join(path, "spectrograms")

        stats = batch_generate_spectrograms(
            input_dir=path,
            output_dir=output,
            viz_type=viz_type,
            sample_rate=sample_rate,
            n_mels=n_mels,
            n_mfcc=n_mfcc,
            cmap=cmap,
            recursive=recursive,
            verbose=not quiet,
        )

        if quiet:
            click.echo(f"Generated {stats['files_processed']} spectrograms in {stats['output_dir']}")
    else:
        # Single file mode
        if output is None:
            # Default output: same name with .png extension
            base_name = os.path.splitext(path)[0]
            output = f"{base_name}.png"

        try:
            result = generate_spectrogram(
                audio_path=path,
                output_path=output,
                viz_type=viz_type,
                sample_rate=sample_rate,
                n_mels=n_mels,
                n_mfcc=n_mfcc,
                cmap=cmap,
            )
            if not quiet:
                click.echo(f"Generated {viz_type} spectrogram: {result}")
        except FileNotFoundError as e:
            click.echo(f"Error: {e}")
            raise SystemExit(1)
        except Exception as e:
            click.echo(f"Error generating spectrogram: {e}")
            raise SystemExit(1)


# =============================================================================
# Augment Command
# =============================================================================

def parse_range(value: str) -> tuple:
    """Parse a range string like '0.8-1.2' or '-2,2' into (min, max)."""
    if '-' in value and not value.startswith('-'):
        parts = value.split('-')
        return float(parts[0]), float(parts[1])
    elif ',' in value:
        parts = value.split(',')
        return float(parts[0]), float(parts[1])
    else:
        val = float(value)
        return val, val


@cli.command()
@click.argument('input_dir')
@click.option('--output', '-o', required=True, help='Output directory for augmented files')
@click.option('--add-noise', default=None, help='Add Gaussian noise with SNR range (e.g., "3-30" dB)')
@click.option('--time-stretch', default=None, help='Time stretch range (e.g., "0.8-1.2")')
@click.option('--pitch-shift', default=None, help='Pitch shift range in semitones (e.g., "-2,2")')
@click.option('--gain', default=None, help='Gain range in dB (e.g., "-12,12")')
@click.option('--multiply', default=1, type=int, help='Number of augmented copies to create per file')
@click.option('--sample-rate', default=16000, type=int, help='Target sample rate for output')
@click.option('--recursive/--no-recursive', default=True, help='Search subdirectories')
@click.option('--quiet', is_flag=True, help='Suppress progress output')
def augment(
    input_dir: str,
    output: str,
    add_noise: str,
    time_stretch: str,
    pitch_shift: str,
    gain: str,
    multiply: int,
    sample_rate: int,
    recursive: bool,
    quiet: bool
):
    """
    Augment audio files to expand training datasets.

    Creates augmented copies of audio files with various transformations.
    At least one augmentation option must be specified.

    Examples:
        bioamla augment ./audio --output ./augmented --add-noise 3-30

        bioamla augment ./audio --output ./augmented \\
            --add-noise 3-30 \\
            --time-stretch 0.8-1.2 \\
            --pitch-shift -2,2 \\
            --multiply 5

    Augmentation options:
        --add-noise: Add Gaussian noise with SNR in specified range (dB)
        --time-stretch: Speed up/slow down without changing pitch
        --pitch-shift: Change pitch without changing speed (semitones)
        --gain: Random volume adjustment (dB)
    """
    from bioamla.augment import AugmentationConfig, batch_augment

    # Build configuration from options
    config = AugmentationConfig(
        sample_rate=sample_rate,
        multiply=multiply,
    )

    # Parse augmentation options
    if add_noise:
        config.add_noise = True
        min_snr, max_snr = parse_range(add_noise)
        config.noise_min_snr = min_snr
        config.noise_max_snr = max_snr

    if time_stretch:
        config.time_stretch = True
        min_rate, max_rate = parse_range(time_stretch)
        config.time_stretch_min = min_rate
        config.time_stretch_max = max_rate

    if pitch_shift:
        config.pitch_shift = True
        min_semi, max_semi = parse_range(pitch_shift)
        config.pitch_shift_min = min_semi
        config.pitch_shift_max = max_semi

    if gain:
        config.gain = True
        min_db, max_db = parse_range(gain)
        config.gain_min_db = min_db
        config.gain_max_db = max_db

    # Check that at least one augmentation is enabled
    if not any([config.add_noise, config.time_stretch, config.pitch_shift, config.gain]):
        click.echo("Error: At least one augmentation option must be specified")
        click.echo("Use --help for available options")
        raise SystemExit(1)

    try:
        stats = batch_augment(
            input_dir=input_dir,
            output_dir=output,
            config=config,
            recursive=recursive,
            verbose=not quiet,
        )

        if quiet:
            click.echo(
                f"Created {stats['files_created']} augmented files from "
                f"{stats['files_processed']} source files in {stats['output_dir']}"
            )
    except FileNotFoundError as e:
        click.echo(f"Error: {e}")
        raise SystemExit(1)
    except Exception as e:
        click.echo(f"Error during augmentation: {e}")
        raise SystemExit(1)


# =============================================================================
# iNaturalist Command Group
# =============================================================================

@cli.group()
def inat():
    """iNaturalist integration commands."""
    pass


@inat.command('download')
@click.argument('output_dir')
@click.option('--taxon-ids', default=None, help='Comma-separated list of taxon IDs (e.g., "3" for birds, "3,20978" for multiple)')
@click.option('--taxon-csv', default=None, type=click.Path(exists=True), help='Path to CSV file with taxon_id column')
@click.option('--taxon-name', default=None, help='Filter by taxon name (e.g., "Aves" for birds)')
@click.option('--place-id', type=int, default=None, help='Filter by place ID (e.g., 1 for United States)')
@click.option('--user-id', default=None, help='Filter by observer username')
@click.option('--project-id', default=None, help='Filter by iNaturalist project ID or slug')
@click.option('--quality-grade', default='research', help='Quality grade: research, needs_id, or casual')
@click.option('--sound-license', default=None, help='Comma-separated list of sound licenses (e.g., "cc-by, cc-by-nc, cc0")')
@click.option('--start-date', default=None, help='Start date for observations (YYYY-MM-DD)')
@click.option('--end-date', default=None, help='End date for observations (YYYY-MM-DD)')
@click.option('--obs-per-taxon', type=int, default=100, help='Number of observations to download per taxon ID')
@click.option('--organize-by-taxon/--no-organize-by-taxon', default=True, help='Organize files into subdirectories by species')
@click.option('--include-inat-metadata', is_flag=True, help='Include additional iNaturalist metadata fields in CSV')
@click.option('--file-extensions', default=None, help='Comma-separated list of file extensions to filter (e.g., "wav,mp3")')
@click.option('--delay', type=float, default=1.0, help='Delay between downloads in seconds (rate limiting)')
@click.option('--quiet', is_flag=True, help='Suppress progress output')
def inat_download(
    output_dir: str,
    taxon_ids: str,
    taxon_csv: str,
    taxon_name: str,
    place_id: int,
    user_id: str,
    project_id: str,
    quality_grade: str,
    sound_license: str,
    start_date: str,
    end_date: str,
    obs_per_taxon: int,
    organize_by_taxon: bool,
    include_inat_metadata: bool,
    file_extensions: str,
    delay: float,
    quiet: bool
):
    """Download audio observations from iNaturalist."""
    from bioamla.inat import download_inat_audio

    taxon_ids_list = None
    if taxon_ids:
        taxon_ids_list = [int(tid.strip()) for tid in taxon_ids.split(",")]

    extensions_list = None
    if file_extensions:
        extensions_list = [ext.strip() for ext in file_extensions.split(",")]

    sound_license_list = None
    if sound_license:
        sound_license_list = [lic.strip() for lic in sound_license.split(",")]

    stats = download_inat_audio(
        output_dir=output_dir,
        taxon_ids=taxon_ids_list,
        taxon_csv=taxon_csv,
        taxon_name=taxon_name,
        place_id=place_id,
        user_id=user_id,
        project_id=project_id,
        quality_grade=quality_grade,
        sound_license=sound_license_list,
        d1=start_date,
        d2=end_date,
        obs_per_taxon=obs_per_taxon,
        organize_by_taxon=organize_by_taxon,
        include_inat_metadata=include_inat_metadata,
        file_extensions=extensions_list,
        delay_between_downloads=delay,
        verbose=not quiet
    )

    if quiet:
        click.echo(f"Downloaded {stats['total_sounds']} audio files to {stats['output_dir']}")


@inat.command('search')
@click.option('--place-id', type=int, default=None, help='Filter by place ID (e.g., 1 for United States)')
@click.option('--project-id', default=None, help='Filter by iNaturalist project ID or slug')
@click.option('--taxon-id', type=int, default=None, help='Filter by parent taxon ID (e.g., 20979 for Amphibia)')
@click.option('--quality-grade', default='research', help='Quality grade: research, needs_id, or casual')
@click.option('--output', '-o', default=None, help='Output file path for CSV (optional)')
@click.option('--quiet', is_flag=True, help='Suppress progress output')
def inat_search(
    place_id: int,
    project_id: str,
    taxon_id: int,
    quality_grade: str,
    output: str,
    quiet: bool
):
    """Search for taxa with observations in a place or project."""
    from bioamla.inat import get_taxa

    if not place_id and not project_id:
        raise click.UsageError("At least one of --place-id or --project-id must be provided")

    taxa = get_taxa(
        place_id=place_id,
        project_id=project_id,
        quality_grade=quality_grade,
        taxon_id=taxon_id,
        verbose=not quiet
    )

    if output:
        import csv
        with open(output, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['taxon_id', 'name', 'common_name', 'observation_count'])
            writer.writeheader()
            writer.writerows(taxa)
        click.echo(f"Saved {len(taxa)} taxa to {output}")
    else:
        click.echo(f"\n{'Taxon ID':<12} {'Scientific Name':<30} {'Common Name':<25} {'Obs Count':<10}")
        click.echo("-" * 80)
        for t in taxa:
            click.echo(f"{t['taxon_id']:<12} {t['name']:<30} {t['common_name']:<25} {t['observation_count']:<10}")


@inat.command('stats')
@click.argument('project_id')
@click.option('--output', '-o', default=None, help='Output file path for JSON (optional)')
@click.option('--quiet', is_flag=True, help='Suppress progress output, print only JSON')
def inat_stats(
    project_id: str,
    output: str,
    quiet: bool
):
    """Get statistics for an iNaturalist project."""
    import json

    from bioamla.inat import get_project_stats

    stats = get_project_stats(
        project_id=project_id,
        verbose=not quiet
    )

    if output:
        with open(output, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2)
        click.echo(f"Saved project stats to {output}")
    elif quiet:
        click.echo(json.dumps(stats, indent=2))
    else:
        click.echo(f"\nProject: {stats['title']}")
        click.echo(f"URL: {stats['url']}")
        click.echo(f"Type: {stats['project_type']}")
        if stats['place']:
            click.echo(f"Place: {stats['place']}")
        click.echo(f"Created: {stats['created_at']}")
        click.echo("\nStatistics:")
        click.echo(f"  Observations: {stats['observation_count']}")
        click.echo(f"  Species: {stats['species_count']}")
        click.echo(f"  Observers: {stats['observers_count']}")


# =============================================================================
# Dataset Command Group
# =============================================================================

@cli.group()
def dataset():
    """Dataset management commands."""
    pass


@dataset.command('merge')
@click.argument('output_dir')
@click.argument('dataset_paths', nargs=-1, required=True)
@click.option('--metadata-filename', default='metadata.csv', help='Name of metadata CSV file in each dataset')
@click.option('--overwrite', is_flag=True, help='Overwrite existing files instead of skipping')
@click.option('--no-organize', is_flag=True, help='Preserve original directory structure instead of organizing by category')
@click.option('--target-format', default=None, help='Convert all audio files to this format (wav, mp3, flac, etc.)')
@click.option('--quiet', is_flag=True, help='Suppress progress output')
def dataset_merge(
    output_dir: str,
    dataset_paths: tuple,
    metadata_filename: str,
    overwrite: bool,
    no_organize: bool,
    target_format: str,
    quiet: bool
):
    """Merge multiple audio datasets into a single dataset."""
    from bioamla.datasets import merge_datasets as do_merge

    stats = do_merge(
        dataset_paths=list(dataset_paths),
        output_dir=output_dir,
        metadata_filename=metadata_filename,
        skip_existing=not overwrite,
        organize_by_category=not no_organize,
        target_format=target_format,
        verbose=not quiet
    )

    if quiet:
        msg = f"Merged {stats['datasets_merged']} datasets: {stats['total_files']} total files"
        if target_format:
            msg += f", {stats['files_converted']} converted"
        click.echo(msg)


@dataset.command('license')
@click.argument('path')
@click.option('--template', '-t', default=None, help='Template file to prepend to the license file')
@click.option('--output', '-o', default='LICENSE', help='Output filename for the license file')
@click.option('--metadata-filename', default='metadata.csv', help='Name of metadata CSV file')
@click.option('--batch', is_flag=True, help='Process all datasets in directory (each subdirectory with metadata.csv)')
@click.option('--quiet', is_flag=True, help='Suppress progress output')
def dataset_license(
    path: str,
    template: str,
    output: str,
    metadata_filename: str,
    batch: bool,
    quiet: bool
):
    """Generate license/attribution file from dataset metadata.

    Reads attribution data from metadata CSV and generates a formatted LICENSE file.

    Required CSV columns: file_name, attr_id, attr_lic, attr_url, attr_note
    """
    from pathlib import Path as PathLib

    from bioamla.license import (
        generate_license_for_dataset,
        generate_licenses_for_directory,
    )

    path_obj = PathLib(path)
    template_path = PathLib(template) if template else None

    if template_path and not template_path.exists():
        click.echo(f"Error: Template file '{template}' not found.")
        raise SystemExit(1)

    if batch:
        # Process all datasets in directory
        if not path_obj.is_dir():
            click.echo(f"Error: Path '{path}' is not a directory.")
            raise SystemExit(1)

        if not quiet:
            click.echo(f"Scanning directory for datasets: {path}")

        try:
            stats = generate_licenses_for_directory(
                audio_dir=path_obj,
                template_path=template_path,
                output_filename=output,
                metadata_filename=metadata_filename
            )
        except FileNotFoundError as e:
            click.echo(f"Error: {e}")
            raise SystemExit(1)

        if stats['datasets_found'] == 0:
            click.echo("No datasets found (no directories with metadata.csv)")
            raise SystemExit(1)

        if not quiet:
            click.echo(f"\nProcessed {stats['datasets_found']} dataset(s):")
            click.echo(f"  Successful: {stats['datasets_processed']}")
            click.echo(f"  Failed: {stats['datasets_failed']}")

            for result in stats['results']:
                if result['status'] == 'success':
                    click.echo(f"  - {result['dataset_name']}: {result['attributions_count']} attributions")
                else:
                    click.echo(f"  - {result['dataset_name']}: FAILED - {result.get('error', 'Unknown error')}")
        else:
            click.echo(f"Generated {stats['datasets_processed']} license files")

        if stats['datasets_failed'] > 0:
            raise SystemExit(1)

    else:
        # Process single dataset
        if not path_obj.is_dir():
            click.echo(f"Error: Path '{path}' is not a directory.")
            raise SystemExit(1)

        csv_path = path_obj / metadata_filename
        if not csv_path.exists():
            click.echo(f"Error: Metadata file '{csv_path}' not found.")
            raise SystemExit(1)

        if not quiet:
            click.echo(f"Generating license file for: {path}")

        try:
            stats = generate_license_for_dataset(
                dataset_path=path_obj,
                template_path=template_path,
                output_filename=output,
                metadata_filename=metadata_filename
            )
        except (FileNotFoundError, ValueError) as e:
            click.echo(f"Error: {e}")
            raise SystemExit(1)

        if not quiet:
            click.echo(f"License file generated: {stats['output_path']}")
            click.echo(f"  Attributions: {stats['attributions_count']}")
            click.echo(f"  File size: {stats['file_size']:,} bytes")
        else:
            click.echo(f"Generated {output} with {stats['attributions_count']} attributions")


# =============================================================================
# HuggingFace Hub Command Group
# =============================================================================

def _get_folder_size(path: str) -> int:
    """Calculate the total size of a folder in bytes."""
    import os
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            if os.path.isfile(filepath):
                total_size += os.path.getsize(filepath)
    return total_size


def _count_files(path: str) -> int:
    """Count the total number of files in a folder."""
    import os
    count = 0
    for dirpath, dirnames, filenames in os.walk(path):
        count += len(filenames)
    return count


def _is_large_folder(path: str, size_threshold_gb: float = 5.0, file_count_threshold: int = 1000) -> bool:
    """
    Determine if a folder should be uploaded using upload_large_folder.

    A folder is considered 'large' if:
    - Total size exceeds size_threshold_gb (default: 5 GB), OR
    - Total file count exceeds file_count_threshold (default: 1000 files)
    """
    size_threshold_bytes = size_threshold_gb * 1024 * 1024 * 1024
    folder_size = _get_folder_size(path)
    file_count = _count_files(path)
    return folder_size > size_threshold_bytes or file_count > file_count_threshold


@cli.group()
def hf():
    """HuggingFace Hub commands for pushing models and datasets."""
    pass


@hf.command('push-model')
@click.argument('path')
@click.argument('repo_id')
@click.option('--private/--public', default=False, help='Make the repository private (default: public)')
@click.option('--commit-message', default=None, help='Custom commit message for the push')
def hf_push_model(
    path: str,
    repo_id: str,
    private: bool,
    commit_message: str
):
    """Push a model folder to the HuggingFace Hub.

    Uploads the entire contents of PATH folder to the Hub as a model.
    Automatically uses upload_large_folder for folders >5GB or >1000 files.
    """
    import os

    from huggingface_hub import HfApi

    if not os.path.isdir(path):
        click.echo(f"Error: Path '{path}' does not exist or is not a directory.")
        raise SystemExit(1)

    click.echo(f"Pushing model folder {path} to HuggingFace Hub: {repo_id}...")

    try:
        api = HfApi()
        api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)

        if _is_large_folder(path):
            click.echo("Large folder detected, using optimized upload method...")
            api.upload_large_folder(
                folder_path=path,
                repo_id=repo_id,
                repo_type="model",
            )
        else:
            api.upload_folder(
                folder_path=path,
                repo_id=repo_id,
                repo_type="model",
                commit_message=commit_message or "Upload model",
            )

        click.echo(f"Successfully pushed model to: https://huggingface.co/{repo_id}")

    except Exception as e:
        click.echo(f"Error pushing to HuggingFace Hub: {e}")
        click.echo("Make sure you are logged in with 'huggingface-cli login'.")
        raise SystemExit(1)


@hf.command('push-dataset')
@click.argument('path')
@click.argument('repo_id')
@click.option('--private/--public', default=False, help='Make the repository private (default: public)')
@click.option('--commit-message', default=None, help='Custom commit message for the push')
def hf_push_dataset(
    path: str,
    repo_id: str,
    private: bool,
    commit_message: str
):
    """Push a dataset folder to the HuggingFace Hub.

    Uploads the entire contents of PATH folder to the Hub as a dataset.
    Automatically uses upload_large_folder for folders >5GB or >1000 files.
    """
    import os

    from huggingface_hub import HfApi

    if not os.path.isdir(path):
        click.echo(f"Error: Path '{path}' does not exist or is not a directory.")
        raise SystemExit(1)

    click.echo(f"Pushing dataset folder {path} to HuggingFace Hub: {repo_id}...")

    try:
        api = HfApi()
        api.create_repo(repo_id=repo_id, repo_type="dataset", private=private, exist_ok=True)

        if _is_large_folder(path):
            click.echo("Large folder detected, using optimized upload method...")
            api.upload_large_folder(
                folder_path=path,
                repo_id=repo_id,
                repo_type="dataset",
            )
        else:
            api.upload_folder(
                folder_path=path,
                repo_id=repo_id,
                repo_type="dataset",
                commit_message=commit_message or "Upload dataset",
            )

        click.echo(f"Successfully pushed dataset to: https://huggingface.co/datasets/{repo_id}")

    except Exception as e:
        click.echo(f"Error pushing to HuggingFace Hub: {e}")
        click.echo("Make sure you are logged in with 'huggingface-cli login'.")
        raise SystemExit(1)


if __name__ == '__main__':
    cli()
