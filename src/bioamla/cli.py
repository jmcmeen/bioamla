from typing import Dict

import click


@click.group()
def cli():
    """Bioamla CLI"""
    pass

@cli.command()
def devices():
    """
    Display comprehensive device information including CUDA and GPU details.

    Retrieves and displays information about available compute devices,
    focusing on CUDA-capable GPUs that can be used for machine learning
    inference and training tasks.
    """
    from bioamla.core.diagnostics import get_device_info
    device_info = get_device_info()

    click.echo("Devices:")
    click.echo(f'CUDA available: {device_info["cuda_available"]}')
    click.echo(f'Current device: {device_info["current_device"]}')
    click.echo(f'Device count: {device_info["device_count"]}')

    for device in device_info['devices']:
        click.echo(f'  - Index: {device["index"]}, Name: {device["name"]}')

@cli.command()
@click.argument('url', required=True)
@click.argument('output_dir', required=False, default='.')
def download(url: str, output_dir: str):
    """
    Download a file from the specified URL to the target directory.

    Downloads a file from the given URL and saves it to the specified output
    directory. If no output directory is provided, downloads to the current
    working directory.

    Args:
        url (str): The URL of the file to download
        output_dir (str): Directory where the file should be saved.
                         Defaults to current directory if not specified.
    """
    #TODO update to filename for output

    import os

    from novus_pytils.files import download_file

    if output_dir == '.':
        output_dir = os.getcwd()

    download_file(url, output_dir)

@cli.command()
@click.argument('filepath', required=False, default='.')
def audio(filepath: str):
    """
    Display audio files from a specified directory.

    Args:
        filepath (str): The path to the directory to search for audio files.
                        Defaults to the current directory if not provided.
    """
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

@cli.command()
@click.argument('file_path')
@click.argument('output_path', required=False, default='.')
def unzip(file_path: str, output_path: str):
    """
    Extract a ZIP archive to the specified output directory.

    Extracts the contents of a ZIP file to the target directory.
    If no output path is specified, extracts to the current working directory.

    Args:
        file_path (str): Path to the ZIP file to extract
        output_path (str): Directory where the ZIP contents should be extracted.
                          Defaults to current directory if not specified.
    """
    from novus_pytils.compression import extract_zip_file
    if output_path == '.':
        import os
        output_path = os.getcwd()

    extract_zip_file(file_path, output_path)


@cli.command()
@click.argument('source_path')
@click.argument('output_file')
def zip(source_path: str, output_file: str):
    """
    Create a ZIP archive from a file or directory.

    Compresses the specified file or directory into a ZIP archive.

    Args:
        source_path (str): Path to the file or directory to compress
        output_file (str): Path for the output ZIP file
    """
    import os

    from novus_pytils.compression import create_zip_file, zip_directory

    if os.path.isdir(source_path):
        zip_directory(source_path, output_file)
    else:
        create_zip_file([source_path], output_file)

    click.echo(f"Created {output_file}")

@cli.command()
def version():
    """
    Display the current version of the bioamla package.

    This command retrieves and displays the version information
    for the installed bioamla package.
    """
    from bioamla.core.diagnostics import get_bioamla_version
    click.echo(f"bioamla v{get_bioamla_version()}")

@cli.command()
@click.option('--training-dir', default='.', help='Directory to save training outputs')
@click.option('--base-model', default='MIT/ast-finetuned-audioset-10-10-0.4593', help='Base model to fine-tune')
@click.option('--train-dataset', default='bioamla/scp-frogs', help='Training dataset from HuggingFace Hub') #TODO lets make this something else
@click.option('--split', default='train', help='Dataset split to use')
@click.option('--category-id-column', default='target', help='Column name for category IDs')
@click.option('--category-label-column', default='category', help='Column name for category labels')
@click.option('--report-to', default='tensorboard', help='Where to report metrics')
@click.option('--learning-rate', default=5.0e-5, type=float, help='Learning rate for training')
@click.option('--push-to-hub/--no-push-to-hub', default=False, help='Whether to push model to HuggingFace Hub')
@click.option('--num-train-epochs', default=1, type=int, help='Number of training epochs')
@click.option('--per-device-train-batch-size', default=1, type=int, help='Training batch size per device')
@click.option('--eval-strategy', default='epoch', help='Evaluation strategy')
@click.option('--save-strategy', default='epoch', help='Model save strategy')
@click.option('--eval-steps', default=1, type=int, help='Number of steps between evaluations')
@click.option('--save-steps', default=1, type=int, help='Number of steps between saves')
@click.option('--load-best-model-at-end/--no-load-best-model-at-end', default=True, help='Load best model at end of training')
@click.option('--metric-for-best-model', default='accuracy', help='Metric to use for best model selection')
@click.option('--logging-strategy', default='steps', help='Logging strategy')
@click.option('--logging-steps', default=100, type=int, help='Number of steps between logging')
def ast_finetune(
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
    logging_steps: int
):
    """
    Fine-tune an Audio Spectrogram Transformer (AST) model using a YAML configuration.

    Performs complete fine-tuning workflow including data loading, preprocessing,
    augmentation, model configuration, training, and evaluation. The process includes
    automatic dataset normalization calculation and comprehensive metrics tracking.

    Args:
        config_filepath (str): Path to the YAML configuration file containing
                             training parameters, dataset information, and model settings.
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
    from datasets import Audio, ClassLabel, Dataset, DatasetDict, load_dataset
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


    # Load a pre-existing dataset from the HuggingFace Hub
    dataset = load_dataset(train_dataset, split=split)

    # get target value - class name mappings
    import pandas as pd
    if isinstance(dataset, Dataset):
        selected_data = dataset.select_columns([category_id_column, category_label_column])
        df = pd.DataFrame(selected_data.to_dict())
        unique_indices = np.unique(df[category_id_column], return_index=True)[1]
        class_names = df.iloc[unique_indices][category_label_column].to_list()
    elif isinstance(dataset, DatasetDict):
        # For DatasetDict, use the first available split to get class names
        first_split_name = list(dataset.keys())[0]
        first_split = dataset[first_split_name]
        selected_data = first_split.select_columns([category_id_column, category_label_column])
        df = pd.DataFrame(selected_data.to_dict())
        unique_indices = np.unique(df[category_id_column], return_index=True)[1]
        class_names = df.iloc[unique_indices][category_label_column].to_list()
    else:
        raise TypeError("Dataset must be a Dataset or DatasetDict instance")

    # cast target and audio column
    dataset = dataset.cast_column("target", ClassLabel(names=class_names))
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000)) #TODO bad

    # rename the target feature
    dataset = dataset.rename_column("target", "labels")
    if isinstance(dataset, Dataset):
        num_labels = len(np.unique(list(dataset["labels"])))
    elif isinstance(dataset, DatasetDict) and "train" in dataset:
        num_labels = len(np.unique(list(dataset["train"]["labels"])))
    else:
        raise TypeError("Unable to determine number of labels from dataset")

    # Define the pretrained model and instantiate the feature extractor
    pretrained_model = base_model
    feature_extractor = ASTFeatureExtractor.from_pretrained(pretrained_model)
    model_input_name = feature_extractor.model_input_names[0]
    SAMPLING_RATE = feature_extractor.sampling_rate

    # Preprocessing function
    def preprocess_audio(batch):
        """
        Preprocess audio batch for AST model input without augmentations.

        Args:
            batch: Batch containing audio data and labels

        Returns:
            dict: Processed batch with feature-extracted audio and labels
        """
        wavs = [audio["array"] for audio in batch["input_values"]]
        inputs = feature_extractor(wavs, sampling_rate=SAMPLING_RATE, return_tensors="pt")
        return {model_input_name: inputs.get(model_input_name), "labels": list(batch["labels"])}

    # we use the esc50 train split
    if isinstance(dataset, DatasetDict):
        first_split = list(dataset.keys())[0]
        features = dataset[first_split].features
    else:
        features = dataset.features

    if features and "labels" in features:
        label2id = features["labels"]._str2int  # we add the mapping from INTs to STRINGs
    else:
        raise ValueError("Labels feature not found in dataset")

    # split training data
    if isinstance(dataset, Dataset):
        dataset = dataset.train_test_split(
            test_size=0.2, shuffle=True, seed=0, stratify_by_column="labels")
    elif isinstance(dataset, DatasetDict) and "test" not in dataset:
        dataset = dataset["train"].train_test_split(
            test_size=0.2, shuffle=True, seed=0, stratify_by_column="labels")

    # Define audio augmentations
    audio_augmentations = Compose([
        AddGaussianSNR(min_snr_db=10, max_snr_db=20),
        Gain(min_gain_db=-6, max_gain_db=6),
        GainTransition(min_gain_db=-6, max_gain_db=6, min_duration=0.01, max_duration=0.3, duration_unit="fraction"),
        ClippingDistortion(min_percentile_threshold=0, max_percentile_threshold=30, p=0.5),
        TimeStretch(min_rate=0.8, max_rate=1.2),
        PitchShift(min_semitones=-4, max_semitones=4),
    ], p=0.8, shuffle=True)

    # Preprocessing with augmentations
    def preprocess_audio_with_transforms(batch):
        """
        Preprocess audio batch for AST model input with applied augmentations.

        Args:
            batch: Batch containing audio data and labels

        Returns:
            dict: Processed batch with augmented, feature-extracted audio and labels
        """
        wavs = [audio_augmentations(audio["array"], sample_rate=SAMPLING_RATE) for audio in batch["input_values"]]
        inputs = feature_extractor(wavs, sampling_rate=SAMPLING_RATE, return_tensors="pt")
        return {model_input_name: inputs.get(model_input_name), "labels": list(batch["labels"])}

    dataset = dataset.cast_column("audio", Audio(sampling_rate=feature_extractor.sampling_rate))
    dataset = dataset.rename_column("audio", "input_values")

    # calculate values for normalization
    feature_extractor.do_normalize = False  # we set normalization to False in order to calculate the mean + std of the dataset
    mean = []
    std = []

    # we use the transformation w/o augmentation on the training dataset to calculate the mean + std
    if isinstance(dataset, DatasetDict) and "train" in dataset:
        train_dataset = dataset["train"]
        train_dataset.set_transform(preprocess_audio, output_all_columns=False)
        for sample in train_dataset:
            if isinstance(sample, dict) and model_input_name in sample:
                cur_mean = torch.mean(sample[model_input_name])
                cur_std = torch.std(sample[model_input_name])
                mean.append(cur_mean)
                std.append(cur_std)
    else:
        raise ValueError("Expected DatasetDict with 'train' split")

    feature_extractor.mean = float(np.mean(mean))
    feature_extractor.std = float(np.mean(std))
    feature_extractor.do_normalize = True

    print("Calculated mean and std:", feature_extractor.mean, feature_extractor.std)

    # Apply transforms
    if isinstance(dataset, DatasetDict):
        if "train" in dataset:
            dataset["train"].set_transform(preprocess_audio_with_transforms, output_all_columns=False)
        if "test" in dataset:
            dataset["test"].set_transform(preprocess_audio, output_all_columns=False)
    else:
        raise ValueError("Expected DatasetDict for transform application")

    # Load configuration from the pretrained model
    config = ASTConfig.from_pretrained(pretrained_model)
    config.num_labels = num_labels
    config.label2id = label2id
    config.id2label = {v: k for k, v in label2id.items()}

    # Initialize the model with the updated configuration
    model = ASTForAudioClassification.from_pretrained(pretrained_model, config=config, ignore_mismatched_sizes=True)
    model.init_weights()

    # Configure training arguments
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
        logging_steps=logging_steps
    )

    # Define evaluation metrics
    accuracy = evaluate.load("accuracy")
    recall = evaluate.load("recall")
    precision = evaluate.load("precision")
    f1 = evaluate.load("f1")

    AVERAGE = "macro" if config.num_labels > 2 else "binary"

    # setup metrics function
    def compute_metrics(eval_pred) -> Dict[str, float]:
        """
        Compute evaluation metrics for the AST model training.

        Args:
            eval_pred: Evaluation prediction object containing predictions and labels

        Returns:
            dict: Dictionary containing accuracy, precision, recall, and F1 metrics
        """
        # get predictions and scores
        logits = eval_pred.predictions
        predictions = np.argmax(logits, axis=1)

        # compute metrics
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

    # setup trainer
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

    # start a training
    trainer.train()

    create_directory(best_model_path)
    trainer.save_model(best_model_path)

    # torch.save(model.state_dict(), best_model_path + "/pytorch_model.bin")
    # model.save_pretrained(model_dir)

@cli.command()
@click.argument('filepath')
@click.argument('model_path')
@click.argument('sample_rate')
def ast_predict(filepath, model_path, sample_rate):
    """
    Perform AST model prediction on a single audio file.

    Args:
        filepath: Path to the audio file to classify
        model_path: Path to the pre-trained AST model
        sample_rate: Target sample rate for audio preprocessing
    """
    from bioamla.core.ast import wav_ast_inference
    prediction = wav_ast_inference(filepath, model_path, int(sample_rate))
    click.echo(f"{prediction}")

@cli.command()
@click.argument('directory')
@click.option('--output-csv', default='output.csv', help='Output CSV file name')
@click.option('--model-path', default='bioamla/scp-frogs', help='AST model to use for inference')
@click.option('--resample-freq', default=16000, type=int, help='Resampling frequency')
@click.option('--clip-seconds', default=1, type=int, help='Duration of audio clips in seconds')
@click.option('--overlap-seconds', default=0, type=int, help='Overlap between clips in seconds')
@click.option('--restart/--no-restart', default=False, help='Whether to restart from existing results')
def ast_batch_inference(
    directory: str,
    output_csv: str,
    model_path: str,
    resample_freq: int,
    clip_seconds: int,
    overlap_seconds: int,
    restart: bool
):
    """
    Run batch AST inference on a directory of WAV files.

    Loads an AST model and processes all WAV files in the specified directory,
    generating predictions and saving results to a CSV file. Supports resumable
    operations by checking for existing results and skipping already processed files.

    Args:
        directory (str): Directory containing WAV files to process
        output_csv (str): Output CSV file name
        model (str): AST model to use for inference
        resample_freq (int): Resampling frequency
        clip_seconds (int): Duration of audio clips in seconds
        overlap_seconds (int): Overlap between clips in seconds
        restart (bool): Whether to restart from existing results
    """
    import os
    import time

    import pandas as pd
    import torch
    from novus_pytils.files import file_exists, get_files_by_extension

    from bioamla.core.ast import load_pretrained_ast_model, wave_file_batch_inference

    output_csv = os.path.join(directory, output_csv)
    print("Output csv: " + output_csv)

    wave_files = get_files_by_extension(directory=directory, extensions=['.wav'], recursive=True)

    if(len(wave_files) == 0):
        print("No wave files found in directory: " + directory)
        return
    else:
        print("Found " + str(len(wave_files)) + " wave files in directory: " + directory)

    print("Restart: " + str(restart))
    if restart:
        #if file exists, read file names from file and remove from wave files
        if file_exists(output_csv):
            print("file exists: " + output_csv)
            df = pd.read_csv(output_csv)
            #filenames exist more than once get the unique ones
            processed_files = set(df['filepath'])
            print("Found " + str(len(processed_files)) + " processed files")

            print("Removing processed files from wave files")
            # Todo use sets
            for filepath in processed_files:
                wave_files.remove(filepath)

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
    model = load_pretrained_ast_model(model_path)

    # Type cast to indicate this is a PyTorch module with eval() and to() methods
    from torch.nn import Module
    if not isinstance(model, Module):
        raise TypeError("Model must be a PyTorch Module")

    model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device: " + device)
    model.to(device)

    # start timercurrent
    start_time = time.time()
    #format start time
    time_string = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time))
    print("Start batch inference at " + time_string)
    wave_file_batch_inference(wave_files=wave_files,
                              model=model,
                              freq=resample_freq,
                              clip_seconds=clip_seconds,
                              overlap_seconds=overlap_seconds,
                              output_csv=output_csv)

    # end timer
    end_time = time.time()
    time_string = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end_time))
    print("End batch inference at " + time_string)
    print("Elapsed time: " + str(end_time - start_time))

@cli.command()
@click.argument('filepath')
def wave(filepath: str):
    """
    Extract and display metadata from a WAV audio file.

    Analyzes the specified WAV file and extracts comprehensive metadata
    including audio properties, file characteristics, and technical details.

    Args:
        filepath (str): Path to the WAV file to analyze
    """
    from novus_pytils.audio.wave import get_wav_metadata
    metadata = get_wav_metadata(filepath)
    click.echo(f"{metadata}")


@cli.command()
@click.argument('output_dir')
@click.option('--taxon-ids', default=None, help='Comma-separated list of taxon IDs (e.g., "3" for birds, "3,20978" for multiple)')
@click.option('--taxon-csv', default=None, type=click.Path(exists=True), help='Path to CSV file with taxon_id column')
@click.option('--taxon-name', default=None, help='Filter by taxon name (e.g., "Aves" for birds)')
@click.option('--place-id', type=int, default=None, help='Filter by place ID (e.g., 1 for United States)')
@click.option('--user-id', default=None, help='Filter by observer username')
@click.option('--project-id', default=None, help='Filter by iNaturalist project ID or slug')
@click.option('--quality-grade', default='research', help='Quality grade: research, needs_id, or casual')
@click.option('--sound-license', default=None, help='Filter by sound license (e.g., cc-by, cc-by-nc, cc0)')
@click.option('--start-date', default=None, help='Start date for observations (YYYY-MM-DD)')
@click.option('--end-date', default=None, help='End date for observations (YYYY-MM-DD)')
@click.option('--obs-per-taxon', type=int, default=100, help='Number of observations to download per taxon ID')
@click.option('--organize-by-taxon/--no-organize-by-taxon', default=True, help='Organize files into subdirectories by species')
@click.option('--include-inat-metadata', is_flag=True, help='Include additional iNaturalist metadata fields in CSV')
@click.option('--file-extensions', default=None, help='Comma-separated list of file extensions to filter (e.g., "wav,mp3")')
@click.option('--delay', type=float, default=1.0, help='Delay between downloads in seconds (rate limiting)')
@click.option('--quiet', is_flag=True, help='Suppress progress output')
def inat_audio(
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
    """
    Download audio observations from iNaturalist.

    Downloads audio files from iNaturalist observations matching the specified
    filters. Creates a metadata.csv file with observation details including
    species, location, observer, and licensing information.

    Examples:

        Download bird sounds from the US:
        bioamla inat-audio ./birds --taxon-id 3 --place-id 1

        Download frog sounds with specific license:
        bioamla inat-audio ./frogs --taxon-name Anura --sound-license cc-by

        Download from a CSV file of taxon IDs:
        bioamla inat-audio ./sounds --taxon-csv taxa.csv --obs-per-taxon 10

        Download 50 observations per taxon without subdirectories:
        bioamla inat-audio ./sounds --obs-per-taxon 50 --no-organize-by-taxon
    """
    from bioamla.core.inat import download_inat_audio

    # Parse comma-separated taxon IDs into a list of integers
    taxon_ids_list = None
    if taxon_ids:
        taxon_ids_list = [int(tid.strip()) for tid in taxon_ids.split(",")]

    # Parse comma-separated file extensions into a list
    extensions_list = None
    if file_extensions:
        extensions_list = [ext.strip() for ext in file_extensions.split(",")]

    stats = download_inat_audio(
        output_dir=output_dir,
        taxon_ids=taxon_ids_list,
        taxon_csv=taxon_csv,
        taxon_name=taxon_name,
        place_id=place_id,
        user_id=user_id,
        project_id=project_id,
        quality_grade=quality_grade,
        sound_license=sound_license,
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


@cli.command()
@click.option('--place-id', type=int, default=None, help='Filter by place ID (e.g., 1 for United States)')
@click.option('--project-id', default=None, help='Filter by iNaturalist project ID or slug')
@click.option('--taxon-id', type=int, default=None, help='Filter by parent taxon ID (e.g., 20979 for Amphibia)')
@click.option('--quality-grade', default='research', help='Quality grade: research, needs_id, or casual')
@click.option('--output', '-o', default=None, help='Output file path for CSV (optional)')
@click.option('--quiet', is_flag=True, help='Suppress progress output')
def inat_taxa_search(
    place_id: int,
    project_id: str,
    taxon_id: int,
    quality_grade: str,
    output: str,
    quiet: bool
):
    """
    Search for taxa with observations in a place or project.

    Uses the iNaturalist species_counts API for efficient retrieval.

    Examples:

        Find amphibian taxa in a project:
        bioamla inat-taxa-search --project-id appalachia-bioacoustics --taxon-id 20979

        Find all bird taxa in a place:
        bioamla inat-taxa-search --place-id 1 --taxon-id 3

        Export results to CSV:
        bioamla inat-taxa-search --project-id my-project -o taxa.csv
    """
    from bioamla.core.inat import get_taxa

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


@cli.command()
@click.argument('project_id')
@click.option('--output', '-o', default=None, help='Output file path for JSON (optional)')
@click.option('--quiet', is_flag=True, help='Suppress progress output, print only JSON')
def inat_project_stats(
    project_id: str,
    output: str,
    quiet: bool
):
    """
    Get statistics for an iNaturalist project.

    Fetches project information including observation counts, species counts,
    and observer information.

    Examples:

        Get stats for a project:
        bioamla inat-project-stats appalachia-bioacoustics

        Export stats to JSON:
        bioamla inat-project-stats appalachia-bioacoustics -o stats.json

        Get JSON output only:
        bioamla inat-project-stats appalachia-bioacoustics --quiet
    """
    import json

    from bioamla.core.inat import get_project_stats

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


@cli.command()
@click.argument('output_dir')
@click.argument('dataset_paths', nargs=-1, required=True)
@click.option('--metadata-filename', default='metadata.csv', help='Name of metadata CSV file in each dataset')
@click.option('--overwrite', is_flag=True, help='Overwrite existing files instead of skipping')
@click.option('--no-organize', is_flag=True, help='Preserve original directory structure instead of organizing by category')
@click.option('--target-format', default=None, help='Convert all audio files to this format (wav, mp3, flac, etc.)')
@click.option('--quiet', is_flag=True, help='Suppress progress output')
def merge_datasets(
    output_dir: str,
    dataset_paths: tuple,
    metadata_filename: str,
    overwrite: bool,
    no_organize: bool,
    target_format: str,
    quiet: bool
):
    """
    Merge multiple audio datasets into a single dataset.

    Combines audio files and metadata from multiple dataset directories into
    a single output directory. By default, files are organized into subdirectories
    based on the 'category' field in metadata.

    Examples:

        Merge two datasets:
        bioamla merge-datasets ./merged ./birds_v1 ./birds_v2

        Merge multiple datasets with overwrite:
        bioamla merge-datasets ./merged ./dataset1 ./dataset2 ./dataset3 --overwrite

        Merge preserving original directory structure:
        bioamla merge-datasets ./merged ./data1 ./data2 --no-organize

        Merge and convert all files to WAV:
        bioamla merge-datasets ./merged ./data1 ./data2 --target-format wav

        Merge with custom metadata filename:
        bioamla merge-datasets ./merged ./data1 ./data2 --metadata-filename data.csv
    """
    from bioamla.core.datasets import merge_datasets as do_merge

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


@cli.command()
@click.argument('dataset_path')
@click.argument('target_format')
@click.option('--metadata-filename', default='metadata.csv', help='Name of metadata CSV file')
@click.option('--keep-original', is_flag=True, help='Keep original files after conversion (default: delete)')
@click.option('--quiet', is_flag=True, help='Suppress progress output')
def convert_audio(
    dataset_path: str,
    target_format: str,
    metadata_filename: str,
    keep_original: bool,
    quiet: bool
):
    """
    Convert all audio files in a dataset to a specified format.

    Converts audio files and updates the metadata.csv with new filenames.
    Original files are deleted by default. The attr_note field is updated
    to indicate "modified clip from original source".

    Supported formats: wav, mp3, m4a, aac, flac, ogg, wma

    Examples:

        Convert dataset to WAV:
        bioamla convert-audio ./my_dataset wav

        Convert to MP3 and keep originals:
        bioamla convert-audio ./my_dataset mp3 --keep-original

        Convert with custom metadata filename:
        bioamla convert-audio ./my_dataset flac --metadata-filename data.csv
    """
    from bioamla.core.datasets import convert_filetype

    stats = convert_filetype(
        dataset_path=dataset_path,
        target_format=target_format,
        metadata_filename=metadata_filename,
        keep_original=keep_original,
        verbose=not quiet
    )

    if quiet:
        click.echo(f"Converted {stats['files_converted']} files to {target_format}")


if __name__ == '__main__':
    cli()
