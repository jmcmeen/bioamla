import click
from typing import Dict


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
    from novus_pytils.files import download_file
    import os
    
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
def version():
    """
    Display the current version of the bioamla package.
    
    This command retrieves and displays the version information
    for the installed bioamla package.
    """
    from bioamla.core.diagnostics import get_bioamla_version
    click.echo(f"bioamla v{get_bioamla_version()}")  
     
@cli.command()
@click.argument('filepath')
def ast(filepath: str):
    """
    Create a new AST project directory with configuration templates.
    
    Creates a new directory at the specified path and copies default AST
    configuration files (YAML templates) into it. These configuration files
    can be customized for specific training and inference tasks.
    
    Args:
        filepath (str): Path where the new AST project directory should be created.
                       Must not already exist as a directory.
    
    Raises:
        ValueError: If the specified directory already exists.
    """
    from novus_pytils.files import directory_exists, create_directory, copy_files
    from novus_pytils.text.yaml import get_yaml_files
    from pathlib import Path
    
    # TODO handle existing directory logic
    
    module_dir = Path(__file__).parent
    config_dir = module_dir.joinpath("config")

    if directory_exists(filepath):
        raise ValueError("Existing directory")

    create_directory(filepath)
    config_files = get_yaml_files(str(config_dir))
    
    copy_files(config_files, filepath)

    click.echo(f"AST project created at {filepath}")
    
@cli.command()
@click.argument('config_filepath')
def ast_finetune(config_filepath: str):
    """
    Fine-tune an Audio Spectrogram Transformer (AST) model using a YAML configuration.
    
    Performs complete fine-tuning workflow including data loading, preprocessing,
    augmentation, model configuration, training, and evaluation. The process includes
    automatic dataset normalization calculation and comprehensive metrics tracking.
    
    Args:
        config_filepath (str): Path to the YAML configuration file containing
                             training parameters, dataset information, and model settings.
    
    The function performs the following operations:
    1. Loads dataset from HuggingFace Hub
    2. Prepares class labels and mappings
    3. Configures audio preprocessing and augmentations
    4. Calculates dataset normalization parameters
    5. Sets up model configuration and training arguments
    6. Trains the model with evaluation metrics
    7. Saves the fine-tuned model
    """
    from datasets import Audio, ClassLabel, load_dataset, Dataset, DatasetDict
    from transformers import ASTFeatureExtractor, ASTConfig, ASTForAudioClassification, TrainingArguments, Trainer
    from audiomentations import Compose, AddGaussianSNR, GainTransition, Gain, ClippingDistortion, TimeStretch, PitchShift
    import torch
    import evaluate
    import numpy as np
    from novus_pytils.files import create_directory
    from novus_pytils.text.yaml import load_yaml
    train_args = load_yaml(config_filepath)
    
    # Load a pre-existing dataset from the HuggingFace Hub
    dataset = load_dataset(train_args["train_dataset"], split=train_args["split"])

    # get target value - class name mappings
    import pandas as pd
    if isinstance(dataset, Dataset):
        selected_data = dataset.select_columns([train_args["category_id_column"], train_args["category_label_column"]])
        df = pd.DataFrame(selected_data.to_dict())
        unique_indices = np.unique(df[train_args["category_id_column"]], return_index=True)[1]
        class_names = df.iloc[unique_indices][train_args["category_label_column"]].to_list()
    elif isinstance(dataset, DatasetDict):
        # For DatasetDict, use the first available split to get class names
        first_split_name = list(dataset.keys())[0]
        first_split = dataset[first_split_name]
        selected_data = first_split.select_columns([train_args["category_id_column"], train_args["category_label_column"]])
        df = pd.DataFrame(selected_data.to_dict())
        unique_indices = np.unique(df[train_args["category_id_column"]], return_index=True)[1]
        class_names = df.iloc[unique_indices][train_args["category_label_column"]].to_list()
    else:
        raise TypeError("Dataset must be a Dataset or DatasetDict instance")

    # cast target and audio column
    dataset = dataset.cast_column("target", ClassLabel(names=class_names))
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000)) #TODO bad

    # rename the target feature
    dataset = dataset.rename_column("target", "labels")
    if isinstance(dataset, Dataset):
        num_labels = len(np.unique([item for item in dataset["labels"]]))
    elif isinstance(dataset, DatasetDict) and "train" in dataset:
        num_labels = len(np.unique([item for item in dataset["train"]["labels"]]))
    else:
        raise TypeError("Unable to determine number of labels from dataset")

    # Define the pretrained model and instantiate the feature extractor
    pretrained_model = train_args["base_model"]
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
        output_dir=train_args["output_dir"],
        logging_dir=train_args["logging_dir"],
        report_to=train_args["report_to"],
        learning_rate=train_args["learning_rate"],
        push_to_hub=train_args["push_to_hub"],
        num_train_epochs=train_args["num_train_epochs"],
        per_device_train_batch_size=train_args["per_device_train_batch_size"],
        eval_strategy=train_args["eval_strategy"],
        save_strategy=train_args["save_strategy"],
        eval_steps=train_args["eval_steps"],
        save_steps=train_args["save_steps"],
        load_best_model_at_end=train_args["load_best_model_at_end"],
        metric_for_best_model=train_args["metric_for_best_model"],
        logging_strategy=train_args["logging_strategy"],
        logging_steps=train_args["logging_steps"]
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
        args=training_args,  # we use our configured training arguments
        train_dataset=train_data,
        eval_dataset=eval_data,
        compute_metrics=compute_metrics,  # we the metrics function from above
    )

    # start a training
    trainer.train()

    create_directory(train_args["best_model_path"])
    torch.save(model.state_dict(), train_args["best_model_path"] + "/pytorch_model.bin")
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
@click.argument('config_filepath')
def ast_batch_inference(config_filepath: str):
    """
    Run batch AST inference on a directory of WAV files using a YAML configuration.
    
    Loads an AST model and processes all WAV files in the specified directory,
    generating predictions and saving results to a CSV file. Supports resumable
    operations by checking for existing results and skipping already processed files.
    
    Args:
        config_filepath (str): Path to the YAML configuration file containing
                             all necessary parameters for batch inference.
    
    The function performs the following operations:
    1. Loads configuration from YAML file
    2. Discovers WAV files in the target directory
    3. Handles resumable operations (if restart=True in config)
    4. Loads the specified AST model
    5. Runs batch inference with timing information
    6. Saves results to CSV file with predictions for each audio segment
    """
    from novus_pytils.files import get_files_by_extension, file_exists
    from novus_pytils.text.yaml import load_yaml
    from bioamla.core.ast import load_pretrained_ast_model, wave_file_batch_inference
    import torch
    import time
    import pandas as pd
    import os
    print ("Loading config file: " + config_filepath)
    config = load_yaml(config_filepath)

    output_csv = os.path.join(config['directory'], config['output_csv'])
    print("Output csv: " + output_csv)

    wave_files = get_files_by_extension(config["directory"], ['.wav'])

    if(len(wave_files) == 0):
        print("No wave files found in directory: " + config["directory"])
        return
    else:
        print("Found " + str(len(wave_files)) + " wave files in directory: " + config["directory"])

    if config['restart']:
        print("Restart: " + str(config['restart']))

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

    print("Loading model: " + config["model"])
    model = load_pretrained_ast_model(config["model"])
    
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
                              freq=config["resample_freq"], 
                              clip_seconds=config["clip_seconds"], 
                              overlap_seconds=config["overlap_seconds"],
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

if __name__ == '__main__':
    cli()