"""
AST Model Fine-tuning Command
=============================

Command-line tool for fine-tuning Audio Spectrogram Transformer (AST) models on custom datasets.
This utility supports loading datasets from HuggingFace Hub, applying audio augmentations,
and training with comprehensive evaluation metrics.

Usage:
    ast-finetune CONFIG_FILEPATH

Examples:
    ast-finetune ./training_config.yml        # Fine-tune with local config
    ast-finetune /path/to/config.yml          # Fine-tune with absolute config path

Configuration File Format:
    The YAML configuration file should include:
    - train_dataset: HuggingFace dataset identifier
    - split: Dataset split to use for training
    - category_id_column: Column name for category IDs
    - category_label_column: Column name for category labels
    - base_model: Pre-trained model to fine-tune from
    - output_dir: Directory for training outputs
    - logging_dir: Directory for training logs
    - best_model_path: Path to save the best model
    - Training hyperparameters (learning_rate, num_train_epochs, etc.)
"""

import click
from datasets import Audio, ClassLabel, load_dataset
from transformers import ASTFeatureExtractor, ASTConfig, ASTForAudioClassification, TrainingArguments, Trainer
from audiomentations import Compose, AddGaussianSNR, GainTransition, Gain, ClippingDistortion, TimeStretch, PitchShift
import torch
import evaluate
import numpy as np
from novus_pytils.files import create_directory
from novus_pytils.config.yaml import load_yaml

@click.command()
@click.argument('config_filepath')
def main(config_filepath: str):
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
    train_args = load_yaml(config_filepath)
    
    # Load a pre-existing dataset from the HuggingFace Hub
    dataset = load_dataset(train_args["train_dataset"], split=train_args["split"])

    # get target value - class name mappings
    df = dataset.select_columns([train_args["category_id_column"], train_args["category_label_column"]]).to_pandas()
    class_names = df.iloc[np.unique(df[train_args["category_id_column"]], return_index=True)[1]][train_args["category_label_column"]].to_list()

    # cast target and audio column
    dataset = dataset.cast_column("target", ClassLabel(names=class_names))
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000)) #TODO bad

    # rename the target feature
    dataset = dataset.rename_column("target", "labels")
    num_labels = len(np.unique(dataset["labels"]))

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
    label2id = dataset.features["labels"]._str2int  # we add the mapping from INTs to STRINGs

    # split training data
    if "test" not in dataset:
        dataset = dataset.train_test_split(
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
    dataset["train"].set_transform(preprocess_audio, output_all_columns=False)
    for i, (audio_input, labels) in enumerate(dataset["train"]):
        cur_mean = torch.mean(dataset["train"][i][audio_input])
        cur_std = torch.std(dataset["train"][i][audio_input])
        mean.append(cur_mean)
        std.append(cur_std)

    feature_extractor.mean = np.mean(mean)
    feature_extractor.std = np.mean(std)
    feature_extractor.do_normalize = True

    print("Calculated mean and std:", feature_extractor.mean, feature_extractor.std)

    # Apply transforms
    dataset["train"].set_transform(preprocess_audio_with_transforms, output_all_columns=False)
    dataset["test"].set_transform(preprocess_audio, output_all_columns=False)

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
    def compute_metrics(eval_pred):
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
        metrics = accuracy.compute(predictions=predictions, references=eval_pred.label_ids)
        metrics.update(precision.compute(predictions=predictions, references=eval_pred.label_ids, average=AVERAGE))
        metrics.update(recall.compute(predictions=predictions, references=eval_pred.label_ids, average=AVERAGE))
        metrics.update(f1.compute(predictions=predictions, references=eval_pred.label_ids, average=AVERAGE))

        return metrics

    # setup trainer
    trainer = Trainer(
        model=model,
        args=training_args,  # we use our configured training arguments
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        compute_metrics=compute_metrics,  # we the metrics function from above
    )

    # start a training
    trainer.train()

    create_directory(train_args["best_model_path"])
    torch.save(model.state_dict(), train_args["best_model_path"] + "/pytorch_model.bin")
    # model.save_pretrained(model_dir)

if __name__ == "__main__":
    main()