"""
AST Model Training
==================

This module provides functionality for fine-tuning Audio Spectrogram Transformer (AST)
models on custom audio classification datasets. It extracts and consolidates
training logic for better modularity and testability.

Example usage:
    from bioamla.training import ASTTrainer, TrainingConfig

    config = TrainingConfig(
        base_model="MIT/ast-finetuned-audioset-10-10-0.4593",
        train_dataset="bioamla/scp-frogs",
        output_dir="./training_output"
    )

    trainer = ASTTrainer(config)
    trainer.train()
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

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

from bioamla.core.logger import get_logger

logger = get_logger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for AST model training."""

    # Model settings
    base_model: str = "MIT/ast-finetuned-audioset-10-10-0.4593"

    # Dataset settings
    train_dataset: str = "bioamla/scp-frogs"
    split: str = "train"
    category_id_column: str = "target"
    category_label_column: str = "category"

    # Output settings
    output_dir: str = "./training_output"
    push_to_hub: bool = False

    # Training hyperparameters
    learning_rate: float = 5.0e-5
    num_train_epochs: int = 1
    per_device_train_batch_size: int = 1

    # Evaluation settings
    eval_strategy: str = "epoch"
    save_strategy: str = "epoch"
    eval_steps: int = 1
    save_steps: int = 1
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "accuracy"

    # Logging settings
    report_to: str = "tensorboard"
    logging_strategy: str = "steps"
    logging_steps: int = 100


@dataclass
class AugmentationConfig:
    """Configuration for audio augmentations."""

    enabled: bool = True
    probability: float = 0.8

    # SNR augmentation
    min_snr_db: float = 10.0
    max_snr_db: float = 20.0

    # Gain augmentation
    min_gain_db: float = -6.0
    max_gain_db: float = 6.0

    # Clipping distortion
    clipping_probability: float = 0.5
    min_percentile_threshold: int = 0
    max_percentile_threshold: int = 30

    # Time stretch
    min_rate: float = 0.8
    max_rate: float = 1.2

    # Pitch shift
    min_semitones: int = -4
    max_semitones: int = 4


def create_audio_augmentations(config: Optional[AugmentationConfig] = None) -> Compose:
    """
    Create an audio augmentation pipeline.

    Args:
        config: Augmentation configuration. Uses defaults if None.

    Returns:
        Compose: Audiomentations pipeline
    """
    if config is None:
        config = AugmentationConfig()

    return Compose([
        AddGaussianSNR(min_snr_db=config.min_snr_db, max_snr_db=config.max_snr_db),
        Gain(min_gain_db=config.min_gain_db, max_gain_db=config.max_gain_db),
        GainTransition(
            min_gain_db=config.min_gain_db,
            max_gain_db=config.max_gain_db,
            min_duration=0.01,
            max_duration=0.3,
            duration_unit="fraction"
        ),
        ClippingDistortion(
            min_percentile_threshold=config.min_percentile_threshold,
            max_percentile_threshold=config.max_percentile_threshold,
            p=config.clipping_probability
        ),
        TimeStretch(min_rate=config.min_rate, max_rate=config.max_rate),
        PitchShift(min_semitones=config.min_semitones, max_semitones=config.max_semitones),
    ], p=config.probability, shuffle=True)


def compute_metrics(
    eval_pred,
    accuracy_metric,
    precision_metric,
    recall_metric,
    f1_metric,
    average: str = "macro"
) -> Dict[str, float]:
    """
    Compute evaluation metrics for AST model training.

    Args:
        eval_pred: Evaluation prediction object containing predictions and labels
        accuracy_metric: Loaded accuracy metric
        precision_metric: Loaded precision metric
        recall_metric: Loaded recall metric
        f1_metric: Loaded F1 metric
        average: Averaging method for multi-class metrics

    Returns:
        dict: Dictionary containing accuracy, precision, recall, and F1 metrics
    """
    logits = eval_pred.predictions
    predictions = np.argmax(logits, axis=1)

    # Compute metrics
    accuracy_result = accuracy_metric.compute(
        predictions=predictions,
        references=eval_pred.label_ids
    )
    metrics: Dict[str, float] = accuracy_result if accuracy_result else {}

    precision_result = precision_metric.compute(
        predictions=predictions,
        references=eval_pred.label_ids,
        average=average
    )
    if precision_result:
        metrics.update(precision_result)

    recall_result = recall_metric.compute(
        predictions=predictions,
        references=eval_pred.label_ids,
        average=average
    )
    if recall_result:
        metrics.update(recall_result)

    f1_result = f1_metric.compute(
        predictions=predictions,
        references=eval_pred.label_ids,
        average=average
    )
    if f1_result:
        metrics.update(f1_result)

    return metrics


def calculate_dataset_statistics(
    dataset,
    feature_extractor,
    preprocess_fn,
    model_input_name: str
) -> tuple:
    """
    Calculate mean and standard deviation for dataset normalization.

    Args:
        dataset: The training dataset
        feature_extractor: The AST feature extractor
        preprocess_fn: Preprocessing function
        model_input_name: Name of the model input field

    Returns:
        tuple: (mean, std) values for normalization
    """
    mean_values = []
    std_values = []

    dataset.set_transform(preprocess_fn, output_all_columns=False)

    for sample in dataset:
        if isinstance(sample, dict) and model_input_name in sample:
            cur_mean = torch.mean(sample[model_input_name])
            cur_std = torch.std(sample[model_input_name])
            mean_values.append(cur_mean)
            std_values.append(cur_std)

    mean = float(np.mean(mean_values))
    std = float(np.mean(std_values))

    logger.info(f"Calculated normalization: mean={mean:.4f}, std={std:.4f}")
    return mean, std


class ASTTrainer:
    """
    High-level trainer for AST models.

    This class encapsulates the full training workflow including:
    - Dataset loading and preprocessing
    - Feature extraction
    - Model initialization
    - Training loop with evaluation
    - Model saving
    """

    def __init__(
        self,
        config: TrainingConfig,
        augmentation_config: Optional[AugmentationConfig] = None
    ):
        """
        Initialize the trainer.

        Args:
            config: Training configuration
            augmentation_config: Optional augmentation configuration
        """
        self.config = config
        self.augmentation_config = augmentation_config or AugmentationConfig()
        self.audio_augmentations = create_audio_augmentations(self.augmentation_config)

        # These will be set during setup
        self.feature_extractor = None
        self.model = None
        self.dataset = None
        self.trainer = None

    def setup(self):
        """Set up the training environment, dataset, and model."""
        import pandas as pd
        from datasets import Audio, ClassLabel, Dataset, DatasetDict, load_dataset
        from transformers import ASTConfig, ASTFeatureExtractor, ASTForAudioClassification

        logger.info(f"Loading dataset: {self.config.train_dataset}")

        # Load dataset
        dataset = load_dataset(self.config.train_dataset, split=self.config.split)

        # Get target value - class name mappings
        if isinstance(dataset, Dataset):
            selected_data = dataset.select_columns([
                self.config.category_id_column,
                self.config.category_label_column
            ])
            df = pd.DataFrame(selected_data.to_dict())
            unique_indices = np.unique(df[self.config.category_id_column], return_index=True)[1]
            class_names = df.iloc[unique_indices][self.config.category_label_column].to_list()
        elif isinstance(dataset, DatasetDict):
            first_split_name = list(dataset.keys())[0]
            first_split = dataset[first_split_name]
            selected_data = first_split.select_columns([
                self.config.category_id_column,
                self.config.category_label_column
            ])
            df = pd.DataFrame(selected_data.to_dict())
            unique_indices = np.unique(df[self.config.category_id_column], return_index=True)[1]
            class_names = df.iloc[unique_indices][self.config.category_label_column].to_list()
        else:
            raise TypeError("Dataset must be a Dataset or DatasetDict instance")

        # Cast columns
        dataset = dataset.cast_column("target", ClassLabel(names=class_names))
        dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
        dataset = dataset.rename_column("target", "labels")

        # Determine number of labels
        if isinstance(dataset, Dataset):
            num_labels = len(np.unique(list(dataset["labels"])))
        elif isinstance(dataset, DatasetDict) and "train" in dataset:
            num_labels = len(np.unique(list(dataset["train"]["labels"])))
        else:
            raise TypeError("Unable to determine number of labels from dataset")

        # Initialize feature extractor
        self.feature_extractor = ASTFeatureExtractor.from_pretrained(self.config.base_model)
        model_input_name = self.feature_extractor.model_input_names[0]
        sampling_rate = self.feature_extractor.sampling_rate

        # Get label mappings
        if isinstance(dataset, DatasetDict):
            first_split = list(dataset.keys())[0]
            features = dataset[first_split].features
        else:
            features = dataset.features

        if features and "labels" in features:
            label2id = features["labels"]._str2int
        else:
            raise ValueError("Labels feature not found in dataset")

        # Split dataset if needed
        if isinstance(dataset, Dataset):
            dataset = dataset.train_test_split(
                test_size=0.2, shuffle=True, seed=0, stratify_by_column="labels"
            )
        elif isinstance(dataset, DatasetDict) and "test" not in dataset:
            dataset = dataset["train"].train_test_split(
                test_size=0.2, shuffle=True, seed=0, stratify_by_column="labels"
            )

        # Preprocessing functions
        def preprocess_audio(batch):
            wavs = [audio["array"] for audio in batch["input_values"]]
            inputs = self.feature_extractor(
                wavs, sampling_rate=sampling_rate, return_tensors="pt"
            )
            return {model_input_name: inputs.get(model_input_name), "labels": list(batch["labels"])}

        def preprocess_audio_with_transforms(batch):
            wavs = [
                self.audio_augmentations(audio["array"], sample_rate=sampling_rate)
                for audio in batch["input_values"]
            ]
            inputs = self.feature_extractor(
                wavs, sampling_rate=sampling_rate, return_tensors="pt"
            )
            return {model_input_name: inputs.get(model_input_name), "labels": list(batch["labels"])}

        # Cast and rename audio column
        dataset = dataset.cast_column(
            "audio", Audio(sampling_rate=self.feature_extractor.sampling_rate)
        )
        dataset = dataset.rename_column("audio", "input_values")

        # Calculate normalization statistics
        self.feature_extractor.do_normalize = False
        if isinstance(dataset, DatasetDict) and "train" in dataset:
            mean, std = calculate_dataset_statistics(
                dataset["train"],
                self.feature_extractor,
                preprocess_audio,
                model_input_name
            )
            self.feature_extractor.mean = mean
            self.feature_extractor.std = std
        else:
            raise ValueError("Expected DatasetDict with 'train' split")

        self.feature_extractor.do_normalize = True

        # Apply transforms
        if isinstance(dataset, DatasetDict):
            if "train" in dataset:
                dataset["train"].set_transform(preprocess_audio_with_transforms, output_all_columns=False)
            if "test" in dataset:
                dataset["test"].set_transform(preprocess_audio, output_all_columns=False)
        else:
            raise ValueError("Expected DatasetDict for transform application")

        self.dataset = dataset

        # Initialize model
        config = ASTConfig.from_pretrained(self.config.base_model)
        config.num_labels = num_labels
        config.label2id = label2id
        config.id2label = {v: k for k, v in label2id.items()}

        self.model = ASTForAudioClassification.from_pretrained(
            self.config.base_model,
            config=config,
            ignore_mismatched_sizes=True
        )
        self.model.init_weights()

        logger.info(f"Model initialized with {num_labels} labels")

    def train(self) -> Dict[str, Any]:
        """
        Run the training loop.

        Returns:
            dict: Training results
        """
        import evaluate
        from bioamla.core.utils import create_directory
        from transformers import Trainer, TrainingArguments

        if self.model is None:
            self.setup()

        output_dir = Path(self.config.output_dir)
        runs_dir = output_dir / "runs"
        logs_dir = output_dir / "logs"
        best_model_path = output_dir / "best_model"

        # Load metrics
        accuracy = evaluate.load("accuracy")
        recall = evaluate.load("recall")
        precision = evaluate.load("precision")
        f1 = evaluate.load("f1")

        average = "macro" if self.model.config.num_labels > 2 else "binary"

        def metrics_fn(eval_pred):
            return compute_metrics(
                eval_pred, accuracy, precision, recall, f1, average
            )

        # Configure training arguments
        training_args = TrainingArguments(
            output_dir=str(runs_dir),
            logging_dir=str(logs_dir),
            report_to=self.config.report_to,
            learning_rate=self.config.learning_rate,
            push_to_hub=self.config.push_to_hub,
            num_train_epochs=self.config.num_train_epochs,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            eval_strategy=self.config.eval_strategy,
            save_strategy=self.config.save_strategy,
            eval_steps=self.config.eval_steps,
            save_steps=self.config.save_steps,
            load_best_model_at_end=self.config.load_best_model_at_end,
            metric_for_best_model=self.config.metric_for_best_model,
            logging_strategy=self.config.logging_strategy,
            logging_steps=self.config.logging_steps
        )

        # Create trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.dataset.get("train"),
            eval_dataset=self.dataset.get("test"),
            compute_metrics=metrics_fn,
        )

        # Train
        logger.info("Starting training...")
        result = self.trainer.train()

        # Save best model
        create_directory(str(best_model_path))
        self.trainer.save_model(str(best_model_path))
        logger.info(f"Saved best model to {best_model_path}")

        return {
            "training_loss": result.training_loss,
            "metrics": result.metrics,
            "model_path": str(best_model_path)
        }
