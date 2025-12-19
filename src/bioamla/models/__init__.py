"""
Models Package
==============

Machine learning model inference, training, and embedding extraction.

This package provides the model layer for bioamla, including:
- Model inference (AST, BirdNET, OpenSoundscape)
- Model training and fine-tuning
- Embedding extraction for clustering workflows
- Model evaluation utilities
"""

from bioamla.models.inference import (
    InferenceConfig,
    wave_file_batch_inference,
    segmented_wave_file_inference,
)

from bioamla.models.training import (
    TrainingConfig,
    train_model,
)

from bioamla.models.evaluate import (
    EvaluationResult,
    compute_metrics,
    evaluate_predictions,
)

__all__ = [
    # inference
    "InferenceConfig",
    "wave_file_batch_inference",
    "segmented_wave_file_inference",
    # training
    "TrainingConfig",
    "train_model",
    # evaluate
    "EvaluationResult",
    "compute_metrics",
    "evaluate_predictions",
]
