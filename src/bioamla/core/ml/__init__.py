"""
ML Models Package
=================

This package provides a unified interface for various audio classification
models used in bioacoustics analysis.

Supported Models:
- AST (Audio Spectrogram Transformer)
- BirdNET (bird species classification)
- OpenSoundscape (ResNet-based CNNs)
- Custom CNN (transfer learning)

Example:
    >>> from bioamla.models import load_model, list_models
    >>>
    >>> # List available model types
    >>> print(list_models())
    ['ast', 'birdnet', 'opensoundscape']
    >>>
    >>> # Load an AST model
    >>> model = load_model("ast", "MIT/ast-finetuned-audioset-10-10-0.4593")
    >>> results = model.predict("audio.wav")
    >>>
    >>> # Extract embeddings
    >>> embeddings = model.extract_embeddings("audio.wav")
"""

# API response models (safe to import, no circular deps)
from bioamla.core.ml.responses import (
    AudioClassificationResponse,
    Base64AudioRequest,
    ErrorResponse,
)
from bioamla.core.ml.responses import PredictionResult as APIPredictionResult

# Config (safe, no circular deps)
from bioamla.core.ml.config import DefaultConfig


def _ensure_models_registered():
    """Import model modules to trigger @register_model decorators."""
    # These imports register the models via @register_model decorator
    from bioamla.core.ml import ast_model  # noqa: F401
    from bioamla.core.ml import birdnet  # noqa: F401
    from bioamla.core.ml import opensoundscape  # noqa: F401


def __getattr__(name):
    """Lazy import to avoid circular dependencies."""
    # Base classes and utilities
    if name in (
        "AudioDataset", "BaseAudioModel", "BatchPredictionResult",
        "ModelBackend", "ModelConfig", "PredictionResult",
        "create_dataloader", "register_model"
    ):
        from bioamla.core.ml import base
        return getattr(base, name)

    # These functions need models registered first
    if name == "list_models":
        _ensure_models_registered()
        from bioamla.core.ml import base
        return base.list_models

    if name == "get_model_class":
        _ensure_models_registered()
        from bioamla.core.ml import base
        return base.get_model_class

    # Model implementations
    if name == "ASTModel":
        from bioamla.core.ml.ast_model import ASTModel
        return ASTModel
    if name in ("BirdNETModel", "BirdNETEncoder"):
        from bioamla.core.ml import birdnet
        return getattr(birdnet, name)
    if name in ("OpenSoundscapeModel", "SpectrogramCNN"):
        from bioamla.core.ml import opensoundscape
        return getattr(opensoundscape, name)

    # Training utilities
    if name in ("ModelTrainer", "SpectrogramDataset", "TrainingConfig", "TrainingMetrics", "train_model"):
        from bioamla.core.ml import trainer
        return getattr(trainer, name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    # Base classes
    "BaseAudioModel",
    "ModelBackend",
    "ModelConfig",
    "PredictionResult",
    "BatchPredictionResult",
    "AudioDataset",
    # Model implementations
    "ASTModel",
    "BirdNETModel",
    "BirdNETEncoder",
    "OpenSoundscapeModel",
    "SpectrogramCNN",
    # Training
    "ModelTrainer",
    "TrainingConfig",
    "TrainingMetrics",
    "SpectrogramDataset",
    "train_model",
    # Registry functions
    "register_model",
    "get_model_class",
    "list_models",
    "create_dataloader",
    "load_model",
    # API responses
    "APIPredictionResult",
    "AudioClassificationResponse",
    "Base64AudioRequest",
    "ErrorResponse",
]


def load_model(
    model_type: str,
    model_path: str,
    config: "ModelConfig" = None,
    **kwargs
) -> "BaseAudioModel":
    """
    Load a model by type and path.

    This is the main entry point for loading models in bioamla.

    Args:
        model_type: Type of model ("ast", "birdnet", "opensoundscape").
        model_path: Path to model file or HuggingFace identifier.
        config: Optional model configuration.
        **kwargs: Additional arguments passed to model.load().

    Returns:
        Loaded model instance.

    Example:
        >>> model = load_model("ast", "MIT/ast-finetuned-audioset-10-10-0.4593")
        >>> results = model.predict("audio.wav")
    """
    _ensure_models_registered()
    from bioamla.ml.base import get_model_class
    model_class = get_model_class(model_type)
    model = model_class(config)
    model.load(model_path, **kwargs)
    return model


def predict_file(
    filepath: str,
    model_type: str = "ast",
    model_path: str = "MIT/ast-finetuned-audioset-10-10-0.4593",
    min_confidence: float = 0.0,
    top_k: int = 1,
    **kwargs
) -> list:
    """
    Quick prediction on a single file.

    Convenience function for one-off predictions without
    manually loading the model.

    Args:
        filepath: Path to audio file.
        model_type: Type of model to use.
        model_path: Path to model.
        min_confidence: Minimum confidence threshold.
        top_k: Number of top predictions per segment.
        **kwargs: Additional model arguments.

    Returns:
        List of prediction results.
    """
    from bioamla.ml.base import ModelConfig
    config = ModelConfig(
        min_confidence=min_confidence,
        top_k=top_k,
    )

    model = load_model(model_type, model_path, config, **kwargs)
    return model.predict(filepath)


def extract_embeddings(
    filepath: str,
    model_type: str = "ast",
    model_path: str = "MIT/ast-finetuned-audioset-10-10-0.4593",
    layer: str = None,
    **kwargs
):
    """
    Extract embeddings from an audio file.

    Args:
        filepath: Path to audio file.
        model_type: Type of model to use.
        model_path: Path to model.
        layer: Optional layer to extract from.
        **kwargs: Additional model arguments.

    Returns:
        Embedding vectors as numpy array.
    """
    model = load_model(model_type, model_path, **kwargs)
    return model.extract_embeddings(filepath, layer=layer)
