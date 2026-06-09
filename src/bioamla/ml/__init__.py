"""
bioamla.ml — machine-learning domain.
=====================================

Audio Spectrogram Transformer (AST) inference, training, and embeddings, on top
of the device / base-model foundations.

PyTorch / torchaudio / transformers ship in the base install but are imported
lazily inside functions/methods so this package imports fast. Load / inference
failures raise :class:`~bioamla.exceptions.ModelError`.

Example:
    >>> from bioamla.ml import ASTInference
    >>> inference = ASTInference(model_path="bioamla/scp-frogs")
    >>> result = inference.predict("audio.wav")
    >>> print(result.predicted_label, result.confidence)
"""

# Imports are grouped logically below but kept import-sorted (ruff/isort):
# device + base are the preserved foundations; ast / ast_model / ast_service /
# inference / embedding / batch are the folded ml-domain modules.
from bioamla.ml.ast import (
    InferenceConfig,
    ast_predict,
    ast_predict_batch,
    extract_features,
    get_cached_feature_extractor,
    load_pretrained_ast_model,
    segmented_wave_file_inference,
    wav_ast_inference,
    wave_file_batch_inference,
)
from bioamla.ml.ast_model import ASTModel

# --- AST service-level operations --------------------------------------------
from bioamla.ml.ast_service import (
    EvaluationResult,
    TrainResult,
    evaluate_directory,
    extract_embeddings_file,
    get_model_info,
    predict_file,
)
from bioamla.ml.base import (
    AudioDataset,
    BaseAudioModel,
    BatchPredictionResult,
    ModelBackend,
    ModelConfig,
    PredictionResult,
    create_dataloader,
    get_model_class,
    list_models,
    register_model,
)

# --- Batch wrappers ----------------------------------------------------------
from bioamla.ml.batch import batch_embed_files, batch_predict_files
from bioamla.ml.device import (
    DeviceContext,
    get_current_device_index,
    get_device,
    get_device_count,
    get_device_info,
    get_device_name,
    get_device_string,
    is_cuda_available,
    move_to_device,
)

# --- Embeddings --------------------------------------------------------------
from bioamla.ml.embedding import (
    BatchEmbeddingResult,
    EmbeddingConfig,
    EmbeddingExtractor,
    EmbeddingResult,
    extract_embeddings,
    extract_embeddings_batch,
    get_ast_model_info,
    load_embeddings,
    save_embeddings,
)

# --- High-level inference ----------------------------------------------------
from bioamla.ml.inference import (
    ASTInference,
    ASTPredictionResult,
    BatchInferenceConfig,
    run_batch_inference,
)

# --- Preprocessing / augmentation --------------------------------------------
from bioamla.ml.preprocessing import AugmentationConfig, BioamlaPreprocessor

# --- Training -----------------------------------------------------------------
from bioamla.ml.training import train_ast

__all__ = [
    # Device
    "get_device",
    "get_device_string",
    "move_to_device",
    "is_cuda_available",
    "get_device_count",
    "get_current_device_index",
    "get_device_name",
    "get_device_info",
    "DeviceContext",
    # Base
    "ModelBackend",
    "ModelConfig",
    "PredictionResult",
    "BatchPredictionResult",
    "BaseAudioModel",
    "AudioDataset",
    "create_dataloader",
    "register_model",
    "get_model_class",
    "list_models",
    # AST core
    "ASTModel",
    "InferenceConfig",
    "ast_predict",
    "ast_predict_batch",
    "extract_features",
    "get_cached_feature_extractor",
    "load_pretrained_ast_model",
    "segmented_wave_file_inference",
    "wave_file_batch_inference",
    "wav_ast_inference",
    # AST service-level
    "predict_file",
    "evaluate_directory",
    "extract_embeddings_file",
    "get_model_info",
    "EvaluationResult",
    "TrainResult",
    "train_ast",
    # Inference
    "ASTInference",
    "ASTPredictionResult",
    "BatchInferenceConfig",
    "run_batch_inference",
    # Embeddings
    "EmbeddingConfig",
    "EmbeddingResult",
    "BatchEmbeddingResult",
    "EmbeddingExtractor",
    "extract_embeddings",
    "extract_embeddings_batch",
    "save_embeddings",
    "load_embeddings",
    "get_ast_model_info",
    # Batch
    "batch_predict_files",
    "batch_embed_files",
    # Preprocessing / augmentation
    "AugmentationConfig",
    "BioamlaPreprocessor",
]
