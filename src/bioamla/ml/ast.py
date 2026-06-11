"""
Audio Spectrogram Transformer (AST) — load, predict, feature extraction.
========================================================================

Low-level building blocks for using Audio Spectrogram Transformer models
(HuggingFace ``transformers``) for audio classification: model loading,
single-clip and batched prediction, feature extraction, and whole-file
prediction. Segmented / multi-file inference lives in
:class:`bioamla.ml.inference.ASTInference` and the ``bioamla.ml.batch`` wrappers.

PyTorch / transformers ship in the base install but are imported lazily inside
each function so this module imports fast. Model-load / inference failures raise
:class:`~bioamla.exceptions.ModelError`.

Performance features:
- Cached feature extractor (avoids recreation per call)
- Half-precision (FP16) model loading
- ``torch.compile()`` for optimized model execution
"""

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from bioamla.exceptions import ModelError

if TYPE_CHECKING:
    import torch
    from transformers import ASTFeatureExtractor, AutoModelForAudioClassification


def _require_torch():
    """Import and return the torch module."""
    import torch

    return torch


def _require_transformers():
    """Import and return AST transformers symbols."""
    from transformers import ASTFeatureExtractor, AutoModelForAudioClassification

    return ASTFeatureExtractor, AutoModelForAudioClassification


def _torchaudio_helpers():
    """Lazily import the torchaudio waveform helpers."""
    from bioamla.audio.torchaudio import (
        load_waveform_tensor,
        resample_waveform_tensor,
        split_waveform_tensor,
    )

    return load_waveform_tensor, resample_waveform_tensor, split_waveform_tensor


@dataclass
class InferenceConfig:
    """Configuration for optimized batch inference."""

    batch_size: int = 8
    use_fp16: bool = False
    use_compile: bool = False
    num_workers: int = 1


@lru_cache(maxsize=4)
def get_cached_feature_extractor(model_path: str | None = None) -> "ASTFeatureExtractor":
    """
    Get a cached AST feature extractor instance.

    Uses an LRU cache to avoid recreating the feature extractor on every call.
    If loading from a model path fails (e.g. missing ``preprocessor_config.json``),
    falls back to the default :class:`ASTFeatureExtractor`.

    Args:
        model_path: Optional path to load a feature extractor from a specific
            model. If None or if loading fails, uses the default extractor.

    Returns:
        ASTFeatureExtractor: Cached feature extractor instance.

    """
    ASTFeatureExtractor, _ = _require_transformers()
    if model_path:
        try:
            return ASTFeatureExtractor.from_pretrained(model_path)
        except OSError:
            # Model doesn't have preprocessor_config.json, use default
            pass
    return ASTFeatureExtractor()


def load_pretrained_ast_model(
    model_path: str, use_fp16: bool = False, use_compile: bool = False
) -> "AutoModelForAudioClassification":
    """
    Load a pre-trained AST model from a path or HuggingFace identifier.

    Args:
        model_path: Path to the model directory or HuggingFace model identifier.
        use_fp16: If True, load the model in half precision (FP16).
        use_compile: If True, wrap the model with ``torch.compile()`` (PyTorch 2.0+).

    Returns:
        The loaded AST model, ready for inference.

    Raises:
        ModelError: If the model cannot be loaded.
    """
    torch = _require_torch()
    _, AutoModelForAudioClassification = _require_transformers()

    is_local_path = Path(model_path).exists() or model_path.startswith(("./", "../"))
    torch_dtype = torch.float16 if use_fp16 else None

    try:
        if is_local_path:
            model = AutoModelForAudioClassification.from_pretrained(
                model_path, device_map="auto", local_files_only=True, torch_dtype=torch_dtype
            )
        else:
            model = AutoModelForAudioClassification.from_pretrained(
                model_path, device_map="auto", torch_dtype=torch_dtype
            )
    except Exception as e:
        raise ModelError(f"Failed to load AST model from {model_path}: {e}") from e

    if use_compile and hasattr(torch, "compile"):
        model = torch.compile(model)

    return model


def extract_features(
    waveform_tensor: "torch.Tensor",
    sample_rate: int,
    feature_extractor: Optional["ASTFeatureExtractor"] = None,
    device: Optional["torch.device"] = None,
) -> "torch.Tensor":
    """
    Extract AST input features from an audio waveform tensor.

    Args:
        waveform_tensor: Audio waveform as a tensor.
        sample_rate: Sampling rate of the audio.
        feature_extractor: Optional cached feature extractor.
        device: Optional device for the output tensor.

    Returns:
        Extracted features (``input_values``) ready for model input.

    """
    torch = _require_torch()

    waveform_np = waveform_tensor.squeeze().numpy()

    if feature_extractor is None:
        feature_extractor = get_cached_feature_extractor()

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    inputs = feature_extractor(
        waveform_np, sampling_rate=sample_rate, padding="max_length", return_tensors="pt"
    ).to(device)

    return inputs.input_values


def ast_predict(input_values: "torch.Tensor", model: "AutoModelForAudioClassification") -> str:
    """
    Run an AST model on preprocessed features and return the predicted label.

    Args:
        input_values: Preprocessed audio features from the feature extractor.
        model: The AST model to use for prediction.

    Returns:
        The predicted class label.

    Raises:
        ModelError: If inference fails.
    """
    torch = _require_torch()
    try:
        with torch.inference_mode():
            outputs = model(input_values)
        predicted_class_idx = outputs.logits.argmax(-1).item()
        return model.config.id2label[predicted_class_idx]
    except Exception as e:
        raise ModelError(f"AST prediction failed: {e}") from e


def ast_predict_batch(
    input_values: "torch.Tensor", model: "AutoModelForAudioClassification"
) -> list[str]:
    """
    Run an AST model on a batch of preprocessed features.

    Args:
        input_values: Batched preprocessed audio features.
        model: The AST model to use for prediction.

    Returns:
        Predicted class labels, one per item in the batch.

    Raises:
        ModelError: If inference fails.
    """
    torch = _require_torch()
    try:
        with torch.inference_mode():
            outputs = model(input_values)
        predicted_indices = outputs.logits.argmax(dim=-1).cpu().tolist()
        return [model.config.id2label[idx] for idx in predicted_indices]
    except Exception as e:
        raise ModelError(f"AST batch prediction failed: {e}") from e


def wav_ast_inference(wave_path: str, model_path: str, sample_rate: int) -> str:
    """
    Run AST inference on a single audio file and return one prediction.

    Loads the file, resamples, loads the model, and returns a single predicted
    label for the entire file.

    Args:
        wave_path: Path to the audio file to classify.
        model_path: Path to the pre-trained AST model.
        sample_rate: Target sampling rate for preprocessing.

    Returns:
        The predicted class label.

    Raises:
        ModelError: If loading or inference fails.
    """
    torch = _require_torch()
    load_waveform_tensor, resample_waveform_tensor, _ = _torchaudio_helpers()

    try:
        waveform, orig_freq = load_waveform_tensor(wave_path)
    except Exception as e:
        raise ModelError(f"Failed to load audio {wave_path}: {e}") from e

    waveform = resample_waveform_tensor(waveform, orig_freq, sample_rate)
    feature_extractor = get_cached_feature_extractor(model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_values = extract_features(waveform, sample_rate, feature_extractor, device)
    model = load_pretrained_ast_model(model_path)
    return ast_predict(input_values, model)
