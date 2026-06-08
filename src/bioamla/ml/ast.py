"""
Audio Spectrogram Transformer (AST) — load, predict, segmented inference.
========================================================================

Functions for using Audio Spectrogram Transformer models (HuggingFace
``transformers``) for audio classification: model loading, single-clip
prediction, batched prediction, feature extraction, and segmented / batch
file inference.

PyTorch / transformers / pandas are optional extras (``bioamla[ml]``). They are
imported lazily inside each function so this module imports on a slim install;
calling any function without them raises
:class:`~bioamla.exceptions.DependencyError`. Model-load / inference failures
raise :class:`~bioamla.exceptions.ModelError`.

Performance features:
- Cached feature extractor (avoids recreation per segment)
- Batched inference (process multiple segments in one forward pass)
- Half-precision (FP16/BF16) inference support
- ``torch.compile()`` for optimized model execution
- Parallel file loading with a thread pool
"""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from bioamla.exceptions import DependencyError, InvalidInputError, ModelError

if TYPE_CHECKING:
    import pandas as pd
    import torch
    from transformers import ASTFeatureExtractor, AutoModelForAudioClassification


def _require_torch():
    """Import and return the torch module, or raise DependencyError."""
    try:
        import torch
    except ImportError as e:
        raise DependencyError("AST requires torch — install bioamla[ml]") from e
    return torch


def _require_transformers():
    """Import and return AST transformers symbols, or raise DependencyError."""
    try:
        from transformers import ASTFeatureExtractor, AutoModelForAudioClassification
    except ImportError as e:
        raise DependencyError("AST requires transformers — install bioamla[ml]") from e
    return ASTFeatureExtractor, AutoModelForAudioClassification


def _require_pandas():
    """Import and return the pandas module, or raise DependencyError."""
    try:
        import pandas as pd
    except ImportError as e:
        raise DependencyError("AST inference requires pandas — install bioamla[ml]") from e
    return pd


def _torchaudio_helpers():
    """Lazily import the torchaudio waveform helpers."""
    try:
        from bioamla.audio.torchaudio import (
            load_waveform_tensor,
            resample_waveform_tensor,
            split_waveform_tensor,
        )
    except ImportError as e:
        raise DependencyError("AST inference requires torchaudio — install bioamla[ml]") from e
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

    Raises:
        DependencyError: If transformers is not installed.
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
        DependencyError: If torch / transformers are not installed.
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

    Raises:
        DependencyError: If torch / transformers are not installed.
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


def ast_predict(input_values, model: "AutoModelForAudioClassification") -> str:
    """
    Run an AST model on preprocessed features and return the predicted label.

    Args:
        input_values: Preprocessed audio features from the feature extractor.
        model: The AST model to use for prediction.

    Returns:
        The predicted class label.

    Raises:
        DependencyError: If torch is not installed.
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
        DependencyError: If torch is not installed.
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
        DependencyError: If torch / transformers / torchaudio are not installed.
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


def segmented_wave_file_inference(
    filepath: str,
    model: "AutoModelForAudioClassification",
    freq: int,
    segment_duration: int,
    segment_overlap: int,
    config: InferenceConfig | None = None,
    feature_extractor: Optional["ASTFeatureExtractor"] = None,
    device: Optional["torch.device"] = None,
) -> "pd.DataFrame":
    """
    Run AST inference on a single file by splitting it into segments.

    Loads the file, splits it into (optionally overlapping) segments, and runs
    AST inference on each segment.

    Args:
        filepath: Path to the audio file.
        model: Pre-trained AST model.
        freq: Target sampling frequency for preprocessing.
        segment_duration: Duration of each segment in seconds.
        segment_overlap: Overlap between consecutive segments in seconds.
        config: Optional performance configuration.
        feature_extractor: Optional cached feature extractor.
        device: Optional inference device.

    Returns:
        A DataFrame with columns ``['filepath', 'start', 'stop', 'prediction']``.

    Raises:
        DependencyError: If torch / transformers / torchaudio / pandas are missing.
        ModelError: If loading or inference fails.
    """
    pd = _require_pandas()
    load_waveform_tensor, resample_waveform_tensor, split_waveform_tensor = _torchaudio_helpers()

    if config is None:
        config = InferenceConfig()

    if feature_extractor is None:
        feature_extractor = get_cached_feature_extractor()

    if device is None:
        device = next(model.parameters()).device

    try:
        waveform, orig_freq = load_waveform_tensor(filepath)
    except Exception as e:
        raise ModelError(f"Failed to load audio {filepath}: {e}") from e

    waveform = resample_waveform_tensor(waveform, orig_freq, freq)
    segments = split_waveform_tensor(waveform, freq, segment_duration, segment_overlap)

    if config.batch_size > 1:
        rows = _process_segments_batched(
            filepath, segments, model, freq, config, feature_extractor, device
        )
    else:
        rows = _process_segments_sequential(
            filepath, segments, model, freq, feature_extractor, device
        )

    return pd.DataFrame(rows, columns=["filepath", "start", "stop", "prediction"])


def wave_file_batch_inference(
    wave_files: list,
    model: "AutoModelForAudioClassification",
    freq: int,
    segment_duration: int,
    segment_overlap: int,
    output_csv: str,
    config: InferenceConfig | None = None,
    feature_extractor: Optional["ASTFeatureExtractor"] = None,
) -> None:
    """
    Run segmented AST inference over multiple files, appending results to a CSV.

    Each file is processed in segments; per-segment results are appended to
    ``output_csv`` (columns: ``filepath, start, stop, prediction``). No header
    is written, matching the historical behavior — callers that need a header
    should write one first.

    Args:
        wave_files: Paths to the audio files to process.
        model: Pre-trained AST model.
        freq: Target sampling frequency for preprocessing.
        segment_duration: Duration of each segment in seconds.
        segment_overlap: Overlap between consecutive segments in seconds.
        output_csv: Output CSV path (results are appended).
        config: Optional performance configuration.
        feature_extractor: Optional cached feature extractor.

    Raises:
        DependencyError: If torch / transformers / torchaudio / pandas are missing.
        ModelError: If loading or inference fails.
    """
    if config is None:
        config = InferenceConfig()

    if feature_extractor is None:
        feature_extractor = get_cached_feature_extractor()
    device = next(model.parameters()).device

    if config.num_workers > 1:
        _batch_inference_parallel(
            wave_files,
            model,
            freq,
            segment_duration,
            segment_overlap,
            output_csv,
            config,
            feature_extractor,
            device,
        )
    else:
        _batch_inference_sequential(
            wave_files,
            model,
            freq,
            segment_duration,
            segment_overlap,
            output_csv,
            config,
            feature_extractor,
            device,
        )


def _batch_inference_sequential(
    wave_files: list,
    model: "AutoModelForAudioClassification",
    freq: int,
    segment_duration: int,
    segment_overlap: int,
    output_csv: str,
    config: InferenceConfig,
    feature_extractor: "ASTFeatureExtractor",
    device: "torch.device",
) -> None:
    """Sequential batch inference processing."""
    for filepath in wave_files:
        df = segmented_wave_file_inference(
            filepath,
            model,
            freq,
            segment_duration,
            segment_overlap,
            config=config,
            feature_extractor=feature_extractor,
            device=device,
        )
        df.to_csv(output_csv, mode="a", header=False, index=False)


def _batch_inference_parallel(
    wave_files: list,
    model: "AutoModelForAudioClassification",
    freq: int,
    segment_duration: int,
    segment_overlap: int,
    output_csv: str,
    config: InferenceConfig,
    feature_extractor: "ASTFeatureExtractor",
    device: "torch.device",
) -> None:
    """Parallel batch inference with a thread pool for file loading."""
    pd = _require_pandas()
    load_waveform_tensor, resample_waveform_tensor, split_waveform_tensor = _torchaudio_helpers()

    def load_and_preprocess(filepath: str):
        """Load and preprocess a single file."""
        waveform, orig_freq = load_waveform_tensor(filepath)
        waveform = resample_waveform_tensor(waveform, orig_freq, freq)
        segments = split_waveform_tensor(waveform, freq, segment_duration, segment_overlap)
        return filepath, segments

    with ThreadPoolExecutor(max_workers=config.num_workers) as executor:
        for filepath, segments in executor.map(load_and_preprocess, wave_files):
            if not segments:
                continue

            rows = _process_segments_batched(
                filepath, segments, model, freq, config, feature_extractor, device
            )

            df = pd.DataFrame(rows, columns=["filepath", "start", "stop", "prediction"])
            df.to_csv(output_csv, mode="a", header=False, index=False)


def _process_segments_sequential(
    filepath: str,
    segments: list[tuple["torch.Tensor", int, int]],
    model: "AutoModelForAudioClassification",
    freq: int,
    feature_extractor: "ASTFeatureExtractor",
    device: "torch.device",
) -> list[dict]:
    """Process segments one at a time."""
    rows = []
    for waveform, start, stop in segments:
        input_values = extract_features(waveform, freq, feature_extractor, device)
        prediction = ast_predict(input_values, model)
        rows.append({"filepath": filepath, "start": start, "stop": stop, "prediction": prediction})
    return rows


def _process_segments_batched(
    filepath: str,
    segments: list[tuple["torch.Tensor", int, int]],
    model: "AutoModelForAudioClassification",
    freq: int,
    config: InferenceConfig,
    feature_extractor: "ASTFeatureExtractor",
    device: "torch.device",
) -> list[dict]:
    """Process segments in batches for improved GPU utilization."""
    # Validate arguments before requiring the heavy optional dependency, so bad
    # input is reported even on a slim (no-torch) install.
    if config.batch_size <= 0:
        raise InvalidInputError(f"batch_size must be positive, got {config.batch_size}")

    torch = _require_torch()

    rows = []

    for batch_start in range(0, len(segments), config.batch_size):
        batch_segments = segments[batch_start : batch_start + config.batch_size]

        batch_inputs = []
        batch_metadata = []

        for waveform, start, stop in batch_segments:
            waveform_np = waveform.squeeze().numpy()
            inputs = feature_extractor(
                waveform_np, sampling_rate=freq, padding="max_length", return_tensors="pt"
            )
            batch_inputs.append(inputs.input_values)
            batch_metadata.append((start, stop))

        stacked_inputs = torch.cat(batch_inputs, dim=0).to(device)

        if config.use_fp16 and device.type == "cuda":
            stacked_inputs = stacked_inputs.half()

        with torch.inference_mode():
            outputs = model(stacked_inputs)

        predicted_indices = outputs.logits.argmax(dim=-1).cpu().tolist()

        for idx, (start, stop) in enumerate(batch_metadata):
            prediction = model.config.id2label[predicted_indices[idx]]
            rows.append(
                {"filepath": filepath, "start": start, "stop": stop, "prediction": prediction}
            )

    return rows
