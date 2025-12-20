"""
Audio Spectrogram Transformer (AST) Model Processing
====================================================

This module provides functionality for using Audio Spectrogram Transformer models
for audio classification tasks. It includes batch processing, segmented inference,
and feature extraction capabilities specifically designed for bioacoustic analysis.

The AST models used here are based on Hugging Face's transformers library and are
particularly effective for environmental sound classification tasks.

Performance optimizations include:
- Cached feature extractor (avoids recreation per segment)
- Batched inference (process multiple segments in one forward pass)
- Half-precision (FP16/BF16) inference support
- torch.compile() for optimized model execution
- Parallel file loading with multiprocessing

"""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import lru_cache
from typing import List, Optional, Tuple

import pandas as pd
import torch
from transformers import ASTFeatureExtractor, AutoModelForAudioClassification

from bioamla.core.audio.torchaudio import (
    load_waveform_tensor,
    resample_waveform_tensor,
    split_waveform_tensor,
)


@dataclass
class InferenceConfig:
    """Configuration for optimized batch inference."""

    batch_size: int = 8
    use_fp16: bool = False
    use_compile: bool = False
    num_workers: int = 1


@lru_cache(maxsize=4)
def get_cached_feature_extractor(model_path: Optional[str] = None) -> ASTFeatureExtractor:
    """
    Get a cached feature extractor instance.

    Uses LRU cache to avoid recreating the feature extractor on every call.
    If loading from a model path fails (e.g., missing preprocessor_config.json),
    falls back to the default ASTFeatureExtractor.

    Args:
        model_path: Optional path to load feature extractor from a specific model.
                   If None or if loading fails, uses the default ASTFeatureExtractor.

    Returns:
        ASTFeatureExtractor: Cached feature extractor instance
    """
    if model_path:
        try:
            return ASTFeatureExtractor.from_pretrained(model_path)
        except OSError:
            # Model doesn't have preprocessor_config.json, use default
            pass
    return ASTFeatureExtractor()


def wave_file_batch_inference(
    wave_files: list,
    model: AutoModelForAudioClassification,
    freq: int,
    segment_duration: int,
    segment_overlap: int,
    output_csv: str,
    config: Optional[InferenceConfig] = None,
    feature_extractor: Optional[ASTFeatureExtractor] = None,
) -> None:
    """
    Perform batch inference on multiple audio files using an AST model.

    This function processes a list of audio files, performs segmented inference
    on each file, and appends the results to a CSV file. Each file is processed
    in segments to handle long audio recordings efficiently.

    Args:
        wave_files (list): List of file paths to audio files to process
        model (AutoModelForAudioClassification): Pre-trained AST model for inference
        freq (int): Target sampling frequency for audio preprocessing
        segment_duration (int): Duration of each audio segment in seconds
        segment_overlap (int): Overlap between consecutive segments in seconds
        output_csv (str): Path to output CSV file for results
        config (InferenceConfig): Optional configuration for performance optimizations
        feature_extractor (ASTFeatureExtractor): Optional pre-loaded feature extractor

    Returns:
        None: Results are written directly to the CSV file

    Note:
        Results are appended to the CSV file, so the file will grow with each
        processed audio file. The CSV format includes: filepath, start, stop, prediction
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
    model: AutoModelForAudioClassification,
    freq: int,
    segment_duration: int,
    segment_overlap: int,
    output_csv: str,
    config: InferenceConfig,
    feature_extractor: ASTFeatureExtractor,
    device: torch.device,
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
    model: AutoModelForAudioClassification,
    freq: int,
    segment_duration: int,
    segment_overlap: int,
    output_csv: str,
    config: InferenceConfig,
    feature_extractor: ASTFeatureExtractor,
    device: torch.device,
) -> None:
    """Parallel batch inference with multiprocessing for file loading."""

    def load_and_preprocess(filepath: str) -> Tuple[str, List[Tuple[torch.Tensor, int, int]]]:
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


def segmented_wave_file_inference(
    filepath: str,
    model: AutoModelForAudioClassification,
    freq: int,
    segment_duration: int,
    segment_overlap: int,
    config: Optional[InferenceConfig] = None,
    feature_extractor: Optional[ASTFeatureExtractor] = None,
    device: Optional[torch.device] = None,
) -> pd.DataFrame:
    """
    Perform inference on a single audio file by processing it in segments.

    This function loads an audio file, splits it into overlapping segments,
    and runs AST model inference on each segment. This approach allows for
    processing of long audio recordings while maintaining temporal resolution.

    Args:
        filepath (str): Path to the audio file to process
        model (AutoModelForAudioClassification): Pre-trained AST model
        freq (int): Target sampling frequency for preprocessing
        segment_duration (int): Duration of each segment in seconds
        segment_overlap (int): Overlap between consecutive segments in seconds
        config (InferenceConfig): Optional configuration for performance optimizations
        feature_extractor (ASTFeatureExtractor): Optional cached feature extractor
        device (torch.device): Optional device for inference

    Returns:
        pd.DataFrame: DataFrame with columns ['filepath', 'start', 'stop', 'prediction']
                     containing the inference results for each segment
    """
    if config is None:
        config = InferenceConfig()

    if feature_extractor is None:
        feature_extractor = get_cached_feature_extractor()

    if device is None:
        device = next(model.parameters()).device

    waveform, orig_freq = load_waveform_tensor(filepath)
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


def _process_segments_sequential(
    filepath: str,
    segments: List[Tuple[torch.Tensor, int, int]],
    model: AutoModelForAudioClassification,
    freq: int,
    feature_extractor: ASTFeatureExtractor,
    device: torch.device,
) -> List[dict]:
    """Process segments one at a time (legacy behavior)."""
    rows = []
    for waveform, start, stop in segments:
        input_values = extract_features(waveform, freq, feature_extractor, device)
        prediction = ast_predict(input_values, model)
        rows.append({"filepath": filepath, "start": start, "stop": stop, "prediction": prediction})
    return rows


def _process_segments_batched(
    filepath: str,
    segments: List[Tuple[torch.Tensor, int, int]],
    model: AutoModelForAudioClassification,
    freq: int,
    config: InferenceConfig,
    feature_extractor: ASTFeatureExtractor,
    device: torch.device,
) -> List[dict]:
    """Process segments in batches for improved GPU utilization."""
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


def wav_ast_inference(wave_path: str, model_path: str, sample_rate: int):
    """
    Perform AST inference on a single audio file.

    This is a convenience function that loads an audio file, preprocesses it,
    loads the specified AST model, and returns a single prediction for the
    entire audio file.

    Args:
        wave_path (str): Path to the audio file to classify
        model_path (str): Path to the pre-trained AST model
        sample_rate (int): Target sampling rate for preprocessing

    Returns:
        str: Predicted class label for the audio file
    """
    waveform, orig_freq = load_waveform_tensor(wave_path)
    waveform = resample_waveform_tensor(waveform, orig_freq, sample_rate)
    feature_extractor = get_cached_feature_extractor(model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_values = extract_features(waveform, sample_rate, feature_extractor, device)
    model = load_pretrained_ast_model(model_path)
    return ast_predict(input_values, model)


def ast_predict(input_values, model: AutoModelForAudioClassification) -> str:
    """
    Perform prediction using an AST model on preprocessed audio features.

    This function runs the actual model inference and returns the predicted
    class label. It uses torch.inference_mode() for optimized inference.

    Args:
        input_values: Preprocessed audio features from the feature extractor
        model (AutoModelForAudioClassification): The AST model to use for prediction

    Returns:
        str: The predicted class label (e.g., species name or sound type)
    """
    with torch.inference_mode():
        outputs = model(input_values)

    predicted_class_idx = outputs.logits.argmax(-1).item()
    return model.config.id2label[predicted_class_idx]


def ast_predict_batch(
    input_values: torch.Tensor, model: AutoModelForAudioClassification
) -> List[str]:
    """
    Perform batched prediction using an AST model.

    Args:
        input_values: Batched preprocessed audio features (B, seq_len, features)
        model (AutoModelForAudioClassification): The AST model to use for prediction

    Returns:
        List[str]: List of predicted class labels for each input in the batch
    """
    with torch.inference_mode():
        outputs = model(input_values)

    predicted_indices = outputs.logits.argmax(dim=-1).cpu().tolist()
    return [model.config.id2label[idx] for idx in predicted_indices]


def extract_features(
    waveform_tensor: torch.Tensor,
    sample_rate: int,
    feature_extractor: Optional[ASTFeatureExtractor] = None,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Extract features from audio waveform for AST model input.

    This function converts a waveform tensor into the feature representation
    expected by AST models using the appropriate feature extractor.

    Args:
        waveform_tensor (torch.Tensor): Audio waveform as a tensor
        sample_rate (int): Sampling rate of the audio
        feature_extractor (ASTFeatureExtractor): Optional cached feature extractor
        device (torch.device): Optional device for the output tensor

    Returns:
        torch.Tensor: Extracted features ready for model input

    Note:
        The function automatically detects and uses CUDA if available,
        otherwise falls back to CPU processing.
    """
    waveform_tensor = waveform_tensor.squeeze().numpy()

    if feature_extractor is None:
        feature_extractor = get_cached_feature_extractor()

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    inputs = feature_extractor(
        waveform_tensor, sampling_rate=sample_rate, padding="max_length", return_tensors="pt"
    ).to(device)

    return inputs.input_values


def load_pretrained_ast_model(
    model_path: str, use_fp16: bool = False, use_compile: bool = False
) -> AutoModelForAudioClassification:
    """
    Load a pre-trained AST model from a given path with optional optimizations.

    This function loads an Audio Spectrogram Transformer model that has been
    pre-trained for audio classification tasks. The model is automatically
    distributed across available devices.

    Args:
        model_path (str): Path to the pre-trained model directory or Hugging Face model identifier
        use_fp16 (bool): If True, convert model to half-precision (FP16) for faster inference
        use_compile (bool): If True, use torch.compile() for optimized execution (PyTorch 2.0+)

    Returns:
        AutoModelForAudioClassification: The loaded AST model ready for inference

    Note:
        The device_map="auto" parameter automatically distributes the model
        across available GPUs if present, otherwise uses CPU.
    """
    from pathlib import Path

    is_local_path = Path(model_path).exists() or model_path.startswith(("./", "../"))

    torch_dtype = torch.float16 if use_fp16 else None

    if is_local_path:
        model = AutoModelForAudioClassification.from_pretrained(
            model_path, device_map="auto", local_files_only=True, torch_dtype=torch_dtype
        )
    else:
        model = AutoModelForAudioClassification.from_pretrained(
            model_path, device_map="auto", torch_dtype=torch_dtype
        )

    if use_compile and hasattr(torch, "compile"):
        model = torch.compile(model)

    return model
