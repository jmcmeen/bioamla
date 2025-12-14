"""
Audio Spectrogram Transformer (AST) Model Processing
====================================================

This module provides functionality for using Audio Spectrogram Transformer models
for audio classification tasks. It includes batch processing, segmented inference,
and feature extraction capabilities specifically designed for bioacoustic analysis.

The AST models used here are based on Hugging Face's transformers library and are
particularly effective for environmental sound classification tasks.

"""

import pandas as pd
import torch
from transformers import ASTFeatureExtractor, AutoModelForAudioClassification

from bioamla.core.torchaudio import (
    load_waveform_tensor,
    resample_waveform_tensor,
    split_waveform_tensor,
)


def wave_file_batch_inference(wave_files: list, model: AutoModelForAudioClassification,
                              freq: int, clip_seconds: int, overlap_seconds: int,
                              output_csv: str) -> None:
    """
    Perform batch inference on multiple audio files using an AST model.

    This function processes a list of audio files, performs segmented inference
    on each file, and appends the results to a CSV file. Each file is processed
    in segments to handle long audio recordings efficiently.

    Args:
        wave_files (list): List of file paths to audio files to process
        model (AutoModelForAudioClassification): Pre-trained AST model for inference
        freq (int): Target sampling frequency for audio preprocessing
        clip_seconds (int): Duration of each audio segment in seconds
        overlap_seconds (int): Overlap between consecutive segments in seconds
        output_csv (str): Path to output CSV file for results

    Returns:
        None: Results are written directly to the CSV file

    Note:
        Results are appended to the CSV file, so the file will grow with each
        processed audio file. The CSV format includes: filepath, start, stop, prediction
    """
    for filepath in wave_files:
        df = segmented_wave_file_inference(filepath, model, freq, clip_seconds, overlap_seconds)
        # results = pd.concat([results, df]) # TODO this should just return a dict. pandaing should go somewhere else
        df.to_csv(output_csv, mode='a', header=False, index=False)

def segmented_wave_file_inference(filepath: str, model: AutoModelForAudioClassification,
                                  freq: int, clip_seconds: int, overlap_seconds: int) -> pd.DataFrame:
    """
    Perform inference on a single audio file by processing it in segments.

    This function loads an audio file, splits it into overlapping segments,
    and runs AST model inference on each segment. This approach allows for
    processing of long audio recordings while maintaining temporal resolution.

    Args:
        filepath (str): Path to the audio file to process
        model (AutoModelForAudioClassification): Pre-trained AST model
        freq (int): Target sampling frequency for preprocessing
        clip_seconds (int): Duration of each segment in seconds
        overlap_seconds (int): Overlap between consecutive segments in seconds

    Returns:
        pd.DataFrame: DataFrame with columns ['filepath', 'start', 'stop', 'prediction']
                     containing the inference results for each segment
    """
    rows = []
    waveform, orig_freq = load_waveform_tensor(filepath)
    waveform = resample_waveform_tensor(waveform, orig_freq, freq)
    waveforms = split_waveform_tensor(waveform, freq, clip_seconds, overlap_seconds)

    for waveform in waveforms:
        input_values = extract_features(waveform[0], freq)
        prediction = ast_predict(input_values, model)
        rows.append({'filepath': filepath, 'start': waveform[1], 'stop': waveform[2], 'prediction': prediction})
    return pd.DataFrame(rows, columns=['filepath', 'start', 'stop', 'prediction'])

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
    input_values = extract_features(waveform, sample_rate)
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

def extract_features(waveform_tensor: torch.Tensor, sample_rate: int):
    """
    Extract features from audio waveform for AST model input.

    This function converts a waveform tensor into the feature representation
    expected by AST models using the appropriate feature extractor.

    Args:
        waveform_tensor (torch.Tensor): Audio waveform as a tensor
        sample_rate (int): Sampling rate of the audio

    Returns:
        torch.Tensor: Extracted features ready for model input

    Note:
        The function automatically detects and uses CUDA if available,
        otherwise falls back to CPU processing.
    """
    waveform_tensor = waveform_tensor.squeeze().numpy()
    feature_extractor = ASTFeatureExtractor()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    inputs = feature_extractor(waveform_tensor, sampling_rate=sample_rate, padding="max_length", return_tensors="pt").to(device)
    return inputs.input_values

def load_pretrained_ast_model(model_path: str) -> AutoModelForAudioClassification:
    """
    Load a pre-trained AST model from a given path.

    This function loads an Audio Spectrogram Transformer model that has been
    pre-trained for audio classification tasks. The model is automatically
    distributed across available devices.

    Args:
        model_path (str): Path to the pre-trained model directory or Hugging Face model identifier

    Returns:
        AutoModelForAudioClassification: The loaded AST model ready for inference

    Note:
        The device_map="auto" parameter automatically distributes the model
        across available GPUs if present, otherwise uses CPU.
    """
    import os

    # Check if model_path looks like a local path (contains path separators or starts with . or /)
    is_local_path = os.path.sep in model_path or model_path.startswith(('.', '/'))

    if is_local_path:
        return AutoModelForAudioClassification.from_pretrained(model_path, device_map="auto", local_files_only=True)
    else:
        return AutoModelForAudioClassification.from_pretrained(model_path, device_map="auto")
