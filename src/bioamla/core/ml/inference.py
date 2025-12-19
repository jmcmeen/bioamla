"""
AST Model Inference
===================

This module provides functionality for running inference with Audio Spectrogram
Transformer (AST) models. It consolidates inference logic for better modularity
and testability.

Example usage:
    from bioamla.inference import ASTInference, BatchInferenceConfig

    inference = ASTInference(model_path="./my_model")
    result = inference.predict("audio.wav")

    # For batch processing
    config = BatchInferenceConfig(
        model_path="./my_model",
        input_dir="./audio_files",
        output_csv="results.csv"
    )
    results = run_batch_inference(config)
"""

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
from transformers import ASTFeatureExtractor, AutoModelForAudioClassification

from bioamla.core.files import TextFile

from bioamla.core.device import get_device
from bioamla.core.logger import get_logger
from bioamla.core.audio.torchaudio import (
    load_waveform_tensor,
    resample_waveform_tensor,
    split_waveform_tensor,
)

logger = get_logger(__name__)


@dataclass
class BatchInferenceConfig:
    """Configuration for batch inference."""

    model_path: str
    input_dir: str
    output_csv: str
    clip_length: int = 10
    overlap: int = 0
    sample_rate: int = 16000
    recursive: bool = True
    file_extensions: Optional[List[str]] = None
    verbose: bool = True


@dataclass
class PredictionResult:
    """Result from a single prediction."""

    filepath: str
    start_time: float
    end_time: float
    predicted_label: str
    confidence: float
    logits: Optional[List[float]] = None


class ASTInference:
    """
    High-level inference class for AST models.

    This class handles model loading, feature extraction, and prediction
    for audio classification tasks.
    """

    def __init__(
        self,
        model_path: str,
        sample_rate: int = 16000,
        device: Optional[Union[str, torch.device]] = None
    ):
        """
        Initialize the inference engine.

        Args:
            model_path: Path to the trained AST model
            sample_rate: Target sample rate for audio preprocessing
            device: Device to use for inference (auto-detected if None)
        """
        self.model_path = model_path
        self.sample_rate = sample_rate
        self.device = device if device else get_device()

        if isinstance(self.device, str):
            self.device = torch.device(self.device)

        # Load model and feature extractor
        logger.info(f"Loading model from {model_path}")
        self.model = AutoModelForAudioClassification.from_pretrained(
            model_path, device_map="auto"
        )
        self.feature_extractor = ASTFeatureExtractor.from_pretrained(model_path)

        # Get label mapping
        self.id2label = self.model.config.id2label
        self.label2id = self.model.config.label2id

        logger.info(f"Model loaded with {len(self.id2label)} classes")

    def predict(
        self,
        audio_path: str,
        return_logits: bool = False
    ) -> PredictionResult:
        """
        Run inference on a single audio file.

        Args:
            audio_path: Path to the audio file
            return_logits: If True, include raw logits in the result

        Returns:
            PredictionResult with prediction details
        """
        # Load and preprocess audio
        waveform, orig_sr = load_waveform_tensor(audio_path)

        if orig_sr != self.sample_rate:
            waveform = resample_waveform_tensor(waveform, orig_sr, self.sample_rate)

        # Extract features
        inputs = self._extract_features(waveform)

        # Run inference
        with torch.inference_mode():
            outputs = self.model(inputs)

        logits = outputs.logits[0]
        probs = torch.softmax(logits, dim=-1)
        predicted_idx = logits.argmax(-1).item()
        confidence = probs[predicted_idx].item()
        predicted_label = self.id2label[predicted_idx]

        duration = waveform.shape[1] / self.sample_rate

        return PredictionResult(
            filepath=audio_path,
            start_time=0.0,
            end_time=duration,
            predicted_label=predicted_label,
            confidence=confidence,
            logits=logits.cpu().tolist() if return_logits else None
        )

    def predict_segments(
        self,
        audio_path: str,
        clip_length: int = 10,
        overlap: int = 0,
        return_logits: bool = False
    ) -> List[PredictionResult]:
        """
        Run inference on audio file in segments.

        Args:
            audio_path: Path to the audio file
            clip_length: Duration of each segment in seconds
            overlap: Overlap between segments in seconds
            return_logits: If True, include raw logits in results

        Returns:
            List of PredictionResult for each segment
        """
        results = []

        # Load and preprocess audio
        waveform, orig_sr = load_waveform_tensor(audio_path)

        if orig_sr != self.sample_rate:
            waveform = resample_waveform_tensor(waveform, orig_sr, self.sample_rate)

        # Split into segments
        segments = split_waveform_tensor(waveform, self.sample_rate, clip_length, overlap)

        for segment, start_sample, end_sample in segments:
            # Extract features
            inputs = self._extract_features(segment)

            # Run inference
            with torch.inference_mode():
                outputs = self.model(inputs)

            logits = outputs.logits[0]
            probs = torch.softmax(logits, dim=-1)
            predicted_idx = logits.argmax(-1).item()
            confidence = probs[predicted_idx].item()
            predicted_label = self.id2label[predicted_idx]

            start_time = start_sample / self.sample_rate
            end_time = end_sample / self.sample_rate

            results.append(PredictionResult(
                filepath=audio_path,
                start_time=start_time,
                end_time=end_time,
                predicted_label=predicted_label,
                confidence=confidence,
                logits=logits.cpu().tolist() if return_logits else None
            ))

        return results

    def _extract_features(self, waveform: torch.Tensor) -> torch.Tensor:
        """Extract features from waveform tensor."""
        # Convert to numpy and squeeze
        waveform_np = waveform.squeeze().numpy()

        # Extract features
        inputs = self.feature_extractor(
            waveform_np,
            sampling_rate=self.sample_rate,
            padding="max_length",
            return_tensors="pt"
        )

        # Move to device
        return inputs.input_values.to(self.device)


def run_batch_inference(config: BatchInferenceConfig) -> Dict[str, Any]:
    """
    Run batch inference on a directory of audio files.

    Args:
        config: Batch inference configuration

    Returns:
        dict: Summary statistics
    """
    from bioamla.core.utils import get_audio_files

    input_path = Path(config.input_dir)
    output_path = Path(config.output_csv)

    if not input_path.exists():
        raise FileNotFoundError(f"Input directory not found: {config.input_dir}")

    # Get audio files
    extensions = config.file_extensions or [".wav", ".mp3", ".m4a", ".flac", ".ogg"]
    audio_files = get_audio_files(str(input_path), extensions)

    if not audio_files:
        raise ValueError(f"No audio files found in {config.input_dir}")

    if config.verbose:
        print(f"Found {len(audio_files)} audio files")

    # Initialize inference engine
    inference = ASTInference(
        model_path=config.model_path,
        sample_rate=config.sample_rate
    )

    stats = {
        "total_files": len(audio_files),
        "total_segments": 0,
        "files_processed": 0,
        "files_failed": 0,
        "output_csv": str(output_path)
    }

    # Initialize CSV
    fieldnames = ["filepath", "start_time", "end_time", "predicted_label", "confidence"]
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with TextFile(output_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f.handle, fieldnames=fieldnames)
        writer.writeheader()

    # Process files
    for i, audio_file in enumerate(audio_files):
        if config.verbose and (i + 1) % 10 == 0:
            print(f"Processing file {i + 1}/{len(audio_files)}: {audio_file}")

        try:
            results = inference.predict_segments(
                audio_file,
                clip_length=config.clip_length,
                overlap=config.overlap
            )

            # Append to CSV
            with TextFile(output_path, mode="a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f.handle, fieldnames=fieldnames)
                for result in results:
                    writer.writerow({
                        "filepath": result.filepath,
                        "start_time": result.start_time,
                        "end_time": result.end_time,
                        "predicted_label": result.predicted_label,
                        "confidence": result.confidence
                    })

            stats["total_segments"] += len(results)
            stats["files_processed"] += 1

        except Exception as e:
            logger.warning(f"Failed to process {audio_file}: {e}")
            if config.verbose:
                print(f"  Error: {e}")
            stats["files_failed"] += 1

    if config.verbose:
        print("\nBatch inference complete!")
        print(f"  Files processed: {stats['files_processed']}")
        print(f"  Files failed: {stats['files_failed']}")
        print(f"  Total segments: {stats['total_segments']}")
        print(f"  Output: {stats['output_csv']}")

    return stats
