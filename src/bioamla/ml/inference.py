"""
AST Model Inference
===================

High-level inference for Audio Spectrogram Transformer (AST) models:
:class:`ASTInference` covers whole-file prediction (:meth:`ASTInference.predict`,
:meth:`ASTInference.predict_topk`) and segmented prediction
(:meth:`ASTInference.predict_segments`). Directory / CSV batch wrappers live in
:mod:`bioamla.ml.batch`.

PyTorch / transformers / torchaudio ship in the base install but are imported
lazily so this module imports fast. Load / inference failures raise
:class:`~bioamla.exceptions.ModelError`.

Example:
    from bioamla.ml import ASTInference

    inference = ASTInference(model_path="./my_model")
    result = inference.predict("audio.wav")
    segments = inference.predict_segments("audio.wav", clip_length=3, overlap=0)
"""

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Union

from bioamla.exceptions import ModelError

if TYPE_CHECKING:
    import torch

logger = logging.getLogger(__name__)


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
class ASTPredictionResult:
    """Result from a single AST inference (one clip or segment).

    Distinct from :class:`bioamla.ml.base.PredictionResult`; this is the flat,
    inference-oriented shape returned by :class:`ASTInference` and consumed by
    the ``models ast predict`` CLI command.
    """

    filepath: str
    start_time: float
    end_time: float
    predicted_label: str
    confidence: float
    logits: list[float] | None = None
    top_k_labels: list[str] | None = None
    top_k_scores: list[float] | None = None


class ASTInference:
    """
    High-level inference engine for AST models.

    Handles model loading, feature extraction, and prediction for audio
    classification.

    Raises:
        ModelError: If the model cannot be loaded.
    """

    def __init__(
        self,
        model_path: str,
        sample_rate: int = 16000,
        device: Union[str, "torch.device"] | None = None,
    ):
        """
        Initialize the inference engine and load the model.

        Args:
            model_path: Path to the trained AST model or a HuggingFace identifier.
            sample_rate: Target sample rate for audio preprocessing.
            device: Inference device (auto-detected if None).
        """
        torch = _require_torch()
        ASTFeatureExtractor, AutoModelForAudioClassification = _require_transformers()
        from bioamla.ml.device import get_device

        self.model_path = model_path
        self.sample_rate = sample_rate
        self.device = device if device else get_device()
        if isinstance(self.device, str):
            self.device = torch.device(self.device)

        logger.info(f"Loading model from {model_path}")
        try:
            self.model = AutoModelForAudioClassification.from_pretrained(
                model_path, device_map="auto"
            )
            self.feature_extractor = ASTFeatureExtractor.from_pretrained(model_path)
        except Exception as e:
            raise ModelError(f"Failed to load AST model from {model_path}: {e}") from e

        self.id2label = self.model.config.id2label
        self.label2id = self.model.config.label2id
        logger.info(f"Model loaded with {len(self.id2label)} classes")

    def predict(self, audio_path: str, return_logits: bool = False) -> ASTPredictionResult:
        """
        Run inference on a single audio file (treated as one clip).

        Args:
            audio_path: Path to the audio file.
            return_logits: Include raw logits in the result if True.

        Returns:
            An :class:`ASTPredictionResult`.

        Raises:
            ModelError: If loading the audio or running inference fails.
        """
        torch = _require_torch()
        load_waveform_tensor, resample_waveform_tensor, _ = _torchaudio_helpers()

        try:
            waveform, orig_sr = load_waveform_tensor(audio_path)
        except Exception as e:
            raise ModelError(f"Failed to load audio {audio_path}: {e}") from e

        if orig_sr != self.sample_rate:
            waveform = resample_waveform_tensor(waveform, orig_sr, self.sample_rate)

        inputs = self._extract_features(waveform)

        try:
            with torch.inference_mode():
                outputs = self.model(inputs)
        except Exception as e:
            raise ModelError(f"AST inference failed for {audio_path}: {e}") from e

        logits = outputs.logits[0]
        probs = torch.softmax(logits, dim=-1)
        predicted_idx = logits.argmax(-1).item()
        confidence = probs[predicted_idx].item()
        predicted_label = self.id2label[predicted_idx]
        duration = waveform.shape[1] / self.sample_rate

        return ASTPredictionResult(
            filepath=audio_path,
            start_time=0.0,
            end_time=duration,
            predicted_label=predicted_label,
            confidence=confidence,
            logits=logits.cpu().tolist() if return_logits else None,
        )

    def predict_topk(
        self,
        audio_path: str,
        *,
        top_k: int = 5,
        min_confidence: float = 0.0,
    ) -> ASTPredictionResult:
        """
        Run inference on a single file, returning the top-k labels and scores.

        The top-1 label/confidence populate ``predicted_label``/``confidence`` as
        usual; ``top_k_labels`` / ``top_k_scores`` hold the ranked top-k
        predictions whose probability is at least ``min_confidence``.

        Args:
            audio_path: Path to the audio file.
            top_k: Number of top predictions to keep.
            min_confidence: Drop predictions below this probability.

        Returns:
            An :class:`ASTPredictionResult` with top-k fields populated.

        Raises:
            ModelError: If loading the audio or running inference fails.
        """
        torch = _require_torch()
        load_waveform_tensor, resample_waveform_tensor, _ = _torchaudio_helpers()

        try:
            waveform, orig_sr = load_waveform_tensor(audio_path)
        except Exception as e:
            raise ModelError(f"Failed to load audio {audio_path}: {e}") from e

        if orig_sr != self.sample_rate:
            waveform = resample_waveform_tensor(waveform, orig_sr, self.sample_rate)

        inputs = self._extract_features(waveform)

        try:
            with torch.inference_mode():
                outputs = self.model(inputs)
        except Exception as e:
            raise ModelError(f"AST inference failed for {audio_path}: {e}") from e

        logits = outputs.logits[0]
        probs = torch.softmax(logits, dim=-1)
        duration = waveform.shape[1] / self.sample_rate

        k = min(max(top_k, 1), probs.shape[-1])
        top_scores, top_indices = torch.topk(probs, k)
        labels: list[str] = []
        scores: list[float] = []
        for score, idx in zip(top_scores.cpu().tolist(), top_indices.cpu().tolist(), strict=False):
            if score >= min_confidence:
                labels.append(self.id2label[idx])
                scores.append(float(score))

        predicted_idx = int(top_indices[0].item())
        return ASTPredictionResult(
            filepath=audio_path,
            start_time=0.0,
            end_time=duration,
            predicted_label=self.id2label[predicted_idx],
            confidence=float(top_scores[0].item()),
            top_k_labels=labels,
            top_k_scores=scores,
        )

    def predict_segments(
        self, audio_path: str, clip_length: int = 10, overlap: int = 0, return_logits: bool = False
    ) -> list[ASTPredictionResult]:
        """
        Run inference on an audio file split into segments.

        Args:
            audio_path: Path to the audio file.
            clip_length: Duration of each segment in seconds.
            overlap: Overlap between segments in seconds.
            return_logits: Include raw logits in each result if True.

        Returns:
            A list of :class:`ASTPredictionResult`, one per segment.

        Raises:
            ModelError: If loading the audio or running inference fails.
        """
        torch = _require_torch()
        load_waveform_tensor, resample_waveform_tensor, split_waveform_tensor = (
            _torchaudio_helpers()
        )

        try:
            waveform, orig_sr = load_waveform_tensor(audio_path)
        except Exception as e:
            raise ModelError(f"Failed to load audio {audio_path}: {e}") from e

        if orig_sr != self.sample_rate:
            waveform = resample_waveform_tensor(waveform, orig_sr, self.sample_rate)

        segments = split_waveform_tensor(waveform, self.sample_rate, clip_length, overlap)

        results = []
        for segment, start_sample, end_sample in segments:
            inputs = self._extract_features(segment)

            try:
                with torch.inference_mode():
                    outputs = self.model(inputs)
            except Exception as e:
                raise ModelError(f"AST inference failed for {audio_path}: {e}") from e

            logits = outputs.logits[0]
            probs = torch.softmax(logits, dim=-1)
            predicted_idx = logits.argmax(-1).item()
            confidence = probs[predicted_idx].item()
            predicted_label = self.id2label[predicted_idx]

            results.append(
                ASTPredictionResult(
                    filepath=audio_path,
                    start_time=start_sample / self.sample_rate,
                    end_time=end_sample / self.sample_rate,
                    predicted_label=predicted_label,
                    confidence=confidence,
                    logits=logits.cpu().tolist() if return_logits else None,
                )
            )

        return results

    def _extract_features(self, waveform: "torch.Tensor") -> "torch.Tensor":
        """Extract feature-extractor input values from a waveform tensor."""
        waveform_np = waveform.squeeze().numpy()
        inputs = self.feature_extractor(
            waveform_np, sampling_rate=self.sample_rate, padding="max_length", return_tensors="pt"
        )
        return inputs.input_values.to(self.device)
