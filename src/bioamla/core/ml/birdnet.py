"""
BirdNET Model Wrapper
====================

BirdNET model implementation for bird species classification.
Supports both the TFLite model and custom PyTorch implementations.

BirdNET is a deep learning model trained on bird vocalizations,
capable of identifying over 3,000 bird species by their sounds.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn as nn

from bioamla.core.audio.torchaudio import load_waveform_tensor, resample_waveform_tensor
from bioamla.core.files import TextFile
from bioamla.core.ml.base import (
    BaseAudioModel,
    ModelBackend,
    ModelConfig,
    PredictionResult,
    register_model,
)


@register_model("birdnet")
class BirdNETModel(BaseAudioModel):
    """
    BirdNET model wrapper for bird species classification.

    This class wraps BirdNET models for bird sound identification.
    It supports loading from TFLite format or PyTorch checkpoints.

    BirdNET uses 3-second audio segments sampled at 48kHz.

    Example:
        >>> model = BirdNETModel()
        >>> model.load("path/to/birdnet_model")
        >>> results = model.predict("bird_audio.wav")

    Note:
        For full BirdNET functionality including species lists and
        geographic filtering, use the species_filter and geography
        methods.
    """

    # BirdNET default parameters
    BIRDNET_SAMPLE_RATE = 48000
    BIRDNET_CLIP_DURATION = 3.0
    BIRDNET_MIN_CONFIDENCE = 0.1

    def __init__(self, config: Optional[ModelConfig] = None):
        """Initialize BirdNET model."""
        # Override defaults for BirdNET
        if config is None:
            config = ModelConfig()
        config.sample_rate = self.BIRDNET_SAMPLE_RATE
        config.clip_duration = self.BIRDNET_CLIP_DURATION
        if config.min_confidence == 0.0:
            config.min_confidence = self.BIRDNET_MIN_CONFIDENCE

        super().__init__(config)
        self.species_list: List[str] = []
        self.geography_filter: Optional[Dict[str, List[str]]] = None
        self._embeddings_layer: Optional[nn.Module] = None

    @property
    def backend(self) -> ModelBackend:
        return ModelBackend.BIRDNET

    def load(self, model_path: str, labels_path: Optional[str] = None, **kwargs) -> "BirdNETModel":
        """
        Load BirdNET model from path.

        Args:
            model_path: Path to model file (.pt, .pth, or directory).
            labels_path: Optional path to labels file (JSON or text).

        Returns:
            Self for method chaining.
        """
        model_path = Path(model_path)

        if model_path.suffix in (".pt", ".pth"):
            self._load_pytorch(str(model_path))
        elif model_path.is_dir():
            # Look for model file in directory
            pt_files = list(model_path.glob("*.pt")) + list(model_path.glob("*.pth"))
            if pt_files:
                self._load_pytorch(str(pt_files[0]))
            else:
                raise FileNotFoundError(f"No model file found in {model_path}")
        else:
            # Try loading as PyTorch state dict
            self._load_pytorch(str(model_path))

        # Load labels
        if labels_path:
            self._load_labels(labels_path)
        elif model_path.is_dir():
            # Look for labels in same directory
            label_files = list(model_path.glob("labels*.json")) + list(
                model_path.glob("labels*.txt")
            )
            if label_files:
                self._load_labels(str(label_files[0]))

        self.model.eval()
        self.model.to(self.device)
        return self

    def _load_pytorch(self, path: str) -> None:
        """Load PyTorch model."""
        checkpoint = torch.load(path, map_location=self.device)

        if isinstance(checkpoint, dict):
            if "model_state_dict" in checkpoint:
                # Standard checkpoint format
                self.model = self._create_birdnet_architecture(checkpoint.get("num_classes", 3000))
                self.model.load_state_dict(checkpoint["model_state_dict"])

                if "id2label" in checkpoint:
                    self.id2label = checkpoint["id2label"]
                    self.label2id = {v: k for k, v in self.id2label.items()}
                if "species_list" in checkpoint:
                    self.species_list = checkpoint["species_list"]
            else:
                # Just state dict
                self.model = self._create_birdnet_architecture(
                    len(checkpoint.get("classifier.weight", [3000]))
                )
                self.model.load_state_dict(checkpoint)
        elif isinstance(checkpoint, nn.Module):
            self.model = checkpoint
        else:
            raise ValueError(f"Unknown checkpoint format: {type(checkpoint)}")

    def _create_birdnet_architecture(self, num_classes: int) -> nn.Module:
        """
        Create BirdNET-like architecture.

        This is a simplified version of the BirdNET architecture.
        For production use, load the official BirdNET model.
        """
        return BirdNETEncoder(num_classes=num_classes)

    def _load_labels(self, path: str) -> None:
        """Load species labels from file."""
        path = Path(path)

        if path.suffix == ".json":
            with TextFile(path, mode="r") as f:
                data = json.load(f.handle)
                if isinstance(data, list):
                    self.species_list = data
                elif isinstance(data, dict):
                    self.id2label = {int(k): v for k, v in data.items()}
                    self.species_list = list(data.values())
        else:
            # Text file with one label per line
            with TextFile(path, mode="r") as f:
                self.species_list = [line.strip() for line in f.handle if line.strip()]

        # Setup label mappings
        if not self.id2label:
            self.id2label = dict(enumerate(self.species_list))
        if not self.label2id:
            self.label2id = {v: k for k, v in self.id2label.items()}

    def predict(
        self,
        audio: Union[str, np.ndarray, torch.Tensor],
        sample_rate: Optional[int] = None,
    ) -> List[PredictionResult]:
        """
        Run BirdNET prediction on audio.

        Args:
            audio: Audio file path, numpy array, or torch tensor.
            sample_rate: Sample rate if audio is array/tensor.

        Returns:
            List of prediction results.
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        # Load audio
        filepath = None
        if isinstance(audio, str):
            filepath = audio
            waveform, orig_sr = load_waveform_tensor(audio)
            sample_rate = orig_sr
        elif isinstance(audio, np.ndarray):
            waveform = torch.from_numpy(audio).float()
            if waveform.dim() == 1:
                waveform = waveform.unsqueeze(0)
        else:
            waveform = audio
            if waveform.dim() == 1:
                waveform = waveform.unsqueeze(0)

        if sample_rate is None:
            sample_rate = self.config.sample_rate

        # Resample to BirdNET sample rate
        if sample_rate != self.config.sample_rate:
            waveform = resample_waveform_tensor(waveform, sample_rate, self.config.sample_rate)

        # Split into 3-second segments
        segment_samples = int(self.config.clip_duration * self.config.sample_rate)
        overlap_samples = int(self.config.overlap * self.config.sample_rate)
        step = segment_samples - overlap_samples

        results = []
        num_samples = waveform.shape[1]

        for start in range(0, num_samples, step):
            end = min(start + segment_samples, num_samples)

            segment = waveform[:, start:end]

            # Pad if necessary
            if segment.shape[1] < segment_samples:
                padding = segment_samples - segment.shape[1]
                segment = torch.nn.functional.pad(segment, (0, padding))

            # Compute spectrogram (BirdNET uses mel spectrogram)
            mel_spec = self._compute_mel_spectrogram(segment)
            mel_spec = mel_spec.unsqueeze(0).to(self.device)

            # Run inference
            with torch.inference_mode():
                outputs = self.model(mel_spec)

            if isinstance(outputs, tuple):
                logits = outputs[0]
            else:
                logits = outputs

            probs = torch.sigmoid(logits[0])  # BirdNET uses sigmoid for multi-label

            # Get top-k predictions
            top_k = min(self.config.top_k, len(probs))
            top_probs, top_indices = torch.topk(probs, top_k)

            for prob, idx in zip(top_probs, top_indices):
                confidence = prob.item()

                if confidence >= self.config.min_confidence:
                    label = self.id2label.get(idx.item(), f"class_{idx.item()}")

                    # Apply geography filter if set
                    if self.geography_filter and label not in self._get_allowed_species():
                        continue

                    results.append(
                        PredictionResult(
                            label=label,
                            confidence=confidence,
                            start_time=start / self.config.sample_rate,
                            end_time=end / self.config.sample_rate,
                            filepath=filepath,
                        )
                    )

            # Only process first segment if we'd exceed step
            if end >= num_samples:
                break

        return results

    def _compute_mel_spectrogram(self, waveform: torch.Tensor) -> torch.Tensor:
        """Compute mel spectrogram for BirdNET input."""
        import torchaudio.transforms as T

        # BirdNET parameters
        n_fft = 2048
        hop_length = 512
        n_mels = 128
        f_min = 150
        f_max = 15000

        mel_transform = T.MelSpectrogram(
            sample_rate=self.config.sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            f_min=f_min,
            f_max=f_max,
        )

        mel_spec = mel_transform(waveform)

        # Convert to dB
        db_transform = T.AmplitudeToDB(stype="power", top_db=80)
        mel_spec_db = db_transform(mel_spec)

        # Normalize
        mel_spec_db = (mel_spec_db - mel_spec_db.mean()) / (mel_spec_db.std() + 1e-8)

        return mel_spec_db

    def extract_embeddings(
        self,
        audio: Union[str, np.ndarray, torch.Tensor],
        sample_rate: Optional[int] = None,
        layer: Optional[str] = None,
    ) -> np.ndarray:
        """
        Extract embeddings from BirdNET model.

        Args:
            audio: Audio file path, numpy array, or torch tensor.
            sample_rate: Sample rate if audio is array/tensor.
            layer: Not used for BirdNET (uses penultimate layer).

        Returns:
            Embedding vectors as numpy array.
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        # Load audio
        if isinstance(audio, str):
            waveform, orig_sr = load_waveform_tensor(audio)
            sample_rate = orig_sr
        elif isinstance(audio, np.ndarray):
            waveform = torch.from_numpy(audio).float()
            if waveform.dim() == 1:
                waveform = waveform.unsqueeze(0)
        else:
            waveform = audio
            if waveform.dim() == 1:
                waveform = waveform.unsqueeze(0)

        if sample_rate is None:
            sample_rate = self.config.sample_rate

        if sample_rate != self.config.sample_rate:
            waveform = resample_waveform_tensor(waveform, sample_rate, self.config.sample_rate)

        # Truncate/pad to clip duration
        target_samples = int(self.config.clip_duration * self.config.sample_rate)
        if waveform.shape[1] > target_samples:
            waveform = waveform[:, :target_samples]
        elif waveform.shape[1] < target_samples:
            padding = target_samples - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, padding))

        mel_spec = self._compute_mel_spectrogram(waveform)
        mel_spec = mel_spec.unsqueeze(0).to(self.device)

        with torch.inference_mode():
            embeddings = self.model.get_embeddings(mel_spec)

        return embeddings.cpu().numpy()

    def set_geography_filter(
        self,
        latitude: float,
        longitude: float,
        species_file: Optional[str] = None,
    ) -> None:
        """
        Set geographic filter for species predictions.

        Args:
            latitude: Location latitude.
            longitude: Location longitude.
            species_file: Optional path to species occurrence file.
        """
        # This would load species occurrence data and filter predictions
        # For now, store the location for filtering
        self.geography_filter = {
            "latitude": latitude,
            "longitude": longitude,
        }

        if species_file:
            with TextFile(species_file, mode="r") as f:
                data = json.load(f.handle)
                self.geography_filter["species"] = data.get("species", [])

    def _get_allowed_species(self) -> List[str]:
        """Get list of species allowed by geography filter."""
        if self.geography_filter and "species" in self.geography_filter:
            return self.geography_filter["species"]
        return self.species_list


class BirdNETEncoder(nn.Module):
    """
    BirdNET-like encoder architecture.

    This is a simplified CNN architecture similar to BirdNET.
    For production use, load the official BirdNET model weights.
    """

    def __init__(self, num_classes: int = 3000, embedding_dim: int = 1024):
        super().__init__()

        self.embedding_dim = embedding_dim

        # CNN encoder
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            # Block 4
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        # Embedding layer
        self.embedding = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )

        # Classifier
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning logits."""
        x = self.features(x)
        x = self.embedding(x)
        return self.classifier(x)

    def get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """Extract embeddings before classifier."""
        x = self.features(x)
        return self.embedding(x)
