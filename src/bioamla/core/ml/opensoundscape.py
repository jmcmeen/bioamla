"""
OpenSoundscape CNN Wrapper
==========================

CNN model implementations inspired by OpenSoundscape library.
Provides ResNet18 and ResNet50 architectures for audio classification.

OpenSoundscape-style models use spectrogram images as input to standard
image classification networks (transfer learning from ImageNet).
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models

from bioamla.core.ml.base import (
    BaseAudioModel,
    ModelBackend,
    ModelConfig,
    PredictionResult,
    register_model,
)
from bioamla.core.audio.torchaudio import load_waveform_tensor, resample_waveform_tensor


class SpectrogramCNN(nn.Module):
    """
    CNN wrapper for spectrogram classification.

    Wraps a pretrained image classification network (ResNet18/50)
    for audio spectrogram classification.
    """

    SUPPORTED_ARCHITECTURES = ["resnet18", "resnet50", "efficientnet_b0"]

    def __init__(
        self,
        num_classes: int,
        architecture: str = "resnet18",
        pretrained: bool = True,
        embedding_dim: Optional[int] = None,
    ):
        """
        Initialize the spectrogram CNN.

        Args:
            num_classes: Number of output classes.
            architecture: Base architecture ("resnet18", "resnet50", "efficientnet_b0").
            pretrained: Use ImageNet pretrained weights.
            embedding_dim: Optional embedding dimension (uses architecture default if None).
        """
        super().__init__()

        self.architecture = architecture
        self.num_classes = num_classes

        # Create backbone
        if architecture == "resnet18":
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            self.backbone = models.resnet18(weights=weights)
            self._embedding_dim = 512
        elif architecture == "resnet50":
            weights = models.ResNet50_Weights.DEFAULT if pretrained else None
            self.backbone = models.resnet50(weights=weights)
            self._embedding_dim = 2048
        elif architecture == "efficientnet_b0":
            weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
            self.backbone = models.efficientnet_b0(weights=weights)
            self._embedding_dim = 1280
        else:
            raise ValueError(f"Unsupported architecture: {architecture}")

        self.embedding_dim = embedding_dim or self._embedding_dim

        # Modify first conv layer to accept single channel (spectrogram)
        self._modify_input_layer()

        # Replace classifier
        self._replace_classifier()

    def _modify_input_layer(self) -> None:
        """Modify first conv layer for single-channel input."""
        if self.architecture.startswith("resnet"):
            # ResNet has 3-channel input, average weights for 1-channel
            old_conv = self.backbone.conv1
            self.backbone.conv1 = nn.Conv2d(
                1, old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=old_conv.bias is not None,
            )
            # Initialize with averaged weights
            with torch.no_grad():
                self.backbone.conv1.weight = nn.Parameter(
                    old_conv.weight.mean(dim=1, keepdim=True)
                )
        elif self.architecture.startswith("efficientnet"):
            old_conv = self.backbone.features[0][0]
            self.backbone.features[0][0] = nn.Conv2d(
                1, old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=old_conv.bias is not None,
            )
            with torch.no_grad():
                self.backbone.features[0][0].weight = nn.Parameter(
                    old_conv.weight.mean(dim=1, keepdim=True)
                )

    def _replace_classifier(self) -> None:
        """Replace the classifier head."""
        if self.architecture.startswith("resnet"):
            self.backbone.fc = nn.Identity()
        elif self.architecture.startswith("efficientnet"):
            self.backbone.classifier = nn.Identity()

        # Add new classifier
        self.classifier = nn.Sequential(
            nn.Linear(self._embedding_dim, self.embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(self.embedding_dim, self.num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning logits."""
        features = self.backbone(x)
        return self.classifier(features)

    def get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """Extract embeddings before classifier."""
        features = self.backbone(x)
        # Return features after first linear layer
        return self.classifier[0](features)

    def freeze_backbone(self) -> None:
        """Freeze backbone weights for transfer learning."""
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self) -> None:
        """Unfreeze backbone weights for fine-tuning."""
        for param in self.backbone.parameters():
            param.requires_grad = True


@register_model("opensoundscape")
class OpenSoundscapeModel(BaseAudioModel):
    """
    OpenSoundscape-style CNN model for audio classification.

    This class implements spectrogram-based classification using
    pretrained ResNet or EfficientNet architectures, similar to
    the OpenSoundscape library approach.

    Example:
        >>> model = OpenSoundscapeModel()
        >>> model.load("path/to/model.pt")
        >>> results = model.predict("audio.wav")

        # Or create new model for training
        >>> model = OpenSoundscapeModel.create(
        ...     num_classes=10,
        ...     architecture="resnet18",
        ...     class_names=["class1", "class2", ...]
        ... )
    """

    def __init__(self, config: Optional[ModelConfig] = None):
        """Initialize OpenSoundscape model."""
        super().__init__(config)

        # Spectrogram parameters
        self.n_fft = 2048
        self.hop_length = 512
        self.n_mels = 128
        self.f_min = 0
        self.f_max = None  # Nyquist
        self.target_height = 224
        self.target_width = 224

    @property
    def backend(self) -> ModelBackend:
        return ModelBackend.OPENSOUNDSCAPE

    @classmethod
    def create(
        cls,
        num_classes: int,
        architecture: str = "resnet18",
        class_names: Optional[List[str]] = None,
        pretrained: bool = True,
        config: Optional[ModelConfig] = None,
    ) -> "OpenSoundscapeModel":
        """
        Create a new OpenSoundscape model for training.

        Args:
            num_classes: Number of output classes.
            architecture: Base architecture ("resnet18", "resnet50").
            class_names: Optional list of class names.
            pretrained: Use ImageNet pretrained weights.
            config: Model configuration.

        Returns:
            New OpenSoundscapeModel instance.
        """
        instance = cls(config)

        instance.model = SpectrogramCNN(
            num_classes=num_classes,
            architecture=architecture,
            pretrained=pretrained,
        )

        # Setup label mappings
        if class_names:
            instance.id2label = {i: name for i, name in enumerate(class_names)}
        else:
            instance.id2label = {i: f"class_{i}" for i in range(num_classes)}
        instance.label2id = {v: k for k, v in instance.id2label.items()}

        instance.model.to(instance.device)
        return instance

    def load(
        self,
        model_path: str,
        architecture: Optional[str] = None,
        **kwargs
    ) -> "OpenSoundscapeModel":
        """
        Load model from path.

        Args:
            model_path: Path to model file (.pt, .pth).
            architecture: Architecture type (inferred from checkpoint if not specified).

        Returns:
            Self for method chaining.
        """
        checkpoint = torch.load(model_path, map_location=self.device)

        if isinstance(checkpoint, dict):
            # Load from checkpoint dict
            arch = architecture or checkpoint.get("architecture", "resnet18")
            num_classes = checkpoint.get("num_classes", len(checkpoint.get("id2label", {})))

            self.model = SpectrogramCNN(
                num_classes=num_classes,
                architecture=arch,
                pretrained=False,
            )

            if "model_state_dict" in checkpoint:
                self.model.load_state_dict(checkpoint["model_state_dict"])
            elif "state_dict" in checkpoint:
                self.model.load_state_dict(checkpoint["state_dict"])

            if "id2label" in checkpoint:
                self.id2label = checkpoint["id2label"]
                self.label2id = {v: k for k, v in self.id2label.items()}

            # Load spectrogram parameters if saved
            if "spectrogram_config" in checkpoint:
                spec_config = checkpoint["spectrogram_config"]
                self.n_fft = spec_config.get("n_fft", self.n_fft)
                self.hop_length = spec_config.get("hop_length", self.hop_length)
                self.n_mels = spec_config.get("n_mels", self.n_mels)

        elif isinstance(checkpoint, nn.Module):
            self.model = checkpoint
        else:
            # Try loading as state dict directly
            raise ValueError(f"Unknown checkpoint format. Please use a dict with 'model_state_dict'.")

        self.model.to(self.device)
        self.model.eval()
        return self

    def predict(
        self,
        audio: Union[str, np.ndarray, torch.Tensor],
        sample_rate: Optional[int] = None,
    ) -> List[PredictionResult]:
        """
        Run prediction on audio.

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

        # Resample if needed
        if sample_rate != self.config.sample_rate:
            waveform = resample_waveform_tensor(waveform, sample_rate, self.config.sample_rate)

        # Split into segments
        segment_samples = int(self.config.clip_duration * self.config.sample_rate)
        overlap_samples = int(self.config.overlap * self.config.sample_rate)
        step = max(1, segment_samples - overlap_samples)

        results = []
        num_samples = waveform.shape[1]

        for start in range(0, num_samples, step):
            end = min(start + segment_samples, num_samples)
            segment = waveform[:, start:end]

            # Pad if necessary
            if segment.shape[1] < segment_samples:
                padding = segment_samples - segment.shape[1]
                segment = torch.nn.functional.pad(segment, (0, padding))

            # Compute spectrogram
            spectrogram = self._compute_spectrogram(segment)
            spectrogram = spectrogram.unsqueeze(0).to(self.device)

            # Run inference
            with torch.inference_mode():
                outputs = self.model(spectrogram)

            logits = outputs[0]
            probs = torch.softmax(logits, dim=-1)

            # Get top-k predictions
            top_k = min(self.config.top_k, len(probs))
            top_probs, top_indices = torch.topk(probs, top_k)

            for prob, idx in zip(top_probs, top_indices):
                confidence = prob.item()

                if confidence >= self.config.min_confidence:
                    label = self.id2label.get(idx.item(), f"class_{idx.item()}")
                    results.append(PredictionResult(
                        label=label,
                        confidence=confidence,
                        start_time=start / self.config.sample_rate,
                        end_time=end / self.config.sample_rate,
                        filepath=filepath,
                    ))

            if end >= num_samples:
                break

        return results

    def _compute_spectrogram(self, waveform: torch.Tensor) -> torch.Tensor:
        """Compute mel spectrogram for CNN input."""
        import torchaudio.transforms as T

        # Compute mel spectrogram
        mel_transform = T.MelSpectrogram(
            sample_rate=self.config.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            f_min=self.f_min,
            f_max=self.f_max or self.config.sample_rate // 2,
        )

        mel_spec = mel_transform(waveform)

        # Convert to dB
        db_transform = T.AmplitudeToDB(stype="power", top_db=80)
        mel_spec_db = db_transform(mel_spec)

        # Normalize to [0, 1]
        mel_spec_db = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min() + 1e-8)

        # Resize to target dimensions
        mel_spec_db = torch.nn.functional.interpolate(
            mel_spec_db.unsqueeze(0),
            size=(self.target_height, self.target_width),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

        return mel_spec_db

    def extract_embeddings(
        self,
        audio: Union[str, np.ndarray, torch.Tensor],
        sample_rate: Optional[int] = None,
        layer: Optional[str] = None,
    ) -> np.ndarray:
        """
        Extract embeddings from the model.

        Args:
            audio: Audio file path, numpy array, or torch tensor.
            sample_rate: Sample rate if audio is array/tensor.
            layer: Not used (extracts from penultimate layer).

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

        spectrogram = self._compute_spectrogram(waveform)
        spectrogram = spectrogram.unsqueeze(0).to(self.device)

        with torch.inference_mode():
            embeddings = self.model.get_embeddings(spectrogram)

        return embeddings.cpu().numpy()

    def _save_pytorch(self, path: str) -> str:
        """Save model in PyTorch format with metadata."""
        if self.model is None:
            raise RuntimeError("No model loaded")

        state = {
            "model_state_dict": self.model.state_dict(),
            "id2label": self.id2label,
            "label2id": self.label2id,
            "architecture": self.model.architecture,
            "num_classes": self.model.num_classes,
            "spectrogram_config": {
                "n_fft": self.n_fft,
                "hop_length": self.hop_length,
                "n_mels": self.n_mels,
                "f_min": self.f_min,
                "f_max": self.f_max,
            },
            "config": {
                "sample_rate": self.config.sample_rate,
                "clip_duration": self.config.clip_duration,
                "backend": self.backend.value,
            }
        }

        torch.save(state, path)
        return path

    def _get_dummy_input(self) -> torch.Tensor:
        """Get dummy input for model export."""
        return torch.randn(1, 1, self.target_height, self.target_width).to(self.device)

    def freeze_backbone(self) -> None:
        """Freeze backbone for transfer learning."""
        if self.model is not None:
            self.model.freeze_backbone()

    def unfreeze_backbone(self) -> None:
        """Unfreeze backbone for fine-tuning."""
        if self.model is not None:
            self.model.unfreeze_backbone()
