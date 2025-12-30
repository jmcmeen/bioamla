from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import torch
from opensoundscape.ml.cnn import CNN as OSSCNN


class CNNAdapter:
    """Adapter for OpenSoundscape CNN class.

    Provides a bioamla-compatible interface for training and running inference
    with CNN-based audio classifiers using OpenSoundscape.

    Example:
        >>> # Create a new model for training
        >>> adapter = CNNAdapter.create(
        ...     classes=["bird", "frog", "insect"],
        ...     architecture="resnet18",
        ...     sample_duration=3.0,
        ... )
        >>> adapter.train(train_df, val_df, epochs=10)
        >>> adapter.save("model.pt")

        >>> # Load and run inference
        >>> adapter = CNNAdapter.load("model.pt")
        >>> predictions = adapter.predict(["audio1.wav", "audio2.wav"])
    """

    def __init__(self, cnn: OSSCNN, architecture: Optional[str] = None) -> None:
        """Initialize adapter with an OpenSoundscape CNN.

        Args:
            cnn: OpenSoundscape CNN instance.
            architecture: Architecture name (stored separately since OSS doesn't preserve it).
        """
        self._cnn = cnn
        self._architecture = architecture or self._infer_architecture()
        self._disable_bandpass()

    def _infer_architecture(self) -> str:
        """Infer architecture from network class name.

        Returns:
            Inferred architecture name.
        """
        network_class = type(self._cnn.network).__name__
        # Map common network class names to architecture strings
        arch_map = {
            "ResNet": "resnet",
            "EfficientNet": "efficientnet",
            "DenseNet": "densenet",
            "VGG": "vgg",
            "InceptionV3": "inception_v3",
        }
        for class_name, arch in arch_map.items():
            if class_name in network_class:
                return arch
        return "unknown"

    def _disable_bandpass(self) -> None:
        """Disable bandpass filter in preprocessor pipeline.

        OSS applies bandpass by default which can cause errors if frequency
        range doesn't match the audio. We disable it for inference flexibility.
        """
        if hasattr(self._cnn.preprocessor.pipeline, "bandpass"):
            self._cnn.preprocessor.pipeline.bandpass.bypass = True

    @classmethod
    def create(
        cls,
        classes: List[str],
        architecture: str = "resnet18",
        sample_duration: float = 3.0,
        sample_rate: int = 16000,
        single_target: bool = True,
    ) -> "CNNAdapter":
        """Create a new CNN model for training.

        Args:
            classes: List of class names.
            architecture: Model architecture (resnet18, resnet50, efficientnet_b0, etc.).
            sample_duration: Duration of audio clips in seconds.
            sample_rate: Target sample rate.
            single_target: If True, uses single-label classification.

        Returns:
            New CNNAdapter instance.
        """
        cnn = OSSCNN(
            architecture=architecture,
            classes=classes,
            sample_duration=sample_duration,
            single_target=single_target,
        )

        # Configure sample rate in preprocessor
        cnn.preprocessor.pipeline.load_audio.set(sample_rate=sample_rate)

        return cls(cnn, architecture=architecture)

    @classmethod
    def load(cls, path: str) -> "CNNAdapter":
        """Load a saved CNN model.

        Args:
            path: Path to saved model file.

        Returns:
            CNNAdapter with loaded model.
        """
        cnn = OSSCNN.load(path)
        return cls(cnn)

    def save(self, path: str) -> str:
        """Save the model to a file.

        Args:
            path: Path to save model.

        Returns:
            Path where model was saved.
        """
        self._cnn.save(path)
        return path

    def train(
        self,
        train_df: pd.DataFrame,
        validation_df: Optional[pd.DataFrame] = None,
        epochs: int = 10,
        batch_size: int = 32,
        num_workers: int = 0,
        save_path: str = ".",
        learning_rate: Optional[float] = None,
        freeze_feature_extractor: bool = False,
    ) -> Dict[str, Any]:
        """Train the model.

        Args:
            train_df: Training dataframe with file paths as index and
                class columns with 0/1 labels.
            validation_df: Validation dataframe (same format as train_df).
            epochs: Number of training epochs.
            batch_size: Batch size for training.
            num_workers: Number of workers for data loading.
            save_path: Directory to save model checkpoints.
            learning_rate: Learning rate (uses default if None).
            freeze_feature_extractor: If True, freezes backbone weights.

        Returns:
            Dict with training history/metrics.
        """
        # Optionally set learning rate
        if learning_rate is not None:
            for param_group in self._cnn.optimizer.param_groups:
                param_group["lr"] = learning_rate

        # Optionally freeze feature extractor
        if freeze_feature_extractor:
            self._cnn.freeze_feature_extractor()

        # Train the model
        self._cnn.train(
            train_df=train_df,
            validation_df=validation_df,
            epochs=epochs,
            batch_size=batch_size,
            num_workers=num_workers,
            save_path=save_path,
            progress_bar=True,
        )

        # Return training history if available
        return {
            "epochs": epochs,
            "save_path": save_path,
        }

    def predict(
        self,
        samples: Union[str, List[str], pd.DataFrame],
        batch_size: int = 1,
        num_workers: int = 0,
        activation: Optional[str] = "softmax",
        split_files_into_clips: bool = True,
    ) -> pd.DataFrame:
        """Run prediction on audio files.

        Args:
            samples: Audio file path(s) or DataFrame with file paths as index.
            batch_size: Batch size for inference.
            num_workers: Number of workers for data loading.
            activation: Activation function ("softmax", "sigmoid", or None).
            split_files_into_clips: If True, splits long files into clips.

        Returns:
            DataFrame with predictions (columns are class names, values are scores).
        """
        # Handle single file path
        if isinstance(samples, str):
            samples = [samples]

        predictions = self._cnn.predict(
            samples=samples,
            batch_size=batch_size,
            num_workers=num_workers,
            activation_layer=activation,
            split_files_into_clips=split_files_into_clips,
            progress_bar=False,
        )

        return predictions

    def extract_embeddings(
        self,
        samples: Union[str, List[str], pd.DataFrame],
        batch_size: int = 1,
        num_workers: int = 0,
        target_layer: Optional[str] = None,
    ) -> np.ndarray:
        """Extract embeddings from audio files.

        Args:
            samples: Audio file path(s) or DataFrame with file paths as index.
            batch_size: Batch size for inference.
            num_workers: Number of workers for data loading.
            target_layer: Layer to extract embeddings from (default: embedding layer).

        Returns:
            Embeddings as numpy array.
        """
        if isinstance(samples, str):
            samples = [samples]

        embeddings = self._cnn.embed(
            samples=samples,
            target_layer=target_layer,
            progress_bar=False,
            return_dfs=False,
            avgpool=True,
        )

        return embeddings

    def freeze_backbone(self) -> None:
        """Freeze the feature extractor weights."""
        self._cnn.freeze_feature_extractor()

    def unfreeze(self) -> None:
        """Unfreeze all model weights."""
        self._cnn.unfreeze()

    @property
    def classes(self) -> List[str]:
        """Get list of class names."""
        return list(self._cnn.classes)

    @property
    def num_classes(self) -> int:
        """Get number of classes."""
        return len(self._cnn.classes)

    @property
    def architecture(self) -> str:
        """Get model architecture name."""
        return self._architecture

    @property
    def sample_duration(self) -> float:
        """Get sample duration in seconds."""
        return self._cnn.preprocessor.sample_duration

    @property
    def device(self) -> torch.device:
        """Get the device the model is on."""
        return next(self._cnn.network.parameters()).device

    def to(self, device: Union[str, torch.device]) -> "CNNAdapter":
        """Move model to specified device.

        Args:
            device: Target device (e.g., "cuda", "cpu").

        Returns:
            Self for method chaining.
        """
        self._cnn.to(device)
        return self

    def eval(self) -> "CNNAdapter":
        """Set model to evaluation mode.

        Returns:
            Self for method chaining.
        """
        self._cnn.eval()
        return self

    def config(self) -> Dict[str, Any]:
        """Get model configuration.

        Returns:
            Dict with model configuration.
        """
        return {
            "architecture": self.architecture,
            "classes": self.classes,
            "num_classes": self.num_classes,
            "sample_duration": self.sample_duration,
        }
