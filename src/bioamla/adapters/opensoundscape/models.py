from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd


@dataclass
class PredictionResult:
    """Result from a model prediction.

    Attributes:
        label: Predicted class label.
        confidence: Confidence score (0-1).
        start_time: Start time in seconds.
        end_time: End time in seconds.
        filepath: Source audio file path.
    """

    label: str
    confidence: float
    start_time: float
    end_time: float
    filepath: Optional[str] = None


class BirdNETAdapter:
    """Adapter for BirdNET model from bioacoustics-model-zoo.

    BirdNET is a deep learning model trained to identify over 6,000 bird species
    by their vocalizations. It processes 3-second audio clips at 48kHz.

    Note: Requires ai_edge_litert (tflite) to be installed.

    Example:
        >>> adapter = BirdNETAdapter()
        >>> predictions = adapter.predict(["audio1.wav", "audio2.wav"])
        >>> embeddings = adapter.extract_embeddings(["audio.wav"])
    """

    # BirdNET default parameters
    SAMPLE_RATE = 48000
    SAMPLE_DURATION = 3.0

    def __init__(self) -> None:
        """Initialize BirdNET adapter.

        Raises:
            ImportError: If ai_edge_litert (tflite) is not installed.
        """
        self._model = None
        self._initialize_model()

    def _initialize_model(self) -> None:
        """Initialize the BirdNET model."""
        try:
            from bioacoustics_model_zoo import BirdNET
            self._model = BirdNET()
        except ImportError as e:
            raise ImportError(
                "BirdNET requires ai_edge_litert (tflite). "
                "Install with: pip install ai-edge-litert"
            ) from e

    def predict(
        self,
        samples: Union[str, List[str], pd.DataFrame],
        min_confidence: float = 0.1,
        batch_size: int = 1,
        num_workers: int = 0,
        overlap: float = 0.0,
    ) -> pd.DataFrame:
        """Run BirdNET prediction on audio files.

        Args:
            samples: Audio file path(s) or DataFrame with file paths as index.
            min_confidence: Minimum confidence threshold for predictions.
            batch_size: Batch size for inference.
            num_workers: Number of workers for data loading.
            overlap: Overlap between clips (0-1).

        Returns:
            DataFrame with predictions. Columns are species names,
            index is (file, start_time, end_time).
        """
        if isinstance(samples, str):
            samples = [samples]

        predictions = self._model.predict(
            samples,
            batch_size=batch_size,
            num_workers=num_workers,
            overlap_fraction=overlap,
        )

        # Filter by min_confidence after prediction
        if min_confidence > 0:
            predictions = predictions.where(predictions >= min_confidence)

        return predictions

    def predict_with_results(
        self,
        samples: Union[str, List[str]],
        min_confidence: float = 0.1,
        top_k: int = 1,
        overlap: float = 0.0,
    ) -> List[PredictionResult]:
        """Run prediction and return as PredictionResult list.

        Args:
            samples: Audio file path(s).
            min_confidence: Minimum confidence threshold.
            top_k: Number of top predictions per segment.
            overlap: Overlap between clips (0-1).

        Returns:
            List of PredictionResult objects.
        """
        if isinstance(samples, str):
            samples = [samples]

        df = self.predict(
            samples,
            min_confidence=min_confidence,
            overlap=overlap,
        )

        results = []
        for (filepath, start_time, end_time), row in df.iterrows():
            # Get top-k predictions for this segment
            top_scores = row.nlargest(top_k)
            for label, confidence in top_scores.items():
                if confidence >= min_confidence:
                    results.append(
                        PredictionResult(
                            label=label,
                            confidence=float(confidence),
                            start_time=float(start_time),
                            end_time=float(end_time),
                            filepath=str(filepath),
                        )
                    )

        return results

    def extract_embeddings(
        self,
        samples: Union[str, List[str]],
        batch_size: int = 1,
        num_workers: int = 0,
    ) -> np.ndarray:
        """Extract embeddings from audio files.

        Args:
            samples: Audio file path(s).
            batch_size: Batch size for inference.
            num_workers: Number of workers for data loading.

        Returns:
            Embeddings as numpy array with shape (n_samples, embedding_dim).
        """
        if isinstance(samples, str):
            samples = [samples]

        embeddings = self._model.embed(
            samples,
            batch_size=batch_size,
            num_workers=num_workers,
        )

        return embeddings

    @property
    def sample_rate(self) -> int:
        """Get expected sample rate."""
        return self.SAMPLE_RATE

    @property
    def sample_duration(self) -> float:
        """Get expected sample duration in seconds."""
        return self.SAMPLE_DURATION

    @property
    def classes(self) -> List[str]:
        """Get list of class names."""
        if hasattr(self._model, 'classes'):
            return list(self._model.classes)
        return []

    @property
    def num_classes(self) -> int:
        """Get number of classes."""
        return len(self.classes)

    def config(self) -> Dict[str, Any]:
        """Get model configuration.

        Returns:
            Dict with model configuration.
        """
        return {
            "model": "BirdNET",
            "sample_rate": self.sample_rate,
            "sample_duration": self.sample_duration,
            "num_classes": self.num_classes,
        }


class PerchAdapter:
    """Adapter for Perch model from bioacoustics-model-zoo.

    Perch is a model trained on bird vocalizations that produces high-quality
    embeddings suitable for downstream classification tasks.

    Note: Requires tensorflow to be installed.

    Example:
        >>> adapter = PerchAdapter()
        >>> embeddings = adapter.extract_embeddings(["audio.wav"])
    """

    # Perch default parameters
    SAMPLE_RATE = 32000
    SAMPLE_DURATION = 5.0

    def __init__(self, version: str = "perch") -> None:
        """Initialize Perch adapter.

        Args:
            version: Model version ("perch" or "perch2").

        Raises:
            ImportError: If tensorflow is not installed.
        """
        self._model = None
        self._version = version
        self._initialize_model()

    def _initialize_model(self) -> None:
        """Initialize the Perch model."""
        try:
            if self._version == "perch2":
                from bioacoustics_model_zoo import Perch2
                self._model = Perch2()
            else:
                from bioacoustics_model_zoo import Perch
                self._model = Perch()
        except ImportError as e:
            raise ImportError(
                "Perch requires tensorflow. "
                "Install with: pip install tensorflow"
            ) from e

    def extract_embeddings(
        self,
        samples: Union[str, List[str]],
        batch_size: int = 1,
        num_workers: int = 0,
    ) -> np.ndarray:
        """Extract embeddings from audio files.

        Args:
            samples: Audio file path(s).
            batch_size: Batch size for inference.
            num_workers: Number of workers for data loading.

        Returns:
            Embeddings as numpy array with shape (n_samples, embedding_dim).
        """
        if isinstance(samples, str):
            samples = [samples]

        embeddings = self._model.embed(
            samples,
            batch_size=batch_size,
            num_workers=num_workers,
        )

        return embeddings

    def predict(
        self,
        samples: Union[str, List[str]],
        batch_size: int = 1,
        num_workers: int = 0,
    ) -> pd.DataFrame:
        """Run prediction on audio files.

        Note: Perch is primarily an embedding model. This method returns
        logits/scores if available, otherwise raises NotImplementedError.

        Args:
            samples: Audio file path(s).
            batch_size: Batch size for inference.
            num_workers: Number of workers for data loading.

        Returns:
            DataFrame with predictions if available.

        Raises:
            NotImplementedError: If the model doesn't support direct prediction.
        """
        if isinstance(samples, str):
            samples = [samples]

        if hasattr(self._model, 'predict'):
            return self._model.predict(
                samples,
                batch_size=batch_size,
                num_workers=num_workers,
            )
        else:
            raise NotImplementedError(
                "Perch is an embedding model. Use extract_embeddings() instead, "
                "then train a classifier on the embeddings."
            )

    @property
    def sample_rate(self) -> int:
        """Get expected sample rate."""
        return self.SAMPLE_RATE

    @property
    def sample_duration(self) -> float:
        """Get expected sample duration in seconds."""
        return self.SAMPLE_DURATION

    @property
    def version(self) -> str:
        """Get model version."""
        return self._version

    def config(self) -> Dict[str, Any]:
        """Get model configuration.

        Returns:
            Dict with model configuration.
        """
        return {
            "model": f"Perch ({self._version})",
            "sample_rate": self.sample_rate,
            "sample_duration": self.sample_duration,
        }


class HawkEarsAdapter:
    """Adapter for HawkEars model from bioacoustics-model-zoo.

    HawkEars is a PyTorch-based model for bird species classification,
    designed as an alternative to BirdNET that runs on PyTorch.

    Example:
        >>> adapter = HawkEarsAdapter()
        >>> predictions = adapter.predict(["audio.wav"])
        >>> embeddings = adapter.extract_embeddings(["audio.wav"])
    """

    def __init__(self, variant: str = "default") -> None:
        """Initialize HawkEars adapter.

        Args:
            variant: Model variant ("default", "embedding", "low_band", "v010").

        Raises:
            ImportError: If timm or torchaudio is not installed.
        """
        self._model = None
        self._variant = variant
        self._initialize_model()

    def _initialize_model(self) -> None:
        """Initialize the HawkEars model."""
        try:
            if self._variant == "embedding":
                from bioacoustics_model_zoo import HawkEars_Embedding
                self._model = HawkEars_Embedding()
            elif self._variant == "low_band":
                from bioacoustics_model_zoo import HawkEars_Low_Band
                self._model = HawkEars_Low_Band()
            elif self._variant == "v010":
                from bioacoustics_model_zoo import HawkEars_v010
                self._model = HawkEars_v010()
            else:
                from bioacoustics_model_zoo import HawkEars
                self._model = HawkEars()
        except ImportError as e:
            raise ImportError(
                "HawkEars requires timm and torchaudio. "
                "Install with: pip install timm torchaudio"
            ) from e

    def predict(
        self,
        samples: Union[str, List[str]],
        batch_size: int = 1,
        num_workers: int = 0,
        min_confidence: float = 0.1,
    ) -> pd.DataFrame:
        """Run prediction on audio files.

        Args:
            samples: Audio file path(s).
            batch_size: Batch size for inference.
            num_workers: Number of workers for data loading.
            min_confidence: Minimum confidence threshold.

        Returns:
            DataFrame with predictions.
        """
        if isinstance(samples, str):
            samples = [samples]

        predictions = self._model.predict(
            samples,
            batch_size=batch_size,
            num_workers=num_workers,
        )

        return predictions

    def extract_embeddings(
        self,
        samples: Union[str, List[str]],
        batch_size: int = 1,
        num_workers: int = 0,
    ) -> np.ndarray:
        """Extract embeddings from audio files.

        Args:
            samples: Audio file path(s).
            batch_size: Batch size for inference.
            num_workers: Number of workers for data loading.

        Returns:
            Embeddings as numpy array.
        """
        if isinstance(samples, str):
            samples = [samples]

        embeddings = self._model.embed(
            samples,
            batch_size=batch_size,
            num_workers=num_workers,
        )

        return embeddings

    @property
    def sample_rate(self) -> int:
        """Get expected sample rate."""
        if hasattr(self._model, 'sample_rate'):
            return self._model.sample_rate
        return 16000  # Default

    @property
    def sample_duration(self) -> float:
        """Get expected sample duration in seconds."""
        if hasattr(self._model, 'sample_duration'):
            return self._model.sample_duration
        return 5.0  # Default

    @property
    def classes(self) -> List[str]:
        """Get list of class names."""
        if hasattr(self._model, 'classes'):
            return list(self._model.classes)
        return []

    @property
    def variant(self) -> str:
        """Get model variant."""
        return self._variant

    def config(self) -> Dict[str, Any]:
        """Get model configuration.

        Returns:
            Dict with model configuration.
        """
        return {
            "model": f"HawkEars ({self._variant})",
            "sample_rate": self.sample_rate,
            "sample_duration": self.sample_duration,
            "num_classes": len(self.classes),
        }


def check_model_availability() -> Dict[str, bool]:
    """Check which model zoo models are available.

    Returns:
        Dict mapping model names to availability status.
    """
    availability = {}

    # Check BirdNET (requires tflite)
    try:
        from bioacoustics_model_zoo import BirdNET
        BirdNET()
        availability["BirdNET"] = True
    except ImportError:
        availability["BirdNET"] = False

    # Check Perch (requires tensorflow)
    try:
        from bioacoustics_model_zoo import Perch
        Perch()
        availability["Perch"] = True
    except ImportError:
        availability["Perch"] = False

    # Check HawkEars (requires timm)
    try:
        from bioacoustics_model_zoo import HawkEars
        HawkEars()
        availability["HawkEars"] = True
    except ImportError:
        availability["HawkEars"] = False

    return availability
