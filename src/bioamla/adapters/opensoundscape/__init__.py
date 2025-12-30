"""OpenSoundscape adapters for bioamla.

This package provides adapters that wrap OpenSoundscape functionality
to provide bioamla-compatible interfaces. Only the services layer
should import from this package - core code should NOT import from here.

Example:
    >>> from bioamla.adapters.opensoundscape import AudioAdapter, SpectrogramAdapter
    >>> audio = AudioAdapter.from_file("audio.wav", sample_rate=16000)
    >>> filtered = audio.bandpass(500, 5000)

    >>> from bioamla.adapters.opensoundscape import BioamlaPreprocessor
    >>> preprocessor = BioamlaPreprocessor(sample_duration=3.0)
    >>> spectrogram = preprocessor.process_file("audio.wav")

    >>> from bioamla.adapters.opensoundscape import CNNAdapter
    >>> model = CNNAdapter.create(classes=["bird", "frog"], architecture="resnet18")
    >>> model.train(train_df, val_df, epochs=10)

    >>> from bioamla.adapters.opensoundscape import BirdNETAdapter
    >>> birdnet = BirdNETAdapter()
    >>> predictions = birdnet.predict(["audio.wav"])

    >>> from bioamla.adapters.opensoundscape import ribbit_detect_preset
    >>> detections, metadata = ribbit_detect_preset("frog.wav", "spring_peeper")
"""

from bioamla.adapters.opensoundscape.audio import AudioAdapter
from bioamla.adapters.opensoundscape.cnn import CNNAdapter
from bioamla.adapters.opensoundscape.models import (
    BirdNETAdapter,
    HawkEarsAdapter,
    PerchAdapter,
    PredictionResult,
    check_model_availability,
)
from bioamla.adapters.opensoundscape.preprocessing import (
    AugmentationConfig,
    BioamlaPreprocessor,
)
from bioamla.adapters.opensoundscape.ribbit import (
    RIBBIT_PRESETS,
    RibbitDetection,
    get_ribbit_preset,
    list_ribbit_presets,
    ribbit_detect,
    ribbit_detect_preset,
    ribbit_detect_samples,
)
from bioamla.adapters.opensoundscape.spectrogram import SpectrogramAdapter

__all__ = [
    "AudioAdapter",
    "AugmentationConfig",
    "BioamlaPreprocessor",
    "BirdNETAdapter",
    "CNNAdapter",
    "HawkEarsAdapter",
    "PerchAdapter",
    "PredictionResult",
    "RIBBIT_PRESETS",
    "RibbitDetection",
    "SpectrogramAdapter",
    "check_model_availability",
    "get_ribbit_preset",
    "list_ribbit_presets",
    "ribbit_detect",
    "ribbit_detect_preset",
    "ribbit_detect_samples",
]
