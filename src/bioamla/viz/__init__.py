"""
bioamla.viz — audio visualization domain.

Spectrogram and waveform rendering built on matplotlib and librosa. matplotlib
(with the ``Agg`` backend) and torchaudio are imported lazily inside the
functions that need them, so this package imports cleanly on a slim install.

Example:
    >>> from bioamla.viz import generate_spectrogram
    >>> generate_spectrogram("recording.wav", "out.png", viz_type="mel")
"""

from bioamla.viz.batch import batch_generate_spectrograms
from bioamla.viz.core import (
    VisualizationType,
    WindowType,
    compute_mel_spectrogram,
    compute_stft,
    generate_spectrogram,
    spectrogram_to_db,
    spectrogram_to_image,
)

__all__ = [
    "generate_spectrogram",
    "compute_stft",
    "compute_mel_spectrogram",
    "spectrogram_to_db",
    "spectrogram_to_image",
    "batch_generate_spectrograms",
    "VisualizationType",
    "WindowType",
]
