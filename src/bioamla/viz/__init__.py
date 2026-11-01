"""
bioamla.viz — audio visualization domain.

Spectrogram and waveform rendering built on matplotlib and librosa. matplotlib
(with the ``Agg`` backend) is imported lazily inside the functions that need it,
so importing this package stays fast.

Example:
    >>> from bioamla.viz import generate_spectrogram, generate_spectrogram_video
    >>> generate_spectrogram("recording.wav", "out.png", viz_type="mel")
    >>> generate_spectrogram_video("recording.wav", "out.mp4")
    >>> generate_trajectory_video("recording.wav", "trajectory.mp4")
"""

from bioamla._extras import require_extra

# Fail early, with the install command, if the [viz] extra is missing.
require_extra("viz", "bioamla.viz")

from bioamla.viz.batch import batch_generate_spectrograms
from bioamla.viz.core import (
    SpectrogramBox,
    VisualizationType,
    WindowType,
    annotation_boxes,
    check_picture_settings,
    compute_mel_spectrogram,
    compute_stft,
    generate_spectrogram,
    spectrogram_to_db,
    spectrogram_to_image,
)
from bioamla.viz.trajectory_video import TrajectoryVideoResult, generate_trajectory_video
from bioamla.viz.video import SpectrogramVideoResult, generate_spectrogram_video

__all__ = [
    "SpectrogramBox",
    "annotation_boxes",
    "check_picture_settings",
    "generate_spectrogram",
    "generate_spectrogram_video",
    "SpectrogramVideoResult",
    "generate_trajectory_video",
    "TrajectoryVideoResult",
    "compute_stft",
    "compute_mel_spectrogram",
    "spectrogram_to_db",
    "spectrogram_to_image",
    "batch_generate_spectrograms",
    "VisualizationType",
    "WindowType",
]
