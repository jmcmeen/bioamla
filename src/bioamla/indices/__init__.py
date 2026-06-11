"""Acoustic indices for soundscape ecology.

Compute standard ecoacoustic indices (ACI, ADI, AEI, BIO, NDSI) plus spectral and
temporal entropy from audio signals.

Example:
    >>> from bioamla.audio import load_audio_data
    >>> from bioamla.indices import compute_all_indices
    >>> audio = load_audio_data("recording.wav")
    >>> idx = compute_all_indices(audio.samples, audio.sample_rate, include_entropy=True)
    >>> print(idx.aci, idx.ndsi)
"""

from bioamla.indices.compute import (
    AVAILABLE_INDICES,
    INDEX_DESCRIPTIONS,
    AcousticIndices,
    batch_compute_indices,
    compute_aci,
    compute_adi,
    compute_aei,
    compute_all_indices,
    compute_bio,
    compute_index,
    compute_indices_from_file,
    compute_ndsi,
    describe_index,
    spectral_entropy,
    temporal_entropy,
    temporal_indices,
)

__all__ = [
    "AcousticIndices",
    "compute_aci",
    "compute_adi",
    "compute_aei",
    "compute_bio",
    "compute_ndsi",
    "compute_all_indices",
    "compute_index",
    "compute_indices_from_file",
    "batch_compute_indices",
    "temporal_indices",
    "spectral_entropy",
    "temporal_entropy",
    "describe_index",
    "AVAILABLE_INDICES",
    "INDEX_DESCRIPTIONS",
]
