"""Acoustic indices for soundscape ecology.

Compute standard ecoacoustic indices (ACI, ADI, AEI, BIO, NDSI) plus spectral and
temporal entropy from audio signals.

Example:
    >>> from bioamla.audio import load_audio_data
    >>> from bioamla.indices import compute_all_indices
    >>> audio = load_audio_data("recording.wav")
    >>> idx = compute_all_indices(audio.samples, audio.sample_rate, include_entropy=True)
    >>> print(idx.aci, idx.ndsi)

For one file, :func:`index_file`, :func:`temporal_index_file` and
:func:`index_values_file` load it, compute and optionally save the result:

    >>> from bioamla.indices import index_file
    >>> index_file("recording.wav", output="recording-indices.json").indices.aci
"""

from bioamla._extras import require_extra

# Fail early, with the install command, if the [indices] extra is missing.
require_extra("indices", "bioamla.indices")

from bioamla.indices.batch import batch_index_files, batch_indices
from bioamla.indices.compute import (
    AVAILABLE_INDICES,
    DEFAULT_INDICES,
    INDEX_DESCRIPTIONS,
    AcousticIndices,
    batch_compute_indices,
    check_index_names,
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
from bioamla.indices.files import (
    SEGMENT_COLUMNS,
    FileIndexValues,
    FileIndices,
    FileTemporalIndices,
    index_file,
    index_values_file,
    temporal_index_file,
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
    "batch_index_files",
    "batch_indices",
    "check_index_names",
    "temporal_indices",
    "spectral_entropy",
    "temporal_entropy",
    "describe_index",
    "AVAILABLE_INDICES",
    "DEFAULT_INDICES",
    "INDEX_DESCRIPTIONS",
    # One file: load, compute, optionally save
    "index_file",
    "temporal_index_file",
    "index_values_file",
    "FileIndices",
    "FileTemporalIndices",
    "FileIndexValues",
    "SEGMENT_COLUMNS",
]
