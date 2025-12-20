"""
Analysis Package
================

Analysis domain for bioacoustic data including:
- Acoustic indices (ACI, ADI, AEI, BIO, NDSI)
- Clustering and dimensionality reduction
- Dataset exploration utilities
"""

from bioamla.core.analysis.indices import (
    AcousticIndices,
    compute_aci,
    compute_adi,
    compute_aei,
    compute_bio,
    compute_ndsi,
    compute_all_indices,
    compute_indices_from_file,
    batch_compute_indices,
    temporal_indices,
    spectral_entropy,
    temporal_entropy,
)

__all__ = [
    # indices
    "AcousticIndices",
    "compute_aci",
    "compute_adi",
    "compute_aei",
    "compute_bio",
    "compute_ndsi",
    "compute_all_indices",
    "compute_indices_from_file",
    "batch_compute_indices",
    "temporal_indices",
    "spectral_entropy",
    "temporal_entropy",
]
