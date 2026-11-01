"""Operations for the indices domain: thin adapters over :mod:`bioamla.indices`.

See :mod:`bioamla.operations`. Each operation maps its request onto the public
library calls a script would make, and returns their result. Indices of one file are
a ``job``: their cost grows with the recording's length.
"""

from __future__ import annotations

from bioamla.common.config import config_dataclass
from bioamla.indices.files import (
    FileIndexValues,
    FileIndices,
    FileTemporalIndices,
    index_file,
    index_values_file,
    temporal_index_file,
)
from bioamla.operations import OperationContext, PathStr, operation


@config_dataclass
class IndicesComputeRequest:
    """Every acoustic index of one file.

    Attributes:
        file: The audio file.
        n_fft: FFT window size.
        aci_min_freq: ACI low-frequency bound (Hz).
        aci_max_freq: ACI high-frequency bound (Hz), or ``None`` for the Nyquist frequency.
        bio_min_freq: BIO low-frequency bound (Hz).
        bio_max_freq: BIO high-frequency bound (Hz).
        db_threshold: dB threshold for ADI and AEI.
        output: A JSON file to save the indices to.
    """

    file: PathStr
    n_fft: int = 512
    aci_min_freq: float = 0.0
    aci_max_freq: float | None = None
    bio_min_freq: float = 2000.0
    bio_max_freq: float = 8000.0
    db_threshold: float = -50.0
    output: PathStr | None = None


@operation("indices.compute", request=IndicesComputeRequest, kind="job")
def indices_compute(request: IndicesComputeRequest, context: OperationContext) -> FileIndices:
    """Compute every acoustic index of one file."""
    return index_file(
        request.file,
        n_fft=request.n_fft,
        aci_min_freq=request.aci_min_freq,
        aci_max_freq=request.aci_max_freq,
        bio_min_freq=request.bio_min_freq,
        bio_max_freq=request.bio_max_freq,
        db_threshold=request.db_threshold,
        output=request.output,
    )


@config_dataclass
class IndicesTemporalRequest:
    """Acoustic indices of one file over consecutive windows.

    Attributes:
        file: The audio file.
        window_duration: Window length (s).
        n_fft: FFT window size.
        output: A JSON file to save the segment rows to.
    """

    file: PathStr
    window_duration: float = 60.0
    n_fft: int = 512
    output: PathStr | None = None


@operation("indices.temporal", request=IndicesTemporalRequest, kind="job")
def indices_temporal(
    request: IndicesTemporalRequest, context: OperationContext
) -> FileTemporalIndices:
    """Compute acoustic indices of one file over consecutive windows."""
    return temporal_index_file(
        request.file,
        window_duration=request.window_duration,
        n_fft=request.n_fft,
        output=request.output,
    )


@config_dataclass
class IndexBandRequest:
    """One frequency-banded index (ACI or BIO) of one file.

    Attributes:
        file: The audio file.
        min_freq: Low-frequency bound (Hz).
        max_freq: High-frequency bound (Hz), or ``None`` for the Nyquist frequency.
        n_fft: FFT window size.
    """

    file: PathStr
    min_freq: float = 0.0
    max_freq: float | None = None
    n_fft: int = 512


@config_dataclass
class BioIndexRequest:
    """The Bioacoustic Index (BIO) of one file.

    Attributes:
        file: The audio file.
        min_freq: Low-frequency bound (Hz).
        max_freq: High-frequency bound (Hz).
        n_fft: FFT window size.
    """

    file: PathStr
    min_freq: float = 2000.0
    max_freq: float = 8000.0
    n_fft: int = 512


@config_dataclass
class IndexThresholdRequest:
    """One threshold index (ADI or AEI) of one file.

    Attributes:
        file: The audio file.
        db_threshold: dB threshold.
        n_fft: FFT window size.
    """

    file: PathStr
    db_threshold: float = -50.0
    n_fft: int = 512


@config_dataclass
class IndexRequest:
    """An index of one file that only takes the FFT size (entropy).

    Attributes:
        file: The audio file.
        n_fft: FFT window size.
    """

    file: PathStr
    n_fft: int = 512


@config_dataclass
class NdsiRequest:
    """The Normalized Difference Soundscape Index (NDSI) of one file.

    Attributes:
        file: The audio file.
        n_fft: FFT window size.
    """

    file: PathStr
    n_fft: int = 1024


@operation("indices.aci", request=IndexBandRequest, kind="job")
def indices_aci(request: IndexBandRequest, context: OperationContext) -> FileIndexValues:
    """Compute the Acoustic Complexity Index (ACI) of one file."""
    return index_values_file(
        request.file,
        ["aci"],
        n_fft=request.n_fft,
        min_freq=request.min_freq,
        max_freq=request.max_freq,
    )


@operation("indices.bio", request=BioIndexRequest, kind="job")
def indices_bio(request: BioIndexRequest, context: OperationContext) -> FileIndexValues:
    """Compute the Bioacoustic Index (BIO) of one file."""
    return index_values_file(
        request.file,
        ["bio"],
        n_fft=request.n_fft,
        min_freq=request.min_freq,
        max_freq=request.max_freq,
    )


@operation("indices.adi", request=IndexThresholdRequest, kind="job")
def indices_adi(request: IndexThresholdRequest, context: OperationContext) -> FileIndexValues:
    """Compute the Acoustic Diversity Index (ADI) of one file."""
    return index_values_file(
        request.file, ["adi"], n_fft=request.n_fft, db_threshold=request.db_threshold
    )


@operation("indices.aei", request=IndexThresholdRequest, kind="job")
def indices_aei(request: IndexThresholdRequest, context: OperationContext) -> FileIndexValues:
    """Compute the Acoustic Evenness Index (AEI) of one file."""
    return index_values_file(
        request.file, ["aei"], n_fft=request.n_fft, db_threshold=request.db_threshold
    )


@operation("indices.ndsi", request=NdsiRequest, kind="job")
def indices_ndsi(request: NdsiRequest, context: OperationContext) -> FileIndexValues:
    """Compute the Normalized Difference Soundscape Index (NDSI) of one file."""
    return index_values_file(request.file, ["ndsi"], n_fft=request.n_fft)


@operation("indices.entropy", request=IndexRequest, kind="job")
def indices_entropy(request: IndexRequest, context: OperationContext) -> FileIndexValues:
    """Compute the spectral and temporal entropy of one file."""
    return index_values_file(request.file, ["h_spectral", "h_temporal"], n_fft=request.n_fft)


__all__ = [
    "BioIndexRequest",
    "IndexBandRequest",
    "IndexRequest",
    "IndexThresholdRequest",
    "IndicesComputeRequest",
    "IndicesTemporalRequest",
    "NdsiRequest",
    "indices_adi",
    "indices_aci",
    "indices_aei",
    "indices_bio",
    "indices_compute",
    "indices_entropy",
    "indices_ndsi",
    "indices_temporal",
]
