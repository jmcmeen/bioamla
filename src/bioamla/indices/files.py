"""Acoustic indices for one audio file: load it, compute, and optionally save.

These are the file-level calls behind the ``bioamla indices`` commands. Each loads the
file once and returns a result object with ``to_dict()``; the ``output`` argument, where
there is one, also writes the result as JSON.

Example:
    >>> from bioamla.indices import index_file, index_values_file
    >>> found = index_file("recording.wav", output="recording-indices.json")
    >>> found.indices.ndsi
    >>> index_values_file("recording.wav", ["aci"], min_freq=1000).values["aci"]
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bioamla.exceptions import InvalidInputError, ProcessingError
from bioamla.indices.compute import (
    AcousticIndices,
    check_index_names,
    compute_all_indices,
    compute_index,
    temporal_indices,
)

#: The columns of a :func:`temporal_index_file` segment row.
SEGMENT_COLUMNS = ("segment", "start_time", "aci", "adi", "aei", "bio", "ndsi")


def _write_json(data: Any, output: str | Path) -> str:
    path = Path(output)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2))
    except OSError as e:
        raise ProcessingError(f"Failed to write {output}: {e}") from e
    return str(output)


def _load(file: str | Path) -> Any:
    from bioamla.audio import load_audio_data

    return load_audio_data(file)


@dataclass
class FileIndices:
    """All acoustic indices of one file.

    Attributes:
        file: The audio file.
        indices: The indices.
        output: The JSON file they were written to, or ``None`` if not saved.
    """

    file: str
    indices: AcousticIndices
    output: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """``{"file", "indices", "output"}``."""
        return {"file": self.file, "indices": self.indices.to_dict(), "output": self.output}


def index_file(
    file: str | Path,
    *,
    n_fft: int = 512,
    aci_min_freq: float = 0.0,
    aci_max_freq: float | None = None,
    bio_min_freq: float = 2000.0,
    bio_max_freq: float = 8000.0,
    db_threshold: float = -50.0,
    output: str | Path | None = None,
) -> FileIndices:
    """Compute every acoustic index of one file, entropy included.

    Args:
        file: The audio file.
        n_fft: FFT window size.
        aci_min_freq: ACI low-frequency bound (Hz).
        aci_max_freq: ACI high-frequency bound (Hz); ``None`` (or 0) for the Nyquist
            frequency.
        bio_min_freq: BIO low-frequency bound (Hz).
        bio_max_freq: BIO high-frequency bound (Hz).
        db_threshold: dB threshold for ADI and AEI.
        output: A JSON file to write ``{"filepath", <index>: value, ...}`` to (the
            file format ``indices compute -o`` has always written), or ``None`` or an
            empty string to save nothing.

    Returns:
        The file's indices, and the output path if saved.

    Raises:
        NotFoundError: If the file doesn't exist.
        AudioLoadError: If the file can't be read.
        ProcessingError: If the output can't be written.
    """
    audio = _load(file)
    kwargs: dict[str, Any] = {}
    if aci_max_freq:
        kwargs["aci_max_freq"] = aci_max_freq
    indices = compute_all_indices(
        audio.samples,
        audio.sample_rate,
        n_fft=n_fft,
        aci_min_freq=aci_min_freq,
        bio_min_freq=bio_min_freq,
        bio_max_freq=bio_max_freq,
        db_threshold=db_threshold,
        include_entropy=True,
        **kwargs,
    )
    saved = _write_json({"filepath": str(file), **indices.to_dict()}, output) if output else None
    return FileIndices(file=str(file), indices=indices, output=saved)


@dataclass
class FileTemporalIndices:
    """Acoustic indices of one file over consecutive windows.

    Attributes:
        file: The audio file.
        window_duration: Window length in seconds.
        segments: One row per window, with the :data:`SEGMENT_COLUMNS`.
        output: The JSON file the rows were written to, or ``None`` if not saved.
    """

    file: str
    window_duration: float
    segments: list[dict[str, Any]]
    output: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """``{"file", "window_duration", "segments", "output"}``."""
        return {
            "file": self.file,
            "window_duration": self.window_duration,
            "segments": self.segments,
            "output": self.output,
        }


def temporal_index_file(
    file: str | Path,
    *,
    window_duration: float = 60.0,
    n_fft: int = 512,
    output: str | Path | None = None,
) -> FileTemporalIndices:
    """Compute the main indices of one file over consecutive, non-overlapping windows.

    A trailing part shorter than ``window_duration`` is left out.

    Args:
        file: The audio file.
        window_duration: Window length in seconds.
        n_fft: FFT window size.
        output: A JSON file to write the segment rows to, or ``None`` or an empty
            string to save nothing.

    Returns:
        The rows, one per window, and the output path if saved.

    Raises:
        InvalidInputError: If ``window_duration`` is shorter than one sample.
        NotFoundError: If the file doesn't exist.
        AudioLoadError: If the file can't be read.
        ProcessingError: If the output can't be written.
    """
    audio = _load(file)
    windows = temporal_indices(
        audio.samples,
        audio.sample_rate,
        window_duration=window_duration,
        hop_duration=window_duration,
        n_fft=n_fft,
    )
    rows = [
        {
            "segment": i,
            "start_time": w["start_time"],
            **{k: w[k] for k in SEGMENT_COLUMNS[2:]},
        }
        for i, w in enumerate(windows)
    ]
    saved = _write_json(rows, output) if output else None
    return FileTemporalIndices(
        file=str(file), window_duration=window_duration, segments=rows, output=saved
    )


@dataclass
class FileIndexValues:
    """Selected acoustic indices of one file.

    Attributes:
        file: The audio file.
        values: Index name -> value, in the order asked for.
    """

    file: str
    values: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        """``{"file", "values"}``."""
        return {"file": self.file, "values": self.values}


def index_values_file(
    file: str | Path, names: list[str], *, n_fft: int | None = None, **params: Any
) -> FileIndexValues:
    """Compute selected indices of one file, loading it once.

    Args:
        file: The audio file.
        names: Index names, from :data:`~bioamla.indices.AVAILABLE_INDICES`.
        n_fft: FFT window size, or ``None`` for each index's default (see
            :func:`~bioamla.indices.compute_index`).
        **params: Parameters of a single index, as
            :func:`~bioamla.indices.compute_index` takes them (``min_freq``,
            ``max_freq``, ``db_threshold``, ...). They mean different things to
            different indices (``max_freq`` is ACI's band edge but ADI's top band), so
            they're only allowed with one name.

    Returns:
        The values by name.

    Raises:
        InvalidInputError: If a name isn't a known index, or ``params`` are given
            with more than one name.
        NotFoundError: If the file doesn't exist.
        AudioLoadError: If the file can't be read.
    """
    # compute_index checks names too, but only after the (possibly long) load.
    check_index_names(list(names))
    if params and len(names) > 1:
        raise InvalidInputError(
            f"Index parameters ({', '.join(sorted(params))}) apply to one index; "
            f"compute {', '.join(names)} separately"
        )
    audio = _load(file)
    values = {
        name: compute_index(audio.samples, audio.sample_rate, name, n_fft=n_fft, **params)
        for name in names
    }
    return FileIndexValues(file=str(file), values=values)


__all__ = [
    "SEGMENT_COLUMNS",
    "FileIndexValues",
    "FileIndices",
    "FileTemporalIndices",
    "index_file",
    "index_values_file",
    "temporal_index_file",
]
