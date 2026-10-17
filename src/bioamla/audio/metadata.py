"""
Recorder Metadata
=================

Field recorders embed deployment details inside the WAV container: AudioMoth writes a
comment in the RIFF ``INFO`` block, Wildlife Acoustics writes a ``wamd`` chunk, bat
detectors and others write GUANO (``guan``), broadcast tools write ``bext``. riffy
decodes each standard; this module flattens whichever ones a file carries into one
:class:`RecordingMetadata` with the fields bioamla datasets care about: when, where,
and which device.

Only WAV files carry this metadata. Every function raises :class:`NotFoundError` for
a missing file, :class:`InvalidInputError` for a non-WAV path, and
:class:`AudioLoadError` when the file cannot be parsed.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from riffy import RecordingMetadata as RiffyRecordingMetadata

#: Column order used by :meth:`RecordingMetadata.to_dict` and CSV writers.
RECORDING_METADATA_FIELDS = (
    "recorded_at",
    "latitude",
    "longitude",
    "device_make",
    "device_model",
    "device_id",
    "gain",
    "temperature_c",
    "battery_voltage",
    "metadata_sources",
)


@dataclass
class RecordingMetadata:
    """
    Recorder-embedded metadata flattened across standards.

    Every field is ``None`` when no standard in the file provides it. When several
    do, GUANO wins, then AudioMoth, then ``bext``, then WAMD.

    Attributes:
        recorded_at: Start of the recording. Timezone-aware when the recorder
            stated an offset (GUANO, AudioMoth); naive for ``bext`` and WAMD.
        latitude: WGS84 latitude in decimal degrees.
        longitude: WGS84 longitude in decimal degrees.
        device_make: Manufacturer, e.g. ``"Open Acoustic Devices"``.
        device_model: Model, e.g. ``"AudioMoth"``, ``"SM4"``.
        device_id: Serial number or device ID.
        gain: Gain setting as the recorder wrote it, e.g. ``"medium"``, ``"2"``.
        temperature_c: Temperature in degrees Celsius at recording time.
        battery_voltage: Battery voltage at recording time.
        sources: Standards found in the file, e.g. ``("info", "audiomoth")``.
    """

    recorded_at: datetime | None = None
    latitude: float | None = None
    longitude: float | None = None
    device_make: str | None = None
    device_model: str | None = None
    device_id: str | None = None
    gain: str | None = None
    temperature_c: float | None = None
    battery_voltage: float | None = None
    sources: tuple[str, ...] = field(default_factory=tuple)

    @property
    def is_empty(self) -> bool:
        """``True`` when the file carried no recognised recorder metadata."""
        return not self.sources

    def to_dict(self) -> dict[str, Any]:
        """Flat, CSV/JSON-friendly dict in :data:`RECORDING_METADATA_FIELDS` order.

        ``recorded_at`` becomes an ISO 8601 string and ``sources`` a ``+``-joined
        string under the key ``metadata_sources``.
        """
        d = asdict(self)
        d["recorded_at"] = self.recorded_at.isoformat() if self.recorded_at else None
        d["metadata_sources"] = "+".join(d.pop("sources"))
        return {k: d[k] for k in RECORDING_METADATA_FIELDS}

    @classmethod
    def from_riffy(cls, meta: RiffyRecordingMetadata) -> RecordingMetadata:
        """Flatten riffy's per-standard view into one record."""
        guano, moth, bext, wamd = meta.guano, meta.audiomoth, meta.bext, meta.wamd

        recorded_at = _first(
            guano.timestamp if guano else None,
            moth.timestamp if moth else None,
            _bext_datetime(bext) if bext else None,
            _parse_iso(wamd.timestamp) if wamd else None,
        )
        position = _first(
            guano.loc_position if guano else None,
            wamd.loc_position if wamd else None,
        )
        latitude, longitude = position if position else (None, None)

        return cls(
            recorded_at=recorded_at,
            latitude=latitude,
            longitude=longitude,
            device_make=_first(
                guano.make if guano else None,
                "Open Acoustic Devices" if moth else None,
                "Wildlife Acoustics" if wamd else None,
            ),
            device_model=_first(
                guano.model if guano else None,
                "AudioMoth" if moth else None,
                wamd.model if wamd else None,
            ),
            device_id=_first(
                guano.serial if guano else None,
                moth.device_id if moth else None,
                wamd.serial if wamd else None,
            ),
            gain=_first(
                _guano_gain(guano) if guano else None,
                moth.gain if moth else None,
            ),
            temperature_c=_first(
                guano.temperature_ext if guano else None,
                guano.temperature_int if guano else None,
                moth.temperature_c if moth else None,
            ),
            battery_voltage=moth.battery_voltage if moth else None,
            sources=tuple(meta.sources),
        )


def _guano_gain(guano: Any) -> str | None:
    """Gain from Wildlife Acoustics' ``WA|Song Meter|Audio settings`` JSON field."""
    raw = guano.get("WA", "Song Meter|Audio settings")
    if not raw:
        return None
    try:
        settings = json.loads(raw)
    except ValueError:
        return None
    if isinstance(settings, list):
        settings = settings[0] if settings else None
    if isinstance(settings, dict) and settings.get("gain") is not None:
        return str(settings["gain"])
    return None


#: Wildlife Acoustics / AudioMoth style file names: ``PREFIX_YYYYMMDD_HHMMSS.wav``.
_FILENAME_TIMESTAMP_RE = re.compile(r"(?:^|_)(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})(?=\.|_|$)")


def timestamp_from_filename(filepath: str | Path) -> datetime | None:
    """Naive datetime from a ``PREFIX_YYYYMMDD_HHMMSS`` file name, or ``None``.

    Song Meter and AudioMoth units name files this way in local time. Used only as
    a fallback when no embedded standard carries a timestamp.
    """
    m = _FILENAME_TIMESTAMP_RE.search(Path(filepath).stem)
    if not m:
        return None
    try:
        return datetime(*(int(g) for g in m.groups()))
    except ValueError:
        return None


def _first(*values: Any) -> Any:
    """First value that is not ``None`` (empty strings count as missing)."""
    for v in values:
        if v is not None and v != "":
            return v
    return None


def _parse_iso(text: str | None) -> datetime | None:
    if not text:
        return None
    try:
        return datetime.fromisoformat(text.strip())
    except ValueError:
        return None


def _bext_datetime(bext: Any) -> datetime | None:
    """Combine bext ``origination_date`` (yyyy-mm-dd) and ``origination_time``.

    The time separator is ``:`` per EBU 3285 but ``-`` and ``_`` appear in the wild.
    """
    date, time = (bext.origination_date or "").strip(), (bext.origination_time or "").strip()
    if not date:
        return None
    time = time.replace("-", ":").replace("_", ":") or "00:00:00"
    try:
        return datetime.fromisoformat(f"{date}T{time}")
    except ValueError:
        return None


def get_recording_metadata(filepath: str) -> RecordingMetadata:
    """
    Read recorder-embedded metadata from a WAV file, flattened.

    Args:
        filepath: Path to a WAV file.

    When no embedded standard carries a timestamp but the file is named
    ``PREFIX_YYYYMMDD_HHMMSS.wav`` (Song Meter, AudioMoth), ``recorded_at`` is taken
    from the name as a naive datetime and ``"filename"`` is added to ``sources``.

    Returns:
        :class:`RecordingMetadata`. Check :attr:`RecordingMetadata.is_empty` to
        tell "no metadata in file" from "fields the recorder did not write".

    Raises:
        NotFoundError: If the file does not exist.
        InvalidInputError: If the path is not a WAV file.
        AudioLoadError: If the file cannot be parsed.
    """
    from bioamla.audio._riffy import read_recorder_metadata

    meta = RecordingMetadata.from_riffy(read_recorder_metadata(filepath))
    if meta.recorded_at is None:
        from_name = timestamp_from_filename(filepath)
        if from_name is not None:
            meta.recorded_at = from_name
            meta.sources = (*meta.sources, "filename")
    return meta


def dump_recording_metadata(filepath: str) -> dict[str, Any]:
    """
    Every recorder-metadata field in a WAV file, per standard, JSON-serializable.

    Use this when the flat :func:`get_recording_metadata` view drops something you
    need (custom GUANO fields, bext coding history, iXML). Keys: ``file``,
    ``riff_form``, ``format``, ``sources`` and one entry per standard (``guano``,
    ``info``, ``bext``, ``audiomoth``, ``wamd``, ``ixml``), ``None`` when absent.

    Raises:
        NotFoundError: If the file does not exist.
        InvalidInputError: If the path is not a WAV file.
        AudioLoadError: If the file cannot be parsed.
    """
    from bioamla.audio._riffy import dump_recorder_metadata

    return dump_recorder_metadata(filepath)


__all__ = [
    "RECORDING_METADATA_FIELDS",
    "RecordingMetadata",
    "dump_recording_metadata",
    "get_recording_metadata",
    "timestamp_from_filename",
]
