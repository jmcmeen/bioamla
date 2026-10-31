"""Tests for bioamla.datasets.add_recording_metadata."""

import csv
import wave
from datetime import UTC, datetime, timedelta, timezone

import pytest
from riffy import GuanoMetadata, WAVParser

from bioamla.datasets import add_recording_metadata
from bioamla.datasets._metadata import write_metadata_csv
from bioamla.exceptions import NotFoundError


def _pcm_wav(path) -> None:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(8000)
        w.writeframes(b"\x00" * 800)


def _guano_wav(path, *, serial: str, when: datetime, position=(36.52, -82.21)) -> None:
    src = path.with_name("src_" + path.name)
    _pcm_wav(src)
    parser = WAVParser(src)
    g = GuanoMetadata(version="1.0")
    g.make = "Wildlife Acoustics, Inc."
    g.model = "Song Meter Micro"
    g.serial = serial
    g.timestamp = when
    g.loc_position = position
    g.write_to_parser(parser)
    parser.write_wav(path)
    src.unlink()


@pytest.fixture
def dataset(tmp_path):
    (tmp_path / "site1").mkdir()
    tz = timezone(timedelta(hours=-4))
    _guano_wav(
        tmp_path / "site1" / "r1.wav", serial="SMU00001", when=datetime(2023, 6, 14, 5, tzinfo=tz)
    )
    _guano_wav(
        tmp_path / "site1" / "r2.wav",
        serial="SMU00002",
        when=datetime(2023, 6, 15, 5, tzinfo=tz),
        position=(36.60, -82.30),
    )
    _pcm_wav(tmp_path / "plain.wav")
    (tmp_path / "clip.flac").write_bytes(b"fLaC")
    (tmp_path / "broken.wav").write_bytes(b"not riff")
    rows = [
        {"file_name": "site1/r1.wav", "label": "frog"},
        {"file_name": "site1/r2.wav", "label": "frog"},
        {"file_name": "plain.wav", "label": "bird"},
        {"file_name": "clip.flac", "label": "bird"},
        {"file_name": "broken.wav", "label": "bird"},
        {"file_name": "missing.wav", "label": "bird"},
    ]
    write_metadata_csv(tmp_path / "metadata.csv", rows, merge_existing=False)
    return tmp_path


def _read(path):
    with open(path, newline="") as f:
        return {r["file_name"]: r for r in csv.DictReader(f)}


class TestAddRecordingMetadata:
    def test_fills_columns(self, dataset) -> None:
        counts = add_recording_metadata(str(dataset))
        assert counts["rows"] == 6
        assert counts["updated"] == 3  # r1, r2, plain
        assert counts["with_metadata"] == 2
        assert counts["skipped"] == 3  # flac, broken, missing
        assert len(counts["errors"]) == 1 and "broken.wav" in counts["errors"][0]

        rows = _read(dataset / "metadata.csv")
        assert rows["site1/r1.wav"]["recorded_at"] == "2023-06-14T05:00:00-04:00"
        assert rows["site1/r1.wav"]["device_id"] == "SMU00001"
        assert rows["site1/r1.wav"]["device_model"] == "Song Meter Micro"
        assert float(rows["site1/r2.wav"]["latitude"]) == pytest.approx(36.60)
        assert rows["site1/r1.wav"]["metadata_sources"] == "guano"
        assert rows["plain.wav"]["metadata_sources"] == ""
        assert rows["plain.wav"]["recorded_at"] == ""
        assert rows["clip.flac"]["recorded_at"] == ""
        # existing columns survive, recorder columns follow them in documented order
        assert rows["site1/r1.wav"]["label"] == "frog"
        with open(dataset / "metadata.csv", newline="") as f:
            header = next(csv.reader(f))
        assert header.index("label") < header.index("recorded_at")
        # latitude/longitude are catalog columns already and keep their place;
        # the rest follow recorded_at in RECORDING_METADATA_FIELDS order.
        assert header[header.index("recorded_at") :] == [
            "recorded_at",
            "device_make",
            "device_model",
            "device_id",
            "gain",
            "temperature_c",
            "battery_voltage",
            "metadata_sources",
        ]
        assert rows["missing.wav"]["label"] == "bird"

    def test_idempotent_and_overwrite(self, dataset) -> None:
        add_recording_metadata(str(dataset))
        second = add_recording_metadata(str(dataset))
        assert second["updated"] == 1  # only plain.wav, which has no sources recorded
        third = add_recording_metadata(str(dataset), overwrite=True)
        assert third["updated"] == 3

    def test_absolute_paths(self, tmp_path) -> None:
        tz = UTC
        wav = tmp_path / "elsewhere" / "x.wav"
        wav.parent.mkdir()
        _guano_wav(wav, serial="S1", when=datetime(2024, 1, 1, tzinfo=tz))
        ds = tmp_path / "ds"
        ds.mkdir()
        write_metadata_csv(
            ds / "metadata.csv", [{"file_name": str(wav), "label": "a"}], merge_existing=False
        )
        counts = add_recording_metadata(str(ds))
        assert counts["with_metadata"] == 1
        assert _read(ds / "metadata.csv")[str(wav)]["device_id"] == "S1"

    def test_missing_csv_raises(self, tmp_path) -> None:
        with pytest.raises(NotFoundError):
            add_recording_metadata(str(tmp_path))

    def test_nothing_updated_leaves_file_alone(self, tmp_path) -> None:
        p = tmp_path / "metadata.csv"
        write_metadata_csv(p, [{"file_name": "clip.flac", "label": "a"}], merge_existing=False)
        before = p.read_text()
        counts = add_recording_metadata(str(tmp_path))
        assert counts["updated"] == 0
        assert p.read_text() == before

    def test_manifest_from_enriched_dataset(self, dataset) -> None:
        from bioamla.datasets import build_manifest_from_metadata

        add_recording_metadata(str(dataset))
        m = build_manifest_from_metadata(str(dataset), name="scp")
        assert m.recorded_from == "2023-06-14T05:00:00-04:00"
        assert m.recorded_to == "2023-06-15T05:00:00-04:00"
        assert [d["device_id"] for d in m.devices] == ["SMU00001", "SMU00002"]
        assert len(m.sites) == 2
