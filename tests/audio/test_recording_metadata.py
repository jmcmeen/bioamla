"""Tests for bioamla.audio.metadata (recorder metadata via riffy)."""

import wave
from datetime import datetime, timedelta, timezone

import pytest
from riffy import BextMetadata, GuanoMetadata, InfoMetadata, WAVParser
from riffy import RecordingMetadata as RiffyRecordingMetadata
from riffy.metadata.audiomoth import AudioMothMetadata
from riffy.metadata.wamd import WamdMetadata

from bioamla.audio import (
    RECORDING_METADATA_FIELDS,
    RecordingMetadata,
    dump_recording_metadata,
    get_recording_metadata,
)
from bioamla.exceptions import AudioLoadError, InvalidInputError, NotFoundError

AUDIOMOTH_COMMENT = (
    "Recorded at 19:30:00 12/11/2021 (UTC) by AudioMoth 248D9B045EC9EE79 "
    "at medium gain while battery was 4.1V and temperature was 14.0C."
)


def _write_pcm_wav(path, sample_rate=16000, seconds=0.1) -> None:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(b"\x00" * int(sample_rate * seconds) * 2)


@pytest.fixture
def plain_wav(tmp_path):
    path = tmp_path / "plain.wav"
    _write_pcm_wav(path)
    return str(path)


@pytest.fixture
def guano_wav(tmp_path):
    src = tmp_path / "src.wav"
    _write_pcm_wav(src)
    parser = WAVParser(src)
    g = GuanoMetadata(version="1.0")
    g.timestamp = datetime(2024, 5, 3, 21, 15, 0, tzinfo=timezone(timedelta(hours=-4)))
    g.loc_position = (36.52, -82.21)
    g.make = "Wildlife Acoustics"
    g.model = "SM4BAT"
    g.serial = "S4U09876"
    g.temperature_ext = 18.5
    g.write_to_parser(parser)
    out = tmp_path / "guano.wav"
    parser.write_wav(out)
    return str(out)


@pytest.fixture
def audiomoth_wav(tmp_path):
    src = tmp_path / "src.wav"
    _write_pcm_wav(src)
    parser = WAVParser(src)
    info = InfoMetadata()
    info.set("IART", "AudioMoth 248D9B045EC9EE79")
    info.set("ICMT", AUDIOMOTH_COMMENT)
    info.write_to_parser(parser)
    out = tmp_path / "moth.wav"
    parser.write_wav(out)
    return str(out)


class TestGetRecordingMetadata:
    def test_plain_wav_is_empty(self, plain_wav: str) -> None:
        meta = get_recording_metadata(plain_wav)
        assert isinstance(meta, RecordingMetadata)
        assert meta.is_empty
        assert meta.sources == ()
        assert meta.recorded_at is None
        assert meta.latitude is None

    def test_guano(self, guano_wav: str) -> None:
        meta = get_recording_metadata(guano_wav)
        assert meta.sources == ("guano",)
        assert meta.recorded_at == datetime(
            2024, 5, 3, 21, 15, 0, tzinfo=timezone(timedelta(hours=-4))
        )
        assert meta.latitude == pytest.approx(36.52)
        assert meta.longitude == pytest.approx(-82.21)
        assert meta.device_make == "Wildlife Acoustics"
        assert meta.device_model == "SM4BAT"
        assert meta.device_id == "S4U09876"
        assert meta.temperature_c == pytest.approx(18.5)
        assert meta.gain is None
        assert meta.battery_voltage is None

    def test_audiomoth(self, audiomoth_wav: str) -> None:
        meta = get_recording_metadata(audiomoth_wav)
        assert meta.sources == ("info", "audiomoth")
        assert meta.recorded_at == datetime(2021, 11, 12, 19, 30, 0, tzinfo=timezone.utc)
        assert meta.device_make == "Open Acoustic Devices"
        assert meta.device_model == "AudioMoth"
        assert meta.device_id == "248D9B045EC9EE79"
        assert meta.gain == "medium"
        assert meta.battery_voltage == pytest.approx(4.1)
        assert meta.temperature_c == pytest.approx(14.0)
        assert meta.latitude is None

    def test_missing_raises(self, tmp_path) -> None:
        with pytest.raises(NotFoundError):
            get_recording_metadata(str(tmp_path / "nope.wav"))

    def test_non_wav_raises(self, tmp_path) -> None:
        path = tmp_path / "clip.flac"
        path.write_bytes(b"fLaC")
        with pytest.raises(InvalidInputError):
            get_recording_metadata(str(path))

    def test_garbage_raises(self, tmp_path) -> None:
        path = tmp_path / "junk.wav"
        path.write_bytes(b"definitely not riff")
        with pytest.raises(AudioLoadError):
            get_recording_metadata(str(path))


class TestToDict:
    def test_field_order_and_serialization(self, audiomoth_wav: str) -> None:
        d = get_recording_metadata(audiomoth_wav).to_dict()
        assert tuple(d) == RECORDING_METADATA_FIELDS
        assert d["recorded_at"] == "2021-11-12T19:30:00+00:00"
        assert d["metadata_sources"] == "info+audiomoth"
        assert d["latitude"] is None

    def test_empty(self) -> None:
        d = RecordingMetadata().to_dict()
        assert d["recorded_at"] is None
        assert d["metadata_sources"] == ""


class TestFromRiffyPrecedence:
    def test_guano_beats_audiomoth(self) -> None:
        g = GuanoMetadata(version="1.0")
        g.timestamp = datetime(2020, 1, 1, tzinfo=timezone.utc)
        g.serial = "G1"
        moth = AudioMothMetadata.from_comment(AUDIOMOTH_COMMENT)
        meta = RecordingMetadata.from_riffy(RiffyRecordingMetadata(guano=g, audiomoth=moth))
        assert meta.recorded_at == datetime(2020, 1, 1, tzinfo=timezone.utc)
        assert meta.device_id == "G1"
        assert meta.device_model == "AudioMoth"
        # fields GUANO lacks still fall through to AudioMoth
        assert meta.gain == "medium"
        assert meta.battery_voltage == pytest.approx(4.1)
        assert meta.sources == ("guano", "audiomoth")

    def test_bext_datetime(self) -> None:
        bext = BextMetadata(origination_date="2023-07-04", origination_time="05-30-00")
        meta = RecordingMetadata.from_riffy(RiffyRecordingMetadata(bext=bext))
        assert meta.recorded_at == datetime(2023, 7, 4, 5, 30, 0)
        assert meta.sources == ("bext",)

    def test_bext_bad_date_ignored(self) -> None:
        bext = BextMetadata(origination_date="not a date", origination_time="")
        meta = RecordingMetadata.from_riffy(RiffyRecordingMetadata(bext=bext))
        assert meta.recorded_at is None

    def test_wamd_position_and_model(self) -> None:
        w = WamdMetadata()
        w.model = "SM4"
        w.serial = "SMU01234"
        w.loc_position = (10.5, -84.0)
        meta = RecordingMetadata.from_riffy(RiffyRecordingMetadata(wamd=w))
        assert meta.device_make == "Wildlife Acoustics"
        assert meta.device_model == "SM4"
        assert meta.device_id == "SMU01234"
        assert meta.latitude == pytest.approx(10.5)
        assert meta.longitude == pytest.approx(-84.0)

    def test_empty_strings_count_as_missing(self) -> None:
        g = GuanoMetadata(version="1.0")
        g.make = ""
        moth = AudioMothMetadata.from_comment(AUDIOMOTH_COMMENT)
        meta = RecordingMetadata.from_riffy(RiffyRecordingMetadata(guano=g, audiomoth=moth))
        assert meta.device_make == "Open Acoustic Devices"


class TestDumpRecordingMetadata:
    def test_json_safe(self, guano_wav: str) -> None:
        import json

        d = dump_recording_metadata(guano_wav)
        assert d["sources"] == ["guano"]
        assert d["guano"] is not None
        assert d["audiomoth"] is None
        json.dumps(d)  # must not raise

    def test_non_wav_raises(self, tmp_path) -> None:
        path = tmp_path / "clip.mp3"
        path.write_bytes(b"ID3")
        with pytest.raises(InvalidInputError):
            dump_recording_metadata(str(path))
