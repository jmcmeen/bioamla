"""Tests for coordinate redaction (bioamla.datasets.redact) and push_dataset's use of it."""

import csv
import io
import json
from pathlib import Path

import pytest
from click.testing import CliRunner

import bioamla.catalogs.huggingface as hf
from bioamla.catalogs._models import PushResult
from bioamla.cli.commands.catalogs import catalogs
from bioamla.datasets import (
    redact_rows,
    redact_sites,
    redacted_dataset_files,
    write_dataset_card,
)
from bioamla.exceptions import InvalidInputError

FIELDS = ["file_name", "label", "latitude", "longitude", "recorded_at"]
ROWS = [
    {
        "file_name": "a.wav",
        "label": "frog",
        "latitude": "35.123456",
        "longitude": "-82.987654",
        "recorded_at": "2026-05-01T21:00:00",
    },
    {"file_name": "b.wav", "label": "frog", "latitude": "", "longitude": "", "recorded_at": ""},
    {
        "file_name": "c.wav",
        "label": "toad",
        "latitude": "n/a",
        "longitude": "-82.9",
        "recorded_at": "",
    },
]


class _RecordingApi:
    """Fake HfApi that snapshots each uploaded folder (staging dirs are temporary)."""

    def __init__(self) -> None:
        self.uploads: list[dict] = []

    def create_repo(self, **kwargs):
        pass

    def upload_folder(self, *, folder_path, **kwargs):
        base = Path(folder_path)
        files = {
            p.relative_to(base).as_posix(): p.read_bytes() for p in base.rglob("*") if p.is_file()
        }
        self.uploads.append({"files": files, **kwargs})

    upload_large_folder = upload_folder


def _write_csv(path, rows=ROWS, fields=FIELDS):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(content: bytes):
    reader = csv.DictReader(io.StringIO(content.decode()))
    return reader.fieldnames, list(reader)


class TestRedactRows:
    def test_round(self) -> None:
        rows, fields = redact_rows(ROWS, FIELDS, "round", 2)
        assert fields == FIELDS
        assert (rows[0]["latitude"], rows[0]["longitude"]) == ("35.12", "-82.99")
        assert rows[1]["latitude"] == ""
        assert rows[2]["latitude"] == ""  # unparseable: blanked, never published
        assert ROWS[0]["latitude"] == "35.123456"  # input untouched

    def test_round_to_whole_degrees(self) -> None:
        rows, _ = redact_rows(ROWS, FIELDS, "round", 0)
        assert (rows[0]["latitude"], rows[0]["longitude"]) == ("35", "-83")

    def test_strip(self) -> None:
        rows, fields = redact_rows(ROWS, FIELDS, "strip")
        assert fields == ["file_name", "label", "recorded_at"]
        assert all("latitude" not in r and "longitude" not in r for r in rows)

    def test_keep(self) -> None:
        rows, fields = redact_rows(ROWS, FIELDS, "keep")
        assert rows == ROWS and fields == FIELDS

    @pytest.mark.parametrize(("mode", "decimals"), [("blur", 2), ("round", -1), ("round", 7)])
    def test_bad_arguments(self, mode, decimals) -> None:
        with pytest.raises(InvalidInputError):
            redact_rows(ROWS, FIELDS, mode, decimals)


class TestRedactSites:
    SITES = [
        {"latitude": 35.12, "longitude": -82.98, "files": 3},
        {"latitude": 35.14, "longitude": -82.96, "files": 2},
        {"latitude": 36.5, "longitude": -81.0, "files": 1},
    ]

    def test_round_merges_sites(self) -> None:
        assert redact_sites(self.SITES, "round", 1) == [
            {"latitude": 35.1, "longitude": -83.0, "files": 5},
            {"latitude": 36.5, "longitude": -81.0, "files": 1},
        ]

    def test_strip_drops_sites(self) -> None:
        assert redact_sites(self.SITES, "strip") == []


class TestRedactedDatasetFiles:
    def _dataset(self, tmp_path):
        _write_csv(tmp_path / "metadata.csv")
        (tmp_path / "labels.csv").write_text("label,id\nfrog,0\n")  # no coordinates
        return tmp_path

    def test_keep_changes_nothing(self, tmp_path) -> None:
        assert redacted_dataset_files(self._dataset(tmp_path), "keep") == {}

    def test_csv_and_generated_card(self, tmp_path) -> None:
        d = self._dataset(tmp_path)
        write_dataset_card(str(d))
        before = (d / "metadata.csv").read_bytes()

        files = redacted_dataset_files(d, "round", 1)

        assert set(files) == {"metadata.csv", "README.md"}
        fields, rows = _read_csv(files["metadata.csv"])
        assert fields == FIELDS
        assert (rows[0]["latitude"], rows[0]["longitude"]) == ("35.1", "-83.0")
        assert "per-file coordinates published to 0.1 deg" in files["README.md"].decode()
        assert (d / "metadata.csv").read_bytes() == before  # local copy untouched

    def test_manifest_sites(self, tmp_path) -> None:
        d = self._dataset(tmp_path)
        (d / "dataset.json").write_text(
            json.dumps(
                {"name": "demo", "sites": [{"latitude": 35.12, "longitude": -82.99, "files": 1}]}
            )
        )
        manifest = json.loads(redacted_dataset_files(d, "strip")["dataset.json"])
        assert manifest["sites"] == []
        assert manifest["name"] == "demo"

    def test_hand_written_readme_left_alone(self, tmp_path) -> None:
        d = self._dataset(tmp_path)
        (d / "README.md").write_text("# My frogs\n")
        assert "README.md" not in redacted_dataset_files(d, "strip")


class TestPushDatasetRedaction:
    def test_round_uploads_redacted_copies_first(self, tmp_path, monkeypatch) -> None:
        _write_csv(tmp_path / "metadata.csv")
        _pcm_wav(tmp_path / "a.wav")  # no embedded location: uploaded as is
        api = _RecordingApi()
        monkeypatch.setattr(hf, "_get_hf_api", lambda: api)

        hf.push_dataset(str(tmp_path), "me/frogs", coordinates="round", coordinate_decimals=2)

        staged, dataset = api.uploads
        assert list(staged["files"]) == ["metadata.csv"]
        _, rows = _read_csv(staged["files"]["metadata.csv"])
        assert rows[0]["latitude"] == "35.12"
        assert dataset["ignore_patterns"] == ["metadata.csv"]

    def test_keep_uploads_folder_as_is(self, tmp_path, monkeypatch) -> None:
        _write_csv(tmp_path / "metadata.csv")
        api = _RecordingApi()
        monkeypatch.setattr(hf, "_get_hf_api", lambda: api)

        hf.push_dataset(str(tmp_path), "me/frogs")

        (dataset,) = api.uploads
        assert "ignore_patterns" not in dataset

    def test_bad_mode_rejected_before_upload(self, tmp_path) -> None:
        with pytest.raises(InvalidInputError):
            hf.push_dataset(str(tmp_path), "me/frogs", coordinates="fuzz")


class TestPushDatasetCliCoordinates:
    def _fake_push(self, calls):
        def fake_push(path, repo_id, private=False, commit_message=None, **kwargs):
            calls.update(kwargs)
            return PushResult(
                repo_id=repo_id,
                repo_type="dataset",
                url=f"https://huggingface.co/datasets/{repo_id}",
                files_uploaded=1,
                total_size_bytes=1,
            )

        return fake_push

    def test_warns_when_coordinates_kept(self, tmp_path, monkeypatch) -> None:
        _write_csv(tmp_path / "metadata.csv")
        calls: dict = {}
        monkeypatch.setattr(hf, "push_dataset", self._fake_push(calls))
        res = CliRunner().invoke(catalogs, ["hf", "push-dataset", str(tmp_path), "me/frogs"])
        assert res.exit_code == 0, res.output
        assert "Per-file locations in the metadata" in res.output
        assert calls["coordinates"] == "keep"

    def test_passes_mode_and_warns_about_wav_gps(self, tmp_path, monkeypatch) -> None:
        import bioamla.datasets as datasets

        _write_csv(tmp_path / "metadata.csv")
        calls: dict = {}
        monkeypatch.setattr(hf, "push_dataset", self._fake_push(calls))
        monkeypatch.setattr(datasets, "unredacted_location_files", lambda path: ["a.wav"])
        res = CliRunner().invoke(
            catalogs,
            [
                "hf",
                "push-dataset",
                str(tmp_path),
                "me/frogs",
                "--coordinates",
                "round",
                "--coordinate-decimals",
                "1",
            ],
        )
        assert res.exit_code == 0, res.output
        assert (calls["coordinates"], calls["coordinate_decimals"]) == ("round", 1)
        assert "JSON/text, Parquet" in res.output
        assert "per-file locations" not in res.output.lower()


class TestUnredactedLocationFiles:
    def test_lists_json_and_parquet_not_wavs(self, tmp_path) -> None:
        from bioamla.datasets import unredacted_location_files

        _guano_wav(tmp_path / "sub" / "located.wav", position=(36.52, -82.21))
        (tmp_path / "notes.json").write_text('{"Latitude": 35.1}')
        (tmp_path / "table.parquet").write_bytes(b"PAR1")
        assert unredacted_location_files(tmp_path) == ["notes.json", "table.parquet"]


class TestRedactionEdgeCases:
    def test_column_aliases_and_location_text(self) -> None:
        fields = ["file_name", "Lat", "LNG", "location"]
        rows = [
            {"file_name": "a.wav", "Lat": "35.1234", "LNG": "-82.9876", "location": "35.12,-82.98"}
        ]
        rounded, kept = redact_rows(rows, fields, "round", 1)
        assert kept == ["file_name", "Lat", "LNG"]  # place text dropped even when rounding
        assert (rounded[0]["Lat"], rounded[0]["LNG"]) == ("35.1", "-83.0")
        _, stripped = redact_rows(rows, fields, "strip")
        assert stripped == ["file_name"]

    def test_tsv_redacted(self, tmp_path) -> None:
        (tmp_path / "sites.tsv").write_text("file_name\tlatitude\na.wav\t35.123\n")
        files = redacted_dataset_files(tmp_path, "round", 1)
        assert files["sites.tsv"].decode() == "file_name\tlatitude\na.wav\t35.1\n"

    def test_unreadable_table_is_an_error_not_an_upload(self, tmp_path) -> None:
        from bioamla.exceptions import DatasetError

        (tmp_path / "metadata.csv").write_bytes(
            "file_name,latitude\nca\xf1on.wav,35.1\n".encode("latin-1")
        )
        with pytest.raises(DatasetError, match="metadata.csv"):
            redacted_dataset_files(tmp_path, "strip")

    def test_row_with_extra_fields_is_an_error(self, tmp_path) -> None:
        from bioamla.exceptions import DatasetError

        (tmp_path / "metadata.csv").write_text("file_name,latitude\na.wav,35.1,oops\n")
        with pytest.raises(DatasetError, match="more fields"):
            redacted_dataset_files(tmp_path, "round")

    def test_card_without_a_manifest_source_loses_its_sites_line(self, tmp_path) -> None:
        _write_csv(tmp_path / "recordings.csv")
        (tmp_path / "README.md").write_text(
            "# demo\n- **Sites:** 3 (coordinates rounded to 0.01 deg)\n_Generated by bioamla._\n"
        )
        card = redacted_dataset_files(tmp_path, "round", 0)["README.md"].decode()
        assert "Sites" not in card
        assert "_Generated by bioamla._" in card

    def test_glob_characters_in_names_still_excluded(self, tmp_path, monkeypatch) -> None:
        import fnmatch

        _write_csv(tmp_path / "train[2026].csv")
        api = _RecordingApi()
        monkeypatch.setattr(hf, "_get_hf_api", lambda: api)
        hf.push_dataset(str(tmp_path), "me/frogs", coordinates="strip")
        (pattern,) = api.uploads[1]["ignore_patterns"]
        assert fnmatch.fnmatch("train[2026].csv", pattern)

    def test_keep_push_tolerates_unreadable_csv(self, tmp_path, monkeypatch) -> None:
        (tmp_path / "export.csv").write_bytes("name\nca\xf1on\n".encode("latin-1"))
        calls: dict = {}
        monkeypatch.setattr(hf, "push_dataset", TestPushDatasetCliCoordinates()._fake_push(calls))
        res = CliRunner().invoke(catalogs, ["hf", "push-dataset", str(tmp_path), "me/frogs"])
        assert res.exit_code == 0, res.output


def _pcm_wav(path, frames: int = 1600) -> None:
    import wave

    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(bytes(range(256)) * (frames * 2 // 256))


def _guano_wav(path, position=(36.523456, -82.218765), extra=None):
    from riffy import GuanoMetadata, WAVParser

    _pcm_wav(path)
    parser = WAVParser(path)
    guano = GuanoMetadata(version="1.0")
    guano.loc_position = position
    guano.loc_elevation = 812.0
    guano.loc_accuracy = 4.0
    guano.make = "Wildlife Acoustics"
    for (namespace, key), value in (extra or {}).items():
        guano.set(namespace, key, value)
    guano.write_to_parser(parser)
    parser.write_wav(path, overwrite=True)
    return path


def _wamd_wav(path, waypoint="WGS84,35.123456,-82.567891,512.0"):
    from riffy import WamdMetadata, WAVParser
    from riffy.metadata.wamd import GPSFIRST_ID, GPSTRACK_ID, MODEL_ID

    _pcm_wav(path)
    parser = WAVParser(path)
    wamd = WamdMetadata()
    wamd.set_text(MODEL_ID, "SM4BAT-FS")
    wamd.set_text(GPSFIRST_ID, waypoint)
    wamd.set_raw(GPSTRACK_ID, b"\x01\x02track")
    wamd.write_to_parser(parser)
    parser.write_wav(path, overwrite=True)
    return path


def _chunks(path) -> dict[str, bytes]:
    """Every chunk's payload by id (last wins), read without riffy's parser."""
    import struct

    data = Path(path).read_bytes()
    out, pos = {}, 12
    while pos + 8 <= len(data):
        cid, size = (
            data[pos : pos + 4].decode("latin-1"),
            struct.unpack_from("<I", data, pos + 4)[0],
        )
        if cid == "data" and size == 0xFFFFFFFF:
            size = struct.unpack_from("<Q", out["ds64"], 8)[0]
        out[cid] = data[pos + 8 : pos + 8 + size]
        pos += 8 + size + (size & 1)
    return out


def _guano(path):
    from riffy import GuanoMetadata

    return GuanoMetadata.from_bytes(_chunks(path)["guan"])


def _append_chunk(path, chunk_id: str, payload: bytes) -> None:
    """Append a chunk the way recorders do, fixing the RIFF (or RF64 ds64) size."""
    import struct

    pad = b"\x00" if len(payload) & 1 else b""
    with open(path, "r+b") as f:
        f.seek(0, 2)
        f.write(chunk_id.encode() + struct.pack("<I", len(payload)) + payload + pad)
        total = f.tell()
        f.seek(0)
        form = f.read(4)
        if form == b"RF64":
            f.seek(20)  # ds64 is the first chunk; its first field is the RIFF size
            f.write(struct.pack("<Q", total - 8))
        else:
            f.seek(4)
            f.write(struct.pack("<I", total - 8))


def _guano_payload(position=(36.523456, -82.218765)) -> bytes:
    from riffy import GuanoMetadata

    guano = GuanoMetadata(version="1.0")
    guano.loc_position = position
    guano.loc_elevation = 812.0
    guano.make = "Wildlife Acoustics"
    return guano.to_chunk_bytes()


def _audio_bytes(path) -> bytes:
    from riffy import WAVParser

    return WAVParser(path).get_chunk_bytes("data")


class TestRedactWavLocation:
    def test_guano_round(self, tmp_path) -> None:
        from riffy import GuanoMetadata, WAVParser

        from bioamla.datasets import redact_wav_location

        src = _guano_wav(
            tmp_path / "a.wav",
            extra={("WA|Song Meter", "GPS Lat"): "36.5234", ("", "Clock Drift"): "0.2"},
        )
        before = src.read_bytes()
        assert redact_wav_location(src, tmp_path / "out.wav", "round", 2)

        guano = GuanoMetadata.from_parser(WAVParser(tmp_path / "out.wav"))
        assert guano.loc_position == (36.52, -82.22)
        assert guano.loc_elevation is None and guano.loc_accuracy is None
        assert guano.get("WA|Song Meter", "GPS Lat") is None  # vendor GPS removed
        assert guano.get("", "Clock Drift") == "0.2"  # "Clock" isn't a location word
        assert guano.make == "Wildlife Acoustics"
        assert _audio_bytes(tmp_path / "out.wav") == _audio_bytes(src)
        assert src.read_bytes() == before

    def test_guano_strip(self, tmp_path) -> None:
        from riffy import GuanoMetadata, WAVParser

        from bioamla.datasets import redact_wav_location

        src = _guano_wav(tmp_path / "a.wav")
        assert redact_wav_location(src, tmp_path / "out.wav", "strip")
        guano = GuanoMetadata.from_parser(WAVParser(tmp_path / "out.wav"))
        assert not any("Loc" in key for _, key in guano.fields)
        assert guano.make == "Wildlife Acoustics"

    def test_wamd_round_drops_altitude_and_track(self, tmp_path) -> None:
        from riffy import WamdMetadata, WAVParser
        from riffy.metadata.wamd import GPSFIRST_ID, GPSTRACK_ID

        from bioamla.datasets import redact_wav_location

        src = _wamd_wav(tmp_path / "a.wav")
        assert redact_wav_location(src, tmp_path / "out.wav", "round", 1)
        wamd = WamdMetadata.from_parser(WAVParser(tmp_path / "out.wav"))
        assert wamd.get_text(GPSFIRST_ID) == "WGS84,35.1,-82.6"
        assert wamd.get_raw(GPSTRACK_ID) is None
        assert wamd.model == "SM4BAT-FS"

    def test_wamd_strip(self, tmp_path) -> None:
        from riffy import WamdMetadata, WAVParser

        from bioamla.datasets import redact_wav_location

        src = _wamd_wav(tmp_path / "a.wav")
        assert redact_wav_location(src, tmp_path / "out.wav", "strip")
        assert WamdMetadata.from_parser(WAVParser(tmp_path / "out.wav")).loc_position is None

    def test_wav_without_location_is_not_copied(self, tmp_path) -> None:
        from bioamla.datasets import redact_wav_location

        _pcm_wav(tmp_path / "plain.wav")
        assert not redact_wav_location(tmp_path / "plain.wav", tmp_path / "out.wav", "strip")
        assert not (tmp_path / "out.wav").exists()

    def test_unparseable_wav_is_an_error(self, tmp_path) -> None:
        from bioamla.datasets import redact_wav_location
        from bioamla.exceptions import DatasetError

        bad = tmp_path / "bad.wav"
        _pcm_wav(bad)
        with open(bad, "ab") as f:  # a guan chunk cut off mid-payload
            f.write(b"guan" + (400).to_bytes(4, "little") + b"GUANO|Version: 1.0\nLoc Posi")
        with pytest.raises(DatasetError, match="bad.wav.*[Ii]ncomplete chunk: guan"):
            redact_wav_location(bad, tmp_path / "out.wav", "strip")

    def test_not_a_wav_is_an_error(self, tmp_path) -> None:
        from bioamla.datasets import redact_wav_location
        from bioamla.exceptions import DatasetError

        (tmp_path / "bad.wav").write_bytes(b"ID3\x03junk")
        with pytest.raises(DatasetError, match="bad.wav"):
            redact_wav_location(tmp_path / "bad.wav", tmp_path / "out.wav", "strip")


class TestPushRedactsAudio:
    def test_located_wavs_uploaded_redacted(self, tmp_path, monkeypatch) -> None:
        ds = tmp_path / "ds"
        _guano_wav(ds / "frog" / "a.wav")
        _pcm_wav(ds / "frog" / "plain.wav")
        api = _RecordingApi()
        monkeypatch.setattr(hf, "_get_hf_api", lambda: api)

        hf.push_dataset(str(ds), "me/frogs", coordinates="round", coordinate_decimals=1)

        staged, dataset = api.uploads
        assert list(staged["files"]) == ["frog/a.wav"]
        copy = tmp_path / "copy.wav"
        copy.write_bytes(staged["files"]["frog/a.wav"])
        assert _guano(copy).loc_position == (36.5, -82.2)
        assert dataset["ignore_patterns"] == ["frog/a.wav"]  # plain.wav goes up untouched

    def test_keep_push_warns_about_wav_gps(self, tmp_path, monkeypatch) -> None:
        _guano_wav(tmp_path / "a.wav")
        calls: dict = {}
        monkeypatch.setattr(hf, "push_dataset", TestPushDatasetCliCoordinates()._fake_push(calls))
        res = CliRunner().invoke(catalogs, ["hf", "push-dataset", str(tmp_path), "me/frogs"])
        assert res.exit_code == 0, res.output
        assert "recorder metadata" in res.output


class TestStreamingRewrite:
    """Files riffy's full parser rejects or would load whole: redaction streams them."""

    @pytest.mark.parametrize(
        ("subtype", "fmt"), [("FLOAT", "WAV"), ("PCM_24", "WAVEX"), ("PCM_16", "RF64")]
    )
    def test_non_pcm_and_rf64(self, tmp_path, subtype, fmt) -> None:
        import numpy as np
        import soundfile as sf

        from bioamla.datasets import redact_wav_location

        src = tmp_path / "a.wav"
        audio = np.random.default_rng(0).normal(0, 0.1, (8000, 2)).astype(np.float32)
        sf.write(src, audio, 16000, subtype=subtype, format=fmt)
        _append_chunk(src, "guan", _guano_payload())
        before = src.read_bytes()

        assert redact_wav_location(src, tmp_path / "out.wav", "round", 1)

        out = tmp_path / "out.wav"
        assert _guano(out).loc_position == (36.5, -82.2)
        assert _guano(out).loc_elevation is None
        assert _chunks(out)["data"] == _chunks(src)["data"]
        read_back, _ = sf.read(out, dtype="float32")
        np.testing.assert_array_equal(read_back, sf.read(src, dtype="float32")[0])
        assert src.read_bytes() == before

    def test_wamd_in_a_junk_chunk(self, tmp_path) -> None:
        from riffy import WamdMetadata
        from riffy.metadata.wamd import GPSFIRST_ID, VERSION_ID

        from bioamla.datasets import redact_wav_location

        src = tmp_path / "a.wav"
        _pcm_wav(src)
        wamd = WamdMetadata(chunk_id="junk")
        wamd.set_raw(VERSION_ID, b"\x01\x00")
        wamd.set_text(GPSFIRST_ID, "NAD27,35.1234,N,84.5678,W,300")
        _append_chunk(src, "junk", wamd.to_chunk_bytes())

        assert redact_wav_location(src, tmp_path / "out.wav", "round", 2)
        rounded = WamdMetadata.from_bytes(_chunks(tmp_path / "out.wav")["junk"], chunk_id="junk")
        assert rounded.get_text(GPSFIRST_ID) == "NAD27,35.12,-84.57"  # datum kept, altitude gone

    def test_plain_junk_padding_is_left_alone(self, tmp_path) -> None:
        from bioamla.datasets import redact_wav_location

        src = tmp_path / "a.wav"
        _pcm_wav(src)
        _append_chunk(src, "junk", b"\x00" * 28)
        assert not redact_wav_location(src, tmp_path / "out.wav", "strip")

    def test_chunks_after_an_odd_sized_guano_survive(self, tmp_path) -> None:
        from bioamla.datasets import redact_wav_location

        src = tmp_path / "a.wav"
        _pcm_wav(src)
        payload = _guano_payload()
        if not len(payload) & 1:
            payload += b" "
        _append_chunk(src, "guan", payload)
        _append_chunk(src, "LIST", b"INFOICMT\x05\x00\x00\x00note\x00\x00")
        assert redact_wav_location(src, tmp_path / "out.wav", "strip")
        assert _chunks(tmp_path / "out.wav")["LIST"] == _chunks(src)["LIST"]

    def test_field_word_matching(self) -> None:
        from bioamla.audio._riffy import is_location_field

        for key in ("Loc Position", "Loc Elevation", "GPS Fix", "Lat", "Lon", "Site Coords"):
            assert is_location_field(key), key
        for key in ("Long Description", "Mic Position", "Clock Drift", "Make", "Lon Filter"):
            assert not is_location_field(key), key

    def test_wave_suffix_is_redacted(self, tmp_path) -> None:
        from bioamla.datasets import redacted_audio_files

        _guano_wav(tmp_path / "ds" / "rec.wave")
        staged = redacted_audio_files(tmp_path / "ds", "strip", 2, tmp_path / "work")
        assert list(staged) == ["rec.wave"]


class TestMalformedRecorderMetadata:
    def test_guano_without_version_is_detected_and_redacted(self, tmp_path) -> None:
        from bioamla.datasets import redact_wav_location, wavs_with_location

        src = tmp_path / "a.wav"
        _pcm_wav(src)
        _append_chunk(src, "guan", b"Loc Position: 36.523456 -82.218765\nMake: X\n")

        assert wavs_with_location(tmp_path) == ["a.wav"]
        assert redact_wav_location(src, tmp_path / "out" / "a.wav", "round", 1)
        guano = _guano(tmp_path / "out" / "a.wav")
        assert guano.loc_position == (36.5, -82.2)
        assert guano.version == "1.0"

    @pytest.mark.parametrize("mode", ["round", "strip"])
    def test_unreadable_wamd_waypoint_is_removed(self, tmp_path, mode) -> None:
        from riffy import WamdMetadata
        from riffy.metadata.wamd import GPSFIRST_ID

        from bioamla.datasets import redact_wav_location

        _wamd_wav(tmp_path / "a.wav", waypoint="garbage")
        # Rounding can't read it, so it is removed rather than uploaded as-is.
        assert redact_wav_location(tmp_path / "a.wav", tmp_path / "o.wav", mode)
        wamd = WamdMetadata.from_bytes(_chunks(tmp_path / "o.wav")["wamd"])
        assert wamd.get_raw(GPSFIRST_ID) is None
