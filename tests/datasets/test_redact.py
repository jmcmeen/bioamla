"""Tests for coordinate redaction (bioamla.datasets.redact) and push_dataset's use of it."""

import csv
import io
import json

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
    def _fake_api(self, record):
        class FakeApi:
            def create_repo(self, **kwargs):
                pass

            def upload_folder(self, **kwargs):
                record["upload"] = kwargs
                record.setdefault("order", []).append("upload")

            def create_commit(self, **kwargs):
                record["commit"] = kwargs
                record.setdefault("order", []).append("commit")

        return FakeApi()

    def test_round_uploads_redacted_copies_first(self, tmp_path, monkeypatch) -> None:
        _write_csv(tmp_path / "metadata.csv")
        (tmp_path / "a.wav").write_bytes(b"RIFF")
        record: dict = {}
        monkeypatch.setattr(hf, "_get_hf_api", lambda: self._fake_api(record))

        hf.push_dataset(str(tmp_path), "me/frogs", coordinates="round", coordinate_decimals=2)

        assert record["order"] == ["commit", "upload"]
        assert record["upload"]["ignore_patterns"] == ["metadata.csv"]
        (op,) = record["commit"]["operations"]
        assert op.path_in_repo == "metadata.csv"
        _, rows = _read_csv(op.path_or_fileobj)
        assert rows[0]["latitude"] == "35.12"

    def test_keep_uploads_folder_as_is(self, tmp_path, monkeypatch) -> None:
        _write_csv(tmp_path / "metadata.csv")
        record: dict = {}
        monkeypatch.setattr(hf, "_get_hf_api", lambda: self._fake_api(record))

        hf.push_dataset(str(tmp_path), "me/frogs")

        assert "ignore_patterns" not in record["upload"]
        assert "commit" not in record

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
        assert "per-file locations" in res.output
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
        assert "embedded WAV metadata" in res.output
        assert "per-file locations" not in res.output


class TestFilesWithEmbeddedLocation:
    def test_finds_guano_position_only(self, tmp_path) -> None:
        import wave

        from riffy import GuanoMetadata, WAVParser

        from bioamla.datasets import unredacted_location_files

        def pcm(path):
            with wave.open(str(path), "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(16000)
                w.writeframes(b"\x00" * 3200)

        pcm(tmp_path / "plain.wav")
        pcm(tmp_path / "src.wav")
        parser = WAVParser(tmp_path / "src.wav")
        guano = GuanoMetadata(version="1.0")
        guano.loc_position = (36.52, -82.21)
        guano.write_to_parser(parser)
        (tmp_path / "sub").mkdir()
        parser.write_wav(tmp_path / "sub" / "located.wav")

        (tmp_path / "notes.json").write_text('{"Latitude": 35.1}')
        (tmp_path / "table.parquet").write_bytes(b"PAR1")
        assert unredacted_location_files(tmp_path) == [
            "notes.json",
            "sub/located.wav",
            "table.parquet",
        ]


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
        record: dict = {}

        class FakeApi:
            def create_repo(self, **kwargs):
                pass

            def create_commit(self, **kwargs):
                record["commit"] = kwargs

            def upload_folder(self, **kwargs):
                record["upload"] = kwargs

        monkeypatch.setattr(hf, "_get_hf_api", lambda: FakeApi())
        hf.push_dataset(str(tmp_path), "me/frogs", coordinates="strip")
        (pattern,) = record["upload"]["ignore_patterns"]
        assert fnmatch.fnmatch("train[2026].csv", pattern)

    def test_keep_push_tolerates_unreadable_csv(self, tmp_path, monkeypatch) -> None:
        (tmp_path / "export.csv").write_bytes("name\nca\xf1on\n".encode("latin-1"))
        calls: dict = {}
        monkeypatch.setattr(hf, "push_dataset", TestPushDatasetCliCoordinates()._fake_push(calls))
        res = CliRunner().invoke(catalogs, ["hf", "push-dataset", str(tmp_path), "me/frogs"])
        assert res.exit_code == 0, res.output
