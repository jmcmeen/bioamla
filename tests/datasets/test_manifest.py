"""Tests for the dataset manifest (dataset.json)."""

import pytest

from bioamla.datasets import (
    BIOAMLA_DATASET_FORMAT,
    DatasetManifest,
    build_manifest_from_metadata,
    create_label_map,
    load_dataset_manifest,
    save_dataset_manifest,
)
from bioamla.datasets._metadata import write_metadata_csv
from bioamla.exceptions import NotFoundError


def _write_dataset(tmp_path):
    rows = [
        {"file_name": "call/a.wav", "label": "call", "split": "train"},
        {"file_name": "call/b.wav", "label": "call", "split": "test"},
        {"file_name": "chorus/c.wav", "label": "chorus", "split": "train"},
        {"file_name": "chorus/d.wav", "label": "chorus", "split": "val", "source": "xeno_canto"},
    ]
    write_metadata_csv(tmp_path / "metadata.csv", rows, merge_existing=False)
    return tmp_path


class TestBuildManifest:
    def test_label2id_matches_create_label_map(self, tmp_path) -> None:
        d = _write_dataset(tmp_path)
        manifest = build_manifest_from_metadata(str(d), name="demo")
        assert manifest.label2id == create_label_map(["call", "chorus"])
        assert manifest.id2label == {0: "call", 1: "chorus"}

    def test_counts_and_splits(self, tmp_path) -> None:
        d = _write_dataset(tmp_path)
        manifest = build_manifest_from_metadata(str(d))
        assert manifest.class_counts == {"call": 2, "chorus": 2}
        assert manifest.splits == {"train": 2, "test": 1, "val": 1}
        assert {s["source"] for s in manifest.sources} == {"xeno_canto"}

    def test_missing_metadata_raises(self, tmp_path) -> None:
        with pytest.raises(NotFoundError):
            build_manifest_from_metadata(str(tmp_path / "nope"))


class TestManifestRoundTrip:
    def test_save_load_restores_int_keys(self, tmp_path) -> None:
        d = _write_dataset(tmp_path)
        manifest = build_manifest_from_metadata(str(d), name="demo", kind="labeled")
        out = tmp_path / "dataset.json"
        save_dataset_manifest(manifest, str(out))

        loaded = load_dataset_manifest(str(out))
        assert loaded.format == BIOAMLA_DATASET_FORMAT
        assert loaded.name == "demo"
        assert loaded.label2id == manifest.label2id
        # id2label keys come back as ints, not JSON strings.
        assert all(isinstance(k, int) for k in loaded.id2label)

    def test_load_missing_raises(self, tmp_path) -> None:
        with pytest.raises(NotFoundError):
            load_dataset_manifest(str(tmp_path / "missing.json"))

    def test_from_dict_ignores_unknown_keys(self) -> None:
        m = DatasetManifest.from_dict({"name": "x", "bogus": 1, "id2label": {"0": "call"}})
        assert m.name == "x"
        assert m.id2label == {0: "call"}


class TestRecordingSummary:
    def _rows(self):
        return [
            {
                "file_name": "a.wav",
                "label": "x",
                "recorded_at": "2023-06-14T05:00:00-04:00",
                "device_model": "Song Meter Micro",
                "device_id": "SMU00001",
                "latitude": "36.5234",
                "longitude": "-82.2111",
            },
            {
                "file_name": "b.wav",
                "label": "x",
                "recorded_at": "2023-06-15T05:00:00-04:00",
                "device_model": "Song Meter Micro",
                "device_id": "SMU00001",
                "latitude": "36.5241",  # same site after rounding to 0.01
                "longitude": "-82.2149",
            },
            {
                "file_name": "c.wav",
                "label": "y",
                "recorded_at": "2023-05-01T22:00:00-04:00",
                "device_model": "Song Meter Micro",
                "device_id": "SMU00002",
                "latitude": "36.60",
                "longitude": "-82.30",
            },
            {"file_name": "d.wav", "label": "y"},  # no recorder columns
        ]

    def test_summary(self, tmp_path) -> None:
        from bioamla.datasets import summarize_recordings

        s = summarize_recordings(self._rows())
        assert s["recorded_from"] == "2023-05-01T22:00:00-04:00"
        assert s["recorded_to"] == "2023-06-15T05:00:00-04:00"
        assert s["devices"] == [
            {"device_model": "Song Meter Micro", "device_id": "SMU00001", "files": 2},
            {"device_model": "Song Meter Micro", "device_id": "SMU00002", "files": 1},
        ]
        assert s["sites"] == [
            {"latitude": 36.52, "longitude": -82.21, "files": 2},
            {"latitude": 36.6, "longitude": -82.3, "files": 1},
        ]

    def test_empty_when_no_columns(self) -> None:
        from bioamla.datasets import summarize_recordings

        s = summarize_recordings([{"file_name": "a.wav", "label": "x"}])
        assert s == {"recorded_from": None, "recorded_to": None, "devices": [], "sites": []}

    def test_bad_coordinates_ignored(self) -> None:
        from bioamla.datasets import summarize_recordings

        s = summarize_recordings([{"latitude": "north", "longitude": "-82"}])
        assert s["sites"] == []

    def test_manifest_and_card(self, tmp_path) -> None:
        from bioamla.datasets import build_dataset_card

        write_metadata_csv(tmp_path / "metadata.csv", self._rows(), merge_existing=False)
        manifest = build_manifest_from_metadata(str(tmp_path), name="scp")
        assert manifest.recorded_from == "2023-05-01T22:00:00-04:00"
        assert len(manifest.devices) == 2
        assert len(manifest.sites) == 2

        out = tmp_path / "dataset.json"
        save_dataset_manifest(manifest, str(out))
        loaded = load_dataset_manifest(str(out))
        assert loaded.sites == manifest.sites
        assert loaded.devices == manifest.devices

        card = build_dataset_card(manifest)
        assert "## Recording" in card
        assert "2023-05-01T22:00:00-04:00 to 2023-06-15T05:00:00-04:00" in card
        assert "Song Meter Micro SMU00001" in card
        assert "**Sites:** 2" in card

    def test_card_without_recording_section(self, tmp_path) -> None:
        from bioamla.datasets import build_dataset_card

        d = _write_dataset(tmp_path)
        card = build_dataset_card(build_manifest_from_metadata(str(d)))
        assert "## Recording" not in card
