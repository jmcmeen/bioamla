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
