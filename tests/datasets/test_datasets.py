"""Tests for the datasets domain (flattened, exception-based API).

Covers annotation format conversion round-trips, label utilities, dataset
merge on tmp_path CSVs, license generation, and stats.
"""

import csv

import pytest

from bioamla.datasets import (
    BIOAMLA_ANNOTATION_FORMAT,
    Annotation,
    AnnotationSet,
    annotations_to_one_hot,
    create_annotation,
    create_label_map,
    filter_labels,
    generate_clip_labels,
    generate_frame_labels,
    generate_license_for_dataset,
    generate_licenses_for_directory,
    get_dataset_stats,
    get_unique_labels,
    load_bioamla_annotations,
    load_csv_annotations,
    load_label_mapping,
    load_raven_selection_table,
    merge_datasets,
    remap_labels,
    save_bioamla_annotations,
    save_csv_annotations,
    save_label_mapping,
    save_raven_selection_table,
    summarize_annotations,
)
from bioamla.exceptions import (
    AnnotationError,
    InvalidInputError,
    NotFoundError,
)


def _sample_annotations() -> list:
    return [
        Annotation(
            start_time=1.0,
            end_time=2.0,
            low_freq=1000.0,
            high_freq=8000.0,
            label="bird_song",
            channel=1,
            confidence=0.9,
            notes="clear",
        ),
        Annotation(
            start_time=3.5,
            end_time=4.25,
            low_freq=500.0,
            high_freq=4000.0,
            label="frog_croak",
            channel=1,
        ),
        Annotation(start_time=5.0, end_time=6.0, label="bird_song"),
    ]


class TestAnnotationDataclass:
    def test_properties(self) -> None:
        ann = Annotation(start_time=1.0, end_time=3.0, low_freq=1000.0, high_freq=5000.0)
        assert ann.duration == 2.0
        assert ann.bandwidth == 4000.0
        assert ann.center_time == 2.0
        assert ann.center_freq == 3000.0

    def test_to_from_dict_roundtrip(self) -> None:
        ann = Annotation(
            start_time=1.0,
            end_time=2.0,
            low_freq=100.0,
            high_freq=200.0,
            label="x",
            custom_fields={"site": "A"},
        )
        restored = Annotation.from_dict(ann.to_dict())
        assert restored.start_time == ann.start_time
        assert restored.label == ann.label
        assert restored.custom_fields.get("site") == "A"

    def test_overlaps(self) -> None:
        a = Annotation(start_time=0.0, end_time=2.0)
        b = Annotation(start_time=1.0, end_time=3.0)
        c = Annotation(start_time=2.0, end_time=4.0)
        assert a.overlaps_time(b)
        assert not a.overlaps_time(c)

    def test_create_annotation_validation(self) -> None:
        with pytest.raises(AnnotationError):
            create_annotation(start_time=2.0, end_time=1.0)
        with pytest.raises(AnnotationError):
            create_annotation(start_time=0.0, end_time=1.0, low_freq=500.0, high_freq=100.0)


class TestCsvRoundTrip:
    def test_csv_roundtrip(self, tmp_path) -> None:
        anns = _sample_annotations()
        out = tmp_path / "ann.csv"
        save_csv_annotations(anns, str(out))
        loaded = load_csv_annotations(str(out))

        assert len(loaded) == len(anns)
        for orig, got in zip(anns, loaded, strict=False):
            assert got.start_time == pytest.approx(orig.start_time)
            assert got.end_time == pytest.approx(orig.end_time)
            assert got.label == orig.label

    def test_csv_custom_fields_roundtrip(self, tmp_path) -> None:
        anns = [Annotation(start_time=0.0, end_time=1.0, label="a", custom_fields={"site": "X"})]
        out = tmp_path / "ann.csv"
        save_csv_annotations(anns, str(out))
        loaded = load_csv_annotations(str(out))
        assert loaded[0].custom_fields.get("site") == "X"

    def test_load_missing_csv_raises(self, tmp_path) -> None:
        with pytest.raises(NotFoundError):
            load_csv_annotations(str(tmp_path / "missing.csv"))


class TestRavenRoundTrip:
    def test_raven_roundtrip(self, tmp_path) -> None:
        anns = _sample_annotations()
        out = tmp_path / "selections.txt"
        save_raven_selection_table(anns, str(out))
        loaded = load_raven_selection_table(str(out))

        assert len(loaded) == len(anns)
        for orig, got in zip(anns, loaded, strict=False):
            assert got.start_time == pytest.approx(orig.start_time, abs=1e-5)
            assert got.end_time == pytest.approx(orig.end_time, abs=1e-5)
            assert got.label == orig.label

    def test_raven_is_tab_delimited(self, tmp_path) -> None:
        out = tmp_path / "selections.txt"
        save_raven_selection_table(_sample_annotations(), str(out))
        first_line = out.read_text(encoding="utf-8").splitlines()[0]
        assert "\t" in first_line
        assert "Begin Time (s)" in first_line

    def test_load_missing_raven_raises(self, tmp_path) -> None:
        with pytest.raises(NotFoundError):
            load_raven_selection_table(str(tmp_path / "missing.txt"))


class TestBioamlaFormat:
    def test_roundtrip_preserves_annotations_and_metadata(self, tmp_path) -> None:
        anns = _sample_annotations()
        meta = {"audio_file": "rec.wav", "sample_rate": 48000, "duration": 60.0}
        out = tmp_path / "ann.json"
        save_bioamla_annotations(anns, str(out), metadata=meta)

        loaded, loaded_meta = load_bioamla_annotations(str(out))

        assert len(loaded) == len(anns)
        for orig, got in zip(anns, loaded, strict=False):
            assert got.start_time == pytest.approx(orig.start_time, abs=1e-9)
            assert got.label == orig.label
        assert loaded_meta["audio_file"] == "rec.wav"
        assert loaded_meta["sample_rate"] == 48000
        assert loaded_meta["duration"] == pytest.approx(60.0)

    def test_header_records_format_version(self, tmp_path) -> None:
        import json

        out = tmp_path / "ann.json"
        save_bioamla_annotations(_sample_annotations(), str(out))
        data = json.loads(out.read_text(encoding="utf-8"))
        assert data["format"] == BIOAMLA_ANNOTATION_FORMAT
        assert isinstance(data["annotations"], list)

    def test_metadata_cannot_clobber_reserved_keys(self, tmp_path) -> None:
        import json

        out = tmp_path / "ann.json"
        save_bioamla_annotations(
            _sample_annotations(),
            str(out),
            metadata={"format": "evil", "annotations": "evil"},
        )
        data = json.loads(out.read_text(encoding="utf-8"))
        assert data["format"] == BIOAMLA_ANNOTATION_FORMAT
        assert isinstance(data["annotations"], list)

    def test_load_missing_raises(self, tmp_path) -> None:
        with pytest.raises(NotFoundError):
            load_bioamla_annotations(str(tmp_path / "missing.json"))

    def test_load_wrong_shape_raises(self, tmp_path) -> None:
        bad = tmp_path / "bad.json"
        bad.write_text('{"not_annotations": []}', encoding="utf-8")
        with pytest.raises(AnnotationError):
            load_bioamla_annotations(str(bad))


class TestCrossFormatConversion:
    def test_csv_to_raven_to_csv(self, tmp_path) -> None:
        anns = _sample_annotations()
        csv_in = tmp_path / "in.csv"
        raven_mid = tmp_path / "mid.txt"
        csv_out = tmp_path / "out.csv"

        save_csv_annotations(anns, str(csv_in))
        a1 = load_csv_annotations(str(csv_in))
        save_raven_selection_table(a1, str(raven_mid))
        a2 = load_raven_selection_table(str(raven_mid))
        save_csv_annotations(a2, str(csv_out))
        a3 = load_csv_annotations(str(csv_out))

        assert len(a3) == len(anns)
        assert {a.label for a in a3} == {a.label for a in anns}


class TestSummaryAndLabels:
    def test_get_unique_labels(self) -> None:
        assert get_unique_labels(_sample_annotations()) == ["bird_song", "frog_croak"]

    def test_summarize(self) -> None:
        summary = summarize_annotations(_sample_annotations())
        assert summary["total_annotations"] == 3
        assert summary["unique_labels"] == 2
        assert summary["labels"]["bird_song"] == 2

    def test_summarize_empty(self) -> None:
        summary = summarize_annotations([])
        assert summary["total_annotations"] == 0

    def test_create_label_map(self) -> None:
        lm = create_label_map(["b", "a", "a", "c"])
        assert lm == {"a": 0, "b": 1, "c": 2}

    def test_annotations_to_one_hot(self) -> None:
        anns = _sample_annotations()
        lm = create_label_map(get_unique_labels(anns))
        oh = annotations_to_one_hot(anns, lm)
        assert oh.shape == (3, 2)
        assert oh[0, lm["bird_song"]] == 1.0

    def test_generate_clip_labels(self) -> None:
        anns = _sample_annotations()
        lm = create_label_map(get_unique_labels(anns))
        vec = generate_clip_labels(anns, 0.0, 2.5, lm, multi_label=True)
        assert vec[lm["bird_song"]] == 1.0

    def test_generate_clip_labels_invalid(self) -> None:
        with pytest.raises(InvalidInputError):
            generate_clip_labels([], 2.0, 1.0, {})

    def test_generate_frame_labels(self) -> None:
        anns = _sample_annotations()
        lm = create_label_map(get_unique_labels(anns))
        frames = generate_frame_labels(anns, 6.0, 1.0, 0.5, lm)
        assert frames.shape[0] == len(lm)
        assert frames.shape[1] > 0

    def test_generate_frame_labels_invalid(self) -> None:
        with pytest.raises(InvalidInputError):
            generate_frame_labels([], 6.0, 0.0, 0.5, {})


class TestRemapAndFilter:
    def test_remap_labels(self) -> None:
        anns = _sample_annotations()
        mapping = {"bird_song": "bird", "frog_croak": "frog"}
        remapped = remap_labels(anns, mapping)
        assert {a.label for a in remapped} == {"bird", "frog"}
        # original annotations are unchanged
        assert anns[0].label == "bird_song"

    def test_remap_drop_unmapped(self) -> None:
        anns = _sample_annotations()
        remapped = remap_labels(anns, {"bird_song": "bird"}, keep_unmapped=False)
        assert all(a.label == "bird" for a in remapped)
        assert len(remapped) == 2

    def test_filter_labels(self) -> None:
        anns = _sample_annotations()
        only_bird = filter_labels(anns, include_labels={"bird_song"})
        assert all(a.label == "bird_song" for a in only_bird)
        no_frog = filter_labels(anns, exclude_labels={"frog_croak"})
        assert all(a.label != "frog_croak" for a in no_frog)

    def test_label_mapping_roundtrip(self, tmp_path) -> None:
        mapping = {"bird_song": "bird", "frog_croak": "frog"}
        out = tmp_path / "map.csv"
        save_label_mapping(mapping, str(out))
        loaded = load_label_mapping(str(out))
        assert loaded == mapping

    def test_load_missing_mapping_raises(self, tmp_path) -> None:
        with pytest.raises(NotFoundError):
            load_label_mapping(str(tmp_path / "missing.csv"))


class TestAnnotationSet:
    def test_basic_collection_ops(self) -> None:
        aset = AnnotationSet(file_path="x.wav", annotations=_sample_annotations())
        assert len(aset) == 3
        assert aset.get_labels() == {"bird_song", "frog_croak"}
        assert len(aset.filter_by_label("bird_song")) == 2

    def test_merge_overlapping(self) -> None:
        anns = [
            Annotation(start_time=0.0, end_time=2.0, label="a"),
            Annotation(start_time=1.0, end_time=3.0, label="a"),
            Annotation(start_time=5.0, end_time=6.0, label="a"),
        ]
        aset = AnnotationSet(file_path="x.wav", annotations=anns)
        merged = aset.merge_overlapping()
        assert len(merged) == 2
        assert merged[0].start_time == 0.0
        assert merged[0].end_time == 3.0


def _write_dataset(base, name, rows, fieldnames):
    """Create a dataset dir with a metadata.csv and dummy audio files."""
    ds = base / name
    ds.mkdir(parents=True, exist_ok=True)
    with open(ds / "metadata.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
            # create a placeholder audio file matching file_name
            audio_file = ds / row["file_name"]
            audio_file.parent.mkdir(parents=True, exist_ok=True)
            audio_file.write_bytes(b"RIFFfake")
    return ds


class TestMergeDatasets:
    def test_merge_two_datasets(self, tmp_path) -> None:
        fields = ["file_name", "category"]
        ds1 = _write_dataset(
            tmp_path,
            "ds1",
            [{"file_name": "a.wav", "category": "robin"}],
            fields,
        )
        ds2 = _write_dataset(
            tmp_path,
            "ds2",
            [{"file_name": "b.wav", "category": "sparrow"}],
            fields,
        )
        out = tmp_path / "merged"

        stats = merge_datasets(
            [str(ds1), str(ds2)],
            str(out),
            organize_by_category=True,
            verbose=False,
        )

        assert stats["datasets_merged"] == 2
        assert stats["total_files"] == 2
        assert stats["files_copied"] == 2
        assert (out / "metadata.csv").exists()
        # Files organized by sanitized category
        assert (out / "robin" / "a.wav").exists()
        assert (out / "sparrow" / "b.wav").exists()

    def test_merge_no_paths_raises(self, tmp_path) -> None:
        with pytest.raises(InvalidInputError):
            merge_datasets([], str(tmp_path / "out"), verbose=False)

    def test_merge_bad_target_format_raises(self, tmp_path) -> None:
        with pytest.raises(InvalidInputError):
            merge_datasets(
                ["doesnotmatter"],
                str(tmp_path / "out"),
                target_format="xyz",
                verbose=False,
            )

    def test_merge_missing_dataset_raises(self, tmp_path) -> None:
        with pytest.raises(NotFoundError):
            merge_datasets([str(tmp_path / "nope")], str(tmp_path / "out"), verbose=False)

    def test_merge_skip_existing(self, tmp_path) -> None:
        fields = ["file_name", "category"]
        ds1 = _write_dataset(tmp_path, "ds1", [{"file_name": "a.wav", "category": "robin"}], fields)
        out = tmp_path / "merged"
        merge_datasets([str(ds1)], str(out), verbose=False)
        # Merge again: should skip the already-present file
        stats = merge_datasets([str(ds1)], str(out), skip_existing=True, verbose=False)
        assert stats["files_skipped"] >= 1


class TestLicenseGeneration:
    def _attr_fields(self):
        return ["file_name", "attr_id", "attr_lic", "attr_url", "attr_note"]

    def test_generate_license(self, tmp_path) -> None:
        ds = tmp_path / "ds"
        ds.mkdir()
        with open(ds / "metadata.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self._attr_fields())
            writer.writeheader()
            writer.writerow(
                {
                    "file_name": "a.wav",
                    "attr_id": "user1",
                    "attr_lic": "CC-BY",
                    "attr_url": "http://example.com",
                    "attr_note": "n",
                }
            )

        stats = generate_license_for_dataset(ds)
        assert stats["attributions_count"] == 1
        assert (ds / "LICENSE").exists()
        content = (ds / "LICENSE").read_text(encoding="utf-8")
        assert "a.wav" in content
        assert "CC-BY" in content

    def test_generate_license_missing_metadata(self, tmp_path) -> None:
        with pytest.raises(NotFoundError):
            generate_license_for_dataset(tmp_path / "empty")

    def test_generate_license_missing_fields(self, tmp_path) -> None:
        ds = tmp_path / "ds"
        ds.mkdir()
        with open(ds / "metadata.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["file_name", "category"])
            writer.writeheader()
            writer.writerow({"file_name": "a.wav", "category": "x"})
        with pytest.raises(InvalidInputError):
            generate_license_for_dataset(ds)

    def test_generate_licenses_for_directory(self, tmp_path) -> None:
        root = tmp_path / "root"
        for name in ("ds1", "ds2"):
            ds = root / name
            ds.mkdir(parents=True)
            with open(ds / "metadata.csv", "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=self._attr_fields())
                writer.writeheader()
                writer.writerow(
                    {
                        "file_name": "a.wav",
                        "attr_id": "u",
                        "attr_lic": "CC0",
                        "attr_url": "",
                        "attr_note": "",
                    }
                )

        stats = generate_licenses_for_directory(root)
        assert stats["datasets_found"] == 2
        assert stats["datasets_processed"] == 2
        assert stats["datasets_failed"] == 0

    def test_generate_licenses_missing_dir(self, tmp_path) -> None:
        with pytest.raises(NotFoundError):
            generate_licenses_for_directory(tmp_path / "nope")


class TestDatasetStats:
    def test_get_stats(self, tmp_path) -> None:
        ds = tmp_path / "ds"
        ds.mkdir()
        with open(ds / "metadata.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["file_name", "category", "license"])
            writer.writeheader()
            writer.writerow({"file_name": "a.wav", "category": "robin", "license": "CC0"})
            writer.writerow({"file_name": "b.wav", "category": "robin", "license": "CC-BY"})
            writer.writerow({"file_name": "c.wav", "category": "sparrow", "license": "CC0"})

        stats = get_dataset_stats(str(ds))
        assert stats["total_files"] == 3
        assert stats["num_categories"] == 2
        assert stats["categories"]["robin"] == 2
        assert stats["num_licenses"] == 2

    def test_get_stats_missing(self, tmp_path) -> None:
        with pytest.raises(NotFoundError):
            get_dataset_stats(str(tmp_path / "nope"))


class TestAugmentationConfigImport:
    def test_config_importable_on_slim_install(self) -> None:
        # AugmentationConfig must import without audiomentations/torch installed.
        from bioamla.datasets import AugmentationConfig

        cfg = AugmentationConfig(add_noise=True)
        assert cfg.add_noise is True
        assert cfg.sample_rate == 16000
