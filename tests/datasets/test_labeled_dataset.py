"""Tests for extract_labeled_dataset (annotated chunks -> labeled clip dataset)."""

import csv

import numpy as np
import pytest

from bioamla.audio import get_audio_info, save_audio
from bioamla.datasets import (
    Annotation,
    create_label_map,
    extract_labeled_dataset,
    save_bioamla_annotations,
)
from bioamla.exceptions import AnnotationError

SR = 16000


def _write_wav(path, seconds=3.0, sr=SR):
    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False, dtype=np.float32)
    samples = 0.5 * np.sin(2 * np.pi * 440.0 * t)
    save_audio(str(path), samples, sr)
    return path


def _annotations():
    return [
        Annotation(
            start_time=0.0, end_time=0.5, low_freq=300, high_freq=3000, label="call", confidence=0.9
        ),
        Annotation(start_time=1.0, end_time=1.5, low_freq=500, high_freq=5000, label="call"),
        Annotation(start_time=2.0, end_time=2.5, label="chorus"),
    ]


def _make_pair(tmp_path):
    wav = _write_wav(tmp_path / "rec.wav")
    save_bioamla_annotations(_annotations(), str(tmp_path / "rec.json"))
    return wav, tmp_path / "rec.json"


class TestExtractLabeledDataset:
    def test_audiofolder_layout_and_counts(self, tmp_path):
        wav, ann = _make_pair(tmp_path)
        out = tmp_path / "ds"
        result = extract_labeled_dataset(str(wav), str(out), annotations=str(ann), layout="both")

        assert result["clips_written"] == 3
        assert result["files_processed"] == 1
        # One subdir per unique label.
        subdirs = {p.name for p in out.iterdir() if p.is_dir()}
        assert subdirs == {"call", "chorus"}
        assert len(list((out / "call").glob("*.wav"))) == 2
        assert len(list((out / "chorus").glob("*.wav"))) == 1

    def test_metadata_targets_match_label_map(self, tmp_path):
        wav, ann = _make_pair(tmp_path)
        out = tmp_path / "ds"
        extract_labeled_dataset(str(wav), str(out), annotations=str(ann), layout="both")

        with (out / "metadata.csv").open() as f:
            rows = list(csv.DictReader(f))
        expected = create_label_map(["call", "chorus"])
        assert len(rows) == 3
        for row in rows:
            assert int(row["target"]) == expected[row["label"]]
            assert row["source_file"] == "rec.wav"
            assert row["split"] == ""  # populated later by partition

    def test_resample_changes_clip_rate(self, tmp_path):
        wav = _write_wav(tmp_path / "rec.wav", seconds=3.0, sr=44100)
        save_bioamla_annotations(_annotations(), str(tmp_path / "rec.json"))
        out = tmp_path / "ds"
        extract_labeled_dataset(
            str(wav), str(out), annotations=str(tmp_path / "rec.json"), target_sample_rate=16000
        )
        clip = next((out / "call").glob("*.wav"))
        assert get_audio_info(str(clip)).sample_rate == 16000

    def test_bandpass_runs_when_freq_bounds_present(self, tmp_path):
        wav, ann = _make_pair(tmp_path)
        out = tmp_path / "ds"
        result = extract_labeled_dataset(str(wav), str(out), annotations=str(ann), bandpass=True)
        # All three clips still written (the two with freq bounds are filtered).
        assert result["clips_written"] == 3

    def test_out_of_range_annotations_skipped(self, tmp_path):
        # Audio is 3s; two annotations sit entirely beyond it and must be skipped,
        # not written as empty clips.
        wav = _write_wav(tmp_path / "rec.wav", seconds=3.0)
        anns = [
            Annotation(start_time=0.0, end_time=0.5, label="call"),
            Annotation(start_time=5.0, end_time=6.0, label="call"),
            Annotation(start_time=10.0, end_time=11.0, label="chorus"),
        ]
        save_bioamla_annotations(anns, str(tmp_path / "rec.json"))
        out = tmp_path / "ds"
        result = extract_labeled_dataset(str(wav), str(out), annotations=str(tmp_path / "rec.json"))
        assert result["clips_written"] == 1
        assert len(result["skipped"]) == 2
        # "chorus" had only an out-of-range annotation, so it never reaches the label map.
        assert result["labels"] == ["call"]

    def test_bandpass_too_short_clip_falls_back_unfiltered(self, tmp_path):
        # A ~3ms annotation is too short for the filter's padding; the clip should
        # still be written (unfiltered) rather than dropped.
        wav = _write_wav(tmp_path / "rec.wav", seconds=2.0)
        anns = [
            Annotation(start_time=0.0, end_time=0.003, low_freq=300, high_freq=3000, label="call")
        ]
        save_bioamla_annotations(anns, str(tmp_path / "rec.json"))
        out = tmp_path / "ds"
        result = extract_labeled_dataset(
            str(wav), str(out), annotations=str(tmp_path / "rec.json"), bandpass=True
        )
        assert result["clips_written"] == 1
        assert result["skipped"] == []

    def test_flat_layout_no_subdirs(self, tmp_path):
        wav, ann = _make_pair(tmp_path)
        out = tmp_path / "ds"
        extract_labeled_dataset(str(wav), str(out), annotations=str(ann), layout="flat")
        assert not any(p.is_dir() for p in out.iterdir())
        assert (out / "metadata.csv").exists()
        assert len(list(out.glob("*.wav"))) == 3

    def test_audiofolder_only_writes_no_csv(self, tmp_path):
        wav, ann = _make_pair(tmp_path)
        out = tmp_path / "ds"
        result = extract_labeled_dataset(
            str(wav), str(out), annotations=str(ann), layout="audiofolder"
        )
        assert result["metadata_file"] is None
        assert not (out / "metadata.csv").exists()

    def test_include_exclude_filters(self, tmp_path):
        wav, ann = _make_pair(tmp_path)
        out = tmp_path / "ds"
        result = extract_labeled_dataset(
            str(wav), str(out), annotations=str(ann), include_labels={"chorus"}
        )
        assert result["clips_written"] == 1
        assert result["labels"] == ["chorus"]

    def test_directory_source_pairs_by_stem(self, tmp_path):
        data = tmp_path / "raw"
        data.mkdir()
        _write_wav(data / "a.wav")
        _write_wav(data / "b.wav")
        save_bioamla_annotations(_annotations(), str(data / "a.json"))
        save_bioamla_annotations([_annotations()[0]], str(data / "b.json"))
        out = tmp_path / "ds"
        result = extract_labeled_dataset(str(data), str(out), layout="both")
        assert result["files_processed"] == 2
        assert result["clips_written"] == 4  # 3 from a + 1 from b

    def test_missing_annotation_raises(self, tmp_path):
        wav = _write_wav(tmp_path / "rec.wav")
        with pytest.raises(AnnotationError):
            extract_labeled_dataset(str(wav), str(tmp_path / "ds"))

    def test_invalid_layout_raises(self, tmp_path):
        wav, ann = _make_pair(tmp_path)
        with pytest.raises(AnnotationError):
            extract_labeled_dataset(
                str(wav), str(tmp_path / "ds"), annotations=str(ann), layout="x"
            )
