"""
Unit tests for bioamla.annotations module.
"""

import csv
from pathlib import Path

import numpy as np
import pytest

from bioamla.core.annotations import (
    Annotation,
    AnnotationSet,
    annotations_to_one_hot,
    create_label_map,
    filter_labels,
    generate_clip_labels,
    generate_frame_labels,
    get_unique_labels,
    load_csv_annotations,
    load_label_mapping,
    load_raven_selection_table,
    remap_labels,
    save_csv_annotations,
    save_label_mapping,
    save_raven_selection_table,
    summarize_annotations,
)


class TestAnnotation:
    """Tests for Annotation dataclass."""

    def test_basic_annotation(self):
        """Test creating a basic annotation."""
        ann = Annotation(start_time=1.0, end_time=2.0, label="bird")
        assert ann.start_time == 1.0
        assert ann.end_time == 2.0
        assert ann.label == "bird"
        assert ann.duration == 1.0

    def test_annotation_with_frequency(self):
        """Test annotation with frequency bounds."""
        ann = Annotation(
            start_time=0.0, end_time=1.0,
            low_freq=1000, high_freq=8000,
            label="bird_song"
        )
        assert ann.low_freq == 1000
        assert ann.high_freq == 8000
        assert ann.bandwidth == 7000
        assert ann.center_freq == 4500

    def test_duration_property(self):
        """Test duration calculation."""
        ann = Annotation(start_time=5.0, end_time=8.5, label="test")
        assert ann.duration == 3.5

    def test_center_time_property(self):
        """Test center time calculation."""
        ann = Annotation(start_time=2.0, end_time=6.0, label="test")
        assert ann.center_time == 4.0

    def test_overlaps_time(self):
        """Test time overlap detection."""
        ann1 = Annotation(start_time=1.0, end_time=3.0, label="a")
        ann2 = Annotation(start_time=2.0, end_time=4.0, label="b")
        ann3 = Annotation(start_time=4.0, end_time=5.0, label="c")

        assert ann1.overlaps_time(ann2) is True
        assert ann2.overlaps_time(ann1) is True
        assert ann1.overlaps_time(ann3) is False
        assert ann3.overlaps_time(ann1) is False

    def test_overlaps_freq(self):
        """Test frequency overlap detection."""
        ann1 = Annotation(start_time=0, end_time=1, low_freq=1000, high_freq=3000, label="a")
        ann2 = Annotation(start_time=0, end_time=1, low_freq=2000, high_freq=5000, label="b")
        ann3 = Annotation(start_time=0, end_time=1, low_freq=4000, high_freq=6000, label="c")

        assert ann1.overlaps_freq(ann2) is True
        assert ann1.overlaps_freq(ann3) is False

    def test_overlaps_with_no_freq(self):
        """Test overlap when frequency is not set."""
        ann1 = Annotation(start_time=0, end_time=1, label="a")
        ann2 = Annotation(start_time=0, end_time=1, low_freq=1000, high_freq=5000, label="b")

        assert ann1.overlaps_freq(ann2) is True

    def test_contains_time(self):
        """Test time containment check."""
        ann = Annotation(start_time=1.0, end_time=5.0, label="test")
        assert ann.contains_time(3.0) is True
        assert ann.contains_time(0.5) is False
        assert ann.contains_time(5.5) is False
        assert ann.contains_time(1.0) is True
        assert ann.contains_time(5.0) is True

    def test_to_dict(self):
        """Test conversion to dictionary."""
        ann = Annotation(
            start_time=1.0, end_time=2.0,
            low_freq=1000, high_freq=5000,
            label="bird", channel=1,
            custom_fields={"quality": "good"}
        )
        d = ann.to_dict()

        assert d["start_time"] == 1.0
        assert d["end_time"] == 2.0
        assert d["label"] == "bird"
        assert d["quality"] == "good"

    def test_from_dict(self):
        """Test creation from dictionary."""
        data = {
            "start_time": 1.5,
            "end_time": 3.0,
            "low_freq": 2000,
            "high_freq": 6000,
            "label": "frog",
            "custom_field": "value"
        }
        ann = Annotation.from_dict(data)

        assert ann.start_time == 1.5
        assert ann.end_time == 3.0
        assert ann.low_freq == 2000
        assert ann.high_freq == 6000
        assert ann.label == "frog"
        assert ann.custom_fields["custom_field"] == "value"


class TestAnnotationSet:
    """Tests for AnnotationSet class."""

    def test_basic_annotation_set(self):
        """Test creating an annotation set."""
        annotations = [
            Annotation(start_time=0, end_time=1, label="a"),
            Annotation(start_time=2, end_time=3, label="b"),
        ]
        ann_set = AnnotationSet(file_path="test.wav", annotations=annotations)

        assert len(ann_set) == 2
        assert ann_set.file_path == "test.wav"

    def test_get_labels(self):
        """Test getting unique labels."""
        annotations = [
            Annotation(start_time=0, end_time=1, label="bird"),
            Annotation(start_time=1, end_time=2, label="frog"),
            Annotation(start_time=2, end_time=3, label="bird"),
        ]
        ann_set = AnnotationSet(file_path="test.wav", annotations=annotations)

        labels = ann_set.get_labels()
        assert labels == {"bird", "frog"}

    def test_filter_by_label(self):
        """Test filtering by label."""
        annotations = [
            Annotation(start_time=0, end_time=1, label="bird"),
            Annotation(start_time=1, end_time=2, label="frog"),
            Annotation(start_time=2, end_time=3, label="bird"),
        ]
        ann_set = AnnotationSet(file_path="test.wav", annotations=annotations)

        birds = ann_set.filter_by_label("bird")
        assert len(birds) == 2

    def test_filter_by_time_range(self):
        """Test filtering by time range."""
        annotations = [
            Annotation(start_time=0, end_time=2, label="a"),
            Annotation(start_time=3, end_time=5, label="b"),
            Annotation(start_time=6, end_time=8, label="c"),
        ]
        ann_set = AnnotationSet(file_path="test.wav", annotations=annotations)

        filtered = ann_set.filter_by_time_range(1.0, 4.0)
        assert len(filtered) == 2  # First two overlap with range

    def test_sort_by_time(self):
        """Test sorting by time."""
        annotations = [
            Annotation(start_time=5, end_time=6, label="c"),
            Annotation(start_time=1, end_time=2, label="a"),
            Annotation(start_time=3, end_time=4, label="b"),
        ]
        ann_set = AnnotationSet(file_path="test.wav", annotations=annotations)
        ann_set.sort_by_time()

        assert ann_set[0].start_time == 1
        assert ann_set[1].start_time == 3
        assert ann_set[2].start_time == 5

    def test_merge_overlapping(self):
        """Test merging overlapping annotations."""
        annotations = [
            Annotation(start_time=0, end_time=2, label="bird"),
            Annotation(start_time=1, end_time=3, label="bird"),
            Annotation(start_time=5, end_time=6, label="bird"),
        ]
        ann_set = AnnotationSet(file_path="test.wav", annotations=annotations)

        merged = ann_set.merge_overlapping(same_label_only=True)
        assert len(merged) == 2
        assert merged[0].start_time == 0
        assert merged[0].end_time == 3


class TestRavenSelectionTable:
    """Tests for Raven selection table import/export."""

    def test_load_raven_selection_table(self, temp_dir):
        """Test loading a Raven selection table."""
        # Create a mock Raven file
        raven_file = temp_dir / "selections.txt"
        with open(raven_file, "w", newline="") as f:
            writer = csv.writer(f, delimiter="\t")
            writer.writerow(["Selection", "View", "Channel", "Begin Time (s)",
                           "End Time (s)", "Low Freq (Hz)", "High Freq (Hz)", "Annotation"])
            writer.writerow([1, "Spectrogram 1", 1, "1.0", "2.0", "1000", "5000", "bird_song"])
            writer.writerow([2, "Spectrogram 1", 1, "3.5", "4.5", "2000", "6000", "frog_call"])

        annotations = load_raven_selection_table(str(raven_file))

        assert len(annotations) == 2
        assert annotations[0].start_time == 1.0
        assert annotations[0].end_time == 2.0
        assert annotations[0].low_freq == 1000
        assert annotations[0].high_freq == 5000
        assert annotations[0].label == "bird_song"
        assert annotations[1].label == "frog_call"

    def test_save_raven_selection_table(self, temp_dir):
        """Test saving annotations to Raven format."""
        annotations = [
            Annotation(start_time=1.0, end_time=2.0, low_freq=1000,
                      high_freq=5000, label="bird_song"),
            Annotation(start_time=3.0, end_time=4.0, low_freq=2000,
                      high_freq=6000, label="frog_call"),
        ]

        output_path = temp_dir / "output.txt"
        result = save_raven_selection_table(annotations, str(output_path))

        assert Path(result).exists()

        # Verify content
        loaded = load_raven_selection_table(str(output_path))
        assert len(loaded) == 2
        assert loaded[0].label == "bird_song"

    def test_raven_round_trip(self, temp_dir):
        """Test saving and loading preserves data."""
        original = [
            Annotation(start_time=1.5, end_time=2.5, low_freq=1500,
                      high_freq=5500, label="test_label", channel=2),
        ]

        output_path = temp_dir / "round_trip.txt"
        save_raven_selection_table(original, str(output_path))
        loaded = load_raven_selection_table(str(output_path))

        assert len(loaded) == 1
        assert loaded[0].start_time == 1.5
        assert loaded[0].end_time == 2.5
        assert loaded[0].label == "test_label"

    def test_load_missing_file_raises(self):
        """Test that loading a missing file raises error."""
        with pytest.raises(FileNotFoundError):
            load_raven_selection_table("/nonexistent/file.txt")


class TestCSVAnnotations:
    """Tests for CSV annotation import/export."""

    def test_load_csv_annotations(self, temp_dir):
        """Test loading annotations from CSV."""
        csv_file = temp_dir / "annotations.csv"
        with open(csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["start_time", "end_time", "low_freq", "high_freq", "label"])
            writer.writerow([0.5, 1.5, 1000, 5000, "bird"])
            writer.writerow([2.0, 3.0, 2000, 6000, "frog"])

        annotations = load_csv_annotations(str(csv_file))

        assert len(annotations) == 2
        assert annotations[0].start_time == 0.5
        assert annotations[0].label == "bird"

    def test_save_csv_annotations(self, temp_dir):
        """Test saving annotations to CSV."""
        annotations = [
            Annotation(start_time=1.0, end_time=2.0, label="bird"),
            Annotation(start_time=3.0, end_time=4.0, label="frog"),
        ]

        output_path = temp_dir / "output.csv"
        result = save_csv_annotations(annotations, str(output_path))

        assert Path(result).exists()

    def test_csv_with_custom_columns(self, temp_dir):
        """Test loading CSV with custom column names."""
        csv_file = temp_dir / "custom.csv"
        with open(csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["begin", "end", "species"])
            writer.writerow([1.0, 2.0, "bird"])

        annotations = load_csv_annotations(
            str(csv_file),
            start_time_col="begin",
            end_time_col="end",
            label_col="species"
        )

        assert len(annotations) == 1
        assert annotations[0].label == "bird"

    def test_csv_round_trip(self, temp_dir):
        """Test CSV save and load preserves data."""
        original = [
            Annotation(start_time=1.0, end_time=2.0, low_freq=1000,
                      high_freq=5000, label="test", confidence=0.95,
                      notes="test note"),
        ]

        output_path = temp_dir / "round_trip.csv"
        save_csv_annotations(original, str(output_path))
        loaded = load_csv_annotations(str(output_path))

        assert len(loaded) == 1
        assert loaded[0].start_time == 1.0
        assert loaded[0].label == "test"
        assert loaded[0].confidence == 0.95
        assert loaded[0].notes == "test note"


class TestLabelOperations:
    """Tests for label generation and encoding."""

    def test_get_unique_labels(self):
        """Test extracting unique labels."""
        annotations = [
            Annotation(start_time=0, end_time=1, label="bird"),
            Annotation(start_time=1, end_time=2, label="frog"),
            Annotation(start_time=2, end_time=3, label="bird"),
            Annotation(start_time=3, end_time=4, label="insect"),
        ]

        labels = get_unique_labels(annotations)
        assert labels == ["bird", "frog", "insect"]

    def test_create_label_map(self):
        """Test creating label to index mapping."""
        labels = ["bird", "frog", "insect"]
        label_map = create_label_map(labels)

        assert label_map["bird"] == 0
        assert label_map["frog"] == 1
        assert label_map["insect"] == 2

    def test_annotations_to_one_hot(self):
        """Test one-hot encoding of annotations."""
        annotations = [
            Annotation(start_time=0, end_time=1, label="bird"),
            Annotation(start_time=1, end_time=2, label="frog"),
            Annotation(start_time=2, end_time=3, label="bird"),
        ]
        label_map = {"bird": 0, "frog": 1}

        one_hot = annotations_to_one_hot(annotations, label_map)

        assert one_hot.shape == (3, 2)
        assert one_hot[0, 0] == 1.0  # bird
        assert one_hot[0, 1] == 0.0
        assert one_hot[1, 0] == 0.0
        assert one_hot[1, 1] == 1.0  # frog
        assert one_hot[2, 0] == 1.0  # bird

    def test_generate_clip_labels(self):
        """Test generating labels for a clip."""
        annotations = [
            Annotation(start_time=1.0, end_time=3.0, label="bird"),
            Annotation(start_time=2.5, end_time=4.0, label="frog"),
        ]
        label_map = {"bird": 0, "frog": 1}

        # Clip from 2.0 to 3.5 overlaps both annotations
        labels = generate_clip_labels(annotations, 2.0, 3.5, label_map, multi_label=True)

        assert labels[0] == 1.0  # bird
        assert labels[1] == 1.0  # frog

    def test_generate_clip_labels_single(self):
        """Test single-label clip generation."""
        annotations = [
            Annotation(start_time=1.0, end_time=3.0, label="bird"),
            Annotation(start_time=2.0, end_time=5.0, label="frog"),
        ]
        label_map = {"bird": 0, "frog": 1}

        # Clip from 2.0 to 4.0 - frog has more overlap
        labels = generate_clip_labels(annotations, 2.0, 4.0, label_map, multi_label=False)

        assert labels[0] == 0.0  # bird has less overlap
        assert labels[1] == 1.0  # frog has more overlap

    def test_generate_frame_labels(self):
        """Test frame-level label generation."""
        annotations = [
            Annotation(start_time=0.0, end_time=1.0, label="bird"),
        ]
        label_map = {"bird": 0, "frog": 1}

        # 5 second audio, 1 second frames, 0.5 second hop
        labels = generate_frame_labels(
            annotations, total_duration=5.0,
            frame_size=1.0, hop_length=0.5,
            label_map=label_map
        )

        assert labels.shape[0] == 2  # num_classes
        assert labels.shape[1] == 9  # num_frames
        assert labels[0, 0] == 1.0  # Frame 0 overlaps with bird
        assert labels[0, 1] == 1.0  # Frame 1 overlaps with bird
        assert labels[0, 3] == 0.0  # Frame 3 doesn't overlap


class TestLabelRemapping:
    """Tests for label remapping utilities."""

    def test_remap_labels(self):
        """Test basic label remapping."""
        annotations = [
            Annotation(start_time=0, end_time=1, label="bird_song"),
            Annotation(start_time=1, end_time=2, label="bird_call"),
            Annotation(start_time=2, end_time=3, label="frog"),
        ]
        mapping = {"bird_song": "bird", "bird_call": "bird"}

        remapped = remap_labels(annotations, mapping, keep_unmapped=True)

        assert len(remapped) == 3
        assert remapped[0].label == "bird"
        assert remapped[1].label == "bird"
        assert remapped[2].label == "frog"

    def test_remap_labels_drop_unmapped(self):
        """Test label remapping with dropping unmapped labels."""
        annotations = [
            Annotation(start_time=0, end_time=1, label="bird"),
            Annotation(start_time=1, end_time=2, label="unknown"),
        ]
        mapping = {"bird": "aves"}

        remapped = remap_labels(annotations, mapping, keep_unmapped=False)

        assert len(remapped) == 1
        assert remapped[0].label == "aves"

    def test_filter_labels_include(self):
        """Test filtering to include specific labels."""
        annotations = [
            Annotation(start_time=0, end_time=1, label="bird"),
            Annotation(start_time=1, end_time=2, label="frog"),
            Annotation(start_time=2, end_time=3, label="insect"),
        ]

        filtered = filter_labels(annotations, include_labels={"bird", "frog"})

        assert len(filtered) == 2
        labels = {a.label for a in filtered}
        assert labels == {"bird", "frog"}

    def test_filter_labels_exclude(self):
        """Test filtering to exclude specific labels."""
        annotations = [
            Annotation(start_time=0, end_time=1, label="bird"),
            Annotation(start_time=1, end_time=2, label="noise"),
            Annotation(start_time=2, end_time=3, label="frog"),
        ]

        filtered = filter_labels(annotations, exclude_labels={"noise"})

        assert len(filtered) == 2
        labels = {a.label for a in filtered}
        assert "noise" not in labels

    def test_save_and_load_label_mapping(self, temp_dir):
        """Test saving and loading label mapping."""
        mapping = {
            "bird_song": "bird",
            "bird_call": "bird",
            "frog_croak": "frog",
        }

        filepath = temp_dir / "mapping.csv"
        save_label_mapping(mapping, str(filepath))

        loaded = load_label_mapping(str(filepath))

        assert loaded == mapping


class TestSummarizeAnnotations:
    """Tests for annotation summary statistics."""

    def test_summarize_annotations(self):
        """Test generating summary statistics."""
        annotations = [
            Annotation(start_time=0, end_time=1, label="bird"),
            Annotation(start_time=1, end_time=3, label="frog"),
            Annotation(start_time=3, end_time=4, label="bird"),
        ]

        summary = summarize_annotations(annotations)

        assert summary["total_annotations"] == 3
        assert summary["unique_labels"] == 2
        assert summary["labels"]["bird"] == 2
        assert summary["labels"]["frog"] == 1
        assert summary["total_duration"] == 4.0
        assert summary["min_duration"] == 1.0
        assert summary["max_duration"] == 2.0

    def test_summarize_empty_annotations(self):
        """Test summary of empty annotation list."""
        summary = summarize_annotations([])

        assert summary["total_annotations"] == 0
        assert summary["unique_labels"] == 0


@pytest.fixture
def temp_dir(tmp_path):
    """Provide a temporary directory for tests."""
    return tmp_path
