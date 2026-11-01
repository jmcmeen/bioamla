"""Training labels written as NumPy arrays: the array and its label map."""

from bioamla.datasets import (
    Annotation,
    clip_labels_file,
    frame_labels_file,
    save_bioamla_annotations,
)


def _annotations(path):
    save_bioamla_annotations(
        [Annotation(start_time=0.1, end_time=0.6, label="frog"),
         Annotation(start_time=1.0, end_time=1.5, label="toad")],
        str(path),
        metadata={"duration": 2.0},
    )  # fmt: skip
    return path


def test_label_maps_are_named_after_their_data_file(tmp_path):
    ann = _annotations(tmp_path / "a.json")
    one = clip_labels_file(ann, tmp_path / "run.v1", clip_duration=0.5, output_format="numpy")
    two = frame_labels_file(ann, tmp_path / "run.v2", frame_size=0.5, output_format="numpy")
    assert (one.output, one.label_map) == (
        str(tmp_path / "run.v1.npy"),
        str(tmp_path / "run.v1.labels.csv"),
    )
    assert two.label_map == str(tmp_path / "run.v2.labels.csv")
    assert not (tmp_path / "run.labels.csv").exists()  # the name both used to share


def test_a_csv_name_becomes_npy_and_npy_names_are_unchanged(tmp_path):
    ann = _annotations(tmp_path / "a.json")
    csv_named = clip_labels_file(ann, tmp_path / "labels.csv", clip_duration=0.5,
                                 output_format="numpy")  # fmt: skip
    assert (csv_named.output, csv_named.label_map) == (
        str(tmp_path / "labels.npy"),
        str(tmp_path / "labels.labels.csv"),
    )
    plain = clip_labels_file(ann, tmp_path / "plain.npy", clip_duration=0.5, output_format="numpy")
    assert (plain.output, plain.label_map) == (
        str(tmp_path / "plain.npy"),
        str(tmp_path / "plain.labels.csv"),
    )
