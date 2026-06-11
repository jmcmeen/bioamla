"""Tests for the `annotation generate-frame-labels` CLI command."""

import csv

import numpy as np
from click.testing import CliRunner

from bioamla.cli.commands.annotation import annotation
from bioamla.datasets import Annotation, save_bioamla_annotations


def _write_annotations(path, duration=2.0):
    anns = [
        Annotation(start_time=0.0, end_time=0.5, label="call"),
        Annotation(start_time=0.4, end_time=1.0, label="chorus"),
    ]
    save_bioamla_annotations(anns, str(path), metadata={"duration": duration})


class TestGenerateFrameLabelsCli:
    def test_csv_output_shape_and_header(self, tmp_path):
        ann = tmp_path / "a.json"
        _write_annotations(ann, duration=2.0)
        out = tmp_path / "frames.csv"

        res = CliRunner().invoke(
            annotation,
            ["generate-frame-labels", str(ann), str(out), "--frame-size", "0.5", "--quiet"],
        )
        assert res.exit_code == 0, res.output

        with out.open() as f:
            rows = list(csv.reader(f))
        assert rows[0] == ["frame_start", "frame_end", "call", "chorus"]
        # 2.0s / 0.5 hop -> (2.0-0.5)/0.5 + 1 = 4 frames.
        assert len(rows) - 1 == 4
        # Multi-hot values are 0/1 ints.
        assert all(v in ("0", "1") for r in rows[1:] for v in r[2:])

    def test_numpy_output_and_label_map(self, tmp_path):
        ann = tmp_path / "a.json"
        _write_annotations(ann, duration=2.0)
        out = tmp_path / "frames.npy"

        res = CliRunner().invoke(
            annotation,
            [
                "generate-frame-labels",
                str(ann),
                str(out),
                "--frame-size",
                "0.5",
                "--hop-length",
                "0.25",
                "--format",
                "numpy",
                "--quiet",
            ],
        )
        assert res.exit_code == 0, res.output

        arr = np.load(out)
        # (num_classes, num_frames); 2 classes, (2.0-0.5)/0.25 + 1 = 7 frames.
        assert arr.shape == (2, 7)
        label_map = tmp_path / "frames.labels.csv"
        assert label_map.exists()
        with label_map.open() as f:
            rows = list(csv.DictReader(f))
        assert {r["label"] for r in rows} == {"call", "chorus"}

    def test_duration_required_without_metadata(self, tmp_path):
        # A raven file carries no duration metadata, so omitting --audio-duration errors.
        from bioamla.datasets import save_raven_selection_table

        ann = tmp_path / "a.txt"
        save_raven_selection_table(
            [Annotation(start_time=0.0, end_time=0.5, label="call")], str(ann)
        )
        res = CliRunner().invoke(
            annotation,
            ["generate-frame-labels", str(ann), str(tmp_path / "f.csv"), "--frame-size", "0.5"],
        )
        assert res.exit_code != 0
        assert "audio-duration" in res.output

    def test_duration_inferred_from_bioamla_metadata(self, tmp_path):
        ann = tmp_path / "a.json"
        _write_annotations(ann, duration=3.0)
        out = tmp_path / "frames.csv"
        res = CliRunner().invoke(
            annotation,
            ["generate-frame-labels", str(ann), str(out), "--frame-size", "1.0", "--quiet"],
        )
        assert res.exit_code == 0, res.output
        with out.open() as f:
            n_rows = sum(1 for _ in f) - 1
        # 3.0s, frame 1.0, hop 1.0 -> 3 frames.
        assert n_rows == 3
