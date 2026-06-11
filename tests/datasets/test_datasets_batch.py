"""Coverage tests for :mod:`bioamla.datasets.batch`.

Exercises ``batch_convert_annotations`` validation, format auto-detection, and
the per-file conversion wiring through :func:`bioamla.batch.run_batch`.
"""

from __future__ import annotations

import pytest

from bioamla.datasets.annotations import Annotation, save_csv_annotations
from bioamla.datasets.batch import _format_for_path, batch_convert_annotations
from bioamla.exceptions import InvalidInputError, NotFoundError


def _write_csv_annotations(path) -> None:
    anns = [
        Annotation(start_time=0.0, end_time=1.0, label="a"),
        Annotation(start_time=1.0, end_time=2.0, label="b"),
    ]
    save_csv_annotations(anns, str(path))


class TestFormatForPath:
    def test_txt_is_raven(self, tmp_path) -> None:
        assert _format_for_path(tmp_path / "x.txt") == "raven"

    def test_csv_is_csv(self, tmp_path) -> None:
        assert _format_for_path(tmp_path / "x.csv") == "csv"

    def test_other_defaults_to_csv(self, tmp_path) -> None:
        assert _format_for_path(tmp_path / "x.dat") == "csv"


class TestValidation:
    def test_bad_to_format_raises(self, tmp_path) -> None:
        with pytest.raises(InvalidInputError, match="Invalid to_format"):
            batch_convert_annotations(str(tmp_path), str(tmp_path / "out"), to_format="json")

    def test_bad_from_format_raises(self, tmp_path) -> None:
        with pytest.raises(InvalidInputError, match="Invalid from_format"):
            batch_convert_annotations(
                str(tmp_path), str(tmp_path / "out"), to_format="csv", from_format="json"
            )

    def test_missing_input_dir_raises(self, tmp_path) -> None:
        with pytest.raises(NotFoundError, match="Input directory not found"):
            batch_convert_annotations(
                str(tmp_path / "nope"), str(tmp_path / "out"), to_format="csv"
            )


class TestConversion:
    def test_csv_to_raven(self, tmp_path) -> None:
        in_dir = tmp_path / "in"
        in_dir.mkdir()
        _write_csv_annotations(in_dir / "f1.csv")
        _write_csv_annotations(in_dir / "f2.csv")
        out_dir = tmp_path / "out"

        result = batch_convert_annotations(str(in_dir), str(out_dir), to_format="raven")

        assert result.total_files == 2
        assert result.successful == 2
        assert (out_dir / "f1.txt").exists()
        assert (out_dir / "f2.txt").exists()

    def test_explicit_from_format(self, tmp_path) -> None:
        in_dir = tmp_path / "in"
        in_dir.mkdir()
        _write_csv_annotations(in_dir / "f1.csv")
        out_dir = tmp_path / "out"

        result = batch_convert_annotations(
            str(in_dir), str(out_dir), to_format="csv", from_format="csv"
        )
        assert result.successful == 1
        assert (out_dir / "f1.csv").exists()

    def test_recursive_preserves_relative_structure(self, tmp_path) -> None:
        in_dir = tmp_path / "in"
        sub = in_dir / "nested"
        sub.mkdir(parents=True)
        _write_csv_annotations(sub / "deep.csv")
        out_dir = tmp_path / "out"

        result = batch_convert_annotations(
            str(in_dir), str(out_dir), to_format="raven", recursive=True
        )
        assert result.successful == 1
        assert (out_dir / "nested" / "deep.txt").exists()

    def test_progress_callback_invoked(self, tmp_path) -> None:
        in_dir = tmp_path / "in"
        in_dir.mkdir()
        _write_csv_annotations(in_dir / "f1.csv")
        out_dir = tmp_path / "out"
        calls: list[tuple[int, int]] = []

        batch_convert_annotations(
            str(in_dir),
            str(out_dir),
            to_format="raven",
            on_progress=lambda done, total: calls.append((done, total)),
        )
        assert calls  # callback fired at least once
