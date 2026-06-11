"""Coverage tests for :mod:`bioamla.datasets.annotations`.

Covers the Annotation geometry helpers, exclude-labels in
``predictions_to_annotations``, JSON/Parquet writers, directory loading, and the
Raven label auto-detect / custom-field branches not hit by ``test_datasets.py``.
"""

from __future__ import annotations

import json

import pytest

from bioamla.datasets.annotations import (
    Annotation,
    create_annotation,
    load_annotations_from_directory,
    load_csv_annotations,
    load_raven_selection_table,
    predictions_to_annotations,
    save_csv_annotations,
    save_json_annotations,
    save_raven_selection_table,
)
from bioamla.exceptions import AnnotationError, NotFoundError

# ---------------------------------------------------------------------------
# Annotation geometry helpers
# ---------------------------------------------------------------------------


class TestAnnotationGeometry:
    def test_center_freq_and_bandwidth_none(self) -> None:
        a = Annotation(start_time=0.0, end_time=1.0)
        assert a.center_freq is None
        assert a.bandwidth is None

    def test_center_freq_present(self) -> None:
        a = Annotation(start_time=0.0, end_time=1.0, low_freq=100.0, high_freq=300.0)
        assert a.center_freq == 200.0
        assert a.center_time == 0.5

    def test_overlaps_time(self) -> None:
        a = Annotation(start_time=0.0, end_time=1.0)
        b = Annotation(start_time=0.5, end_time=1.5)
        c = Annotation(start_time=2.0, end_time=3.0)
        assert a.overlaps_time(b)
        assert not a.overlaps_time(c)

    def test_overlaps_freq_with_and_without_bounds(self) -> None:
        a = Annotation(start_time=0, end_time=1, low_freq=100, high_freq=200)
        b = Annotation(start_time=0, end_time=1, low_freq=150, high_freq=250)
        d = Annotation(start_time=0, end_time=1, low_freq=300, high_freq=400)
        assert a.overlaps_freq(b)
        assert not a.overlaps_freq(d)
        # Missing bounds -> assume overlap.
        nofreq = Annotation(start_time=0, end_time=1)
        assert a.overlaps_freq(nofreq)
        assert nofreq.overlaps_freq(a)

    def test_overlaps_combined(self) -> None:
        a = Annotation(start_time=0, end_time=1, low_freq=100, high_freq=200)
        b = Annotation(start_time=0.5, end_time=1.5, low_freq=150, high_freq=250)
        assert a.overlaps(b)

    def test_contains_time_and_freq(self) -> None:
        a = Annotation(start_time=0.0, end_time=1.0, low_freq=100.0, high_freq=200.0)
        assert a.contains_time(0.5)
        assert not a.contains_time(2.0)
        assert a.contains_freq(150.0)
        assert not a.contains_freq(500.0)
        nofreq = Annotation(start_time=0, end_time=1)
        assert nofreq.contains_freq(9999.0)  # no bounds -> always True


class TestCreateAnnotationValidation:
    def test_bad_time_raises(self) -> None:
        with pytest.raises(AnnotationError, match="end_time must be greater"):
            create_annotation(1.0, 0.5)

    def test_bad_freq_raises(self) -> None:
        with pytest.raises(AnnotationError, match="high_freq must be greater"):
            create_annotation(0.0, 1.0, low_freq=300.0, high_freq=100.0)


# ---------------------------------------------------------------------------
# predictions_to_annotations exclude branch
# ---------------------------------------------------------------------------


class TestPredictionsExclude:
    def test_exclude_labels_drops_rows(self) -> None:
        rows = [
            {"start": 0, "stop": 1, "prediction": "bird"},
            {"start": 1, "stop": 2, "prediction": "noise"},
        ]
        anns = predictions_to_annotations(rows, exclude_labels=["noise"])
        assert [a.label for a in anns] == ["bird"]


# ---------------------------------------------------------------------------
# JSON / Parquet writers
# ---------------------------------------------------------------------------


def _anns() -> list[Annotation]:
    return [
        Annotation(start_time=0.0, end_time=1.0, label="a", custom_fields={"k": "v"}),
        Annotation(start_time=1.0, end_time=2.0, label="b"),
    ]


class TestWriters:
    def test_save_json(self, tmp_path) -> None:
        out = tmp_path / "anns.json"
        save_json_annotations(_anns(), str(out))
        data = json.loads(out.read_text())
        assert len(data) == 2
        assert data[0]["label"] == "a"

    def test_save_parquet(self, tmp_path) -> None:
        pytest.importorskip("pandas")
        pytest.importorskip("pyarrow")
        from bioamla.datasets.annotations import save_parquet_annotations

        out = tmp_path / "anns.parquet"
        save_parquet_annotations(_anns(), str(out))
        assert out.exists()

        import pandas as pd

        df = pd.read_parquet(str(out))
        assert len(df) == 2


# ---------------------------------------------------------------------------
# Raven auto label-detect + custom field roundtrip
# ---------------------------------------------------------------------------


class TestRavenBranches:
    def test_label_autodetect_species_column(self, tmp_path) -> None:
        path = tmp_path / "sel.txt"
        path.write_text(
            "Selection\tBegin Time (s)\tEnd Time (s)\tSpecies\n"
            "1\t0.0\t1.0\tRobin\n",
        )
        anns = load_raven_selection_table(str(path))
        assert anns[0].label == "Robin"

    def test_skips_rows_with_missing_times(self, tmp_path) -> None:
        path = tmp_path / "sel.txt"
        path.write_text(
            "Selection\tBegin Time (s)\tEnd Time (s)\tAnnotation\n"
            "1\t\t1.0\tRobin\n"
            "2\t0.0\t2.0\tWren\n",
        )
        anns = load_raven_selection_table(str(path))
        assert len(anns) == 1
        assert anns[0].label == "Wren"

    def test_custom_fields_survive_roundtrip(self, tmp_path) -> None:
        anns = [
            Annotation(
                start_time=0.0,
                end_time=1.0,
                low_freq=100.0,
                high_freq=200.0,
                label="x",
                custom_fields={"site": "north"},
            )
        ]
        out = tmp_path / "sel.txt"
        save_raven_selection_table(anns, str(out))
        loaded = load_raven_selection_table(str(out))
        assert loaded[0].custom_fields.get("site") == "north"


class TestCsvLabelColMissingTimes:
    def test_skips_missing_times(self, tmp_path) -> None:
        path = tmp_path / "a.csv"
        path.write_text("start_time,end_time,label\n,1.0,a\n0.0,2.0,b\n")
        anns = load_csv_annotations(str(path))
        assert len(anns) == 1
        assert anns[0].label == "b"


# ---------------------------------------------------------------------------
# Directory loading
# ---------------------------------------------------------------------------


class TestLoadFromDirectory:
    def test_missing_dir_raises(self, tmp_path) -> None:
        with pytest.raises(NotFoundError, match="Directory not found"):
            load_annotations_from_directory(str(tmp_path / "nope"))

    def test_loads_csv_files(self, tmp_path) -> None:
        save_csv_annotations(_anns(), str(tmp_path / "f1.csv"))
        save_csv_annotations(_anns(), str(tmp_path / "f2.csv"))
        results = load_annotations_from_directory(
            str(tmp_path), file_pattern="*.csv", format="csv"
        )
        assert set(results.keys()) == {"f1.csv", "f2.csv"}
        assert len(results["f1.csv"]) == 2

    def test_bad_file_is_skipped(self, tmp_path) -> None:
        # A .txt that isn't a valid Raven table -> load attempt logs and skips.
        (tmp_path / "bad.txt").write_text("garbage,not,raven\n")
        results = load_annotations_from_directory(
            str(tmp_path), file_pattern="*.txt", format="raven"
        )
        # Parsed but yields zero annotations (no recognizable time columns).
        assert results.get("bad.txt") == [] or "bad.txt" not in results
