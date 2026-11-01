"""The public CSV-mode runners in bioamla.batch."""

import csv
from pathlib import Path

import pytest

from bioamla.batch import (
    SegmentInfo,
    load_csv,
    run_csv_merge,
    run_csv_predict_segments,
    run_csv_process,
    run_csv_segment,
    run_csv_transform,
)


@pytest.fixture
def manifest(tmp_path) -> Path:
    for name in ("a.wav", "b.wav"):
        (tmp_path / name).write_bytes(b"x")
    path = tmp_path / "files.csv"
    path.write_text("file_name,site\na.wav,north\nb.wav,south\n")
    return path


def _rows(path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def test_merge_adds_columns_and_clears_stale_ones(manifest, tmp_path):
    out = tmp_path / "out"
    first = run_csv_merge(load_csv(manifest, str(out)), lambda p: {"size": p.stat().st_size})
    assert first.successful == 2
    written = first.metadata["metadata_csv"]
    assert [r["size"] for r in _rows(written)] == ["1", "1"]

    def fail_b(p):
        if p.name == "b.wav":
            raise OSError("boom")
        return {"size": 9}

    again = run_csv_merge(load_csv(written, str(out)), fail_b, clear=("size",))
    assert again.failed == 1
    assert [r["size"] for r in _rows(again.metadata["metadata_csv"])] == ["9", ""]


def test_transform_updates_file_name(manifest, tmp_path):
    out = tmp_path / "out"

    def copy(src: Path, dst: Path) -> Path:
        dst.write_bytes(src.read_bytes())
        return dst

    result = run_csv_transform(load_csv(manifest, str(out)), copy, new_extension=".bin")
    rows = _rows(result.metadata["metadata_csv"])
    assert [r["file_name"] for r in rows] == ["a.bin", "b.bin"]
    assert (out / "a.bin").exists()


def test_process_keeps_rows(manifest, tmp_path):
    seen = []
    result = run_csv_process(load_csv(manifest, str(tmp_path / "out")), seen.append)
    assert sorted(p.name for p in seen) == ["a.wav", "b.wav"]
    assert [r["site"] for r in _rows(result.metadata["metadata_csv"])] == ["north", "south"]


def test_segment_expands_rows(manifest, tmp_path):
    def two_segments(src: Path, seg_dir: Path) -> list[SegmentInfo]:
        return [
            SegmentInfo(seg_dir / f"{src.stem}_{i}.wav", i, float(i), i + 1.0, 1.0)
            for i in range(2)
        ]

    result = run_csv_segment(load_csv(manifest, str(tmp_path / "out")), two_segments)
    rows = _rows(result.metadata["metadata_csv"])
    assert len(rows) == 4
    assert {"parent_file", "segment_id", "start_time", "end_time", "duration"} <= set(rows[0])


def test_predict_segments_expands_rows_without_files(manifest, tmp_path):
    def spans(src: Path) -> list[dict]:
        return [{"start_time": 0.0, "end_time": 1.0, "prediction": "frog", "confidence": 0.9}]

    result = run_csv_predict_segments(load_csv(manifest, str(tmp_path / "out")), spans)
    rows = _rows(result.metadata["metadata_csv"])
    # file_name is re-resolved relative to where the CSV is written.
    assert [(r["file_name"], r["prediction"], r["segment_id"]) for r in rows] == [
        ("../a.wav", "frog", "0"),
        ("../b.wav", "frog", "0"),
    ]
