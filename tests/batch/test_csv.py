"""Tests for CSV-metadata batch mode.

Covers the engine helpers in :mod:`bioamla.batch` (load/run/merge/write,
relative-path resolution, segment row expansion) and the real ``batch indices
calculate --input-file`` CLI round-trip.
"""

import csv
from pathlib import Path

import pytest

from bioamla.batch import (
    InvalidInputError,
    NotFoundError,
    expand_row_for_segments,
    load_csv,
    merge_analysis_results,
    resolve_output_path,
    run_csv_batch,
    update_row_path,
    write_csv,
)


def _make_meta_csv(csv_dir: Path, names) -> Path:
    """Write a metadata CSV with a ``file_name`` column and a ``site`` column."""
    csv_path = csv_dir / "meta.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["file_name", "site"])
        writer.writeheader()
        for i, name in enumerate(names):
            writer.writerow({"file_name": name, "site": f"site_{i}"})
    return csv_path


class TestLoadCsv:
    def test_requires_file_name_column(self, tmp_path):
        bad = tmp_path / "bad.csv"
        bad.write_text("path,site\nfoo.wav,A\n", encoding="utf-8")
        with pytest.raises(InvalidInputError):
            load_csv(str(bad))

    def test_missing_csv_raises(self, tmp_path):
        with pytest.raises(NotFoundError):
            load_csv(str(tmp_path / "missing.csv"))

    def test_paths_resolved_relative_to_csv_dir(self, test_audio_dir):
        csv_dir = Path(test_audio_dir)
        csv_path = _make_meta_csv(csv_dir, ["audio_0.wav", "audio_1.wav"])
        ctx = load_csv(str(csv_path))
        assert len(ctx.rows) == 2
        # file_name is relative; file_path is resolved absolute under csv_dir.
        for row in ctx.rows:
            assert row.file_path.is_absolute()
            assert row.file_path.parent == csv_dir.resolve()
            assert row.file_path.exists()
        # Other columns are preserved as metadata fields.
        assert ctx.rows[0].metadata_fields["site"] == "site_0"


class TestCsvRoundTrip:
    def test_merge_writes_result_columns_in_place(self, test_audio_dir):
        csv_dir = Path(test_audio_dir)
        csv_path = _make_meta_csv(csv_dir, ["audio_0.wav", "audio_1.wav", "audio_2.wav"])
        ctx = load_csv(str(csv_path))  # in-place (no output_dir)

        def _process(row):
            # Fake analysis result merged per row.
            merge_analysis_results(row, {"score": 0.5, "label": "frog"})
            return str(row.file_path)

        result = run_csv_batch(ctx, _process)
        assert result.successful == 3
        assert result.failed == 0

        out_csv = write_csv(ctx)
        assert out_csv == csv_path  # in-place

        with out_csv.open(encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        header = rows[0].keys()
        # file_name first, original column preserved, new columns merged.
        assert list(header)[0] == "file_name"
        assert "site" in header
        assert "score" in header and "label" in header
        assert rows[0]["label"] == "frog"
        # Paths stayed relative to the CSV dir.
        assert rows[0]["file_name"] == "audio_0.wav"

    def test_output_dir_writes_copy_and_resolves_paths(self, test_audio_dir, tmp_path):
        csv_dir = Path(test_audio_dir)
        csv_path = _make_meta_csv(csv_dir, ["audio_0.wav"])
        out_dir = tmp_path / "out"
        ctx = load_csv(str(csv_path), str(out_dir))

        def _process(row):
            merge_analysis_results(row, {"score": 1.0})
            return None

        run_csv_batch(ctx, _process)
        out_csv = write_csv(ctx)
        assert out_csv == out_dir / "meta.csv"
        assert out_csv.exists()
        # Original CSV untouched (no score column).
        with csv_path.open(encoding="utf-8") as f:
            assert "score" not in next(csv.reader(f))

    def test_missing_file_recorded_as_failure(self, test_audio_dir):
        csv_dir = Path(test_audio_dir)
        csv_path = _make_meta_csv(csv_dir, ["audio_0.wav", "does_not_exist.wav"])
        ctx = load_csv(str(csv_path))

        result = run_csv_batch(ctx, lambda row: None, quiet=True)
        assert result.successful == 1
        assert result.failed == 1
        assert any("does_not_exist.wav" in e for e in result.errors)


class TestTransformPathUpdate:
    def test_update_row_path_makes_relative(self, test_audio_dir, tmp_path):
        csv_dir = Path(test_audio_dir)
        csv_path = _make_meta_csv(csv_dir, ["audio_0.wav"])
        out_dir = tmp_path / "converted"
        out_dir.mkdir()
        ctx = load_csv(str(csv_path), str(out_dir))
        row = ctx.rows[0]
        new_path = resolve_output_path(row.file_path, ctx, new_extension=".flac")
        assert new_path.parent == out_dir.resolve()
        update_row_path(row, new_path, ctx)
        # file_name now relative to output_dir.
        assert row.file_name == "audio_0.flac"


class TestSegmentExpansion:
    def test_one_row_expands_into_many(self, test_audio_dir, tmp_path):
        from bioamla.audio.batch import segment_audio_file

        csv_dir = Path(test_audio_dir)
        # audio_0.wav is 1s; segment into 0.4s chunks -> 3 segments.
        csv_path = _make_meta_csv(csv_dir, ["audio_0.wav"])
        out_dir = tmp_path / "segs"
        ctx = load_csv(str(csv_path), str(out_dir))
        parent_row = ctx.rows[0]

        segments = segment_audio_file(
            str(parent_row.file_path), str(out_dir), duration=0.4, overlap=0.0
        )
        assert len(segments) >= 2

        new_rows = expand_row_for_segments(parent_row, segments, ctx)
        assert len(new_rows) == len(segments)
        # Each expanded row carries segment metadata and inherits parent fields.
        for i, row in enumerate(new_rows):
            md = row.metadata_fields
            assert md["parent_file"] == parent_row.file_name
            assert md["segment_id"] == i
            assert "start_time" in md and "end_time" in md and "duration" in md
            assert md["site"] == "site_0"  # inherited parent column

        ctx.rows = new_rows
        for field in ("parent_file", "segment_id", "start_time", "end_time", "duration"):
            if field not in ctx.fieldnames:
                ctx.fieldnames.append(field)
        out_csv = write_csv(ctx)
        with out_csv.open(encoding="utf-8") as f:
            written = list(csv.DictReader(f))
        assert len(written) == len(segments)
        assert "parent_file" in written[0]


class TestIndicesCsvCli:
    """End-to-end CLI round-trip for ``batch indices calculate --input-file``."""

    def test_indices_csv_merges_columns(self, test_audio_dir):
        from click.testing import CliRunner

        from bioamla.cli.commands.batch import batch

        csv_dir = Path(test_audio_dir)
        csv_path = _make_meta_csv(csv_dir, ["audio_0.wav", "audio_1.wav"])

        runner = CliRunner()
        res = runner.invoke(
            batch,
            ["indices", "calculate", "--input-file", str(csv_path), "--quiet"],
        )
        assert res.exit_code == 0, res.output

        with csv_path.open(encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 2
        header = list(rows[0].keys())
        assert header[0] == "file_name"
        assert "site" in header
        # Acoustic index columns were merged.
        assert "aci" in header and "ndsi" in header
        # filepath/filename redundant columns were dropped.
        assert "filepath" not in header and "filename" not in header
        # Paths stayed relative to the CSV directory.
        assert rows[0]["file_name"] == "audio_0.wav"

    def test_missing_file_name_column_errors(self, tmp_path):
        from click.testing import CliRunner

        from bioamla.cli.commands.batch import batch

        bad = tmp_path / "bad.csv"
        bad.write_text("path,site\nfoo.wav,A\n", encoding="utf-8")
        runner = CliRunner()
        res = runner.invoke(batch, ["indices", "calculate", "--input-file", str(bad), "--quiet"])
        assert res.exit_code == 1
        assert "file_name" in res.output
