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

    def test_missing_file_recorded_as_failure(self, test_audio_dir, caplog):
        csv_dir = Path(test_audio_dir)
        csv_path = _make_meta_csv(csv_dir, ["audio_0.wav", "does_not_exist.wav"])
        ctx = load_csv(str(csv_path))

        with caplog.at_level("WARNING", logger="bioamla"):
            result = run_csv_batch(ctx, lambda row: None)
        assert result.successful == 1
        assert result.failed == 1
        assert any("does_not_exist.wav" in e for e in result.errors)
        assert any("does_not_exist.wav" in r.getMessage() for r in caplog.records)


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
            ["index", "--input-file", str(csv_path), "--quiet"],
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
        res = runner.invoke(batch, ["index", "--input-file", str(bad), "--quiet"])
        assert res.exit_code == 1
        assert "file_name" in res.output


class TestPredictSegmentsCsvExpansion:
    """``batch models predict --input-file --segment-seconds`` row expansion.

    Exercises the CLI helper directly with a fake per-file predictor so no model
    download / torch is needed — the point under test is the CSV row expansion.
    """

    def test_expands_one_row_per_segment(self, test_audio_dir, tmp_path):
        from bioamla.batch import BatchConfig
        from bioamla.cli.commands.batch import _run_csv_predict_segments

        csv_path = _make_meta_csv(Path(test_audio_dir), ["audio_0.wav", "audio_1.wav"])
        out_dir = tmp_path / "out"
        config = BatchConfig(input_file=str(csv_path), output_dir=str(out_dir), quiet=True)

        def fake_predict(path):
            return [
                {"start_time": 0.0, "end_time": 3.0, "prediction": "frog", "confidence": 0.9},
                {"start_time": 3.0, "end_time": 6.0, "prediction": "bird", "confidence": 0.8},
            ]

        _run_csv_predict_segments(config, fake_predict)

        out_csv = out_dir / "meta.csv"
        rows = list(csv.DictReader(out_csv.read_text(encoding="utf-8").splitlines()))
        # 2 files x 2 segments each.
        assert len(rows) == 4
        # Original metadata column preserved on every expanded row.
        assert rows[0]["site"] == "site_0" and rows[2]["site"] == "site_1"
        # Per-segment prediction columns merged in.
        assert rows[0]["segment_id"] == "0" and rows[1]["segment_id"] == "1"
        assert rows[0]["prediction"] == "frog" and rows[1]["prediction"] == "bird"
        assert rows[0]["start_time"] == "0.0" and rows[1]["end_time"] == "6.0"
        assert "confidence" in rows[0]
        # Source file_name points at the source audio (segments are sub-spans, not new
        # files), relative to where the CSV was written.
        assert (out_dir / rows[0]["file_name"]).resolve() == (
            Path(test_audio_dir) / "audio_0.wav"
        ).resolve()

    def test_min_confidence_filter_drops_segments(self, test_audio_dir, tmp_path):
        from bioamla.batch import BatchConfig
        from bioamla.cli.commands.batch import _run_csv_predict_segments

        csv_path = _make_meta_csv(Path(test_audio_dir), ["audio_0.wav"])
        out_dir = tmp_path / "out"
        config = BatchConfig(input_file=str(csv_path), output_dir=str(out_dir), quiet=True)

        # Caller does the filtering; here we simulate one surviving segment.
        def fake_predict(path):
            return [{"start_time": 0.0, "end_time": 3.0, "prediction": "frog", "confidence": 0.95}]

        _run_csv_predict_segments(config, fake_predict)

        rows = list(csv.DictReader((out_dir / "meta.csv").read_text(encoding="utf-8").splitlines()))
        assert len(rows) == 1
        assert rows[0]["prediction"] == "frog"


class TestWriteCsvPaths:
    """Paths in a written metadata CSV resolve from where the CSV is written."""

    def _context(self, tmp_path, output_dir):
        from bioamla.batch import load_csv

        src = tmp_path / "in"
        (src / "sub").mkdir(parents=True)
        (src / "sub" / "a.wav").write_bytes(b"RIFF")
        (src / "a.features.csv").write_text("x")
        meta = src / "meta.csv"
        meta.write_text("file_name,site,features_file\nsub/a.wav,A,a.features.csv\n")
        return load_csv(meta, output_dir)

    def test_moved_csv_reroots_relative_paths(self, tmp_path):
        from bioamla.batch import write_csv

        out = tmp_path / "out" / "deeper"
        context = self._context(tmp_path, str(out))
        new_file = (out / "emb.npy").resolve()
        context.rows[0].metadata_fields["embedding_file"] = new_file  # a file just written
        written = write_csv(context)
        (row,) = list(csv.DictReader(written.open()))
        for column in ("file_name", "features_file", "embedding_file"):
            assert (written.parent / row[column]).resolve().name in (
                "a.wav",
                "a.features.csv",
                "emb.npy",
            )
        assert (written.parent / row["file_name"]).resolve() == (
            tmp_path / "in" / "sub" / "a.wav"
        ).resolve()
        assert row["embedding_file"] == "emb.npy"
        assert row["site"] == "A"  # other columns untouched

    def test_in_place_keeps_paths(self, tmp_path):
        from bioamla.batch import write_csv

        context = self._context(tmp_path, None)
        written = write_csv(context)
        (row,) = list(csv.DictReader(written.open()))
        assert (row["file_name"], row["features_file"]) == ("sub/a.wav", "a.features.csv")

    def test_symlinked_folder_is_kept(self, tmp_path):
        from bioamla.batch import load_csv, write_csv

        real = tmp_path / "vault" / "audio"
        real.mkdir(parents=True)
        (real / "a.wav").write_bytes(b"RIFF")
        proj = tmp_path / "proj"
        proj.mkdir()
        (proj / "audio").symlink_to(real)
        (proj / "meta.csv").write_text("file_name\naudio/a.wav\n")
        written = write_csv(load_csv(proj / "meta.csv", str(proj / "out")))
        (row,) = list(csv.DictReader(written.open()))
        assert row["file_name"] == "../audio/a.wav"  # through the link, not into the vault

    def test_output_dir_equal_to_csv_folder_changes_nothing(self, tmp_path):
        from bioamla.batch import load_csv, write_csv

        self._context(tmp_path, None)  # lays out in/sub/a.wav
        meta = tmp_path / "in" / "meta.csv"
        meta.write_text("file_name,features_file\nsub/../sub/a.wav,a.features.csv\n")
        written = write_csv(load_csv(meta, str(tmp_path / "in")))
        (row,) = list(csv.DictReader(written.open()))
        assert row["file_name"] == "sub/../sub/a.wav"  # left exactly as written
