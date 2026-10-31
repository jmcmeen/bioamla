"""Coverage tests for :mod:`bioamla.datasets.annotations`.

Covers the Annotation geometry helpers, exclude-labels in
``predictions_to_annotations``, JSON/Parquet writers, directory loading, and the
Raven label auto-detect / custom-field branches not hit by ``test_datasets.py``.
"""

from __future__ import annotations

import csv
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
            "Selection\tBegin Time (s)\tEnd Time (s)\tSpecies\n1\t0.0\t1.0\tRobin\n",
        )
        anns = load_raven_selection_table(str(path))
        assert anns[0].label == "Robin"
        assert anns[0].confidence is None

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
        results = load_annotations_from_directory(str(tmp_path), file_pattern="*.csv", format="csv")
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


class TestDetectionOutputAsAnnotations:
    """Files written by ``bioamla detect -o`` name the band frequency_low/_high."""

    def test_csv_band_columns(self, tmp_path) -> None:
        path = tmp_path / "det.csv"
        path.write_text(
            "start_time,end_time,confidence,frequency_low,frequency_high,label\n"
            "0.5,0.8,0.3,500.0,5000.0,band_energy\n"
        )
        (ann,) = load_csv_annotations(str(path))
        assert (ann.low_freq, ann.high_freq) == (500.0, 5000.0)
        assert "frequency_low" not in ann.custom_fields

    def test_standard_columns_win(self, tmp_path) -> None:
        path = tmp_path / "a.csv"
        path.write_text(
            "start_time,end_time,low_freq,high_freq,frequency_low,frequency_high,label\n"
            "0.5,0.8,100,200,500,5000,x\n"
        )
        (ann,) = load_csv_annotations(str(path))
        assert (ann.low_freq, ann.high_freq) == (100.0, 200.0)

    def test_json_list_of_detections(self, tmp_path) -> None:
        from bioamla.datasets.annotations import load_bioamla_annotations

        path = tmp_path / "det.json"
        path.write_text(json.dumps([
            {"start_time": 0.5, "end_time": 0.8, "frequency_low": 500.0,
             "frequency_high": 5000.0, "label": "band_energy", "confidence": 0.3},
        ]))  # fmt: skip
        annotations, metadata = load_bioamla_annotations(str(path))
        assert metadata == {}
        assert (annotations[0].low_freq, annotations[0].high_freq) == (500.0, 5000.0)
        assert annotations[0].label == "band_energy"

    def test_json_list_of_non_objects_rejected(self, tmp_path) -> None:
        from bioamla.datasets.annotations import load_bioamla_annotations

        path = tmp_path / "bad.json"
        path.write_text("[1, 2, 3]")
        with pytest.raises(AnnotationError, match="record 0 .* no start_time/end_time"):
            load_bioamla_annotations(str(path))

    def test_json_list_of_points_rejected(self, tmp_path) -> None:
        """Peak detections are points in time (time/amplitude/width), not regions."""
        from bioamla.datasets.annotations import load_bioamla_annotations

        path = tmp_path / "peaks.json"
        path.write_text(json.dumps([{"time": 1.2, "amplitude": 0.5, "width": 0.01}]))
        with pytest.raises(AnnotationError, match="no start_time/end_time"):
            load_bioamla_annotations(str(path))

    def test_empty_standard_column_falls_back_to_detector_column(self, tmp_path) -> None:
        path = tmp_path / "mixed.csv"
        path.write_text(
            "start_time,end_time,low_freq,high_freq,frequency_low,frequency_high,label\n"
            "0.5,0.8,,,500,5000,x\n"
        )
        (ann,) = load_csv_annotations(str(path))
        assert (ann.low_freq, ann.high_freq) == (500.0, 5000.0)

    def test_from_dict_treats_float_nan_as_missing(self) -> None:
        nan = float("nan")
        ann = Annotation.from_dict(
            {"start_time": 0.1, "end_time": 0.2, "low_freq": nan, "high_freq": nan,
             "frequency_low": 500.0, "frequency_high": 5000.0, "confidence": nan}
        )  # fmt: skip
        assert (ann.low_freq, ann.high_freq, ann.confidence) == (500.0, 5000.0, None)
        assert "frequency_low" not in ann.custom_fields


ARBIMON_EXAMPLE = (
    "id,recording,site,year,month,day,hour,min,species,songtype,x1,x2,y1,y2,validated,uri,"
    "score,site_id\n"
    "24735680,sabana_seca1-2014-07-15_05-50.flac,Sabana Seca,2014,7,15,5,50,"
    "Eleutherodactylus juanariveroi,Common Song,42.92266667,44.032,6562.5,8343.75,"
    "(not validated),https://example.org/24735680.png,0.300001416,506\n"
    "24721339,sabana_seca1-2014-06-29_05-10.flac,Sabana Seca,2014,6,29,5,10,"
    "Eleutherodactylus juanariveroi,Common Song,17.36,18.46933333,6562.5,8343.75,"
    "present,,0.30000146,506\n"
)


class TestArbimon:
    @pytest.fixture
    def export(self, tmp_path):
        path = tmp_path / "pattern_matching_rois.csv"
        path.write_text(ARBIMON_EXAMPLE)
        return path

    def test_load(self, export) -> None:
        from bioamla.datasets.annotations import load_arbimon_annotations

        first, second = load_arbimon_annotations(str(export))
        assert (first.start_time, first.end_time) == (42.92266667, 44.032)
        assert (first.low_freq, first.high_freq) == (6562.5, 8343.75)
        assert first.label == "Eleutherodactylus juanariveroi"
        assert first.confidence == pytest.approx(0.300001416)
        assert first.custom_fields["recording"] == "sabana_seca1-2014-07-15_05-50.flac"
        assert first.custom_fields["songtype"] == "Common Song"
        assert second.custom_fields["validated"] == "present"
        assert "uri" not in second.custom_fields  # empty cells aren't kept

    def test_round_trip_is_exact(self, export, tmp_path) -> None:
        from bioamla.datasets.annotations import (
            load_arbimon_annotations,
            save_arbimon_annotations,
        )

        out = tmp_path / "out.csv"
        save_arbimon_annotations(load_arbimon_annotations(str(export)), str(out))
        assert out.read_text().splitlines() == ARBIMON_EXAMPLE.splitlines()

    def test_write_from_plain_annotations(self, tmp_path) -> None:
        from bioamla.datasets.annotations import ARBIMON_COLUMNS, save_arbimon_annotations

        out = tmp_path / "rois.csv"
        anns = [Annotation(start_time=1.5, end_time=2.0, label="Hyla", confidence=0.9),
                Annotation(start_time=3.0, end_time=3.25, low_freq=1000, high_freq=4000,
                           label="Rana", custom_fields={"recording": "other.wav"})]  # fmt: skip
        save_arbimon_annotations(anns, str(out), recording="night.wav")
        rows = list(csv.DictReader(out.open()))
        assert list(rows[0]) == ARBIMON_COLUMNS
        assert rows[0]["id"] == "" and rows[0]["recording"] == "night.wav"  # no made-up ids
        assert (rows[0]["x1"], rows[0]["x2"], rows[0]["y1"], rows[0]["score"]) == (
            "1.5", "2.0", "", "0.9",
        )  # fmt: skip
        assert rows[0]["validated"] == "(not validated)"
        assert rows[1]["recording"] == "other.wav"  # its own recording wins
        assert (rows[1]["y1"], rows[1]["y2"]) == ("1000.0", "4000.0")

    def test_detected_from_header(self, export, tmp_path) -> None:
        from bioamla.datasets._io import detect_annotation_format

        plain = tmp_path / "plain.csv"
        plain.write_text("start_time,end_time,label\n0,1,x\n")
        assert detect_annotation_format(export, read_header=True) == "arbimon"
        assert detect_annotation_format(plain, read_header=True) == "csv"
        assert detect_annotation_format(tmp_path / "new.csv", read_header=True) == "csv"
        # An output path is never read: an existing Arbimon file doesn't change the format.
        assert detect_annotation_format(export) == "csv"

    def test_not_an_export(self, tmp_path) -> None:
        from bioamla.datasets.annotations import load_arbimon_annotations

        path = tmp_path / "x.csv"
        path.write_text("start_time,end_time\n0,1\n")
        with pytest.raises(AnnotationError, match="missing species, x1, x2"):
            load_arbimon_annotations(str(path))

    def test_bad_time(self, tmp_path) -> None:
        from bioamla.datasets.annotations import load_arbimon_annotations

        path = tmp_path / "x.csv"
        path.write_text("species,x1,x2\nHyla,soon,2\n")
        with pytest.raises(AnnotationError, match="line 2: x1/x2"):
            load_arbimon_annotations(str(path))


class TestRavenKeepsDetail:
    def test_confidence_and_band_precision_round_trip(self, tmp_path) -> None:
        path = tmp_path / "sel.txt"
        anns = [Annotation(start_time=1.0, end_time=2.0, low_freq=6562.5, high_freq=8343.75,
                           label="x", confidence=0.300001416)]  # fmt: skip
        save_raven_selection_table(anns, str(path))
        (back,) = load_raven_selection_table(str(path))
        assert (back.low_freq, back.high_freq) == (6562.5, 8343.75)
        assert back.confidence == pytest.approx(0.300001416)
        assert "Confidence" not in back.custom_fields

    def test_no_confidence_column_without_confidences(self, tmp_path) -> None:
        path = tmp_path / "sel.txt"
        save_raven_selection_table([Annotation(start_time=0, end_time=1, label="x")], str(path))
        assert "Confidence" not in path.read_text().splitlines()[0]

    def test_mixed_numeric_and_text_confidence_has_one_column(self, tmp_path) -> None:
        source = tmp_path / "mixed.txt"
        source.write_text(
            "Selection\tBegin Time (s)\tEnd Time (s)\tConfidence\tsite\n"
            "1\t0\t1\t0\tnorth\n"
            "2\t2\t3\thigh\tsouth\n"
            "3\t4\t5\t\twest\n"
        )
        output = tmp_path / "out.txt"
        save_raven_selection_table(load_raven_selection_table(str(source)), str(output))
        with output.open(newline="") as stream:
            reader = csv.DictReader(stream, delimiter="\t")
            rows = list(reader)
        assert reader.fieldnames.count("Confidence") == 1
        assert [row["Confidence"] for row in rows] == ["0.0", "high", ""]
        assert [row["site"] for row in rows] == ["north", "south", "west"]
        loaded = load_raven_selection_table(str(output))
        assert [ann.confidence for ann in loaded] == [0.0, None, None]
        assert loaded[1].custom_fields["Confidence"] == "high"

    @pytest.mark.parametrize("confidence", [0.0, 0.812345678901])
    def test_canonical_confidence_overrides_stale_custom_value(self, tmp_path, confidence) -> None:
        ann = Annotation(0, 1, confidence=confidence, custom_fields={"Confidence": "high"})
        output = tmp_path / "out.txt"
        save_raven_selection_table([ann], str(output))
        (loaded,) = load_raven_selection_table(str(output))
        assert loaded.confidence == confidence
        assert "Confidence" not in loaded.custom_fields
        assert ann.custom_fields == {"Confidence": "high"}

    @pytest.mark.parametrize("include_custom_fields", [True, False])
    def test_missing_and_zero_confidence_with_band_precision(
        self, tmp_path, include_custom_fields
    ) -> None:
        anns = [
            Annotation(0, 1, low_freq=8343.75, high_freq=12345.6789),
            Annotation(
                2,
                3,
                low_freq=0,
                high_freq=100.125,
                confidence=0.0,
                custom_fields={"site": "north"},
            ),
            Annotation(4, 5, confidence=0.812345678901),
        ]
        output = tmp_path / "out.txt"
        save_raven_selection_table(anns, str(output), include_custom_fields=include_custom_fields)
        with output.open(newline="") as stream:
            reader = csv.DictReader(stream, delimiter="\t")
            rows = list(reader)
        assert reader.fieldnames.count("Confidence") == 1
        assert [row["Confidence"] for row in rows] == ["", "0.0", "0.812345678901"]
        assert rows[0]["Low Freq (Hz)"] == "8343.750"
        assert rows[0]["High Freq (Hz)"] == "12345.679"
        assert rows[1]["Low Freq (Hz)"] == "0.000"
        assert rows[2]["Low Freq (Hz)"] == rows[2]["High Freq (Hz)"] == ""
        assert ("site" in reader.fieldnames) is include_custom_fields
        loaded = load_raven_selection_table(str(output))
        assert [ann.confidence for ann in loaded] == [None, 0.0, 0.812345678901]

    def test_empty_table_has_no_confidence_column(self, tmp_path) -> None:
        output = tmp_path / "empty.txt"
        save_raven_selection_table([], str(output))
        assert "Confidence" not in output.read_text().splitlines()[0].split("\t")
        assert load_raven_selection_table(str(output)) == []


class TestArbimonEdgeCases:
    def test_not_utf8(self, tmp_path) -> None:
        from bioamla.datasets.annotations import load_arbimon_annotations

        path = tmp_path / "x.csv"
        path.write_bytes("species,x1,x2,site\nHyla,1,2,Cañas\n".encode("cp1252"))
        with pytest.raises(AnnotationError, match="not UTF-8"):
            load_arbimon_annotations(str(path))

    def test_text_confidence_stays_a_raven_custom_field(self, tmp_path) -> None:
        path = tmp_path / "sel.txt"
        path.write_text(
            "Selection\tView\tChannel\tBegin Time (s)\tEnd Time (s)\tAnnotation\tConfidence\n"
            "1\tSpectrogram 1\t1\t0.5\t1.0\tHyla\thigh\n"
        )
        (ann,) = load_raven_selection_table(str(path))
        assert ann.confidence is None
        assert ann.custom_fields["Confidence"] == "high"

    @pytest.mark.parametrize(
        ("recording", "audio", "match"),
        [
            ("rec_01.wav", "rec_01.wav", True),
            ("REC_01.WAV", "rec_01.wav", True),
            ("rec_01", "rec_01.flac", True),
            ("rec_01.flac", "rec_01.wav", True),  # same recording, converted
            ("site.2020.01.15", "site.2020.01.15.flac", True),
            ("site.2020.01.15.flac", "site.2020.01.16.flac", False),
            ("rec_02.wav", "rec_01.wav", False),
        ],
    )
    def test_recording_matching(self, recording, audio, match) -> None:
        from bioamla.datasets._io import annotations_for_recording

        ann = Annotation(
            start_time=0, end_time=1, label="x", custom_fields={"recording": recording}
        )
        other = Annotation(
            start_time=0, end_time=1, label="y", custom_fields={"recording": "zzz.wav"}
        )
        assert (annotations_for_recording([ann, other], audio) == [ann]) is match

    def test_unnamed_rows_are_all_kept(self) -> None:
        from bioamla.datasets._io import annotations_for_recording

        anns = [Annotation(start_time=0, end_time=1, label="x")]
        assert annotations_for_recording(anns, "anything.wav") == anns

    def test_recording_from_metadata_is_a_file_name(self, tmp_path) -> None:
        from bioamla.datasets._io import save_annotations

        out = tmp_path / "rois.csv"
        anns = [Annotation(start_time=0, end_time=1, label="x")]
        save_annotations(anns, out, "arbimon", metadata={"audio_file": "/data/site/night.wav"})
        (row,) = csv.DictReader(out.open())
        assert row["recording"] == "night.wav"

    def test_missing_recording_warns(self, tmp_path, caplog) -> None:
        from bioamla.datasets._io import save_annotations

        save_annotations(
            [Annotation(start_time=0, end_time=1, label="x")], tmp_path / "r.csv", "arbimon"
        )
        assert "no recording name" in caplog.text
