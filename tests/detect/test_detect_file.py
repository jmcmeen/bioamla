"""detect_file: the library call behind `detect energy|ribbit|accelerating` on one file."""

import csv
import json

import pytest

pytest.importorskip("librosa")

from bioamla.detect import BandLimitedEnergyDetector, FileDetections, detect_file  # noqa: E402
from bioamla.exceptions import AudioLoadError  # noqa: E402


@pytest.fixture
def detector():
    return BandLimitedEnergyDetector(low_freq=200, high_freq=1000)


def test_detects_without_saving(detector, test_audio_path):
    found = detect_file(detector, test_audio_path)
    assert isinstance(found, FileDetections)
    assert found.file == test_audio_path
    assert found.output is None
    assert found.count == len(found.detections) > 0
    assert found.to_dict()["count"] == found.count
    assert found.to_dict()["detections"][0] == found.detections[0].to_dict()


def test_saves_csv(detector, test_audio_path, tmp_path):
    out = tmp_path / "sub" / "found.csv"
    found = detect_file(detector, test_audio_path, output=out)
    assert found.output == str(out)
    rows = list(csv.DictReader(out.open()))
    assert len(rows) == found.count


def test_saves_json_by_extension(detector, test_audio_path, tmp_path):
    out = tmp_path / "found.json"
    found = detect_file(detector, test_audio_path, output=str(out))
    assert json.loads(out.read_text()) == [d.to_dict() for d in found.detections]


def test_missing_file_raises(detector, tmp_path):
    with pytest.raises(AudioLoadError):
        detect_file(detector, tmp_path / "nope.wav")


def test_detect_peaks_file(test_audio_path, tmp_path):
    import numpy as np

    from bioamla.detect import (
        PEAK_EVENT_COLUMNS,
        CWTPeakDetector,
        FilePeaks,
        PeakDetection,
        detect_peaks_file,
    )

    out = tmp_path / "peaks.csv"
    found = detect_peaks_file(CWTPeakDetector(), test_audio_path, output=out)
    assert isinstance(found, FilePeaks)
    assert found.output == str(out)
    rows = list(csv.DictReader(out.open()))
    assert len(rows) == found.count
    assert found.to_dict()["detections"] == [p.to_event() for p in found.peaks]

    p = PeakDetection(time=1.0, amplitude=0.5, width=0.25, prominence=0.8)
    assert p.to_event() == dict(zip(PEAK_EVENT_COLUMNS, (1.0, 1.25, 0.8, 0.5, 0.25), strict=True))
    assert np.isclose(p.to_event()["end_time"], 1.25)


def test_export_peaks_writes_header_for_no_peaks(tmp_path):
    from bioamla.detect import PEAK_EVENT_COLUMNS, export_peaks

    out = export_peaks([], tmp_path / "none.csv")
    assert out.read_text().strip() == ",".join(PEAK_EVENT_COLUMNS)


def test_export_peaks_write_failure(tmp_path):
    from bioamla.detect import export_peaks
    from bioamla.exceptions import DetectionError

    blocker = tmp_path / "file"
    blocker.write_text("")
    with pytest.raises(DetectionError, match="Failed to write"):
        export_peaks([], blocker / "x.csv")


def test_export_peaks_json_by_extension(tmp_path):
    from bioamla.detect import PeakDetection, export_peaks

    peaks = [PeakDetection(time=0.5, amplitude=1.0, width=0.1, prominence=0.3)]
    out = export_peaks(peaks, tmp_path / "p.JSON")
    assert json.loads(out.read_text()) == [peaks[0].to_event()]


def test_empty_output_saves_nothing(detector, test_audio_path, tmp_path, monkeypatch):
    from bioamla.detect import CWTPeakDetector, detect_peaks_file

    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.chdir(work)
    assert detect_file(detector, test_audio_path, output="").output is None
    assert detect_peaks_file(CWTPeakDetector(), test_audio_path, output="").output is None
    assert list(work.iterdir()) == []
