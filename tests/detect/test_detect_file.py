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
