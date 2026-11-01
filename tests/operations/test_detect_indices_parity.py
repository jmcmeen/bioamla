"""Every `detect` and `indices` command can be reproduced with library calls alone.

Each test runs a command with `--json`, then gets the same result by importing
bioamla's public library functions (no CLI, no operations registry) and checks they
agree. If a command ever grows logic the library doesn't expose, these fail.
"""

import json

import numpy as np
import pytest
from click.testing import CliRunner

pytest.importorskip("librosa")

from bioamla.cli.cli import cli  # noqa: E402
from bioamla.common.jsonable import to_jsonable  # noqa: E402


@pytest.fixture
def pulses(tmp_path) -> str:
    """4 s at 16 kHz: 30 ms bursts of a 1.5 kHz tone, 10 per second, over light noise."""
    import scipy.io.wavfile as wav

    sr = 16000
    t = np.arange(4 * sr) / sr
    gate = ((t % 0.1) < 0.03) & (t > 0.5) & (t < 3.5)
    x = 0.6 * np.sin(2 * np.pi * 1500 * t) * gate
    x += 0.01 * np.random.default_rng(0).standard_normal(len(t))
    path = tmp_path / "pulses.wav"
    wav.write(str(path), sr, (x * 32767).astype(np.int16))
    return str(path)


def _cli(*args: str) -> dict:
    result = CliRunner().invoke(cli, ["--json", *args])
    assert result.exit_code == 0, result.output
    return json.loads(result.stdout)["result"]


def _plain(value):
    """``value`` as the JSON envelope would carry it."""
    return json.loads(json.dumps(to_jsonable(value)))


def test_detect_energy(pulses, tmp_path):
    from bioamla.detect import BandLimitedEnergyDetector, detect_file

    out = tmp_path / "cli.json"
    got = _cli("detect", "energy", pulses, "-l", "1000", "-h", "2000", "-o", str(out))
    lib_out = tmp_path / "lib.json"
    found = detect_file(
        BandLimitedEnergyDetector(low_freq=1000, high_freq=2000), pulses, output=lib_out
    )
    assert found.count > 0
    assert got["count"] == found.count
    assert got["detections"] == _plain(found.to_dict()["detections"])
    assert json.loads(out.read_text()) == json.loads(lib_out.read_text())


def test_detect_ribbit(pulses):
    from bioamla.detect import RibbitDetector, detect_file

    got = _cli("detect", "ribbit", pulses, "-p", "10", "-l", "1000", "-h", "2000")
    found = detect_file(RibbitDetector(pulse_rate_hz=10, low_freq=1000, high_freq=2000), pulses)
    assert got["detections"] == _plain(found.to_dict()["detections"])


def test_detect_peaks(pulses, tmp_path):
    from bioamla.detect import CWTPeakDetector, detect_peaks_file

    out = tmp_path / "cli.csv"
    got = _cli("detect", "peaks", pulses, "--snr-threshold", "1", "-o", str(out))
    lib_out = tmp_path / "lib.csv"
    found = detect_peaks_file(CWTPeakDetector(snr_threshold=1), pulses, output=lib_out)
    assert found.count > 0
    assert got["detections"] == _plain(found.to_dict()["detections"])
    assert out.read_text() == lib_out.read_text()


def test_detect_accelerating(pulses):
    from bioamla.detect import AcceleratingPatternDetector, detect_file

    got = _cli("detect", "accelerating", pulses, "--min-pulses", "3", "-a", "1.1")
    found = detect_file(
        AcceleratingPatternDetector(min_pulses=3, acceleration_threshold=1.1), pulses
    )
    assert got["detections"] == _plain(found.to_dict()["detections"])


def test_indices_compute(pulses, tmp_path):
    from bioamla.indices import index_file

    out = tmp_path / "cli.json"
    got = _cli("indices", "compute", pulses, "--aci-high-freq", "6000", "-o", str(out))
    lib_out = tmp_path / "lib.json"
    found = index_file(pulses, aci_max_freq=6000, output=lib_out)
    assert got["indices"] == _plain(found.indices.to_dict())
    assert json.loads(out.read_text()) == json.loads(lib_out.read_text())


def test_indices_temporal(pulses, tmp_path):
    from bioamla.indices import temporal_index_file

    out = tmp_path / "cli.json"
    got = _cli("indices", "temporal", pulses, "--window-seconds", "1", "-o", str(out))
    found = temporal_index_file(pulses, window_duration=1.0)
    assert len(found.segments) == 4
    assert got["segments"] == _plain(found.segments)
    assert json.loads(out.read_text()) == _plain(found.segments)


@pytest.mark.parametrize(
    ("command", "args", "names", "params"),
    [
        ("aci", ["--low-freq", "1000"], ["aci"], {"min_freq": 1000.0, "max_freq": None}),
        ("bio", [], ["bio"], {"min_freq": 2000.0, "max_freq": 8000.0}),
        ("adi", ["--threshold-db", "-40"], ["adi"], {"db_threshold": -40.0}),
        ("aei", [], ["aei"], {"db_threshold": -50.0}),
        ("ndsi", [], ["ndsi"], {}),
        ("entropy", ["--n-fft", "1024"], ["h_spectral", "h_temporal"], {"n_fft": 1024}),
    ],
)
def test_single_indices(pulses, command, args, names, params):
    from bioamla.indices import index_values_file

    got = _cli("indices", command, pulses, *args)
    values = index_values_file(pulses, names, **params).values
    if command == "entropy":
        assert got["indices"] == _plain(values)
    else:
        assert got["index"] == command
        assert got["value"] == _plain(values[command])
