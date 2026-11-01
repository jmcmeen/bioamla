"""Fixtures shared by the pipeline tests."""

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from click.testing import CliRunner

SR = 22050


@pytest.fixture(autouse=True)
def _needs_audio_extra() -> None:
    pytest.importorskip("librosa")


@pytest.fixture
def recording(tmp_path: Path) -> Path:
    """Eight seconds: seven chirped calls, alternating low and high, over a faint hum."""
    t = np.arange(SR * 8) / SR
    x = 0.003 * np.sin(2 * np.pi * 60 * t)
    for i, start in enumerate(np.arange(0.5, 7.5, 1.0)):
        m = (t >= start) & (t < start + 0.4)
        tt = t[m] - start
        f0 = 2000 if i % 2 else 4000
        x[m] += 0.5 * np.sin(2 * np.pi * (f0 * tt + 3000 * tt**2)) * np.hanning(m.sum())
    path = tmp_path / "calls.wav"
    sf.write(path, x, SR)
    return path


@pytest.fixture
def lab(live_server, tmp_path, monkeypatch):
    """The live server added as the remote ``lab`` (in a private user settings file,
    with its token in ``$BIOAMLA_LAB_TOKEN``); yields ``(url, token, root)``."""
    from bioamla.cli.cli import cli

    url, token, root = live_server
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    monkeypatch.setenv("BIOAMLA_LAB_TOKEN", token)
    monkeypatch.delenv("BIOAMLA_REMOTE", raising=False)
    added = CliRunner().invoke(cli, ["remote", "add", "lab", url])
    assert added.exit_code == 0, added.output
    return live_server
