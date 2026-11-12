"""Fixtures shared by the audio tests."""

from __future__ import annotations

import os
import subprocess
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import pytest


@dataclass
class StalledFfmpeg:
    """A fake ffmpeg on the PATH that records its pid, then sleeps for a minute."""

    bin_dir: Path
    pids: Path

    def _pids(self) -> list[str]:
        return self.pids.read_text().split() if self.pids.exists() else []

    @contextmanager
    def expect_killed(self) -> Iterator[None]:
        """The block starts the fake ffmpeg, and every run it starts is killed.

        Each run must have recorded its pid: the fixture's warm-up exec means none is
        killed before it gets that far, so a missing pid is a failure, not a pass.
        """
        before = len(self._pids())
        yield
        new = self._pids()[before:]
        assert new, "the fake ffmpeg never started (or never recorded its pid)"
        for pid in new:
            with pytest.raises(ProcessLookupError):
                os.kill(int(pid), 0)


@pytest.fixture
def stalled_ffmpeg(tmp_path, monkeypatch) -> StalledFfmpeg:
    """A fake ffmpeg that stalls, for the encode and decode time limits.

    The script is run once before it goes on the PATH: macOS checks the first exec of a
    new file, which can take longer than a test's short time limit, so without this the
    code under test could kill it before it recorded its pid.
    """
    if os.name == "nt":
        pytest.skip("a shell-script ffmpeg")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    pids = tmp_path / "pids"
    script = bin_dir / "ffmpeg"
    script.write_text(
        f'#!/bin/sh\n[ "$1" = --warm-up ] && exit 0\necho $$ >> {pids}\nexec sleep 60\n'
    )
    script.chmod(0o755)
    subprocess.run([str(script), "--warm-up"], check=True, timeout=60)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    return StalledFfmpeg(bin_dir, pids)
