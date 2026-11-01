"""The skills eval harness: run a skill's scripts the way an agent does.

Each skill script runs as its own process, with ``BIOAMLA_BIN`` pointing at this
environment's ``bioamla`` and the workspace as its working directory, and its JSON
summary is returned. Fixtures build small synthetic workspaces with known content,
so each skill gets a pass/fail eval that runs in CI.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

SKILLS = Path(__file__).resolve().parents[2] / "skills"
SR = 16000


def bioamla_bin() -> str:
    """This environment's ``bioamla`` command."""
    return str(Path(sys.executable).parent / "bioamla")


def skill_script(skill: str, script: str) -> Path:
    return SKILLS / skill / "scripts" / f"{script}.py"


@pytest.fixture
def run_skill():
    """``run_skill(skill, script, *args, cwd=workspace)`` -> (exit code, parsed stdout).
    The script gets this process's environment as it is at the call."""

    def run(skill: str, script: str, *args: str, cwd: Path) -> tuple[int, dict]:
        env = {**os.environ, "BIOAMLA_BIN": bioamla_bin()}
        proc = subprocess.run(
            [sys.executable, str(skill_script(skill, script)), *map(str, args)],
            cwd=cwd,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        try:
            return proc.returncode, json.loads(proc.stdout)
        except ValueError:
            raise AssertionError(f"no JSON from {script}: {proc.stdout}\n{proc.stderr}") from None

    return run


def tone(seconds: float, freq: float = 1000.0, amp: float = 0.5) -> np.ndarray:
    t = np.arange(int(seconds * SR)) / SR
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


@pytest.fixture
def workspace(tmp_path) -> Path:
    """An empty folder to act as the workspace root."""
    root = tmp_path / "ws"
    root.mkdir()
    return root


def write_wav(path: Path, samples: np.ndarray) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, samples, SR)
    return path
