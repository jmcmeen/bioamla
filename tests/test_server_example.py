"""The container example under examples/server stays consistent with the package."""

import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "server"


def test_the_dockerfile_installs_real_extras_and_keeps_the_server_reachable():
    dockerfile = (EXAMPLE / "Dockerfile").read_text()
    extras = set(
        tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["optional-dependencies"]
    )
    (default,) = re.findall(r"^ARG BIOAMLA_EXTRAS=(\S+)$", dockerfile, re.M)
    assert set(default.split(",")) <= extras, default
    assert {"server", "cli"} <= set(default.split(","))  # serve needs both
    assert '"--host", "0.0.0.0"' in dockerfile  # compose publishes on loopback instead
    assert "USER bioamla" in dockerfile


def test_the_compose_file_publishes_on_loopback_and_mounts_recordings_read_only():
    compose = (EXAMPLE / "compose.yaml").read_text()
    assert '"127.0.0.1:8765:8765"' in compose
    assert ":/recordings:ro" in compose
    assert "bioamla_token" in compose and "file: ./bioamla-token" in compose
    assert (EXAMPLE / ".gitignore").read_text().splitlines()[0] == "bioamla-token"
