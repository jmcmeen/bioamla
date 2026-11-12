"""Helpers shared by this skill's scripts (not by other skills: it isn't the identical
``_bioamla.py``)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from _bioamla import run


class Parser(argparse.ArgumentParser):
    """Bad arguments print ``{"error": ...}`` and exit 1, not argparse's 2, which a step
    uses for a failed check or a refused push."""

    def error(self, message: str):
        print(json.dumps({"error": f"{self.prog}: {message}"}))
        sys.exit(1)


def count(value: str) -> int:
    """An argparse type: a whole number, zero or more."""
    try:
        number = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{value!r} is not a whole number") from None
    if number < 0:
        raise argparse.ArgumentTypeError(f"{value!r} is negative")
    return number


def unredactable_files(dataset: str, root: Path, coordinates: str, decimals: int) -> list[str]:
    """Files in the dataset a redacting push can't vouch for (neither redacted nor known
    to carry no location), which the push refuses unless the person signs off. Asks
    bioamla (``push-dataset --dry-run``), so the rule is the push's own and a remote
    workspace's files are listed on the server."""
    plan = run("catalogs", "hf", "push-dataset", "--dry-run",
               "--coordinates", coordinates, "--coordinate-decimals", str(decimals),
               "--unredacted", "publish", "--", dataset, cwd=root)  # fmt: skip
    return list(plan["unvouched"])
