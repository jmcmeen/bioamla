"""Helpers shared by this skill's scripts (not by other skills: it isn't the identical
``_bioamla.py``)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from _bioamla import remote


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


def unredactable_files(dataset: str, root: Path) -> list[str] | None:
    """Files in the dataset a redacting push can't change but that may hold a location
    (JSON, JSONL or text mentioning a latitude or longitude, and Parquet), which the push
    refuses unless the person signs off; None with a remote, whose files this step can't
    list."""
    if remote(root) is not None:
        return None
    base = (root / dataset).resolve()
    found = []
    for path in sorted(base.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(base).as_posix()
        suffix = path.suffix.lower()
        if suffix in (".json", ".jsonl", ".txt") and rel != "dataset.json":
            text = path.read_text(encoding="utf-8", errors="replace").lower()
            if "latitude" in text or "longitude" in text:
                found.append(rel)
        elif suffix == ".parquet":
            found.append(rel)
    return found
