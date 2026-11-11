"""Argument parsing shared by this skill's scripts (not by other skills: it isn't the
identical ``_bioamla.py``)."""

from __future__ import annotations

import argparse
import json
import sys


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
