"""Inventory the recordings in a workspace folder; flag the ones that need a look.

Runs ``bioamla audio list`` and ``bioamla audio info`` and writes ``inventory.json``
to ``runs/inventory/<timestamp>/``. A file goes to the review list when it is shorter
than ``--min-duration`` or can't be read: an objective check, not a judgement.

Usage (from the workspace root):
    python scripts/inventory.py recordings --min-duration 1.0

Prints a JSON summary: the run folder, counts, and the review list.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _bioamla import BioamlaCommandError, run, run_dir, write_json  # noqa: E402


def inventory(folder: str, min_duration: float, root: str = ".") -> dict:
    """List ``folder`` (workspace-relative) and check each file; returns the summary."""
    listing = run("audio", "list", folder, cwd=root)
    files, review = [], []
    for path in listing["files"]:
        try:
            info = run("audio", "info", path, cwd=root)["info"]
        except BioamlaCommandError as e:
            review.append({"path": path, "reason": f"unreadable: {e}"})
            continue
        row = {
            "path": path,
            "duration": info["duration"],
            "sample_rate": info["sample_rate"],
            "channels": info["channels"],
        }
        files.append(row)
        if info["duration"] < min_duration:
            review.append({"path": path, "reason": f"shorter than {min_duration}s"})
    out = run_dir("inventory", root)
    write_json(out / "inventory.json", {"folder": folder, "files": files, "review": review})
    return {
        "run_dir": str(out.relative_to(root)),
        "files": len(files),
        "review": review,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("folder", help="Folder of recordings, relative to the workspace")
    parser.add_argument("--min-duration", type=float, default=1.0, help="Seconds")
    parser.add_argument("--root", default=".", help="Workspace root (default: here)")
    args = parser.parse_args(argv)
    try:
        summary = inventory(args.folder, args.min_duration, args.root)
    except BioamlaCommandError as e:
        print(json.dumps({"error": str(e)}))
        return 1
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
