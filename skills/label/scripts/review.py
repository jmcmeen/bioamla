"""Queue the labeling candidates that weren't accepted, for a person to review.

Reads a run's ``candidates.json``, ``evidence.json`` and ``decisions.json`` (and
``crops.json`` when the crop step ran) and writes ``review_queue.json``: every candidate
below a threshold, with why it was held back, its evidence, and its clip and image.
Candidates skipped because a person already labeled that time aren't queued.

Usage (from the workspace root):
    python scripts/review.py runs/label/<timestamp>
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _bioamla import BioamlaCommandError, read_json, save_json, use_remote  # noqa: E402

KINDS = {
    "label": "vocabulary",
    "model confidence": "confidence",
    "SNR": "snr",
    "detector score": "detector_score",
}


def kind(reason: str) -> str:
    """A reason's short name, for counting (``snr``, ``confidence``, ...)."""
    return next((k for prefix, k in KINDS.items() if reason.startswith(prefix)), "other")


def queue(run_folder: str, root: str = ".") -> dict:
    """Write ``review_queue.json``; returns the summary."""
    folder = Path(root) / run_folder
    candidates = {c["id"]: c for c in read_json(folder / "candidates.json")["candidates"]}
    evidence = {e["id"]: e for e in read_json(folder / "evidence.json")["candidates"]}
    decided = read_json(folder / "decisions.json")
    crops_file = folder / "crops.json"
    crops = {c["id"]: c for c in read_json(crops_file)["crops"]} if crops_file.exists() else {}

    items = []
    for d in decided["decisions"]:
        if d["accepted"]:
            continue
        c, crop = candidates[d["id"]], crops.get(d["id"], {})
        items.append(
            {
                "id": d["id"],
                "file": c["file"],
                "start_time": c["start_time"],
                "end_time": c["end_time"],
                "low_freq": c["low_freq"],
                "high_freq": c["high_freq"],
                "suggested_label": c["label"],
                "reasons": d["reasons"],
                "evidence": {k: v for k, v in evidence[d["id"]].items() if k != "id"},
                "clip": crop.get("clip"),
                "image": crop.get("image"),
                "box": crop.get("box"),
            }
        )
    save_json(
        (Path(run_folder) / "review_queue.json").as_posix(),
        {
            "annotations": decided["annotations"],
            "thresholds": decided["thresholds"],
            "items": items,
        },
        root,
    )
    reasons = Counter(kind(r) for item in items for r in item["reasons"])
    return {
        "run_dir": run_folder,
        "queued": len(items),
        "queue": (Path(run_folder) / "review_queue.json").as_posix(),
        "with_crops": sum(i["image"] is not None for i in items),
        "reasons": dict(reasons),
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("run_dir", help="The run folder (after annotate.py)")
    p.add_argument("--root", default=".", help="Workspace root (default: here)")
    p.add_argument("--remote", default=None, metavar="NAME:WORKSPACE",
                   help="Run on this server workspace (default: $BIOAMLA_REMOTE or the "
                   "workspace's settings)")  # fmt: skip
    args = p.parse_args(argv)
    use_remote(args.remote)
    try:
        summary = queue(args.run_dir, args.root)
    except FileNotFoundError as e:
        print(json.dumps({"error": f"Run the earlier steps first: {e}"}))
        return 1
    except BioamlaCommandError as e:
        print(json.dumps({"error": str(e)}))
        return 1
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
