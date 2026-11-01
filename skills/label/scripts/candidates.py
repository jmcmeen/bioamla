"""Find labeling candidates: detect sound events and classify them with an AST model.

For each recording in a workspace folder, runs ``bioamla detect energy`` (events in a
frequency band) and ``bioamla models ast predict --segment-seconds`` (a label per
segment), and pairs each detection with the prediction of the segment it overlaps
most. Writes ``candidates.json`` to a new ``runs/label/<timestamp>/`` folder.

Usage (from the workspace root):
    python scripts/candidates.py recordings --model MODEL --labels frog,bird

Prints a JSON summary: ``run_dir`` (pass it to the next steps) and the counts.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _bioamla import BioamlaCommandError, run, run_dir, write_json  # noqa: E402


def overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    """Seconds two spans share."""
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def file_candidates(path: str, args: argparse.Namespace, root: str) -> list[dict]:
    """The candidates in one recording."""
    detected = run(
        "detect", "energy", path,
        "--low-freq", args.low_freq, "--high-freq", args.high_freq,
        "--threshold-db", args.threshold_db, "--min-duration", args.min_duration,
        cwd=root,
    )["detections"]  # fmt: skip
    if not detected:
        return []
    predicted = run(
        "models", "ast", "predict", path, "--model-path", args.model,
        "--segment-seconds", args.segment_seconds,
        cwd=root,
    )["predictions"]  # fmt: skip
    vocabulary = set(args.labels)
    rows = []
    for d in detected:
        best = max(
            predicted,
            key=lambda p: overlap(d["start_time"], d["end_time"], p["start_time"], p["end_time"]),
            default=None,
        )
        label = best["label"] if best else None
        rows.append(
            {
                "file": path,
                "start_time": d["start_time"],
                "end_time": d["end_time"],
                "low_freq": d["frequency_low"],
                "high_freq": d["frequency_high"],
                "detector": "energy",
                "detector_score": d["confidence"],
                "label": label,
                "model_confidence": best["confidence"] if best else None,
                "in_vocabulary": label in vocabulary,
            }
        )
    return rows


def find_candidates(args: argparse.Namespace, root: str = ".") -> dict:
    """Run detection and prediction over the folder; returns the summary."""
    files = run("audio", "list", args.folder, cwd=root)["files"]
    candidates = []
    for path in files:
        candidates.extend(file_candidates(path, args, root))
    for i, c in enumerate(candidates, 1):
        c["id"] = f"c{i:04d}"
    out = run_dir("label", root)
    write_json(
        out / "candidates.json",
        {
            "folder": args.folder,
            "files": files,
            "model": args.model,
            "labels": args.labels,
            "detector": {
                "low_freq": args.low_freq,
                "high_freq": args.high_freq,
                "threshold_db": args.threshold_db,
                "min_duration": args.min_duration,
            },
            "candidates": candidates,
        },
    )
    return {
        "run_dir": str(out.relative_to(root)),
        "files": len(files),
        "candidates": len(candidates),
        "outside_vocabulary": sum(not c["in_vocabulary"] for c in candidates),
    }


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("folder", help="Folder of recordings, relative to the workspace")
    p.add_argument("--model", required=True, help="AST model (Hugging Face ID or path)")
    p.add_argument("--labels", required=True, type=lambda s: [x for x in s.split(",") if x],
                   help="The label vocabulary, comma-separated")  # fmt: skip
    p.add_argument("--low-freq", type=float, default=500.0, help="Detector band low edge (Hz)")
    p.add_argument("--high-freq", type=float, default=5000.0, help="Detector band high edge (Hz)")
    p.add_argument("--threshold-db", type=float, default=-20.0, help="Detection threshold (dB)")
    p.add_argument("--min-duration", type=float, default=0.05, help="Shortest event (s)")
    p.add_argument("--segment-seconds", type=int, default=1, help="Prediction segment (s)")
    p.add_argument("--root", default=".", help="Workspace root (default: here)")
    return p


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        summary = find_candidates(args, args.root)
    except BioamlaCommandError as e:
        print(json.dumps({"error": str(e)}))
        return 1
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
