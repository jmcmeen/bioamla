"""Measure objective evidence for each labeling candidate.

Reads ``candidates.json`` from a run folder and, for each candidate, records the
signal-to-noise ratio in the detector's band (from ``bioamla audio features``), the
detector score, the model's confidence, the event's band and duration, and whether
its label is in the vocabulary. Writes ``evidence.json`` beside the candidates; the
decision step reads it.

SNR is the median frame level inside the event minus the median level of the
recording's frames outside every candidate, in dB. It is null when the recording
has no such frames.

``margin`` is the model's top-1 minus top-2 confidence for the candidate's segment
(``models ast predict --top-k 2``); null for a model with a single label.

Usage (from the workspace root):
    python scripts/evidence.py runs/label/<timestamp>
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _bioamla import BioamlaCommandError, add_remote_option, read_json, run, save_json, use_remote  # noqa: E402


def snr_db(frames: list[dict], start: float, end: float, events: list[tuple]) -> float | None:
    """Median level inside ``[start, end]`` minus the median outside all ``events``."""
    inside = [f["level_db"] for f in frames if start <= f["time"] <= end]
    outside = [f["level_db"] for f in frames if not any(s <= f["time"] <= e for s, e in events)]
    if not inside or not outside:
        return None
    return round(statistics.median(inside) - statistics.median(outside), 2)


def measure(run_folder: str, root: str = ".") -> dict:
    """Write ``evidence.json`` for the run's candidates; returns the summary."""
    folder = Path(root) / run_folder
    data = read_json(folder / "candidates.json")
    band = data["detector"]
    candidates = data["candidates"]
    by_file: dict[str, list[dict]] = {}
    for c in candidates:
        by_file.setdefault(c["file"], []).append(c)

    evidence = []
    for path, found in by_file.items():
        frames = run(
            "audio", "features", path, "--no-denoise",
            "--f-min", band["low_freq"], "--f-max", band["high_freq"],
            cwd=root,
        )["frames"]  # fmt: skip
        spans = [(c["start_time"], c["end_time"]) for c in found]
        for c in found:
            evidence.append(
                {
                    "id": c["id"],
                    "snr_db": snr_db(frames, c["start_time"], c["end_time"], spans),
                    "detector_score": c["detector_score"],
                    "model_confidence": c["model_confidence"],
                    "margin": c.get("margin"),
                    "low_freq": c["low_freq"],
                    "high_freq": c["high_freq"],
                    "duration": round(c["end_time"] - c["start_time"], 4),
                    "in_vocabulary": c["in_vocabulary"],
                }
            )
    evidence.sort(key=lambda e: e["id"])
    save_json((Path(run_folder) / "evidence.json").as_posix(), {"candidates": evidence}, root)
    snrs = [e["snr_db"] for e in evidence if e["snr_db"] is not None]
    return {
        "run_dir": run_folder,
        "candidates": len(evidence),
        "median_snr_db": round(statistics.median(snrs), 2) if snrs else None,
        "unmeasured_snr": len(evidence) - len(snrs),
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("run_dir", help="The run folder from candidates.py")
    p.add_argument("--root", default=".", help="Workspace root (default: here)")
    add_remote_option(p)
    args = p.parse_args(argv)
    use_remote(args.remote)
    try:
        summary = measure(args.run_dir, args.root)
    except BioamlaCommandError as e:
        print(json.dumps({"error": str(e)}))
        return 1
    except FileNotFoundError as e:
        print(json.dumps({"error": f"No candidates in {args.run_dir}: {e}"}))
        return 1
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
