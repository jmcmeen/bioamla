"""Accept labeling candidates that clear the thresholds and write them as annotations.

Reads a run's ``candidates.json`` and ``evidence.json`` and accepts a candidate when
its label is in the vocabulary, the model's confidence is at least
``--min-confidence``, its signal-to-noise ratio is at least ``--min-snr`` dB, and its
detector score is at least ``--min-score``. Accepted candidates are merged into an
annotations CSV (one row per event, with a ``file`` column):

- Rows a person made (``labeled_by`` anything but ``agent``, or empty) are kept exactly
  as they are, and an accepted candidate that overlaps one in time is skipped.
- Earlier agent rows for the recordings in this run are replaced; agent rows for other
  recordings are kept.

Every decision, with its reasons, goes to ``decisions.json`` in the run folder; the
review step reads it.

Usage (from the workspace root):
    python scripts/annotate.py runs/label/<timestamp> --annotations annotations.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _bioamla import read_json, write_json  # noqa: E402

COLUMNS = [
    "file",
    "start_time",
    "end_time",
    "low_freq",
    "high_freq",
    "label",
    "channel",
    "confidence",
    "notes",
    "method",
    "labeled_by",
]
AGENT = "agent"


def reasons_against(c: dict, e: dict, t: argparse.Namespace) -> list[str]:
    """Why a candidate isn't accepted (empty: accepted)."""
    reasons = []
    if not c["in_vocabulary"]:
        reasons.append(f"label {c['label']!r} not in the vocabulary")
    if c["model_confidence"] is None or c["model_confidence"] < t.min_confidence:
        reasons.append(f"model confidence {c['model_confidence']} < {t.min_confidence}")
    if e["snr_db"] is None:
        reasons.append("SNR could not be measured")
    elif e["snr_db"] < t.min_snr:
        reasons.append(f"SNR {e['snr_db']} dB < {t.min_snr} dB")
    if c["detector_score"] < t.min_score:
        reasons.append(f"detector score {c['detector_score']} < {t.min_score}")
    return reasons


def read_rows(path: Path) -> tuple[list[str], list[dict]]:
    """An annotations CSV's columns and rows (none if it doesn't exist yet)."""
    if not path.exists():
        return list(COLUMNS), []
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        columns = list(reader.fieldnames or [])
    return columns + [c for c in COLUMNS if c not in columns], rows


def write_rows(path: Path, columns: list[str], rows: list[dict]) -> None:
    """Write the CSV through a temporary file, so a failure leaves the old one intact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, restval="")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def overlaps(row: dict, c: dict) -> bool:
    try:
        start, end = float(row["start_time"]), float(row["end_time"])
    except (KeyError, TypeError, ValueError):
        return False
    return row.get("file") == c["file"] and start < c["end_time"] and c["start_time"] < end


def annotate(args: argparse.Namespace) -> dict:
    """Decide, merge and write; returns the summary."""
    root = Path(args.root)
    folder = root / args.run_dir
    run = read_json(folder / "candidates.json")
    evidence = {e["id"]: e for e in read_json(folder / "evidence.json")["candidates"]}
    path = root / args.annotations
    columns, existing = read_rows(path)

    covered = set(run["files"])
    human = [r for r in existing if r.get("labeled_by", "") != AGENT]
    kept_agent = [
        r for r in existing if r.get("labeled_by") == AGENT and r.get("file") not in covered
    ]

    decisions, new_rows = [], []
    for c in run["candidates"]:
        reasons = reasons_against(c, evidence[c["id"]], args)
        decision = {"id": c["id"], "accepted": not reasons, "reasons": reasons, "written": False}
        if not reasons:
            if any(overlaps(r, c) for r in human):
                decision["reasons"] = ["overlaps a row a person labeled"]
            else:
                decision["written"] = True
                new_rows.append(
                    {
                        "file": c["file"],
                        "start_time": c["start_time"],
                        "end_time": c["end_time"],
                        "low_freq": c["low_freq"],
                        "high_freq": c["high_freq"],
                        "label": c["label"],
                        "channel": 1,
                        "confidence": c["model_confidence"],
                        "notes": f"candidate {c['id']} in {args.run_dir}",
                        "method": f"{c['detector']}+ast:{run['model']}",
                        "labeled_by": AGENT,
                    }
                )
        decisions.append(decision)

    write_rows(path, columns, human + kept_agent + new_rows)
    thresholds = {
        "min_confidence": args.min_confidence,
        "min_snr": args.min_snr,
        "min_score": args.min_score,
    }
    write_json(
        folder / "decisions.json",
        {"annotations": args.annotations, "thresholds": thresholds, "decisions": decisions},
    )
    accepted = sum(d["accepted"] for d in decisions)
    return {
        "run_dir": args.run_dir,
        "annotations": args.annotations,
        "candidates": len(decisions),
        "accepted": accepted,
        "written": len(new_rows),
        "skipped_overlapping_person": accepted - len(new_rows),
        "not_accepted": len(decisions) - accepted,
        "person_rows_kept": len(human),
    }


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("run_dir", help="The run folder (after candidates.py and evidence.py)")
    p.add_argument("--annotations", default="annotations.csv",
                   help="Annotations CSV to merge into, relative to the workspace")  # fmt: skip
    p.add_argument("--min-confidence", type=float, default=0.8, help="Lowest model confidence")
    p.add_argument("--min-snr", type=float, default=10.0, help="Lowest SNR (dB)")
    p.add_argument("--min-score", type=float, default=0.0, help="Lowest detector score")
    p.add_argument("--root", default=".", help="Workspace root (default: here)")
    return p


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        summary = annotate(args)
    except FileNotFoundError as e:
        print(json.dumps({"error": f"Run the candidates and evidence steps first: {e}"}))
        return 1
    except (OSError, csv.Error) as e:
        print(json.dumps({"error": f"Could not update {args.annotations}: {e}"}))
        return 1
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
