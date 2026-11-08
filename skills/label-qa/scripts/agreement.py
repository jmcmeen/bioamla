"""Measure how well the agent's annotations agree with a person's, where both exist.

Takes a reference CSV: the person's annotations for a held-out subset of the recordings
(every row in it counts as the truth; rows marked ``labeled_by=agent`` are ignored). On
the recordings the reference covers, the agent rows of the checked annotations file are
matched to the reference rows by time overlap (intersection over union of the spans, at
least ``--min-iou``), one to one, best overlaps first. Then:

- a matched pair with the same label is a hit for that label; with different labels it
  is a false alarm for the agent's label and a miss for the person's;
- an unmatched reference row is a miss; an unmatched agent row is a false alarm.

Writes ``agreement.json`` to the run folder: per-label precision, recall and F1, the
confusion matrix (the person's label against the agent's, over matched pairs), the
misses and false alarms, and the micro-averaged scores the gate step reads.

Usage (from the workspace root, after check.py):
    python scripts/agreement.py runs/label-qa/<timestamp> --reference reference.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _bioamla import (  # noqa: E402
    BioamlaCommandError,
    add_remote_option,
    fetch,
    read_json,
    remote,
    save_json,
    use_remote,
)

AGENT = "agent"


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def cell(row: dict, column: str) -> str:
    """A cell as text, empty when the column is missing or the row is short."""
    return (row.get(column) or "").strip()


def usable(row: dict) -> bool:
    """A row with a file, a label and a span."""
    return bool(cell(row, "file") and cell(row, "label") and span(row))


def span(row: dict) -> tuple[float, float] | None:
    try:
        start, end = float(row["start_time"]), float(row["end_time"])
    except (KeyError, TypeError, ValueError):
        return None
    return (start, end) if end > start else None


def iou(a: tuple[float, float], b: tuple[float, float]) -> float:
    shared = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = max(a[1], b[1]) - min(a[0], b[0])
    return shared / union if union > 0 else 0.0


def match(reference: list[dict], agent: list[dict], min_iou: float) -> list[tuple[int, int, float]]:
    """Pairs ``(reference index, agent index, iou)`` on one recording, one to one, best
    overlaps first."""
    pairs = []
    for i, r in enumerate(reference):
        rs = span(r)
        for j, a in enumerate(agent):
            if rs and (s := span(a)):
                pairs.append((iou(rs, s), i, j))
    pairs.sort(reverse=True)
    used_r, used_a, matched = set(), set(), []
    for score, i, j in pairs:
        if score <= 0 or score < min_iou:  # disjoint spans never match, whatever the floor
            break
        if i in used_r or j in used_a:
            continue
        used_r.add(i)
        used_a.add(j)
        matched.append((i, j, round(score, 3)))
    return matched


def scores(tp: int, fp: int, fn: int) -> dict:
    """Precision, recall and F1; None where nothing was scored (no agent rows for
    precision, no reference rows for recall, neither for F1)."""
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else None
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


def agreement(args: argparse.Namespace) -> dict:
    root = Path(args.root)
    rel = Path(args.run_dir)
    checks = read_json(root / rel / "checks.json")
    annotations = args.annotations or checks["annotations"]
    annotated = fetch(annotations, rel / "workspace", root)
    reference_path = fetch(args.reference, rel / "workspace", root)
    if not reference_path.exists():
        raise FileNotFoundError(f"{args.reference} is not in the workspace")
    if not annotated.exists():
        raise FileNotFoundError(f"{annotations} is not in the workspace")

    reference_rows = [r for r in read_csv(reference_path) if cell(r, "labeled_by") != AGENT]
    reference = [r for r in reference_rows if usable(r)]
    if not reference:
        raise ValueError(f"{args.reference} has no usable rows a person labeled")
    files = sorted({cell(r, "file") for r in reference})
    wanted = set(files)
    agent_rows = [r for r in read_csv(annotated) if cell(r, "labeled_by") == AGENT]
    agent = [r for r in agent_rows if cell(r, "file") in wanted]
    # Agent rows without a usable span or label can't be matched: each is a false alarm.
    by_file: dict[str, list[dict]] = defaultdict(list)
    for r in reference:
        by_file[cell(r, "file")].append(r)
    ours_by_file: dict[str, list[dict]] = defaultdict(list)
    for a in agent:
        ours_by_file[cell(a, "file")].append(a)

    tp: dict[str, int] = defaultdict(int)
    fp: dict[str, int] = defaultdict(int)
    fn: dict[str, int] = defaultdict(int)
    confusion: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    misses, false_alarms, agreed, disagreed = [], [], 0, 0
    for path in files:
        refs, ours = by_file[path], ours_by_file[path]
        matched = match(refs, ours, args.min_iou)
        for i, j, _score in matched:
            want, got = cell(refs[i], "label"), cell(ours[j], "label")
            confusion[want][got] += 1
            if want == got:
                tp[want] += 1
                agreed += 1
            else:
                fp[got] += 1
                fn[want] += 1
                disagreed += 1
        matched_refs, matched_ours = {m[0] for m in matched}, {m[1] for m in matched}
        for i, r in enumerate(refs):
            if i not in matched_refs:
                fn[cell(r, "label")] += 1
                misses.append({"file": path, "start_time": cell(r, "start_time"),
                               "end_time": cell(r, "end_time"), "label": cell(r, "label")})  # fmt: skip
        for j, a in enumerate(ours):
            if j not in matched_ours:
                fp[cell(a, "label")] += 1
                false_alarms.append({"file": path, "start_time": cell(a, "start_time"),
                                     "end_time": cell(a, "end_time"), "label": cell(a, "label")})  # fmt: skip

    labels = sorted(set(tp) | set(fp) | set(fn))
    per_label = {label: scores(tp[label], fp[label], fn[label]) for label in labels}
    micro = scores(sum(tp.values()), sum(fp.values()), sum(fn.values()))
    report = {
        "annotations": annotations,
        "reference": args.reference,
        "min_iou": args.min_iou,
        "files": files,
        "reference_rows": len(reference),
        "reference_rows_skipped": len(reference_rows) - len(reference),
        "agent_rows": len(agent),
        "matched": agreed + disagreed,
        "label_agreements": agreed,
        "label_disagreements": disagreed,
        "misses": misses,
        "false_alarms": false_alarms,
        "per_label": per_label,
        "confusion": {k: dict(v) for k, v in confusion.items()},
        "micro": micro,
    }
    save_json((rel / "agreement.json").as_posix(), report, root)
    return {
        "run_dir": rel.as_posix(),
        "reference": args.reference,
        "files": len(files),
        "reference_rows": len(reference),
        "reference_rows_skipped": len(reference_rows) - len(reference),
        "agent_rows": len(agent),
        "matched": agreed + disagreed,
        "label_disagreements": disagreed,
        "misses": len(misses),
        "false_alarms": len(false_alarms),
        "precision": micro["precision"],
        "recall": micro["recall"],
        "f1": micro["f1"],
        "f1_by_label": {k: v["f1"] for k, v in per_label.items()},
        "remote": remote(root),
    }


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("run_dir", help="The run folder (after check.py)")
    p.add_argument("--reference", required=True,
                   help="A person's annotations for a subset of the recordings, relative to the workspace")  # fmt: skip
    p.add_argument("--annotations", default=None,
                   help="The checked annotations CSV (default: the one check.py read)")  # fmt: skip
    p.add_argument("--min-iou", type=float, default=0.5,
                   help="Least time overlap (intersection over union) for a match")  # fmt: skip
    p.add_argument("--root", default=".", help="Workspace root (default: here)")
    add_remote_option(p)
    return p


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    use_remote(args.remote)
    try:
        summary = agreement(args)
    except FileNotFoundError as e:
        print(json.dumps({"error": f"Run check.py first, and check the paths: {e}"}))
        return 1
    except (BioamlaCommandError, ValueError, KeyError, TypeError, csv.Error, OSError) as e:
        print(json.dumps({"error": str(e)}))
        return 1
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
