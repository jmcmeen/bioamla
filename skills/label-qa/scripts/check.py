"""Check an annotation set for problems a person would want to know about.

Reads an annotations CSV (one row per event, with a ``file`` column) and writes
``checks.json`` to a new ``runs/label-qa/<timestamp>/`` folder: one finding per problem,
each with the row it concerns and the rule it broke. The rules:

- ``columns``: the required columns (``file``, ``start_time``, ``end_time``, ``label``)
  are present, and the band columns (``low_freq``, ``high_freq``) come as a pair or not
  at all. Without the required ones nothing else can be checked.
- ``value``: the file cell is filled; times (and frequencies, when a row gives them; a
  band is optional) are finite numbers, not negative, and start before end (low below
  high). A row that fails this is left out of the duplicate and overlap rules, which
  need its numbers.
- ``file``: the recording exists in the workspace (``bioamla audio info`` can read it).
- ``bounds``: the row ends within the recording's duration and its band is below the
  Nyquist frequency.
- ``vocabulary``: the label is one of ``--labels`` (and never empty).
- ``duplicate``: another row has the same file, times, band and label.
- ``overlap``: an agent row (``labeled_by=agent``) overlaps another row in time, the
  same rule the labeling skill applies before it writes a row. Two rows a person made
  may overlap (two species calling at once); an agent's should not.

Usage (from the workspace root):
    python scripts/check.py annotations.csv --labels frog,bird

Prints a JSON summary: ``run_dir`` (pass it to the next steps), the row counts, and the
number of findings per rule.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _bioamla import (  # noqa: E402
    BioamlaCommandError,
    add_remote_option,
    fetch,
    remote,
    run,
    run_dir,
    save_json,
    use_remote,
)

REQUIRED = ("file", "start_time", "end_time", "label")
BAND = ("low_freq", "high_freq")
AGENT = "agent"
TOLERANCE = 0.01  # seconds past the end of a recording that still counts as inside


def finding(row: int, rule: str, message: str, file: str | None = None) -> dict:
    return {"row": row, "file": file or None, "rule": rule, "message": message}


def cell(row: dict, column: str) -> str:
    """A cell as text, empty when the column is missing or the row is short."""
    return (row.get(column) or "").strip()


def number(value: str) -> float | None:
    """A cell as a finite float, or None when empty or not one."""
    if not value:
        return None
    try:
        parsed = float(value)
    except ValueError:
        return None
    return parsed if math.isfinite(parsed) else None


def parse_rows(rows: list[dict], has_band: bool) -> tuple[list[dict], list[dict]]:
    """Each row's usable fields (``file``, ``label``, ``start``, ``end``, ``low``,
    ``high``; numbers are None when unusable; ``ok`` when the row can be compared with
    others) and the ``value`` findings."""
    parsed, findings = [], []
    for n, row in enumerate(rows, start=1):
        file, label = cell(row, "file"), cell(row, "label")
        start, end = number(cell(row, "start_time")), number(cell(row, "end_time"))
        low = number(cell(row, "low_freq")) if has_band else None
        high = number(cell(row, "high_freq")) if has_band else None
        problems = []
        if not file:
            problems.append("file is empty")
        if start is None or end is None:
            problems.append("start_time and end_time must be finite numbers")
        else:
            if start < 0:
                problems.append(f"start_time {start} is negative")
            if end <= start:
                problems.append(f"end_time {end} is not after start_time {start}")
        if has_band and (cell(row, "low_freq") or cell(row, "high_freq")):
            if low is None or high is None:
                problems.append("low_freq and high_freq must both be finite numbers")
            elif low < 0 or high <= low:
                problems.append(f"band {low}-{high} Hz is not ordered")
        for p in problems:
            findings.append(finding(n, "value", p, file))
        parsed.append({"row": n, "file": file, "label": label, "start": start, "end": end,
                       "low": low, "high": high, "ok": not problems,
                       "agent": cell(row, "labeled_by") == AGENT,
                       "by": cell(row, "labeled_by") or "a person"})  # fmt: skip
    return parsed, findings


#: Errors that are about the recording itself; any other error from ``audio info`` (a
#: server that can't be reached, a refused token) is the step's problem, not a finding.
FILE_ERRORS = {"NotFoundError", "AudioLoadError", "InvalidInputError", "AccessDenied"}


def recordings(files: list[str], root: str) -> tuple[dict[str, dict | None], list[dict]]:
    """Each distinct file's duration and sample rate, and a ``file`` finding for each
    that bioamla can't read."""
    info: dict[str, dict | None] = {}
    findings = []
    for path in dict.fromkeys(f for f in files if f):
        try:
            got = run("audio", "info", "--", path, cwd=root)["info"]
        except BioamlaCommandError as e:
            if e.kind not in FILE_ERRORS:
                raise
            info[path] = None
            findings.append(finding(0, "file", f"{path}: {e}", path))
        else:
            info[path] = {"duration": got["duration"], "sample_rate": got["sample_rate"]}
    return info, findings


def spans_overlap(a: dict, b: dict) -> bool:
    """Whether two usable rows share time (the band doesn't matter: the labeling skill
    skips a candidate that overlaps a person's row in time, whatever the band)."""
    return a["start"] < b["end"] and b["start"] < a["end"]


def same(a: dict, b: dict) -> bool:
    """An exact duplicate: reported once under ``duplicate``, not as overlaps."""
    keys = ("start", "end", "low", "high", "label")
    return all(a[k] == b[k] for k in keys)


def check_rows(rows: list[dict], columns: list[str], labels: list[str] | None, root: str
               ) -> tuple[list[dict], dict[str, dict | None]]:  # fmt: skip
    """Every rule over the rows; returns the findings and the recordings' info."""
    missing = [c for c in REQUIRED if c not in columns]
    if missing:
        return [finding(0, "columns", f"missing columns: {', '.join(missing)}")], {}
    findings_first: list[dict] = []
    band_columns = [c for c in BAND if c in columns]
    has_band = len(band_columns) == 2
    if len(band_columns) == 1:  # one without the other: the bands can't be checked
        findings_first.append(finding(0, "columns",
            f"{band_columns[0]} without {next(c for c in BAND if c not in columns)}: "
            "bands come as a pair"))  # fmt: skip
    parsed, findings = parse_rows(rows, has_band)
    findings = findings_first + findings
    info, file_findings = recordings([p["file"] for p in parsed], root)
    findings += file_findings
    vocabulary = set(labels) if labels is not None else None

    for p in parsed:
        rec = info.get(p["file"])
        if rec and p["end"] is not None and p["end"] > rec["duration"] + TOLERANCE:
            findings.append(finding(p["row"], "bounds",
                f"ends at {p['end']} s, after the recording's {rec['duration']} s", p["file"]))  # fmt: skip
        if rec and p["high"] is not None and p["high"] > rec["sample_rate"] / 2:
            findings.append(finding(p["row"], "bounds",
                f"high_freq {p['high']} Hz is above the Nyquist frequency "
                f"{rec['sample_rate'] / 2} Hz", p["file"]))  # fmt: skip
        if not p["label"]:
            findings.append(finding(p["row"], "vocabulary", "empty label", p["file"]))
        elif vocabulary is not None and p["label"] not in vocabulary:
            findings.append(finding(p["row"], "vocabulary",
                f"label {p['label']!r} is not in the vocabulary", p["file"]))  # fmt: skip

    # Duplicates and overlaps: within a recording, among the rows whose values parsed.
    by_file: dict[str, list[dict]] = defaultdict(list)
    for p in parsed:
        if p["ok"]:
            by_file[p["file"]].append(p)
    for group in by_file.values():
        first: dict[tuple, int] = {}
        for p in group:
            key = (p["start"], p["end"], p["low"], p["high"], p["label"])
            if key in first:
                findings.append(finding(p["row"], "duplicate",
                    f"same file, times, band and label as row {first[key]}", p["file"]))  # fmt: skip
            else:
                first[key] = p["row"]
        for p in group:
            if not p["agent"]:
                continue
            for q in group:
                if q is not p and not same(p, q) and spans_overlap(p, q):
                    findings.append(finding(p["row"], "overlap",
                        f"agent row overlaps row {q['row']} ({q['by']})", p["file"]))  # fmt: skip
    findings.sort(key=lambda f: (f["row"], f["rule"]))
    return findings, info


def _discard(folder: Path) -> None:
    """Remove a run folder this step created, and ``runs/label-qa`` and ``runs`` when
    that leaves them empty (they were created for it)."""
    shutil.rmtree(folder, ignore_errors=True)
    for parent in (folder.parent, folder.parent.parent):
        try:
            parent.rmdir()
        except OSError:
            break


def check(args: argparse.Namespace) -> dict:
    """Read the file, run every rule, write ``checks.json``; return the summary."""
    root = Path(args.root)
    folder = run_dir("label-qa", root)
    rel = folder.relative_to(root)
    try:
        path = fetch(args.annotations, rel / "workspace", root)
        if not path.exists():
            raise FileNotFoundError(f"{args.annotations} is not in the workspace")
        with path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            columns = list(reader.fieldnames or [])
        findings, info = check_rows(rows, columns, args.labels, str(root))
    except BaseException:
        _discard(folder)  # a failed check leaves no run behind
        raise
    agent_rows = sum(cell(r, "labeled_by") == AGENT for r in rows)
    report = {
        "annotations": args.annotations,
        "labels": args.labels,
        "rows": len(rows),
        "agent_rows": agent_rows,
        "person_rows": len(rows) - agent_rows,
        "recordings": info,
        "findings": findings,
    }
    save_json((rel / "checks.json").as_posix(), report, root)
    return {
        "run_dir": rel.as_posix(),
        "annotations": args.annotations,
        "rows": len(rows),
        "agent_rows": agent_rows,
        "person_rows": len(rows) - agent_rows,
        "recordings": len(info),
        "findings": len(findings),
        "by_rule": dict(Counter(f["rule"] for f in findings)),
        "rows_with_findings": len({f["row"] for f in findings if f["row"]}),
        "remote": remote(root),
    }


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("annotations", help="Annotations CSV, relative to the workspace")
    p.add_argument("--labels", type=lambda s: [x.strip() for x in s.split(",") if x.strip()],
                   default=None,
                   help="The vocabulary, comma-separated (default: not checked; an empty "
                        "value means no label is allowed)")  # fmt: skip
    p.add_argument("--root", default=".", help="Workspace root (default: here)")
    add_remote_option(p)
    return p


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    use_remote(args.remote)
    try:
        summary = check(args)
    except FileNotFoundError as e:
        print(json.dumps({"error": str(e)}))
        return 1
    except (BioamlaCommandError, ValueError, csv.Error, OSError) as e:
        print(json.dumps({"error": str(e)}))
        return 1
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
