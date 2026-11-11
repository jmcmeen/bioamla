"""Pass or fail a dataset for publishing, from the check step's results.

Reads a run's ``checks.json`` and applies the thresholds the person set:

- ``--max-findings``: the dataset fails when the check found more problems than this
  (default 0: any problem fails it). ``--accept RULE`` leaves out every finding of a
  rule the person has signed off (for example ``location`` when they mean to publish
  exact locations); it's listed in the verdict.
- ``--require-splits``: these splits must have rows (default ``train,test``).
- ``--min-per-class``: every label has at least this many rows in every split
  (default 1).
- ``--max-imbalance``: the largest label has at most this many times the rows of the
  smallest, over the whole dataset (default: not checked).

Writes ``gate.json`` with the verdict, every reason, and what the push step needs (the
dataset, its metadata hash and the coordinate mode the check assumed), and prints it.
Exits 0 when the dataset passes, 2 when it fails, 1 on an error.

Usage (from the workspace root, after check.py):
    python scripts/gate.py runs/dataset/<timestamp> --min-per-class 5 --max-imbalance 10
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _args import Parser, count  # noqa: E402
from _bioamla import (  # noqa: E402
    BioamlaCommandError,
    add_remote_option,
    read_json,
    remote,
    save_json,
    use_remote,
)

PASS, FAIL, ERROR = 0, 2, 1
RULES = ("columns", "split", "license", "attribution", "file", "location", "leakage")


def ratio(value: str) -> float:
    """An argparse type: a finite number of at least 1."""
    try:
        number = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{value!r} is not a number") from None
    if not (math.isfinite(number) and number >= 1):
        raise argparse.ArgumentTypeError(f"{value!r} is not a number of at least 1")
    return number


def names(value: str) -> list[str]:
    return [v.strip() for v in value.split(",") if v.strip()]


def balance_reasons(counts: dict[str, dict[str, int]], args: argparse.Namespace) -> list[str]:
    """Why the splits and labels fail the thresholds."""
    reasons = []
    for split in args.require_splits:
        if not sum(counts.get(split, {}).values()):
            reasons.append(f"the {split} split has no rows")
    labels = sorted({label for c in counts.values() for label in c})
    for split in sorted(counts):
        short = [
            f"{label} ({counts[split].get(label, 0)})"
            for label in labels
            if counts[split].get(label, 0) < args.min_per_class
        ]
        if short:
            reasons.append(f"{split}: fewer than {args.min_per_class} rows for {', '.join(short)}")
    if args.max_imbalance is not None and labels:
        totals: Counter[str] = Counter()
        for c in counts.values():
            totals.update(c)
        most, least = max(totals.values()), min(totals.values())
        if most > args.max_imbalance * least:
            big = max(totals, key=totals.get)
            small = min(totals, key=totals.get)
            reasons.append(
                f"{big} has {most} rows, {most / least:.1f} times {small}'s {least} "
                f"(at most {args.max_imbalance:g} allowed)"
            )
    return reasons


def gate(args: argparse.Namespace) -> dict:
    root = Path(args.root)
    rel = Path(args.run_dir)
    checks = read_json(root / rel / "checks.json")

    reasons = []
    counted = [f for f in checks["findings"] if f["rule"] not in args.accept]
    if len(counted) > args.max_findings:
        by_rule = Counter(f["rule"] for f in counted)
        rules = ", ".join(f"{k} {v}" for k, v in sorted(by_rule.items()))
        reasons.append(f"{len(counted)} findings (at most {args.max_findings} allowed): {rules}")
    reasons += balance_reasons(checks["counts"], args)

    verdict = {
        "passed": not reasons,
        "reasons": reasons,
        "thresholds": {
            "max_findings": args.max_findings,
            "accepted_rules": sorted(args.accept),
            "require_splits": args.require_splits,
            "min_per_class": args.min_per_class,
            "max_imbalance": args.max_imbalance,
        },
        "dataset": checks["dataset"],
        "metadata": checks["metadata"],
        "metadata_sha256": checks["metadata_sha256"],
        "license": checks["license"],
        "license_id": checks["license_id"],
        "coordinates": checks["coordinates"],
        "coordinate_decimals": checks["coordinate_decimals"],
        "rows": checks["rows"],
        "findings": len(checks["findings"]),
        "accepted_findings": len(checks["findings"]) - len(counted),
    }
    save_json((rel / "gate.json").as_posix(), verdict, root)
    return {"run_dir": rel.as_posix(), **verdict, "remote": remote(root)}


def parser() -> argparse.ArgumentParser:
    p = Parser(description=__doc__.splitlines()[0])
    p.add_argument("run_dir", help="The run folder check.py made")
    p.add_argument("--max-findings", type=count, default=0, help="Most check findings allowed")
    p.add_argument("--accept", action="append", choices=RULES, default=[], metavar="RULE",
                   help=f"Leave out findings of this rule, signed off by the person "
                        f"(repeatable; one of {', '.join(RULES)})")  # fmt: skip
    p.add_argument("--require-splits", type=names, default=["train", "test"],
                   help="Splits that must have rows, comma-separated (default train,test)")  # fmt: skip
    p.add_argument("--min-per-class", type=count, default=1,
                   help="Fewest rows of each label in every split (default 1)")  # fmt: skip
    p.add_argument("--max-imbalance", type=ratio, default=None,
                   help="Most rows of the largest label per row of the smallest (default: "
                        "not checked)")  # fmt: skip
    p.add_argument("--root", default=".", help="Workspace root (default: here)")
    add_remote_option(p)
    return p


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    use_remote(args.remote)
    try:
        verdict = gate(args)
    except FileNotFoundError as e:
        print(json.dumps({"error": f"Run check.py first: {e}"}))
        return ERROR
    except (BioamlaCommandError, KeyError, ValueError, OSError) as e:
        print(json.dumps({"error": str(e)}))
        return ERROR
    print(json.dumps(verdict, indent=2))
    return PASS if verdict["passed"] else FAIL


if __name__ == "__main__":
    sys.exit(main())
