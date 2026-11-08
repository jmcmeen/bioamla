"""Pass or fail an annotation set from the check and agreement results.

Reads a run's ``checks.json`` (and ``agreement.json`` when the agreement step ran) and
applies the thresholds the person set:

- ``--max-findings``: the set fails when the check step found more problems than this
  (default 0: any problem fails it);
- ``--min-f1``, ``--min-precision``, ``--min-recall``: the set fails when the agent's
  rows score below these against the reference (micro-averaged over labels); the F1
  threshold applies whenever the agreement step ran, the others only when given;
- ``--require-agreement``: fail when the agreement step didn't run.

Writes ``gate.json`` with the verdict and every reason, and prints it. Exits 0 when the
set passes, 2 when it fails, 1 on an error. The verdict is what to report before anyone
builds a dataset from these annotations.

Usage (from the workspace root, after check.py and agreement.py):
    python scripts/gate.py runs/label-qa/<timestamp> --min-f1 0.8
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _bioamla import (
    BioamlaCommandError,
    add_remote_option,
    read_json,
    remote,
    save_json,
    use_remote,
)  # noqa: E402

PASS, FAIL, ERROR = 0, 2, 1


def gate(args: argparse.Namespace) -> dict:
    root = Path(args.root)
    rel = Path(args.run_dir)
    checks = read_json(root / rel / "checks.json")
    agreement_file = root / rel / "agreement.json"
    agreement = read_json(agreement_file) if agreement_file.exists() else None

    reasons = []
    findings = checks["findings"]
    if len(findings) > args.max_findings:
        by_rule: dict[str, int] = {}
        for f in findings:
            by_rule[f["rule"]] = by_rule.get(f["rule"], 0) + 1
        rules = ", ".join(f"{k} {v}" for k, v in sorted(by_rule.items()))
        reasons.append(f"{len(findings)} findings (at most {args.max_findings} allowed): {rules}")
    if agreement is None:
        if args.require_agreement:
            reasons.append("the agreement step didn't run")
    else:
        micro = agreement["micro"]
        for name, floor in (("f1", args.min_f1), ("precision", args.min_precision),
                            ("recall", args.min_recall)):  # fmt: skip
            if floor is None:
                continue
            if micro[name] is None:  # nothing to score it on: no agent rows for precision
                reasons.append(f"{name} could not be scored (no agent rows on the reference "
                               "recordings)")  # fmt: skip
            elif micro[name] < floor:
                reasons.append(f"{name} {micro[name]:.3f} is below {floor}")

    thresholds = {
        "max_findings": args.max_findings,
        "min_f1": args.min_f1,
        "min_precision": args.min_precision,
        "min_recall": args.min_recall,
        "require_agreement": args.require_agreement,
    }
    verdict = {
        "passed": not reasons,
        "reasons": reasons,
        "thresholds": thresholds,
        "annotations": checks["annotations"],
        "rows": checks["rows"],
        "agent_rows": checks["agent_rows"],
        "findings": len(findings),
        "agreement": None
        if agreement is None
        else {
            "reference": agreement["reference"],
            "reference_rows": agreement["reference_rows"],
            "agent_rows": agreement["agent_rows"],
            **agreement["micro"],
        },  # fmt: skip
    }
    save_json((rel / "gate.json").as_posix(), verdict, root)
    return {"run_dir": rel.as_posix(), **verdict, "remote": remote(root)}


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("run_dir", help="The run folder (after check.py, and agreement.py if used)")
    p.add_argument("--max-findings", type=int, default=0, help="Most check findings allowed")
    p.add_argument("--min-f1", type=float, default=0.8,
                   help="Lowest micro-averaged F1 against the reference")  # fmt: skip
    p.add_argument("--min-precision", type=float, default=None, help="Lowest precision, if set")
    p.add_argument("--min-recall", type=float, default=None, help="Lowest recall, if set")
    p.add_argument("--require-agreement", action="store_true",
                   help="Fail when the agreement step didn't run")  # fmt: skip
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
