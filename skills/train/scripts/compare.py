"""Keep or discard a run against a baseline, from their metrics on the same set.

Reads the candidate's and the baseline's ``eval.json`` (from ``evaluate.py``) and
applies the thresholds the person set:

- ``--metric`` (default ``f1_score``; or ``accuracy``, ``precision``, ``recall``): the
  candidate must beat the baseline by at least ``--min-delta`` (default 0.01).
- ``--max-class-drop`` (default 0.05): no class the baseline was evaluated on may lose
  more F1 than this (a class the candidate never predicts has F1 0).

Refuses to compare runs evaluated on different sets (another ground-truth CSV or a
changed one, another folder or columns, or other files skipped): their numbers don't
measure the same thing (the audio folder is fingerprinted by its files' contents,
hashed on the server with a remote). Writes ``compare.json`` into the candidate's run folder with the verdict, every
reason, both values and each class's change, and earlier verdicts under ``earlier``;
prints it.

Exits 0 to keep the candidate, 2 to discard it (or when the runs can't be compared), 1
on an error.

Usage (from the workspace root, after evaluate.py on both):
    python scripts/compare.py runs/train/<candidate> --baseline runs/train/<baseline>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _bioamla import (  # noqa: E402
    BioamlaCommandError,
    add_remote_option,
    read_json,
    remote,
    save_json,
    use_remote,
)
from _train import METRICS, Parser, fraction  # noqa: E402

KEEP, DISCARD, ERROR = 0, 2, 1
#: What makes two evaluations comparable: the same files, labels and outcome per file.
SAME_SET = ("audio", "audio_sha256", "ground_truth_sha256", "file_column", "label_column", "total_samples",
            "skipped")  # fmt: skip


def evaluation(root: Path, run: str) -> dict:
    path = root / run / "eval.json"
    if not path.exists():
        raise FileNotFoundError(f"{run} has no eval.json: run evaluate.py on it first")
    return read_json(path)


def class_changes(candidate: dict, baseline: dict) -> dict[str, float]:
    """Each class the baseline was evaluated on (support above 0): the candidate's F1
    minus the baseline's."""
    changes = {}
    for label, stats in sorted(baseline["per_class"].items()):
        if not stats.get("support"):
            continue
        theirs = (candidate["per_class"].get(label) or {}).get("f1_score", 0.0)
        changes[label] = round(theirs - stats["f1_score"], 6)
    return changes


def compare(args: argparse.Namespace) -> tuple[int, dict]:
    root = Path(args.root)
    rel = Path(args.run_dir)
    candidate = evaluation(root, args.run_dir)
    baseline = evaluation(root, args.baseline)
    thresholds = {"metric": args.metric, "min_delta": args.min_delta,
                  "max_class_drop": args.max_class_drop}  # fmt: skip
    verdict = {"baseline": Path(args.baseline).as_posix(), "thresholds": thresholds}
    # Earlier verdicts stay in the record, so a later one with looser thresholds shows.
    previous = root / rel / "compare.json"
    if previous.exists():
        old = read_json(previous)
        earlier = old.pop("earlier", [])
        verdict["earlier"] = [*earlier, old]

    differ = [k for k in SAME_SET if candidate["eval_set"].get(k) != baseline["eval_set"].get(k)]
    if differ:
        verdict.update(keep=False, comparable=False,
                       reasons=[f"evaluated on different sets ({', '.join(differ)} differ); "
                                "evaluate both on the same set"])  # fmt: skip
        save_json((rel / "compare.json").as_posix(), verdict, root)
        return DISCARD, verdict

    reasons = []
    ours, theirs = candidate["metrics"][args.metric], baseline["metrics"][args.metric]
    delta = round(ours - theirs, 6)
    if delta < args.min_delta:
        reasons.append(f"{args.metric} {ours:.4f} vs the baseline's {theirs:.4f} "
                       f"({delta:+.4f}; at least {args.min_delta:+.4f} needed)")  # fmt: skip
    changes = class_changes(candidate, baseline)
    for label, change in changes.items():
        if -change > args.max_class_drop:
            reasons.append(f"{label}: F1 fell by {-change:.4f} (at most "
                           f"{args.max_class_drop:.4f} allowed)")  # fmt: skip
    verdict.update(
        keep=not reasons,
        comparable=True,
        reasons=reasons,
        candidate_value=ours,
        baseline_value=theirs,
        delta=delta,
        class_f1_changes=changes,
    )
    save_json((rel / "compare.json").as_posix(), verdict, root)
    return (KEEP if not reasons else DISCARD), verdict


def parser() -> argparse.ArgumentParser:
    p = Parser(description=__doc__.splitlines()[0])
    p.add_argument("run_dir", help="The candidate's run folder (evaluated)")
    p.add_argument("--baseline", required=True, help="The baseline's run folder (evaluated)")
    p.add_argument("--metric", choices=METRICS, default="f1_score",
                   help="The metric to decide on (default f1_score)")  # fmt: skip
    p.add_argument("--min-delta", type=fraction, default=0.01,
                   help="How much the candidate must beat the baseline by (default 0.01)")  # fmt: skip
    p.add_argument("--max-class-drop", type=fraction, default=0.05,
                   help="Most F1 any class may lose (default 0.05)")  # fmt: skip
    p.add_argument("--root", default=".", help="Workspace root (default: here)")
    add_remote_option(p)
    return p


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    use_remote(args.remote)
    try:
        code, verdict = compare(args)
    except (BioamlaCommandError, KeyError, ValueError, OSError) as e:
        print(json.dumps({"error": str(e)}))
        return ERROR
    print(json.dumps({"run_dir": Path(args.run_dir).as_posix(), **verdict,
                      "remote": remote(Path(args.root))}, indent=2))  # fmt: skip
    return code


if __name__ == "__main__":
    sys.exit(main())
