"""Evaluate a model on a fixed evaluation set.

Runs ``bioamla models ast evaluate`` on an audio folder against a ground-truth CSV, for
a trained run's model (``--run``, after ``train.py`` finished) or any model
(``--model``, a Hugging Face ID or a workspace path: a baseline). Writes ``eval.json``
into the run folder (a new ``runs/train/<timestamp>/`` for ``--model``): the overall
metrics, per-class precision, recall, F1 and support, the confusion matrix, and the
evaluation set's identity (the folder and a fingerprint of its files, the CSV and its
SHA-256, the columns, the files evaluated and skipped), so ``compare.py`` only compares runs evaluated on the same set.
Refuses a run that didn't finish training, or that was already evaluated.

Exits 0 when evaluated, 2 when refused, 1 on an error.

Usage (from the workspace root):
    python scripts/evaluate.py --run runs/train/<timestamp> --audio eval/audio --ground-truth eval/labels.csv
    python scripts/evaluate.py --model bioamla/scp-frogs --audio eval/audio --ground-truth eval/labels.csv
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
    run,
    save_json,
    use_remote,
)
from _train import (  # noqa: E402
    METRICS,
    Parser,
    abandon,
    folder_sha256,
    new_run,
    positive,
    sha256,
)

EVALUATED, REFUSED, ERROR = 0, 2, 1


def evaluate(args: argparse.Namespace) -> tuple[int, dict]:
    root = Path(args.root)
    if args.run:
        rel = Path(args.run)
        trained = root / rel / "train.json"
        if not trained.exists():
            return REFUSED, {"evaluated": False, "reason": "the run hasn't trained (train.py)"}
        summary = read_json(trained)
        if not summary.get("trained"):
            return REFUSED, {"evaluated": False, "reason": "the run didn't finish training "
                             f"({summary.get('status')})"}  # fmt: skip
        if (root / rel / "eval.json").exists():
            return REFUSED, {"evaluated": False, "reason": "the run was already evaluated; "
                             "compare it, or train a new run"}  # fmt: skip
        return measure(args, root, rel, summary["model"])
    rel = new_run(root)
    try:
        return measure(args, root, rel, args.model)
    except BaseException:
        abandon(rel, root)
        raise


def measure(args: argparse.Namespace, root: Path, rel: Path, model: str) -> tuple[int, dict]:
    ground_truth = Path(args.ground_truth).as_posix()
    digest = sha256(ground_truth, rel, root)
    audio = folder_sha256(args.audio, root)
    result = run("models", "ast", "evaluate", "--model-path", model, "-g", ground_truth,
                 "--file-column", args.file_column, "--label-column", args.label_column,
                 "--batch-size", str(args.batch_size), "--", args.audio, cwd=root)  # fmt: skip
    metrics = result["metrics"]
    report = {
        "evaluated": True,
        "model": model,
        "eval_set": {
            "audio": Path(args.audio).as_posix(),
            "audio_sha256": audio,
            "ground_truth": ground_truth,
            "ground_truth_sha256": digest,
            "file_column": args.file_column,
            "label_column": args.label_column,
            "total_samples": metrics["total_samples"],
            # The files only: the error text can differ between models.
            "skipped": sorted(s.partition(": ")[0] for s in metrics.get("skipped") or []),
        },
        "metrics": {name: metrics[name] for name in METRICS},
        "per_class": metrics.get("per_class") or {},
        "labels": metrics.get("labels") or [],
        "confusion_matrix": metrics.get("confusion_matrix") or [],
    }
    save_json((rel / "eval.json").as_posix(), report, root)
    return EVALUATED, {"run_dir": rel.as_posix(), **report}


def parser() -> argparse.ArgumentParser:
    p = Parser(description=__doc__.splitlines()[0])
    which = p.add_mutually_exclusive_group(required=True)
    which.add_argument("--run", help="A run folder train.py finished: evaluate its model")
    which.add_argument("--model", help="Evaluate this model instead (a Hugging Face ID or a "
                       "workspace path), in a new run folder")  # fmt: skip
    p.add_argument("--audio", required=True, help="The evaluation audio folder")
    p.add_argument("--ground-truth", required=True,
                   help="CSV of the evaluation files and their true labels")  # fmt: skip
    p.add_argument("--file-column", default="file_name",
                   help="The CSV's file column (default file_name)")  # fmt: skip
    p.add_argument("--label-column", default="label",
                   help="The CSV's label column (default label)")  # fmt: skip
    p.add_argument("--batch-size", type=positive, default=8, help="Inference batch size")
    p.add_argument("--root", default=".", help="Workspace root (default: here)")
    add_remote_option(p)
    return p


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    use_remote(args.remote)
    try:
        code, report = evaluate(args)
    except (BioamlaCommandError, KeyError, ValueError, OSError) as e:
        print(json.dumps({"error": str(e)}))
        return ERROR
    print(json.dumps({**report, "remote": remote(Path(args.root))}, indent=2))
    return code


if __name__ == "__main__":
    sys.exit(main())
