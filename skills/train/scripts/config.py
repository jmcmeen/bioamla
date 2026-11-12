"""Check a training config for a dataset, and start a run with it.

Runs ``bioamla models ast train --dry-run`` with the config, the dataset, and the run's
own training folder and metrics file, so the options are checked exactly as training
will check them: the config file's schema and types, augmentation ranges, the dataset
and the paths. Writes ``config.json`` to a new ``runs/train/<timestamp>/`` folder:
whether the config is valid (and why not), the options training will use, any it would
ignore, and the config file's SHA-256, so ``train.py`` refuses a config that changed
since. Nothing is trained or loaded.

Exits 0 when the config is valid, 2 when it's refused, 1 on an error.

Usage (from the workspace root):
    python scripts/config.py train.toml --dataset datasets/frogs
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
    remote,
    run,
    save_json,
    use_remote,
)
from _train import Parser, abandon, new_run, sha256  # noqa: E402

VALID, REFUSED, ERROR = 0, 2, 1
#: bioamla errors that mean the config, dataset or paths won't do (the person's to fix),
#: rather than a failure to run the check.
REFUSALS = {"ConfigValidationError", "ConfigError", "InvalidInputError", "OptionConflictError",
            "TrainingError", "NotFoundError", "BadParameter", "PermissionError"}  # fmt: skip


def train_args(config: dict) -> list[str]:
    """``ast train``'s arguments for a run's ``config.json`` (without ``--dry-run``)."""
    return [
        "models", "ast", "train",
        "--config", config["config"],
        "--train-dataset", config["dataset"],
        "--training-dir", config["training_dir"],
        "--metrics-file", config["metrics_file"],
    ]  # fmt: skip


def check(args: argparse.Namespace) -> tuple[int, dict]:
    root = Path(args.root)
    rel = new_run(root)
    try:
        return checked(args, root, rel)
    except BaseException:
        abandon(rel, root)
        raise


def checked(args: argparse.Namespace, root: Path, rel: Path) -> tuple[int, dict]:
    record = {
        "config": Path(args.config).as_posix(),
        "dataset": args.dataset,
        "training_dir": (rel / "model").as_posix(),
        "metrics_file": (rel / "metrics.jsonl").as_posix(),
    }
    record["config_sha256"] = sha256(record["config"], root)
    try:
        plan = run(*train_args(record), "--dry-run", cwd=root)
    except BioamlaCommandError as e:
        if e.kind not in REFUSALS:
            raise
        verdict = {"valid": False, "reason": str(e), **record}
        save_json((rel / "config.json").as_posix(), verdict, root)
        return REFUSED, {"run_dir": rel.as_posix(), **verdict}
    verdict = {
        "valid": True,
        **record,
        "model": (rel / "model" / "best_model").as_posix(),
        "options": plan["options"],
        "warnings": plan.get("warnings", []),
    }
    save_json((rel / "config.json").as_posix(), verdict, root)
    return VALID, {"run_dir": rel.as_posix(), **verdict}


def parser() -> argparse.ArgumentParser:
    p = Parser(description=__doc__.splitlines()[0])
    p.add_argument("config", help="The training config file (TOML, from 'bioamla models ast "
                   "init-config'), relative to the workspace")  # fmt: skip
    p.add_argument("--dataset", required=True,
                   help="The training data: a dataset folder, a metadata CSV, or a Hugging "
                        "Face dataset ID")  # fmt: skip
    p.add_argument("--root", default=".", help="Workspace root (default: here)")
    add_remote_option(p)
    return p


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    use_remote(args.remote)
    try:
        code, report = check(args)
    except (BioamlaCommandError, KeyError, ValueError, OSError) as e:
        print(json.dumps({"error": str(e)}))
        return ERROR
    print(json.dumps({**report, "remote": remote(Path(args.root))}, indent=2))
    return code


if __name__ == "__main__":
    sys.exit(main())
