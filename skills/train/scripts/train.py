"""Train an AST model with a checked config, and summarise the run from its metrics.

Reads a run's ``config.json`` (from ``config.py``) and runs ``bioamla models ast train``
with that config, dataset, training folder and ``--metrics-file``. The summary comes
from the metrics file, not from logs: how the run ended (``finished``, ``cancelled`` or
``failed``), its last and best evaluation metrics, its last training loss and the steps
it reached. Writes ``train.json`` and prints it. Refuses, without training, when:

- the config wasn't valid, or the run has already trained (start a new run with
  config.py);
- the config file changed since the check (its SHA-256 differs).

Training can take hours: ``--timeout`` (default 24 hours) is how long to wait. When the
wait ends first (or this script is stopped) training may go on, on a server especially;
nothing is recorded as failed. Run ``train.py`` again: a run that has started is never
trained twice; its result is collected once its metrics file records the end, and until
then it's reported as ``running``.

Exits 0 when the run finished, 2 when it was refused, didn't finish or is still running,
1 on an error.

Usage (from the workspace root, after config.py):
    python scripts/train.py runs/train/<timestamp>
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _bioamla import (  # noqa: E402
    BioamlaCommandError,
    add_remote_option,
    fetch,
    read_json,
    remote,
    run,
    save_json,
    use_remote,
)
from _train import Parser, positive, read_metrics, sha256, summarise  # noqa: E402
from config import train_args  # noqa: E402

FINISHED, REFUSED, ERROR = 0, 2, 1


def started(config: dict, rel: Path, root: Path) -> Path | None:
    """The local copy of the run's metrics file, or None when training never started."""
    local = fetch(config["metrics_file"], rel / "workspace-now", root)
    return local if local.exists() else None


def finish(
    config: dict, rel: Path, root: Path, metrics: Path, error: str | None
) -> tuple[int, dict]:
    """Summarise a run that recorded its end, and write ``train.json``."""
    summary = summarise(read_metrics(metrics))
    finished = summary["status"] == "finished"
    report = {
        "trained": finished,
        **summary,
        "error": error,
        # train_ast saves the best model to <training_dir>/best_model (documented).
        "model": config["model"] if finished else None,
        "config": config["config"],
        "dataset": config["dataset"],
        "metrics_file": config["metrics_file"],
    }
    save_json((rel / "train.json").as_posix(), report, root)
    return (FINISHED if finished else REFUSED), report


def still_running(config: dict, error: str | None) -> tuple[int, dict]:
    """A run whose metrics file has no end yet: it's still training (perhaps on a server,
    or after this script stopped waiting), or it stopped without recording its end.
    Nothing is written, so running train.py again collects the result once it ends."""
    return REFUSED, {
        "trained": False,
        "status": "running",
        "reason": f"{config['metrics_file']} has no end yet: training is still running, or "
                  "stopped without recording it. Run train.py again later to collect the result",
        "error": error,
    }  # fmt: skip


def train(args: argparse.Namespace) -> tuple[int, dict]:
    root = Path(args.root)
    rel = Path(args.run_dir)
    if not (root / rel / "config.json").exists():
        raise FileNotFoundError(f"{rel.as_posix()} has no config.json: run config.py first")
    config = read_json(root / rel / "config.json")
    if not config["valid"]:
        return REFUSED, {"trained": False, "reason": "the config wasn't valid: "
                         f"{config['reason']}"}  # fmt: skip
    if (root / rel / "train.json").exists():
        return REFUSED, {"trained": False, "reason": "this run has already trained; start a "
                         "new one with config.py"}  # fmt: skip
    metrics = started(config, rel, root)
    if metrics is not None:  # started earlier: collect it, don't train again
        if summarise(read_metrics(metrics))["status"] is None:
            return still_running(config, None)
        return finish(config, rel, root, metrics, None)
    if sha256(config["config"], rel, root) != config["config_sha256"]:
        return REFUSED, {"trained": False, "reason": f"{config['config']} changed since the "
                         "check; check it again with config.py"}  # fmt: skip

    os.environ["BIOAMLA_TIMEOUT"] = str(args.timeout)
    error = None
    try:
        run(*train_args(config), cwd=root)
    except BioamlaCommandError as e:
        error = str(e)
    metrics = started(config, rel, root)
    if metrics is None:
        if error:  # refused before it could record anything
            raise BioamlaCommandError(error)
        raise FileNotFoundError(f"training wrote no {config['metrics_file']}")
    if summarise(read_metrics(metrics))["status"] is None:
        return still_running(config, error)  # e.g. this script stopped waiting
    return finish(config, rel, root, metrics, error)


def parser() -> argparse.ArgumentParser:
    p = Parser(description=__doc__.splitlines()[0])
    p.add_argument("run_dir", help="The run folder config.py made")
    p.add_argument("--timeout", type=positive, default=86400,
                   help="Seconds to wait for training (default 86400, a day)")  # fmt: skip
    p.add_argument("--root", default=".", help="Workspace root (default: here)")
    add_remote_option(p)
    return p


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    use_remote(args.remote)
    try:
        code, report = train(args)
    except (BioamlaCommandError, KeyError, ValueError, OSError) as e:
        print(json.dumps({"error": str(e)}))
        return ERROR
    print(json.dumps({"run_dir": Path(args.run_dir).as_posix(), **report,
                      "remote": remote(Path(args.root))}, indent=2))  # fmt: skip
    return code


if __name__ == "__main__":
    sys.exit(main())
