"""Push a dataset that passed the gate to the Hugging Face Hub.

Reads a run's ``gate.json`` and pushes its dataset with ``bioamla catalogs hf
push-dataset``, using the coordinate mode the check assumed (``round``, ``strip`` or
``keep``), so the locations are redacted as checked. Refuses, without pushing, when:

- the gate didn't pass, or hasn't run;
- the dataset's ``metadata.csv`` changed since the check (its SHA-256 differs), so
  what passed isn't what would be published. Check and gate it again.

The repository is private unless ``--public`` is given. Writes ``push.json`` (the
push's result, including bioamla's warnings) and prints it. Exits 0 when pushed, 2
when refused, 1 on an error.

Usage (from the workspace root, after gate.py passed):
    python scripts/push.py runs/dataset/<timestamp> my-org/frog-calls
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _args import Parser  # noqa: E402
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

PUSHED, REFUSED, ERROR = 0, 2, 1


def push(args: argparse.Namespace) -> tuple[int, dict]:
    root = Path(args.root)
    rel = Path(args.run_dir)
    gate_file = root / rel / "gate.json"
    if not gate_file.exists():
        return REFUSED, {"pushed": False, "reason": "the gate hasn't run (gate.py)"}
    verdict = read_json(gate_file)
    if not verdict["passed"]:
        return REFUSED, {"pushed": False, "reason": "the gate didn't pass",
                         "gate_reasons": verdict["reasons"]}  # fmt: skip

    current = fetch(verdict["metadata"], rel / "workspace-now", root)
    if not current.exists():
        raise FileNotFoundError(f"{verdict['metadata']} is not in the workspace")
    digest = hashlib.sha256(current.read_bytes()).hexdigest()
    if digest != verdict["metadata_sha256"]:
        return REFUSED, {
            "pushed": False,
            "reason": f"{verdict['metadata']} changed since the check; check and gate it again",
        }

    options = [
        "--coordinates", verdict["coordinates"],
        "--coordinate-decimals", str(verdict["coordinate_decimals"]),
        "--public" if args.public else "--private",
    ]  # fmt: skip
    if args.commit_message:
        options += ["--commit-message", args.commit_message]
    result = run("catalogs", "hf", "push-dataset", *options, "--",
                 verdict["dataset"], args.repo_id, cwd=root)  # fmt: skip
    report = {
        "pushed": True,
        "repo_id": args.repo_id,
        "public": args.public,
        "dataset": verdict["dataset"],
        "license": verdict["license"],
        "coordinates": verdict["coordinates"],
        "coordinate_decimals": verdict["coordinate_decimals"],
        "result": result,
        "warnings": (result or {}).get("warnings", []) if isinstance(result, dict) else [],
    }
    save_json((rel / "push.json").as_posix(), report, root)
    return PUSHED, report


def parser() -> argparse.ArgumentParser:
    p = Parser(description=__doc__.splitlines()[0])
    p.add_argument("run_dir", help="The run folder check.py and gate.py wrote")
    p.add_argument("repo_id", help="The Hub dataset repository (owner/name)")
    p.add_argument("--public", action="store_true",
                   help="Make the repository public (default: private)")  # fmt: skip
    p.add_argument("--commit-message", default=None, help="The push's commit message")
    p.add_argument("--root", default=".", help="Workspace root (default: here)")
    add_remote_option(p)
    return p


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    use_remote(args.remote)
    try:
        code, report = push(args)
    except (BioamlaCommandError, KeyError, ValueError, OSError) as e:
        print(json.dumps({"error": str(e)}))
        return ERROR
    print(json.dumps({"run_dir": Path(args.run_dir).as_posix(), **report,
                      "remote": remote(Path(args.root))}, indent=2))  # fmt: skip
    return code


if __name__ == "__main__":
    sys.exit(main())
