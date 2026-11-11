"""Check a dataset for what must be settled before it's published.

Reads a dataset's ``metadata.csv`` (one row per file, as ``dataset merge``,
``dataset partition`` and the catalog downloads write it) and writes ``checks.json``
to a new ``runs/dataset/<timestamp>/`` folder: one finding per problem, each with the
row it concerns (0 for the dataset as a whole) and the rule it broke. The rules:

- ``columns``: ``file_name``, ``label`` and ``split`` are present (run
  ``dataset partition`` first), and so is every ``--group-by`` column.
- ``split``: the row has a split. Rows without one are left out of the counts and the
  leakage rule.
- ``license``: the row's license (``attr_lic``, else ``license``, the order
  ``dataset license`` reads them in) is one this check can read: Creative Commons (CC0, BY, BY-SA, BY-NC, BY-NC-SA) or public domain. A
  NoDerivatives license fails: clips, resampling and format changes are derivatives.
  The dataset's own license (``--license``) must keep every condition of each row's:
  Attribution, NonCommercial, and for ShareAlike the same license and version. The
  dataset's license can't be NoDerivatives.
- ``attribution``: a row under an Attribution license says whom to credit
  (``attribution``, ``attr_id`` or ``attr_url``, what ``dataset license`` credits).
- ``location``: with ``--coordinates keep``, per-file locations (coordinate or place
  columns in the metadata, or a position in the recorder metadata of a sample of the
  WAV files) that the push would publish as they are. With ``round`` (the default) or
  ``strip`` the push redacts those, so the rule looks instead for files it can't
  redact: JSON, JSONL or text mentioning a latitude or longitude, and Parquet (on a
  local dataset; a remote workspace's files can't be listed from here).
- ``file``: a sampled WAV file's recorder metadata can't be read (``keep`` only).
- ``leakage``: a value of a grouping column (``--group-by``; by default
  ``source_file`` when the dataset has it) appears in more than one split, so the
  same recording, site or device is both trained and tested on.

Also records the rows per split and label (the gate's balance thresholds use them)
and a SHA-256 of ``metadata.csv`` (the push step refuses a dataset changed since).

Usage (from the workspace root):
    python scripts/check.py datasets/frogs --license CC-BY-NC-4.0 --group-by site

Prints a JSON summary: ``run_dir`` (pass it to the next steps), the row counts, and the
number of findings per rule. Exit 0 when the check ran (findings or not), 1 on an error.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _args import Parser, count  # noqa: E402
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

SKILL = "dataset"
REQUIRED = ("file_name", "label", "split")
# What `bioamla dataset license` credits in ATTRIBUTIONS.md.
CREDIT = ("attribution", "attr_id", "attr_url")
LOCATION_COLUMNS = ("latitude", "lat", "longitude", "lon", "lng", "long", "location", "locality")
COORDINATE_MODES = ("keep", "round", "strip")


class License:
    """A license's conditions: ``by``, ``nc``, ``sa``, ``nd``; none for public domain."""

    def __init__(self, name: str, by=False, nc=False, sa=False, nd=False, version=None):
        self.name, self.by, self.nc, self.sa, self.nd = name, by, nc, sa, nd
        self.version = version  # "4.0", or None when the text gives none

    def __eq__(self, other) -> bool:
        return isinstance(other, License) and self.conditions() == other.conditions()

    __hash__ = None

    def conditions(self) -> tuple[bool, bool, bool, bool]:
        return (self.by, self.nc, self.sa, self.nd)


_CC_URL = re.compile(r"creativecommons\.org/licen[cs]es/([a-z-]+)")
_CC_TEXT = re.compile(r"\bcc[\s_-]*(by(?:[\s_-]*(?:nc|nd|sa))*)\b")
_VERSION = re.compile(r"(?<![\d.])([1-4]\.0)(?![\d.])")


def parse_license(text: str) -> License | None:
    """The license ``text`` names (a name like ``CC BY-NC 4.0`` or ``cc-by-nc``, or a
    creativecommons.org URL), or None when it isn't one this check can read."""
    t = text.strip().lower()
    if not t:
        return None
    if "cc0" in t or "publicdomain" in t or "public domain" in t or t in ("pd", "pdm"):
        return License("public domain")
    match = _CC_URL.search(t) or _CC_TEXT.search(t)
    if not match:
        return None
    parts = set(re.split(r"[\s_-]+", match.group(1)))
    if "by" not in parts:
        return None
    nc, sa, nd = "nc" in parts, "sa" in parts, "nd" in parts
    if sa and nd:
        return None
    version = _VERSION.search(t[match.end() :])
    version = version.group(1) if version else None
    name = "CC BY" + "".join(
        f"-{p.upper()}" for p, on in (("nc", nc), ("nd", nd), ("sa", sa)) if on
    )
    return License(f"{name} {version}" if version else name, by=True, nc=nc, sa=sa, nd=nd,
                   version=version)  # fmt: skip


def hub_id(license: License) -> str:
    """The Hugging Face Hub license id for ``license`` (version 4.0 when it gives none)."""
    if not license.by:
        return "cc0-1.0"
    conditions = "".join(f"-{c}" for c, on in (("nc", license.nc), ("nd", license.nd),
                                                ("sa", license.sa)) if on)  # fmt: skip
    return f"cc-by{conditions}-{license.version or '4.0'}"


def license_problem(source: License, target: License) -> str | None:
    """Why a file under ``source`` can't be published in a dataset under ``target``."""
    if source.nd:
        return f"{source.name} allows no derivatives (clips, resampling and conversion are)"
    if source.sa and target != source:
        return f"{source.name} requires the dataset to use the same license, not {target.name}"
    if source.sa and source.version and target.version and source.version != target.version:
        return f"{source.name} requires the same license version, not {target.name}"
    if source.by and not target.by:
        return f"{source.name} requires attribution, which {target.name} doesn't"
    if source.nc and not target.nc:
        return f"{source.name} is non-commercial, which {target.name} doesn't keep"
    return None


def cell(row: dict, column: str) -> str:
    """A cell as text, empty when the column is missing or the row is short."""
    return (row.get(column) or "").strip()


def finding(row: int, rule: str, message: str, **extra) -> dict:
    return {"row": row, "rule": rule, "message": message, **extra}


def check_licenses(rows: list[dict], target: License) -> list[dict]:
    found = []
    for n, row in enumerate(rows, start=2):  # row 1 is the header
        # attr_lic first, as `bioamla dataset license` reads them.
        text = cell(row, "attr_lic") or cell(row, "license")
        source = parse_license(text)
        if source is None:
            what = f"{text!r} is not a license this check can read" if text else "no license"
            found.append(finding(n, "license", what, file=cell(row, "file_name")))
            continue
        problem = license_problem(source, target)
        if problem:
            found.append(finding(n, "license", problem, file=cell(row, "file_name")))
        if source.by and not any(cell(row, c) for c in CREDIT):
            found.append(
                finding(
                    n,
                    "attribution",
                    f"{source.name} requires attribution, and the row names no one to credit "
                    f"({', '.join(CREDIT)})",
                    file=cell(row, "file_name"),
                )
            )
    return found


def check_leakage(rows: list[dict], columns: list[str]) -> list[dict]:
    found = []
    for column in columns:
        splits: dict[str, set[str]] = defaultdict(set)
        counts: Counter[str] = Counter()
        for row in rows:
            value = cell(row, column)
            if value and cell(row, "split"):
                splits[value].add(cell(row, "split"))
                counts[value] += 1
        for value in sorted(v for v, s in splits.items() if len(s) > 1):
            found.append(
                finding(
                    0,
                    "leakage",
                    f"{column} {value!r} is in {', '.join(sorted(splits[value]))}",
                    column=column,
                    value=value,
                    splits=sorted(splits[value]),
                    rows=counts[value],
                )
            )
    return found


def wav_locations(dataset: str, rows: list[dict], sample: int, root: Path) -> dict:
    """Up to ``sample`` of the dataset's WAV files, read until one has a position."""
    wavs = [cell(r, "file_name") for r in rows if cell(r, "file_name").lower().endswith(".wav")]
    checked, unreadable = [], {}
    for name in wavs[:sample]:
        path = (PurePosixPath(dataset) / name).as_posix()
        try:
            report = run("audio", "metadata", "--", path, cwd=root)
        except BioamlaCommandError as e:
            if e.kind in ("Timeout", "RemoteNotFound") or "can't run" in str(e):
                raise  # bioamla itself failed, not this file
            unreadable[path] = str(e)
            continue
        checked.append(path)
        summary = report.get("summary") or {}
        if summary.get("latitude") is not None or summary.get("longitude") is not None:
            return {"wav_files": len(wavs), "sampled": len(checked), "with_position": path,
                    "unreadable": unreadable}  # fmt: skip
    return {"wav_files": len(wavs), "sampled": len(checked), "with_position": None,
            "unreadable": unreadable}  # fmt: skip


def unredactable_files(dataset: str, root: Path) -> list[str] | None:
    """Files in the dataset a redacting push can't change but that may hold a location
    (JSON, JSONL or text mentioning a latitude or longitude, and Parquet), which the push
    refuses unless the person signs off; None with a remote, whose files this step can't
    list."""
    if remote(root) is not None:
        return None
    base = (root / dataset).resolve()
    found = []
    for path in sorted(base.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(base).as_posix()
        suffix = path.suffix.lower()
        if suffix in (".json", ".jsonl", ".txt") and rel != "dataset.json":
            text = path.read_text(encoding="utf-8", errors="replace").lower()
            if "latitude" in text or "longitude" in text:
                found.append(rel)
        elif suffix == ".parquet":
            found.append(rel)
    return found


def check_rows(rows, columns, args, root: Path) -> tuple[list[dict], dict]:
    """Every rule's findings, and what the location rule saw."""
    found = []
    missing = [c for c in REQUIRED if c not in columns]
    group_by = args.group_by if args.group_by is not None else (
        ["source_file"] if "source_file" in columns else []
    )  # fmt: skip
    missing += [c for c in group_by if c not in columns and c not in missing]
    for column in missing:
        found.append(finding(0, "columns", f"no {column} column"))
    found += check_licenses(rows, args.license)
    if "split" in columns:
        found += check_leakage(rows, [c for c in group_by if c in columns])

    for n, row in enumerate(rows, start=2):
        if "split" in columns and not cell(row, "split"):
            found.append(finding(n, "split", "the row has no split", file=cell(row, "file_name")))

    in_metadata = sorted(
        c for c in columns if c.lower() in LOCATION_COLUMNS and any(cell(r, c) for r in rows)
    )
    locations: dict = {"metadata_columns": in_metadata}
    if args.coordinates == "keep":
        # Only a push that keeps locations publishes these; round and strip redact them.
        wavs = (
            wav_locations(args.dataset, rows, args.sample_wavs, root)
            if "file_name" in columns
            else {}
        )
        locations.update(wavs)
        for path, error in wavs.get("unreadable", {}).items():
            found.append(finding(0, "file", f"can't read {path}: {error}", file=path))
        where = []
        if in_metadata:
            where.append(f"the metadata ({', '.join(in_metadata)})")
        if wavs.get("with_position"):
            where.append(f"WAV recorder metadata (e.g. {wavs['with_position']})")
        if where:
            found.append(
                finding(
                    0,
                    "location",
                    f"per-file locations in {' and '.join(where)} would be published as they "
                    "are (--coordinates keep)",
                )
            )
    else:
        others = unredactable_files(args.dataset, root)
        locations["unredactable_files"] = others
        if others:
            found.append(
                finding(
                    0,
                    "location",
                    f"{len(others)} file(s) the push can't redact may hold locations "
                    f"(JSON, text or Parquet), e.g. {others[0]}",
                    files=others,
                )
            )
    return found, {"locations": locations, "group_by": group_by}


def _discard(folder: Path) -> None:
    """Remove a run folder this step created, and ``runs/dataset`` and ``runs`` when that
    leaves them empty (they were created for it)."""
    shutil.rmtree(folder, ignore_errors=True)
    for parent in (folder.parent, folder.parent.parent):
        try:
            parent.rmdir()
        except OSError:
            break


def check(args: argparse.Namespace) -> dict:
    """Read the metadata, run every rule, write ``checks.json``; return the summary."""
    root = Path(args.root)
    folder = run_dir(SKILL, root)
    rel = folder.relative_to(root)
    metadata = (PurePosixPath(args.dataset) / args.metadata_filename).as_posix()
    try:
        path = fetch(metadata, rel / "workspace", root)
        if not path.exists():
            raise FileNotFoundError(f"{metadata} is not in the workspace")
        data = path.read_bytes()
        reader = csv.DictReader(data.decode("utf-8-sig").splitlines())
        rows = list(reader)
        columns = list(reader.fieldnames or [])
        findings, info = check_rows(rows, columns, args, root)
    except BaseException:
        _discard(folder)  # a failed check leaves no run behind
        raise
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        if cell(row, "split"):
            counts[cell(row, "split")][cell(row, "label")] += 1
    report = {
        "dataset": args.dataset,
        "metadata": metadata,
        "metadata_sha256": hashlib.sha256(data).hexdigest(),
        "license": args.license.name,
        "license_id": hub_id(args.license),
        "coordinates": args.coordinates,
        "coordinate_decimals": args.coordinate_decimals,
        "rows": len(rows),
        "counts": {split: dict(sorted(c.items())) for split, c in sorted(counts.items())},
        **info,
        "findings": findings,
    }
    save_json((rel / "checks.json").as_posix(), report, root)
    return {
        "run_dir": rel.as_posix(),
        "dataset": args.dataset,
        "rows": len(rows),
        "splits": {split: sum(c.values()) for split, c in sorted(counts.items())},
        "labels": len({cell(r, "label") for r in rows}),
        "findings": len(findings),
        "by_rule": dict(Counter(f["rule"] for f in findings)),
        "locations": info["locations"],
        "remote": remote(root),
    }


def target_license(value: str) -> License:
    """An argparse type: a license this check can read, other than NoDerivatives."""
    parsed = parse_license(value)
    if parsed is not None and parsed.nd:
        raise argparse.ArgumentTypeError(
            f"{value!r} is NoDerivatives, which a dataset of clips can't be published under"
        )
    if parsed is None:
        raise argparse.ArgumentTypeError(
            f"{value!r} is not a license this check can read (e.g. CC-BY-4.0, CC-BY-NC-4.0, CC0)"
        )
    return parsed


def decimals(value: str) -> int:
    """An argparse type: 0 to 6 decimal places, as ``push-dataset`` takes."""
    try:
        number = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{value!r} is not a whole number") from None
    if not 0 <= number <= 6:
        raise argparse.ArgumentTypeError(f"{value!r} is not from 0 to 6")
    return number


def parser() -> argparse.ArgumentParser:
    p = Parser(description=__doc__.splitlines()[0])
    p.add_argument(
        "dataset", help="Dataset folder (holding metadata.csv), relative to the workspace"
    )
    p.add_argument("--license", type=target_license, required=True,
                   help="The license the dataset will be published under (e.g. CC-BY-NC-4.0)")  # fmt: skip
    p.add_argument("--group-by", action="append", default=None, metavar="COLUMN",
                   help="A column whose values must stay in one split (repeatable; default "
                        "source_file when present)")  # fmt: skip
    p.add_argument("--coordinates", choices=COORDINATE_MODES, default="round",
                   help="How the push will publish per-file locations (default round)")  # fmt: skip
    p.add_argument("--coordinate-decimals", type=decimals, default=2,
                   help="Decimal places kept by --coordinates round (default 2, about 1 km)")  # fmt: skip
    p.add_argument("--sample-wavs", type=count, default=25,
                   help="WAV files whose recorder metadata is read for a position (default 25)")  # fmt: skip
    p.add_argument("--metadata-filename", default="metadata.csv",
                   help="The metadata CSV in the dataset (default metadata.csv)")  # fmt: skip
    p.add_argument("--root", default=".", help="Workspace root (default: here)")
    add_remote_option(p)
    return p


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    use_remote(args.remote)
    try:
        summary = check(args)
    except (BioamlaCommandError, ValueError, UnicodeDecodeError, csv.Error, OSError) as e:
        print(json.dumps({"error": str(e)}))
        return 1
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
