"""Remix a labeled audio dataset: keep or drop classes, add classes, replace examples.

A *remix* is a derived dataset built from a source dataset directory (one with
a ``metadata.csv``) by selecting classes and optionally splicing in clips from
other directories. The output is a fresh dataset directory organized by label
(``<label>/<file>``) with its own ``metadata.csv`` and, when the rows carry
attribution columns, a generated ``LICENSE``.

Remixes are described by a small TOML recipe so a derived dataset can be
rebuilt from its source rather than stored::

    [remix]
    name = "esc-50-animals"
    source = "../datasets/esc-50"
    keep = ["dog", "cat", "frog"]       # or: drop = ["airplane"]

    [[add]]                             # new class from a clip directory
    label = "spring_peeper"
    source = "../build/spring-peeper-clips"
    max_clips = 40

    [[replace]]                         # swap one class's examples
    label = "frog"
    source = "../build/frog-clips"

Relative paths in a recipe file are resolved against the recipe's own
directory, so a recipe builds the same dataset from any working directory.
The label column is ``label`` when present, else ``category`` (ESC-50 style);
set ``label_column`` to force one. Added and replaced clips get the remix
label in every label-like column (``label``, ``category`` and the detected
column), so downstream tools that prefer ``label`` agree with the remix. Other
columns (``target``, ``fold``, ``attr_*`` ...) pass through unchanged.
"""

from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from bioamla.exceptions import InvalidInputError, NotFoundError, RemixError

logger = logging.getLogger(__name__)

_ATTRIBUTION_SIGNALS = ("attr_id", "attribution", "attr_lic", "license", "attr_url")


@dataclass
class RemixSource:
    """Clips to add to, or substitute into, a remix under one label.

    Attributes:
        label: Label the clips get in the output.
        source: Directory holding the clips and a ``metadata.csv``.
        max_clips: Take at most this many clips (in metadata order); None for all.
    """

    label: str
    source: str
    max_clips: int | None = None


@dataclass
class RemixRecipe:
    """What to build from a source dataset.

    Attributes:
        source: Source dataset directory (must contain ``metadata.csv``).
        name: Output dataset name, used for reporting; defaults to the output dir name.
        label_column: Column holding the class label; auto-detected when None.
        keep: Keep only these labels. Mutually exclusive with ``drop``.
        drop: Remove these labels.
        add: Classes to add from other directories.
        replace: Classes whose examples are replaced from other directories.
    """

    source: str
    name: str | None = None
    label_column: str | None = None
    keep: list[str] | None = None
    drop: list[str] | None = None
    add: list[RemixSource] = field(default_factory=list)
    replace: list[RemixSource] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.keep is not None and self.drop is not None:
            raise InvalidInputError("A remix recipe may set keep or drop, not both")
        if not self.source:
            raise InvalidInputError("A remix recipe needs a source dataset")
        seen: set[str] = set()
        for entry in [*self.add, *self.replace]:
            if entry.label in seen:
                raise InvalidInputError(
                    f"Label {entry.label!r} appears more than once in add/replace"
                )
            seen.add(entry.label)

    @classmethod
    def from_dict(cls, data: dict[str, Any], base_dir: str | Path | None = None) -> RemixRecipe:
        """Build a recipe from parsed TOML/JSON data (``[remix]`` + ``[[add]]``/``[[replace]]``).

        Args:
            data: The parsed recipe.
            base_dir: Directory that relative ``source`` paths are resolved against.
                None leaves them as written.
        """
        unknown = set(data) - {"remix", "add", "replace"}
        if unknown:
            raise InvalidInputError(f"Unknown recipe section(s): {', '.join(sorted(unknown))}")
        remix = data.get("remix")
        if not isinstance(remix, dict):
            raise InvalidInputError("Recipe is missing its [remix] table")
        unknown = set(remix) - {"name", "source", "label_column", "keep", "drop"}
        if unknown:
            raise InvalidInputError(f"Unknown [remix] key(s): {', '.join(sorted(unknown))}")

        def _path(value: Any, where: str) -> str:
            if not isinstance(value, str) or not value:
                raise InvalidInputError(f"{where} must be a non-empty path")
            if base_dir is not None and not Path(value).is_absolute():
                return str(Path(base_dir) / value)
            return value

        def _labels(key: str) -> list[str] | None:
            if key not in remix:
                return None
            value = remix[key]
            if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
                raise InvalidInputError(f"[remix] {key} must be a list of label strings")
            return list(value)

        def _sources(key: str) -> list[RemixSource]:
            out = []
            entries = data.get(key, [])
            if not isinstance(entries, list):
                raise InvalidInputError(f"{key} must be an array of tables ([[{key}]])")
            for i, entry in enumerate(entries):
                where = f"[[{key}]] entry {i + 1}"
                unknown = set(entry) - {"label", "source", "max_clips"}
                if unknown:
                    raise InvalidInputError(f"Unknown {where} key(s): {', '.join(sorted(unknown))}")
                label = entry.get("label")
                if not isinstance(label, str) or not label:
                    raise InvalidInputError(f"{where} needs a label")
                max_clips = entry.get("max_clips")
                if max_clips is not None and (
                    isinstance(max_clips, bool) or not isinstance(max_clips, int) or max_clips < 0
                ):
                    raise InvalidInputError(f"{where} max_clips must be a non-negative integer")
                out.append(
                    RemixSource(label, _path(entry.get("source"), f"{where} source"), max_clips)
                )
            return out

        name = remix.get("name")
        if name is not None and not isinstance(name, str):
            raise InvalidInputError("[remix] name must be a string")
        label_column = remix.get("label_column")
        if label_column is not None and not isinstance(label_column, str):
            raise InvalidInputError("[remix] label_column must be a string")

        return cls(
            source=_path(remix.get("source"), "[remix] source"),
            name=name,
            label_column=label_column,
            keep=_labels("keep"),
            drop=_labels("drop"),
            add=_sources("add"),
            replace=_sources("replace"),
        )

    @classmethod
    def from_toml(cls, path: str | Path) -> RemixRecipe:
        """Load a recipe from a TOML file; relative paths resolve against its directory.

        Raises:
            NotFoundError: If the file does not exist.
            ConfigError: If the TOML cannot be parsed.
            InvalidInputError: If the recipe is malformed.
        """
        from bioamla.common.config import load_toml

        path = Path(path)
        return cls.from_dict(load_toml(path), base_dir=path.parent)


def _detect_label_column(fieldnames: set[str], explicit: str | None) -> str:
    if explicit:
        if explicit not in fieldnames:
            raise InvalidInputError(f"Label column {explicit!r} not found in metadata")
        return explicit
    for candidate in ("label", "category"):
        if candidate in fieldnames:
            return candidate
    raise InvalidInputError("Metadata has neither a 'label' nor a 'category' column")


def _read_rows(directory: Path, metadata_filename: str) -> tuple[list[dict], set[str]]:
    from bioamla.datasets._metadata import read_metadata_csv

    if not directory.is_dir():
        raise NotFoundError(f"Dataset directory does not exist: {directory}")
    metadata_path = directory / metadata_filename
    if not metadata_path.exists():
        raise NotFoundError(f"Metadata file not found: {metadata_path}")
    rows, fieldnames = read_metadata_csv(metadata_path)
    return [r for r in rows if r.get("file_name")], fieldnames


def _check_labels(requested: list[str], present: set[str], what: str) -> None:
    missing = sorted(set(requested) - present)
    if missing:
        raise InvalidInputError(f"Cannot {what} label(s) not in the source: {', '.join(missing)}")


def remix_dataset(
    recipe: RemixRecipe,
    output_dir: str | Path,
    metadata_filename: str = "metadata.csv",
    license_template: str | Path | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """Build a remixed dataset directory from a recipe.

    Steps, in order: filter the source rows by ``keep``/``drop``; for each
    ``replace`` entry drop that label's rows and take the replacement clips;
    for each ``add`` entry take its clips under a label the result does not yet
    have; copy every selected clip to ``<output>/<label>/<basename>``; write
    ``metadata.csv``; generate ``LICENSE`` when any row carries attribution.

    All inputs (source directories, labels, template) are validated before any
    file is copied, so a bad recipe leaves no partial output behind.

    Args:
        recipe: The remix description.
        output_dir: Directory to create. Must not already contain a metadata file.
        metadata_filename: Metadata CSV name in the source and output.
        license_template: Optional header file prepended to the generated LICENSE.
        verbose: Log a summary at INFO level.

    Returns:
        Dict with ``name``, ``output_dir``, ``metadata_file``, ``label_column``,
        ``labels`` (sorted), ``total_files``, ``files_copied``, ``files_missing``,
        ``files_duplicate``, ``kept``, ``dropped``, ``added``, ``replaced`` and
        ``license_file`` (None when no row carried attribution).

    Raises:
        NotFoundError: If a source directory, its metadata, or the template is missing.
        InvalidInputError: If the recipe names labels the data does not have, selects
            nothing, or the output already holds a dataset.
        RemixError: If no clip could be copied, or copying/writing fails.
        LicenseGenerationError: If the LICENSE cannot be written.
    """
    from bioamla.common.files import sanitize_filename
    from bioamla.datasets._metadata import write_metadata_csv
    from bioamla.datasets.licenses import generate_license_for_dataset

    output_path = Path(output_dir)
    if (output_path / metadata_filename).exists():
        raise InvalidInputError(f"Output already contains a dataset: {output_path}")
    template_path = Path(license_template) if license_template else None
    if template_path is not None and not template_path.is_file():
        raise NotFoundError(f"License template not found: {template_path}")

    source_path = Path(recipe.source)
    rows, fieldnames = _read_rows(source_path, metadata_filename)
    label_column = _detect_label_column(fieldnames, recipe.label_column)
    source_labels = {r.get(label_column, "") for r in rows} - {""}

    # (row, directory it is copied from)
    selected: list[tuple[dict, Path]] = []
    dropped = 0
    if recipe.keep is not None:
        _check_labels(recipe.keep, source_labels, "keep")
        wanted = set(recipe.keep)
        for r in rows:
            if r.get(label_column) in wanted:
                selected.append((r, source_path))
            else:
                dropped += 1
    elif recipe.drop is not None:
        _check_labels(recipe.drop, source_labels, "drop")
        unwanted = set(recipe.drop)
        for r in rows:
            if r.get(label_column) in unwanted:
                dropped += 1
            else:
                selected.append((r, source_path))
    else:
        selected = [(r, source_path) for r in rows]

    current_labels = {r.get(label_column, "") for r, _ in selected}

    def _take(entry: RemixSource) -> list[tuple[dict, Path]]:
        directory = Path(entry.source)
        extra_rows, extra_fields = _read_rows(directory, metadata_filename)
        fieldnames.update(extra_fields)
        if entry.max_clips is not None:
            extra_rows = extra_rows[: entry.max_clips]
        taken = []
        for r in extra_rows:
            r = dict(r)
            # Every label-like column gets the remix label so stats/merge/partition,
            # which prefer ``label``, agree with the remix's own grouping.
            for col in {label_column, "label", "category"}:
                if col == label_column or col in r:
                    r[col] = entry.label
            taken.append((r, directory))
        return taken

    replaced = 0
    for entry in recipe.replace:
        _check_labels([entry.label], current_labels, "replace")
        selected = [(r, d) for r, d in selected if r.get(label_column) != entry.label]
        selected.extend(_take(entry))
        replaced += 1

    added = 0
    for entry in recipe.add:
        if entry.label in current_labels:
            raise InvalidInputError(
                f"Label {entry.label!r} is already in the remix; use [[replace]] to swap it"
            )
        selected.extend(_take(entry))
        current_labels.add(entry.label)
        added += 1

    if not selected:
        raise InvalidInputError("The remix selects no clips")

    output_rows: list[dict] = []
    seen: set[str] = set()
    copied = missing = duplicate = 0
    try:
        output_path.mkdir(parents=True, exist_ok=True)
        for row, directory in selected:
            src_file = directory / row["file_name"]
            dest_name = (
                f"{sanitize_filename(row.get(label_column, ''))}/{Path(row['file_name']).name}"
            )
            if dest_name in seen:
                logger.warning(f"Duplicate file name in remix, skipping: {dest_name}")
                duplicate += 1
                continue
            if not src_file.exists():
                logger.warning(f"Source file not found, skipping: {src_file}")
                missing += 1
                continue
            dest = output_path / dest_name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_file, dest)
            seen.add(dest_name)
            copied += 1
            new_row = dict(row)
            new_row["file_name"] = dest_name
            output_rows.append(new_row)

        if not copied:
            raise RemixError(
                f"No clips could be copied: {missing} source file(s) missing, "
                f"{duplicate} duplicate(s); check that file_name paths are relative to the dataset"
            )

        metadata_path = output_path / metadata_filename
        write_metadata_csv(metadata_path, output_rows, set(fieldnames), merge_existing=False)
    except OSError as e:
        raise RemixError(f"Remix failed: {e}") from e

    license_file: str | None = None
    if any(r.get(col) for r in output_rows for col in _ATTRIBUTION_SIGNALS):
        info = generate_license_for_dataset(
            output_path, template_path=template_path, metadata_filename=metadata_filename
        )
        license_file = info["output_path"]
    else:
        logger.info("No attribution columns in metadata; LICENSE not generated")

    labels = sorted({r.get(label_column, "") for r in output_rows} - {""})
    stats = {
        "name": recipe.name or output_path.name,
        "output_dir": str(output_path.absolute()),
        "metadata_file": str(metadata_path),
        "label_column": label_column,
        "labels": labels,
        "total_files": len(output_rows),
        "files_copied": copied,
        "files_missing": missing,
        "files_duplicate": duplicate,
        "kept": len(labels),
        "dropped": dropped,
        "added": added,
        "replaced": replaced,
        "license_file": license_file,
    }
    if verbose:
        logger.info(
            f"Remix {stats['name']}: {len(labels)} labels, {copied} files copied, "
            f"{dropped} source rows dropped, {added} labels added, {replaced} replaced"
        )
    return stats
