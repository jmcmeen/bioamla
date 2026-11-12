"""Catalog calls that save their results, and the checks around them.

These are the multi-step calls behind ``bioamla catalogs`` commands: search
iNaturalist and save the rows, save a project's stats, push a dataset with its card
and location warnings, and save nearby eBird observations.

Example:
    >>> from bioamla.catalogs import inat_search
    >>> inat_search(taxon_name="Lithobates catesbeianus", output="frogs.csv").count
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from bioamla.exceptions import (
    BioamlaError,
    CatalogError,
    InvalidInputError,
    OptionConflictError,
)


def _write(path: str | Path, write: Any) -> None:
    out = Path(path)
    try:
        out.parent.mkdir(parents=True, exist_ok=True)
        write(out)
    except OSError as e:
        raise CatalogError(f"Failed to write {path}: {e}") from e


def inat_observation_rows(observations: list[dict]) -> list[dict[str, Any]]:
    """iNaturalist observations as flat rows: id, names, sound count, date, place, URL."""
    rows = []
    for obs in observations:
        taxon = obs.get("taxon") or {}
        observed_on = obs.get("observed_on") or ""
        if hasattr(observed_on, "strftime"):
            observed_on = observed_on.strftime("%Y-%m-%d")
        rows.append(
            {
                "observation_id": obs.get("id"),
                "scientific_name": taxon.get("name", ""),
                "common_name": taxon.get("preferred_common_name", ""),
                "sound_count": len(obs.get("sounds") or []),
                "observed_on": str(observed_on),
                "location": obs.get("place_guess", "") or "",
                "url": f"https://www.inaturalist.org/observations/{obs.get('id')}",
            }
        )
    return rows


@dataclass
class InatSearch:
    """iNaturalist observations with sounds, as rows.

    Attributes:
        observations: The rows (see :func:`inat_observation_rows`).
        output: The CSV they were saved to, if any.
    """

    observations: list[dict[str, Any]] = field(default_factory=list)
    output: str | None = None

    @property
    def count(self) -> int:
        """The number of observations."""
        return len(self.observations)

    def to_dict(self) -> dict[str, Any]:
        """``{"count", "observations", "output"}``."""
        return {"count": self.count, "observations": self.observations, "output": self.output}


def inat_search(
    *,
    taxon_name: str | None = None,
    taxon_id: int | None = None,
    place_id: int | None = None,
    project_id: str | None = None,
    quality_grade: str = "research",
    limit: int = 20,
    output: str | Path | None = None,
) -> InatSearch:
    """Search iNaturalist for observations with sounds, optionally saving them as CSV.

    Raises:
        OptionConflictError: If no filter is given.
        CatalogError: If the search or the CSV write fails.
    """
    from bioamla.catalogs import inat

    if not any([taxon_name, taxon_id, place_id, project_id]):
        raise OptionConflictError(
            "At least one search filter must be provided: taxon_name (--species), "
            "taxon_id (--taxon-id), place_id (--place-id) or project_id (--project-id)"
        )
    result = inat.search(
        taxon_id=taxon_id,
        taxon_name=taxon_name,
        place_id=place_id,
        project_id=project_id,
        quality_grade=quality_grade,
        per_page=limit,
    )
    rows = inat_observation_rows(result.observations)
    saved = None
    if output and rows:
        from bioamla.batch import write_rows_csv

        _write(output, lambda p: write_rows_csv(p, rows))
        saved = str(output)
    return InatSearch(observations=rows, output=saved)


@dataclass
class ProjectStatsReport:
    """An iNaturalist project's statistics.

    Attributes:
        stats: The project's title, type, place and counts.
        output: The JSON file they were saved to, if any.
    """

    stats: Any
    output: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """``{"stats", "output"}``."""
        return {"stats": self.stats.to_dict(), "output": self.output}


def inat_project_stats(project_id: str, *, output: str | Path | None = None) -> ProjectStatsReport:
    """An iNaturalist project's statistics, optionally saved as JSON.

    Raises:
        CatalogError: If the lookup or the write fails.
    """
    from bioamla.catalogs import inat

    stats = inat.get_project_stats(project_id=project_id)
    if output:
        _write(output, lambda p: p.write_text(json.dumps(stats.to_dict(), indent=2)))
    return ProjectStatsReport(stats=stats, output=str(output) if output else None)


def dataset_push_warnings(
    path: str | Path, coordinates: str = "keep", unredacted: str = "refuse", *, card: bool = False
) -> list[str]:
    """What a dataset push would publish, or leave out, that the uploader may not expect.

    With ``coordinates="keep"``: per-file locations in the metadata or the WAV files'
    recorder metadata. Otherwise: files the push can't vouch for, which may still carry
    a location ``coordinates`` doesn't change: left out with ``unredacted="skip"``,
    published as they are with ``"publish"``, and with ``"refuse"`` a reason the push
    would stop (see :func:`bioamla.catalogs.huggingface.unredacted_files`). ``card``
    says the push writes a generated card over ``README.md`` first.
    """
    from bioamla.datasets import unredacted_location_files

    card = card and _has_card_source(path)  # no card replaces README.md otherwise
    files = unredacted_location_files(path, card=card) if coordinates != "keep" else []
    return _warnings(path, coordinates, unredacted, files)


def _warnings(path: str | Path, coordinates: str, unredacted: str, files: list[str]) -> list[str]:
    """:func:`dataset_push_warnings`, with the unredactable ``files`` already listed."""
    from bioamla.datasets import has_location_columns, wavs_with_location

    if coordinates == "keep":
        where = []
        if has_location_columns(path):
            where.append("the metadata")
        if wavs_with_location(path):
            where.append("the WAV files' recorder metadata")
        if not where:
            return []
        return [
            f"Per-file locations in {' and '.join(where)} will be published as they are. "
            "Use --coordinates round or --coordinates strip to redact them."
        ]
    if not files:
        return []
    if unredacted == "skip":
        return [
            f"Leaving out {len(files)} file(s) the redaction can't vouch for, which may "
            f"carry a location --coordinates doesn't change, e.g. {files[0]}."
        ]
    if unredacted == "refuse":
        return [
            f"{len(files)} file(s) the redaction can't vouch for may carry a location "
            f"--coordinates doesn't change, e.g. {files[0]}; the push refuses them unless "
            "--unredacted skip or publish is given."
        ]
    return [
        f"{len(files)} file(s) the redaction can't vouch for may still carry a location "
        f"that --coordinates does not change, e.g. {files[0]}."
    ]


def _has_card_source(path: str | Path) -> bool:
    """Whether :func:`bioamla.datasets.write_dataset_card` can build a card here."""
    return (Path(path) / "dataset.json").exists() or (Path(path) / "metadata.csv").exists()


@dataclass
class DatasetPush:
    """A dataset pushed to the Hugging Face Hub.

    Attributes:
        result: The push (repository URL, ...).
        card: The dataset card written before the push, if any.
        card_error: Why no card could be written, if it failed.
        warnings: What the push published that may need a second look.
    """

    result: Any
    card: str | None = None
    card_error: str | None = None
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """The push result, plus ``card``, ``card_error`` and ``warnings``."""
        from bioamla.common.jsonable import to_jsonable

        pushed = to_jsonable(self.result)
        return {
            **(pushed if isinstance(pushed, dict) else {"result": pushed}),
            "card": self.card,
            "card_error": self.card_error,
            "warnings": self.warnings,
        }


def push_dataset_folder(
    path: str | Path,
    repo_id: str,
    *,
    private: bool = False,
    commit_message: str | None = None,
    card: bool = True,
    coordinates: str = "keep",
    coordinate_decimals: int = 2,
    unredacted: str = "refuse",
    license: str | None = None,
    on_notice: Any = None,
) -> DatasetPush:
    """Push a dataset folder to the Hugging Face Hub, writing its card first.

    With ``card``, a ``README.md`` dataset card is generated from the folder's
    ``dataset.json`` or ``metadata.csv`` (a failure is reported, not raised).
    ``coordinates`` (``keep``, ``round``, ``strip``) controls the per-file locations in
    what is uploaded; the local folder keeps full precision. With ``round`` or
    ``strip``, ``unredacted`` says what to do with files that redaction can't change
    (see :func:`bioamla.catalogs.huggingface.push_dataset`); ``refuse`` raises before
    the card is written. ``license`` (a Hub license id such as ``cc-by-nc-4.0``) is the
    dataset's, written to the card so the Hub shows it; it needs ``card``.
    ``on_notice(level, message)`` hears about the card and each warning before the
    upload starts.

    Raises:
        InvalidInputError: If ``unredacted="refuse"`` and such files exist, or for a
            ``license`` that isn't a Hub license id or comes without ``card``.
        CatalogError: If the push fails.
    """
    from bioamla.catalogs import huggingface as hf
    from bioamla.datasets import write_dataset_card

    def notice(level: str, message: str) -> None:
        if on_notice is not None:
            on_notice(level, message)

    # Everything is checked, and a refusal raised, before the card is written, so a
    # refused push changes nothing. The plan is what --dry-run reports.
    plan = check_dataset_push(
        path,
        repo_id,
        card=card,
        coordinates=coordinates,
        coordinate_decimals=coordinate_decimals,
        unredacted=unredacted,
        license=license,
    )
    if plan.refused:
        raise InvalidInputError(plan.reason)
    card_path, card_error = None, plan.card_error
    if card_error:
        notice("warning", f"Warning: could not generate dataset card ({card_error})")
    if plan.card:
        try:
            card_path = write_dataset_card(str(path), license=plan.license)
            notice("info", f"Wrote dataset card: {card_path}")
        except (BioamlaError, OSError) as e:
            if plan.license is not None:
                raise  # the license would be lost: don't publish without it
            card_error = str(e)
            notice("warning", f"Warning: could not generate dataset card ({e})")
    found, warnings = plan.unvouched, plan.warnings
    if plan.card and not card_path:
        # No card replaced README.md after all, so it must pass on its own.
        found = hf.unredacted_files(path, coordinates, unredacted, coordinate_decimals)
        warnings = _warnings(path, coordinates, unredacted, found)
    for warning in warnings:
        notice("warning", warning)
    result = hf.push_dataset(
        str(path),
        repo_id,
        private=private,
        commit_message=commit_message,
        coordinates=coordinates,
        coordinate_decimals=coordinate_decimals,
        unredacted=unredacted,
        _unredacted_files=found,
    )
    return DatasetPush(result=result, card=card_path, card_error=card_error, warnings=warnings)


@dataclass
class DatasetPushPlan:
    """What :func:`push_dataset_folder` would do with a folder, found without doing it.

    Attributes:
        path: The dataset folder.
        repo_id: The Hub repository, if given.
        coordinates: The coordinate mode (``keep``, ``round``, ``strip``).
        coordinate_decimals: Decimal places ``round`` keeps.
        unredacted: What happens to files it can't vouch for.
        card: Whether a dataset card would be written to ``README.md`` first.
        card_error: Why no card can be built, when one was asked for (the push then
            warns and goes on without one).
        license: The dataset's license, as a Hub license id, if given.
        redacted: Files whose uploaded copy would be redacted (tables with a location
            column, the manifest, the card, WAVs with a recorder position), apart from
            those in ``unvouched``.
        unvouched: Files it can't vouch for (see
            :func:`bioamla.datasets.unredacted_location_files`).
        left_out: The ``unvouched`` files left out of the upload (``skip``).
        refused: Whether the push would stop before uploading (``refuse`` with
            unvouched files).
        reason: Why it would stop, if it would.
        warnings: The warnings the push would give.
    """

    path: str
    repo_id: str | None
    coordinates: str
    coordinate_decimals: int
    unredacted: str
    card: bool
    license: str | None
    card_error: str | None = None
    redacted: list[str] = field(default_factory=list)
    unvouched: list[str] = field(default_factory=list)
    left_out: list[str] = field(default_factory=list)
    refused: bool = False
    reason: str | None = None
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """The plan's fields, plus ``"dry_run": True``."""
        from dataclasses import asdict

        from bioamla.common.jsonable import to_jsonable

        return {"dry_run": True, **to_jsonable(asdict(self))}


def check_dataset_push(
    path: str | Path,
    repo_id: str | None = None,
    *,
    card: bool = True,
    coordinates: str = "keep",
    coordinate_decimals: int = 2,
    unredacted: str = "refuse",
    license: str | None = None,
) -> DatasetPushPlan:
    """Check a dataset push as :func:`push_dataset_folder` would, without pushing.

    Nothing is written (no card) and nothing is sent to the Hub. A push that
    ``unredacted="refuse"`` would stop is reported (``refused`` and ``reason``), not
    raised, so a caller can read which files are the reason. The push itself runs
    from this plan, so the two can't disagree.

    Raises:
        InvalidInputError: If ``path`` isn't a directory, for an unknown mode, bad
            ``coordinate_decimals`` or a ``license`` that isn't a Hub license id, or
            when ``license`` is given and no card can be built.
        OptionConflictError: For ``license`` without ``card``.
        DatasetError: If a file that may carry locations can't be read to redact it,
            or (with ``license``) the card can't be built.
    """
    from bioamla.catalogs import huggingface as hf
    from bioamla.datasets import dataset_card_text, files_to_redact, hub_license_id

    hf._require_dir(str(path))
    hf._check_modes(coordinates, unredacted, coordinate_decimals)
    if license is not None:
        if not card:
            raise OptionConflictError(
                "--license is written to the dataset card, so it can't go with --no-card"
            )
        license = hub_license_id(license)
    card_text = card_error = None
    if card:
        try:
            card_text = dataset_card_text(str(path), license=license)
        except BioamlaError as e:
            if license is not None:
                raise  # the license would be lost: don't publish without it
            card_error = str(e)
        if license is not None and card_text is None:
            raise InvalidInputError(
                f"--license is written to the dataset card, and {path} has no metadata.csv "
                "or dataset.json to build one from"
            )
    card_due = card_text is not None
    files = hf._unvouched_files(path, coordinates, unredacted, coordinate_decimals, card=card_due)
    redacted = files_to_redact(path, card=card_due) if coordinates != "keep" else []
    refused = bool(files) and unredacted == "refuse"
    return DatasetPushPlan(
        path=str(path),
        repo_id=repo_id,
        coordinates=coordinates,
        coordinate_decimals=coordinate_decimals,
        unredacted=unredacted,
        card=card_due,
        card_error=card_error,
        license=license,
        redacted=[f for f in redacted if f not in set(files)],
        unvouched=files,
        left_out=files if unredacted == "skip" else [],
        refused=refused,
        reason=hf._unvouched_refusal(files, coordinates) if refused else None,
        warnings=_warnings(path, coordinates, unredacted, files),
    )


@dataclass
class NearbyObservations:
    """Recent eBird observations near a place.

    Attributes:
        observations: The observations.
        output: The CSV they were saved to, if any.
    """

    observations: list[Any] = field(default_factory=list)
    output: str | None = None

    @property
    def count(self) -> int:
        """The number of observations."""
        return len(self.observations)

    def to_dict(self) -> dict[str, Any]:
        """``{"count", "observations", "output"}``."""
        return {"count": self.count, "observations": self.observations, "output": self.output}


NEARBY_COLUMNS = [
    "species_code",
    "common_name",
    "scientific_name",
    "location_name",
    "observation_date",
    "how_many",
]


def ebird_nearby(
    lat: float,
    lng: float,
    *,
    distance_km: float = 25,
    days: int = 14,
    limit: int = 20,
    output: str | Path | None = None,
    api_key: str | None = None,
) -> NearbyObservations:
    """Recent eBird observations near a place, optionally saved as CSV.

    The key is ``api_key`` or the ``EBIRD_API_KEY`` environment variable.

    Raises:
        CatalogError: If the request or the write fails, or there is no key.
    """
    from bioamla.catalogs.ebird import EBirdService

    result = EBirdService(api_key=api_key).get_nearby(
        lat=lat, lng=lng, distance_km=distance_km, days=days, limit=limit
    )
    observations = result.observations
    if output:

        def _csv(p: Path) -> None:
            with p.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=NEARBY_COLUMNS, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(obs.to_dict() for obs in observations)

        _write(output, _csv)
    return NearbyObservations(observations=observations, output=str(output) if output else None)


__all__ = [
    "NEARBY_COLUMNS",
    "DatasetPush",
    "InatSearch",
    "NearbyObservations",
    "ProjectStatsReport",
    "DatasetPushPlan",
    "check_dataset_push",
    "dataset_push_warnings",
    "ebird_nearby",
    "inat_observation_rows",
    "inat_project_stats",
    "inat_search",
    "push_dataset_folder",
]
