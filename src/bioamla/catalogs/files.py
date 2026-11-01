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

from bioamla.exceptions import BioamlaError, CatalogError, OptionConflictError


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


def dataset_push_warnings(path: str | Path, coordinates: str = "keep") -> list[str]:
    """What a dataset push would publish that the uploader may not want to.

    With ``coordinates="keep"``: per-file locations in the metadata or the WAV files'
    recorder metadata. Otherwise: files that may still carry a location
    ``coordinates`` doesn't change (JSON or text, Parquet).
    """
    from bioamla.datasets import has_location_columns, unredacted_location_files, wavs_with_location

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
    unredacted = unredacted_location_files(path)
    if not unredacted:
        return []
    return [
        f"{len(unredacted)} file(s) may still carry a location that --coordinates "
        f"does not change (JSON/text, Parquet), e.g. {unredacted[0]}."
    ]


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
    on_notice: Any = None,
) -> DatasetPush:
    """Push a dataset folder to the Hugging Face Hub, writing its card first.

    With ``card``, a ``README.md`` dataset card is generated from the folder's
    ``dataset.json`` or ``metadata.csv`` (a failure is reported, not raised).
    ``coordinates`` (``keep``, ``round``, ``strip``) controls the per-file locations in
    what is uploaded; the local folder keeps full precision. ``on_notice(level,
    message)`` hears about the card and each warning before the upload starts.

    Raises:
        CatalogError: If the push fails.
    """
    from bioamla.catalogs import huggingface as hf
    from bioamla.datasets import write_dataset_card

    def notice(level: str, message: str) -> None:
        if on_notice is not None:
            on_notice(level, message)

    card_path = card_error = None
    if card:
        try:
            card_path = write_dataset_card(str(path))
            if card_path:
                notice("info", f"Wrote dataset card: {card_path}")
        except BioamlaError as e:
            card_error = str(e)
            notice("warning", f"Warning: could not generate dataset card ({e})")
    warnings = dataset_push_warnings(path, coordinates)
    for warning in warnings:
        notice("warning", warning)
    result = hf.push_dataset(
        str(path),
        repo_id,
        private=private,
        commit_message=commit_message,
        coordinates=coordinates,
        coordinate_decimals=coordinate_decimals,
    )
    return DatasetPush(result=result, card=card_path, card_error=card_error, warnings=warnings)


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
    "dataset_push_warnings",
    "ebird_nearby",
    "inat_observation_rows",
    "inat_project_stats",
    "inat_search",
    "push_dataset_folder",
]
