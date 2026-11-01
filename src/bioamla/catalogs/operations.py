"""Operations for ``catalogs``: thin adapters over :mod:`bioamla.catalogs`.

See :mod:`bioamla.operations`. Searches and lookups are ``quick``; downloads, pushes
and pulls are jobs. API keys are never part of a request: an operation takes a key from
the context's secrets (local runs: a command-line flag), else from the environment
where it runs (``EBIRD_API_KEY``, ``XC_API_KEY``, ``HF_TOKEN``). Purging the Hugging Face cache is local-only.
"""

from __future__ import annotations

from typing import Any, Literal

from bioamla.common.config import config_dataclass
from bioamla.operations import OperationContext, PathStr, operation

# --------------------------------------------------------------------------- #
# iNaturalist
# --------------------------------------------------------------------------- #


@config_dataclass
class InatSearchRequest:
    """iNaturalist observations with sounds.

    Attributes:
        species: Scientific name.
        taxon_id: iNaturalist taxon ID.
        place_id: iNaturalist place ID.
        project_id: Project slug or ID.
        quality_grade: ``research``, ``needs_id`` or ``casual``.
        limit: Maximum observations.
        output: A CSV to save them to.
    """

    species: str | None = None
    taxon_id: int | None = None
    place_id: int | None = None
    project_id: str | None = None
    quality_grade: Literal["research", "needs_id", "casual"] = "research"
    limit: int = 20
    output: PathStr | None = None


@operation("catalogs.inat.search", request=InatSearchRequest)
def inat_search_op(request: InatSearchRequest, context: OperationContext) -> Any:
    """Search iNaturalist for observations with sounds."""
    from bioamla.catalogs.files import inat_search

    return inat_search(
        taxon_name=request.species,
        taxon_id=request.taxon_id,
        place_id=request.place_id,
        project_id=request.project_id,
        quality_grade=request.quality_grade,
        limit=request.limit,
        output=request.output,
    )


@config_dataclass
class InatStatsRequest:
    """An iNaturalist project's statistics.

    Attributes:
        project_id: Project slug or ID.
        output: A JSON file to save them to.
    """

    project_id: str
    output: PathStr | None = None


@operation("catalogs.inat.stats", request=InatStatsRequest)
def inat_stats_op(request: InatStatsRequest, context: OperationContext) -> Any:
    """Get an iNaturalist project's statistics."""
    from bioamla.catalogs.files import inat_project_stats

    return inat_project_stats(request.project_id, output=request.output)


@config_dataclass
class InatDownloadRequest:
    """Download iNaturalist observations' sounds.

    Attributes:
        output_dir: Where to save them.
        taxon_ids: Taxon IDs.
        taxon_name: Taxon name.
        place_id: Place ID.
        project_id: Project slug or ID.
        quality_grade: ``research``, ``needs_id``, ``casual`` or ``any``.
        licenses: Sound licenses to accept (``cc0``, ``cc-by``, ...).
        start_date: First observation date (YYYY-MM-DD).
        end_date: Last observation date.
        obs_per_taxon: Maximum observations per taxon.
    """

    output_dir: PathStr
    taxon_ids: list[int] | None = None
    taxon_name: str | None = None
    place_id: int | None = None
    project_id: str | None = None
    quality_grade: Literal["research", "needs_id", "casual", "any"] = "research"
    licenses: list[str] | None = None
    start_date: str | None = None
    end_date: str | None = None
    obs_per_taxon: int = 10


@operation("catalogs.inat.download", request=InatDownloadRequest, kind="job")
def inat_download_op(request: InatDownloadRequest, context: OperationContext) -> Any:
    """Download iNaturalist observations' sounds."""
    from bioamla.catalogs import inat

    return inat.download(
        output_dir=request.output_dir,
        taxon_ids=request.taxon_ids,
        taxon_name=request.taxon_name,
        place_id=request.place_id,
        project_id=request.project_id,
        quality_grade=None if request.quality_grade == "any" else request.quality_grade,
        sound_license=request.licenses,
        d1=request.start_date,
        d2=request.end_date,
        obs_per_taxon=request.obs_per_taxon,
    )


# --------------------------------------------------------------------------- #
# Hugging Face Hub
# --------------------------------------------------------------------------- #


@config_dataclass
class PushModelRequest:
    """Push a model folder to the Hugging Face Hub.

    Attributes:
        path: The model folder.
        repo_id: The Hub repository.
        private: Create it private.
        commit_message: The commit message.
    """

    path: PathStr
    repo_id: str
    private: bool = False
    commit_message: str | None = None


@operation("catalogs.hf.push-model", request=PushModelRequest, kind="job")
def hf_push_model_op(request: PushModelRequest, context: OperationContext) -> Any:
    """Push a model folder to the Hugging Face Hub."""
    from bioamla.catalogs import huggingface as hf

    return hf.push_model(
        request.path,
        request.repo_id,
        private=request.private,
        commit_message=request.commit_message,
    )


@config_dataclass
class PushDatasetRequest:
    """Push a dataset folder to the Hugging Face Hub, writing its card first.

    Attributes:
        path: The dataset folder.
        repo_id: The Hub repository.
        private: Create it private.
        commit_message: The commit message.
        card: Write a README.md dataset card first.
        coordinates: Per-file locations in what is uploaded: ``keep``, ``round``,
            ``strip``.
        coordinate_decimals: Decimal places ``round`` keeps (0-6).
    """

    path: PathStr
    repo_id: str
    private: bool = False
    commit_message: str | None = None
    card: bool = True
    coordinates: Literal["keep", "round", "strip"] = "keep"
    coordinate_decimals: int = 2


@operation("catalogs.hf.push-dataset", request=PushDatasetRequest, kind="job")
def hf_push_dataset_op(request: PushDatasetRequest, context: OperationContext) -> Any:
    """Push a dataset folder to the Hugging Face Hub."""
    from bioamla.catalogs.files import push_dataset_folder

    return push_dataset_folder(
        request.path,
        request.repo_id,
        private=request.private,
        commit_message=request.commit_message,
        card=request.card,
        coordinates=request.coordinates,
        coordinate_decimals=request.coordinate_decimals,
        on_notice=context.notice,
    )


@config_dataclass
class PullDatasetRequest:
    """Pull a Hugging Face audio dataset into label folders and a metadata.csv.

    Attributes:
        repo_id: The Hub dataset.
        dest: Where to write it.
        split: One split (default: all).
        config: Dataset config or subset name.
        audio_column: The audio column (default: detected).
        label_column: The label column (default: detected).
        sample_rate: Resample to this rate (0: keep).
        layout: ``both``, ``audiofolder`` or ``flat``.
        verbose: Log progress.
    """

    repo_id: str
    dest: PathStr
    split: str | None = None
    config: str | None = None
    audio_column: str | None = None
    label_column: str | None = None
    sample_rate: int = 16000
    layout: Literal["both", "audiofolder", "flat"] = "both"
    verbose: bool = True


@operation("catalogs.hf.pull-dataset", request=PullDatasetRequest, kind="job")
def hf_pull_dataset_op(request: PullDatasetRequest, context: OperationContext) -> Any:
    """Pull a Hugging Face audio dataset into label folders."""
    from bioamla.catalogs import huggingface as hf

    return hf.pull_dataset(
        request.repo_id,
        request.dest,
        split=request.split,
        config=request.config,
        audio_column=request.audio_column,
        label_column=request.label_column,
        sample_rate=request.sample_rate or None,
        layout=request.layout,
        verbose=request.verbose,
    )


@config_dataclass
class CacheRequest:
    """The local Hugging Face cache.

    Attributes:
        models: Include cached models.
        datasets: Include cached datasets.
    """

    models: bool = True
    datasets: bool = True


@operation("catalogs.hf.cache", request=CacheRequest)
def hf_cache_op(request: CacheRequest, context: OperationContext) -> dict[str, Any]:
    """List the local Hugging Face cache."""
    from bioamla.catalogs import huggingface as hf

    repos = hf.scan_cache(models=request.models, datasets=request.datasets)
    return {"repos": repos, "total_bytes": sum(r.size_bytes for r in repos)}


@operation("catalogs.hf.cache-purge", request=CacheRequest, remote=False)
def hf_cache_purge_op(request: CacheRequest, context: OperationContext) -> Any:
    """Delete the local Hugging Face cache (local only)."""
    from bioamla.catalogs import huggingface as hf

    return hf.purge_cache(models=request.models, datasets=request.datasets)


# --------------------------------------------------------------------------- #
# Xeno-canto and Macaulay Library
# --------------------------------------------------------------------------- #


@config_dataclass
class XcSearchRequest:
    """Xeno-canto recordings.

    Attributes:
        species: Species name.
        genus: Genus.
        country: Country.
        quality: Quality (A-E).
        sound_type: Sound type (song, call, ...).
        max_results: Maximum results.
    """

    species: str | None = None
    genus: str | None = None
    country: str | None = None
    quality: str | None = None
    sound_type: str | None = None
    max_results: int = 10


@operation("catalogs.xc.search", request=XcSearchRequest)
def xc_search_op(request: XcSearchRequest, context: OperationContext) -> Any:
    """Search Xeno-canto."""
    from bioamla.catalogs import xeno_canto as xc

    return xc.search(
        species=request.species,
        genus=request.genus,
        country=request.country,
        quality=request.quality,
        sound_type=request.sound_type,
        max_results=request.max_results,
    )


@config_dataclass
class XcDownloadRequest:
    """Download Xeno-canto recordings.

    Attributes:
        output_dir: Where to save them.
        species: Species name.
        genus: Genus.
        country: Country.
        quality: Quality filter.
        max_recordings: Maximum recordings.
        delay: Seconds between downloads.
    """

    output_dir: PathStr = "./xc_recordings"
    species: str | None = None
    genus: str | None = None
    country: str | None = None
    quality: str = "A"
    max_recordings: int = 10
    delay: float = 1.0


@operation("catalogs.xc.download", request=XcDownloadRequest, kind="job")
def xc_download_op(request: XcDownloadRequest, context: OperationContext) -> Any:
    """Download Xeno-canto recordings."""
    from bioamla.catalogs import xeno_canto as xc

    return xc.download(
        species=request.species,
        genus=request.genus,
        country=request.country,
        quality=request.quality,
        max_recordings=request.max_recordings,
        output_dir=request.output_dir,
        delay=request.delay,
    )


@config_dataclass
class MlSearchRequest:
    """Macaulay Library recordings (at least one filter).

    Attributes:
        species_code: eBird species code.
        scientific_name: Scientific name.
        common_name: Common name.
        region: Region code (e.g. US-NY).
        country: Country code.
        taxon_code: eBird taxon code.
        hotspot_code: eBird hotspot code.
        min_rating: Minimum rating (1-5).
        max_results: Maximum results.
    """

    species_code: str | None = None
    scientific_name: str | None = None
    common_name: str | None = None
    region: str | None = None
    country: str | None = None
    taxon_code: str | None = None
    hotspot_code: str | None = None
    min_rating: int = 0
    max_results: int = 10


def _ml_filters(request: Any) -> dict[str, Any]:
    return {
        "species_code": request.species_code,
        "scientific_name": request.scientific_name,
        "common_name": request.common_name,
        "region": request.region,
        "country": request.country,
        "taxon_code": request.taxon_code,
        "hotspot_code": request.hotspot_code,
        "min_rating": request.min_rating,
    }


@operation("catalogs.ml.search", request=MlSearchRequest)
def ml_search_op(request: MlSearchRequest, context: OperationContext) -> Any:
    """Search the Macaulay Library."""
    from bioamla.catalogs import macaulay as ml

    return ml.search(**_ml_filters(request), max_results=request.max_results)


@config_dataclass
class MlDownloadRequest(MlSearchRequest):
    """Download Macaulay Library recordings (at least one filter).

    Attributes:
        output_dir: Where to save them.
        max_recordings: Maximum recordings.
    """

    min_rating: int = 3
    max_recordings: int = 10
    output_dir: PathStr = "./ml_recordings"


@operation("catalogs.ml.download", request=MlDownloadRequest, kind="job")
def ml_download_op(request: MlDownloadRequest, context: OperationContext) -> Any:
    """Download Macaulay Library recordings."""
    from bioamla.catalogs import macaulay as ml

    return ml.download(
        **_ml_filters(request),
        max_recordings=request.max_recordings,
        output_dir=request.output_dir,
    )


# --------------------------------------------------------------------------- #
# eBird
# --------------------------------------------------------------------------- #


@config_dataclass
class SpeciesRequest:
    """A species in the eBird taxonomy.

    Attributes:
        name: Common name, scientific name or species code.
    """

    name: str


@operation("catalogs.ebird.species", request=SpeciesRequest)
def ebird_species_op(request: SpeciesRequest, context: OperationContext) -> Any:
    """Look up a species in the eBird taxonomy."""
    from bioamla.catalogs import species

    return species.lookup(request.name, ebird_only=True)


@config_dataclass
class SpeciesSearchRequest:
    """Fuzzy search of the eBird taxonomy.

    Attributes:
        query: What to look for.
        limit: Maximum results.
    """

    query: str
    limit: int = 10


@operation("catalogs.ebird.search", request=SpeciesSearchRequest)
def ebird_search_op(request: SpeciesSearchRequest, context: OperationContext) -> dict[str, Any]:
    """Fuzzy search the eBird taxonomy."""
    from bioamla.catalogs import species

    matches = species.search(request.query, limit=request.limit)
    return {"query": request.query, "count": len(matches), "matches": matches}


@config_dataclass
class ValidateRequest:
    """Is a species expected at a place (recent eBird observations)?

    Attributes:
        species_code: eBird species code.
        lat: Latitude.
        lng: Longitude.
        distance: Search radius (km).
    """

    species_code: str
    lat: float
    lng: float
    distance: float = 50


@operation("catalogs.ebird.validate", request=ValidateRequest)
def ebird_validate_op(request: ValidateRequest, context: OperationContext) -> Any:
    """Check whether a species is expected at a place."""
    from bioamla.catalogs.ebird import EBirdService

    return EBirdService(api_key=context.secret("EBIRD_API_KEY")).validate_species(
        species_code=request.species_code,
        lat=request.lat,
        lng=request.lng,
        distance_km=request.distance,
    )


@config_dataclass
class NearbyRequest:
    """Recent eBird observations near a place.

    Attributes:
        lat: Latitude.
        lng: Longitude.
        distance: Search radius (km).
        days: Days back.
        limit: Maximum observations.
        output: A CSV to save them to.
    """

    lat: float
    lng: float
    distance: float = 25
    days: int = 14
    limit: int = 20
    output: PathStr | None = None


@operation("catalogs.ebird.nearby", request=NearbyRequest)
def ebird_nearby_op(request: NearbyRequest, context: OperationContext) -> Any:
    """Get recent eBird observations near a place."""
    from bioamla.catalogs.files import ebird_nearby

    return ebird_nearby(
        request.lat,
        request.lng,
        distance_km=request.distance,
        days=request.days,
        limit=request.limit,
        output=request.output,
        api_key=context.secret("EBIRD_API_KEY"),
    )
