"""Data models for catalog integrations.

Each catalog source (eBird, iNaturalist, Macaulay, Xeno-canto, species, HuggingFace)
defines its own record and result dataclasses. They are gathered here so the rest of
the ``bioamla.catalogs`` package — and downstream consumers — can import them from a
single place.

The per-catalog ``SearchResult`` / ``DownloadResult`` types have different field
structures, so they are exported under catalog-prefixed names
(``INaturalistSearchResult``, ``MacaulaySearchResult``, ``XenoCantoSearchResult``,
etc.) to avoid name collisions.
"""

from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

# =============================================================================
# API endpoints / constants
# =============================================================================

EBIRD_API_URL = "https://api.ebird.org/v2"
EBIRD_TAXONOMY_URL = "https://api.ebird.org/v2/ref/taxonomy/ebird"
INAT_TAXA_URL = "https://api.inaturalist.org/v1/taxa"
ML_SEARCH_URL = "https://search.macaulaylibrary.org/api/v1/search"
ML_ASSET_URL = "https://cdn.download.ams.birds.cornell.edu/api/v1/asset"
# Xeno-canto API v3 (requires API key)
XC_API_URL = "https://xeno-canto.org/api/3/recordings"


# =============================================================================
# Serialization mixin
# =============================================================================


class ToDictMixin:
    """Mixin adding ``to_dict()`` to dataclasses, handling nested structures."""

    def to_dict(self) -> dict[str, Any]:
        """Convert dataclass to dictionary, recursing into nested structures."""
        if not is_dataclass(self):
            raise TypeError(f"{self.__class__.__name__} is not a dataclass")

        result: dict[str, Any] = {}
        for f in fields(self):
            result[f.name] = self._serialize_value(getattr(self, f.name))
        return result

    def _serialize_value(self, value: Any) -> Any:
        if value is None:
            return None
        if isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, (list, tuple)):
            return [self._serialize_value(v) for v in value]
        if isinstance(value, dict):
            return {k: self._serialize_value(v) for k, v in value.items()}
        if hasattr(value, "to_dict"):
            return value.to_dict()
        if is_dataclass(value):
            return asdict(value)
        return str(value)


# =============================================================================
# eBird models
# =============================================================================


@dataclass
class EBirdObservation(ToDictMixin):
    """Information about an eBird observation."""

    species_code: str
    common_name: str
    scientific_name: str
    location_id: str
    location_name: str
    observation_date: str
    how_many: int | None = None
    latitude: float | None = None
    longitude: float | None = None
    observation_valid: bool = True
    observation_reviewed: bool = False
    location_private: bool = False
    subid: str | None = None
    obs_id: str | None = None

    @classmethod
    def from_api_response(cls, data: dict[str, Any]) -> "EBirdObservation":
        """Create from eBird API response."""
        return cls(
            species_code=data.get("speciesCode", ""),
            common_name=data.get("comName", ""),
            scientific_name=data.get("sciName", ""),
            location_id=data.get("locId", ""),
            location_name=data.get("locName", ""),
            observation_date=data.get("obsDt", ""),
            how_many=data.get("howMany"),
            latitude=data.get("lat"),
            longitude=data.get("lng"),
            observation_valid=data.get("obsValid", True),
            observation_reviewed=data.get("obsReviewed", False),
            location_private=data.get("locationPrivate", False),
            subid=data.get("subId"),
            obs_id=data.get("obsId"),
        )


@dataclass
class EBirdChecklist(ToDictMixin):
    """Information about an eBird checklist."""

    submission_id: str
    location_id: str
    location_name: str
    observation_date: str
    observation_time: str | None = None
    latitude: float | None = None
    longitude: float | None = None
    duration_minutes: int | None = None
    distance_km: float | None = None
    num_observers: int | None = None
    species_count: int = 0
    observations: list[EBirdObservation] = field(default_factory=list)


@dataclass
class EBirdHotspot(ToDictMixin):
    """Information about an eBird hotspot."""

    loc_id: str
    loc_name: str
    country_code: str
    subnational1_code: str
    latitude: float
    longitude: float
    latest_obs_dt: str | None = None
    num_species_all_time: int | None = None

    @classmethod
    def from_api_response(cls, data: dict[str, Any]) -> "EBirdHotspot":
        """Create from eBird API response."""
        return cls(
            loc_id=data.get("locId", ""),
            loc_name=data.get("locName", ""),
            country_code=data.get("countryCode", ""),
            subnational1_code=data.get("subnational1Code", ""),
            latitude=data.get("lat", 0.0),
            longitude=data.get("lng", 0.0),
            latest_obs_dt=data.get("latestObsDt"),
            num_species_all_time=data.get("numSpeciesAllTime"),
        )


@dataclass
class ValidationResult(ToDictMixin):
    """Result of species validation at a location."""

    species_code: str
    is_valid: bool
    nearby_observations: int
    total_species_in_area: int
    most_recent_observation: str | None = None


@dataclass
class NearbyResult(ToDictMixin):
    """Result of nearby observations query."""

    observations: list[EBirdObservation]
    total_count: int
    query_params: dict[str, Any]


@dataclass
class RegionResult(ToDictMixin):
    """Result of regional observations query."""

    observations: list[EBirdObservation]
    total_count: int
    region_code: str


# =============================================================================
# Species models
# =============================================================================


@dataclass
class SpeciesInfo(ToDictMixin):
    """Information about a species (taxonomy record)."""

    scientific_name: str
    common_name: str = ""
    species_code: str = ""
    taxon_id: int | None = None
    family: str = ""
    order: str = ""
    genus: str = ""
    species: str = ""
    category: str = "species"
    source: str = ""

    @classmethod
    def from_ebird_response(cls, data: dict[str, Any]) -> "SpeciesInfo":
        """Create SpeciesInfo from eBird taxonomy API response data."""
        sci_name = data.get("sciName", "")
        parts = sci_name.split()
        return cls(
            scientific_name=sci_name,
            common_name=data.get("comName", ""),
            species_code=data.get("speciesCode", ""),
            family=data.get("familyComName", ""),
            order=data.get("order", ""),
            genus=parts[0] if parts else "",
            species=parts[1] if len(parts) > 1 else "",
            category=data.get("category", "species"),
            source="ebird",
        )

    @classmethod
    def from_inat_response(cls, data: dict[str, Any]) -> "SpeciesInfo":
        """Create SpeciesInfo from iNaturalist taxa API response data."""
        sci_name = data.get("name", "")
        parts = sci_name.split()
        return cls(
            scientific_name=sci_name,
            common_name=data.get("preferred_common_name", ""),
            taxon_id=data.get("id"),
            family=data.get("iconic_taxon_name", ""),
            genus=parts[0] if parts else "",
            species=parts[1] if len(parts) > 1 else "",
            category=data.get("rank", "species"),
            source="inat",
        )


@dataclass
class SearchMatch(ToDictMixin):
    """A species search match result with similarity score."""

    scientific_name: str
    common_name: str
    species_code: str
    family: str
    score: float


# =============================================================================
# iNaturalist models
# =============================================================================


@dataclass
class INaturalistSearchResult(ToDictMixin):
    """Result of an iNaturalist search."""

    total_results: int
    observations: list[dict[str, Any]]
    query_params: dict[str, Any]


@dataclass
class INaturalistDownloadResult(ToDictMixin):
    """Result of an iNaturalist download operation."""

    total_observations: int
    total_sounds: int
    observations_with_multiple_sounds: int
    skipped_existing: int
    failed_downloads: int
    output_dir: str
    metadata_file: str
    errors: list[str] = field(default_factory=list)


@dataclass
class TaxonInfo(ToDictMixin):
    """Information about a taxon."""

    taxon_id: int
    name: str
    common_name: str
    observation_count: int


@dataclass
class ProjectStats(ToDictMixin):
    """Statistics for an iNaturalist project."""

    id: int
    title: str
    description: str
    slug: str
    observation_count: int
    species_count: int
    observers_count: int
    created_at: str
    project_type: str
    place: str
    url: str


@dataclass
class ObservationInfo(ToDictMixin):
    """Information about a single observation."""

    id: int
    taxon_name: str
    common_name: str
    observed_on: str
    location: str
    place_guess: str
    observer: str
    quality_grade: str
    sounds: list[dict[str, Any]]
    url: str


# =============================================================================
# Macaulay Library models
# =============================================================================


@dataclass
class MLRecording(ToDictMixin):
    """Information about a Macaulay Library recording."""

    asset_id: str
    catalog_id: str
    species_code: str
    common_name: str
    scientific_name: str
    rating: int
    duration: float | None
    location: str
    country: str
    user_display_name: str
    download_url: str
    media_type: str = "audio"
    region: str = ""
    latitude: float | None = None
    longitude: float | None = None
    date: str = ""
    preview_url: str = ""

    @classmethod
    def from_api_response(cls, data: dict[str, Any]) -> "MLRecording":
        """Create a recording from Macaulay Library API response data."""
        asset_id = str(data.get("assetId", data.get("catalogId", "")))
        catalog_id = str(data.get("catalogId", ""))
        download_url = data.get("downloadUrl", "")
        if not download_url and asset_id:
            download_url = f"{ML_ASSET_URL}/{asset_id}"

        return cls(
            asset_id=asset_id,
            catalog_id=catalog_id,
            species_code=data.get("speciesCode", ""),
            common_name=data.get("commonName", ""),
            scientific_name=data.get("sciName", ""),
            media_type=data.get("mediaType", "audio"),
            rating=int(float(data.get("rating", 0) or 0)),
            location=data.get("location", ""),
            region=data.get("region", ""),
            country=data.get("country", ""),
            latitude=data.get("latitude"),
            longitude=data.get("longitude"),
            date=data.get("obsDt", ""),
            user_display_name=data.get("userDisplayName", ""),
            download_url=download_url,
            preview_url=data.get("previewUrl", ""),
            duration=data.get("duration"),
        )

    def get_download_url(self) -> str:
        """Get the download URL for this recording."""
        if self.download_url:
            return self.download_url
        return f"{ML_ASSET_URL}/{self.asset_id}"


@dataclass
class MacaulaySearchResult(ToDictMixin):
    """Result of a Macaulay Library search."""

    total_results: int
    recordings: list[MLRecording]
    query_params: dict[str, Any]


@dataclass
class MacaulayDownloadResult(ToDictMixin):
    """Result of a Macaulay Library download operation."""

    total: int
    downloaded: int
    failed: int
    skipped: int
    output_dir: str
    errors: list[str] = field(default_factory=list)


# =============================================================================
# Xeno-canto models
# =============================================================================


@dataclass
class XCRecording(ToDictMixin):
    """Information about a Xeno-canto recording."""

    id: str
    scientific_name: str
    common_name: str
    quality: str
    sound_type: str
    length: str
    location: str
    country: str
    recordist: str
    url: str
    download_url: str
    license: str
    genus: str = ""
    species: str = ""
    subspecies: str = ""
    latitude: float | None = None
    longitude: float | None = None
    date: str = ""
    time: str = ""
    remarks: str = ""

    @classmethod
    def from_api_response(cls, data: dict[str, Any]) -> "XCRecording":
        """Create a recording from Xeno-canto API response data."""
        lat = data.get("lat")
        lng = data.get("lng")

        genus = data.get("gen", "")
        species_epithet = data.get("sp", "")
        subspecies = data.get("ssp", "")

        parts = [genus, species_epithet]
        if subspecies:
            parts.append(subspecies)
        scientific_name = " ".join(parts)

        return cls(
            id=data.get("id", ""),
            genus=genus,
            species=species_epithet,
            subspecies=subspecies,
            scientific_name=scientific_name,
            common_name=data.get("en", ""),
            recordist=data.get("rec", ""),
            country=data.get("cnt", ""),
            location=data.get("loc", ""),
            latitude=float(lat) if lat else None,
            longitude=float(lng) if lng else None,
            sound_type=data.get("type", ""),
            quality=data.get("q", ""),
            length=data.get("length", ""),
            date=data.get("date", ""),
            time=data.get("time", ""),
            url=data.get("url", ""),
            download_url=data.get("file", ""),
            license=data.get("lic", ""),
            remarks=data.get("rmk", ""),
        )


@dataclass
class XenoCantoSearchResult(ToDictMixin):
    """Result of a Xeno-canto search."""

    total_results: int
    recordings: list[XCRecording]
    query_params: dict[str, Any]


@dataclass
class XenoCantoDownloadResult(ToDictMixin):
    """Result of a Xeno-canto download operation."""

    total: int
    downloaded: int
    failed: int
    skipped: int
    output_dir: str
    errors: list[str] = field(default_factory=list)


# =============================================================================
# HuggingFace models
# =============================================================================


@dataclass
class PushResult(ToDictMixin):
    """Result of a HuggingFace Hub push operation."""

    repo_id: str
    repo_type: str
    url: str
    files_uploaded: int
    total_size_bytes: int


@dataclass
class PullResult(ToDictMixin):
    """Result of pulling a HuggingFace dataset into the local bioamla layout."""

    repo_id: str
    dest: str
    url: str
    files_written: int
    labels: list[str]
    splits: dict[str, int]
    metadata_file: str | None


@dataclass
class CachedRepo(ToDictMixin):
    """A repo present in the local HuggingFace cache."""

    repo_id: str
    repo_type: str  # "model" or "dataset"
    size_bytes: int


@dataclass
class PurgeResult(ToDictMixin):
    """Outcome of purging cached HuggingFace repos."""

    deleted: int
    freed_bytes: int
    failures: list[str] = field(default_factory=list)
