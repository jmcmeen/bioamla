# controllers/inaturalist.py
"""
iNaturalist Controller
======================

Controller for iNaturalist data operations.

Orchestrates between CLI/API views and core iNaturalist service functions.
Handles search, download, and taxa management with progress reporting.

Example:
    from bioamla.controllers.inaturalist import INaturalistController

    controller = INaturalistController()

    # Search for sounds
    result = controller.search(taxon_name="Strix varia", per_page=10)

    # Download audio
    result = controller.download(
        output_dir="./sounds",
        taxon_ids=[3],  # Birds
        obs_per_taxon=50,
    )

    # Get taxa for a project
    result = controller.get_taxa(project_id="appalachia-bioacoustics")
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

from .base import BaseController, ControllerResult, ToDictMixin


@dataclass
class SearchResult(ToDictMixin):
    """Result of an iNaturalist search."""

    total_results: int
    observations: List[Dict[str, Any]]
    query_params: Dict[str, Any]


@dataclass
class DownloadResult(ToDictMixin):
    """Result of an iNaturalist download operation."""

    total_observations: int
    total_sounds: int
    skipped_existing: int
    failed_downloads: int
    output_dir: str
    metadata_file: str
    errors: List[str] = field(default_factory=list)


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
    sounds: List[Dict[str, Any]]
    url: str


class INaturalistController(BaseController):
    """
    Controller for iNaturalist operations.

    Provides high-level methods for:
    - Searching for observations with sounds
    - Downloading audio files with metadata
    - Discovering taxa in projects/places
    - Getting project statistics
    """

    def __init__(self):
        """Initialize iNaturalist controller."""
        super().__init__()
        self._download_callback: Optional[Callable[[int, int, str], None]] = None

    def set_download_callback(
        self,
        callback: Callable[[int, int, str], None],
    ) -> None:
        """
        Set callback for download progress updates.

        Callback signature: (current, total, current_file) -> None
        """
        self._download_callback = callback

    # =========================================================================
    # Search
    # =========================================================================

    def search(
        self,
        taxon_id: Optional[int] = None,
        taxon_name: Optional[str] = None,
        place_id: Optional[int] = None,
        quality_grade: Optional[str] = "research",
        per_page: int = 30,
    ) -> ControllerResult[SearchResult]:
        """
        Search for iNaturalist observations with sounds.

        Args:
            taxon_id: Filter by taxon ID
            taxon_name: Filter by taxon name (e.g., "Strix varia")
            place_id: Filter by place ID
            quality_grade: Filter by quality grade ("research", "needs_id", "casual")
            per_page: Number of results to return

        Returns:
            Result with search results
        """
        try:
            from bioamla.core.services.inaturalist import search_inat_sounds

            observations = search_inat_sounds(
                taxon_id=taxon_id,
                taxon_name=taxon_name,
                place_id=place_id,
                quality_grade=quality_grade,
                per_page=per_page,
            )

            result = SearchResult(
                total_results=len(observations),
                observations=observations,
                query_params={
                    "taxon_id": taxon_id,
                    "taxon_name": taxon_name,
                    "place_id": place_id,
                    "quality_grade": quality_grade,
                },
            )

            return ControllerResult.ok(
                data=result,
                message=f"Found {len(observations)} observations with sounds",
            )
        except Exception as e:
            return ControllerResult.fail(f"Search failed: {e}")

    # =========================================================================
    # Download
    # =========================================================================

    def download(
        self,
        output_dir: str,
        taxon_ids: Optional[List[int]] = None,
        taxon_csv: Optional[str] = None,
        taxon_name: Optional[str] = None,
        place_id: Optional[int] = None,
        user_id: Optional[str] = None,
        project_id: Optional[str] = None,
        quality_grade: Optional[str] = "research",
        sound_license: Optional[List[str]] = None,
        d1: Optional[str] = None,
        d2: Optional[str] = None,
        obs_per_taxon: int = 100,
        organize_by_taxon: bool = True,
        include_metadata: bool = True,
        file_extensions: Optional[List[str]] = None,
    ) -> ControllerResult[DownloadResult]:
        """
        Download audio files from iNaturalist.

        Args:
            output_dir: Directory to save audio files
            taxon_ids: List of taxon IDs to download
            taxon_csv: Path to CSV file with taxon IDs
            taxon_name: Filter by taxon name
            place_id: Filter by place ID
            user_id: Filter by observer username
            project_id: Filter by project ID or slug
            quality_grade: Filter by quality grade
            sound_license: Filter by license(s) (e.g., ["cc-by", "cc0"])
            d1: Start date (YYYY-MM-DD)
            d2: End date (YYYY-MM-DD)
            obs_per_taxon: Max observations per taxon
            organize_by_taxon: Organize files into subdirectories by species
            include_metadata: Include extended iNaturalist metadata
            file_extensions: Filter by file extensions (e.g., ["wav", "mp3"])

        Returns:
            Result with download statistics
        """
        try:
            from bioamla.core.services.inaturalist import download_inat_audio

            stats = download_inat_audio(
                output_dir=output_dir,
                taxon_ids=taxon_ids,
                taxon_csv=taxon_csv,
                taxon_name=taxon_name,
                place_id=place_id,
                user_id=user_id,
                project_id=project_id,
                quality_grade=quality_grade,
                sound_license=sound_license,
                d1=d1,
                d2=d2,
                obs_per_taxon=obs_per_taxon,
                organize_by_taxon=organize_by_taxon,
                include_inat_metadata=include_metadata,
                file_extensions=file_extensions,
                verbose=False,  # We handle progress ourselves
            )

            result = DownloadResult(
                total_observations=stats["total_observations"],
                total_sounds=stats["total_sounds"],
                skipped_existing=stats["skipped_existing"],
                failed_downloads=stats["failed_downloads"],
                output_dir=stats["output_dir"],
                metadata_file=stats["metadata_file"],
            )

            message = f"Downloaded {result.total_sounds} audio files"
            if result.skipped_existing:
                message += f" (skipped {result.skipped_existing} existing)"

            return ControllerResult.ok(
                data=result,
                message=message,
                stats=stats,
            )
        except Exception as e:
            return ControllerResult.fail(f"Download failed: {e}")

    def download_from_observations(
        self,
        observation_ids: List[int],
        output_dir: str,
        organize_by_taxon: bool = True,
    ) -> ControllerResult[DownloadResult]:
        """
        Download audio from specific observation IDs.

        Args:
            observation_ids: List of observation IDs
            output_dir: Directory to save audio files
            organize_by_taxon: Organize by species

        Returns:
            Result with download statistics
        """
        try:
            from bioamla.core.services.inaturalist import (
                get_observation_sounds,
                _download_file,
            )
            from bioamla.core.files import sanitize_filename
            from pyinaturalist import get_observations

            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            total_sounds = 0
            failed = 0
            errors = []

            for obs_id in observation_ids:
                try:
                    # Get observation details
                    response = get_observations(id=obs_id)
                    results = response.get("results", [])

                    if not results:
                        errors.append(f"Observation {obs_id} not found")
                        continue

                    obs = results[0]
                    sounds = obs.get("sounds", [])
                    taxon = obs.get("taxon", {})
                    species_name = taxon.get("name", "unknown")
                    safe_species = sanitize_filename(species_name)

                    if organize_by_taxon:
                        species_dir = output_path / safe_species
                        species_dir.mkdir(exist_ok=True)
                    else:
                        species_dir = output_path

                    for sound in sounds:
                        sound_id = sound.get("id")
                        file_url = sound.get("file_url")

                        if not file_url:
                            continue

                        # Get extension from URL
                        ext = Path(file_url).suffix or ".mp3"
                        filename = f"inat_{obs_id}_sound_{sound_id}{ext}"
                        filepath = species_dir / filename

                        if _download_file(file_url, filepath, verbose=False):
                            total_sounds += 1
                        else:
                            failed += 1

                except Exception as e:
                    errors.append(f"Observation {obs_id}: {e}")
                    failed += 1

            result = DownloadResult(
                total_observations=len(observation_ids),
                total_sounds=total_sounds,
                skipped_existing=0,
                failed_downloads=failed,
                output_dir=str(output_path),
                metadata_file="",
                errors=errors,
            )

            return ControllerResult.ok(
                data=result,
                message=f"Downloaded {total_sounds} sounds from {len(observation_ids)} observations",
            )
        except Exception as e:
            return ControllerResult.fail(f"Download failed: {e}")

    # =========================================================================
    # Taxa Discovery
    # =========================================================================

    def get_taxa(
        self,
        place_id: Optional[int] = None,
        project_id: Optional[str] = None,
        quality_grade: Optional[str] = "research",
        parent_taxon_id: Optional[int] = None,
    ) -> ControllerResult[List[TaxonInfo]]:
        """
        Get taxa with observations in a place or project.

        Args:
            place_id: Filter by place ID
            project_id: Filter by project ID or slug
            quality_grade: Filter by quality grade
            parent_taxon_id: Filter by parent taxon (e.g., 20979 for Amphibia)

        Returns:
            Result with list of taxa
        """
        if not place_id and not project_id:
            return ControllerResult.fail("At least one of place_id or project_id must be provided")

        try:
            from bioamla.core.services.inaturalist import get_taxa

            taxa_list = get_taxa(
                place_id=place_id,
                project_id=project_id,
                quality_grade=quality_grade,
                taxon_id=parent_taxon_id,
                verbose=False,
            )

            results = [
                TaxonInfo(
                    taxon_id=t["taxon_id"],
                    name=t["name"],
                    common_name=t["common_name"],
                    observation_count=t["observation_count"],
                )
                for t in taxa_list
            ]

            return ControllerResult.ok(
                data=results,
                message=f"Found {len(results)} taxa",
                raw_data=taxa_list,
            )
        except Exception as e:
            return ControllerResult.fail(f"Failed to get taxa: {e}")

    def export_taxa_csv(
        self,
        taxa: List[TaxonInfo],
        output_path: str,
    ) -> ControllerResult[str]:
        """
        Export taxa list to CSV file.

        Args:
            taxa: List of TaxonInfo objects
            output_path: Path to output CSV file

        Returns:
            Result with output path
        """
        try:
            import csv

            output = Path(output_path)
            output.parent.mkdir(parents=True, exist_ok=True)

            with open(output, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=["taxon_id", "name", "common_name", "observation_count"],
                )
                writer.writeheader()
                for taxon in taxa:
                    writer.writerow(
                        {
                            "taxon_id": taxon.taxon_id,
                            "name": taxon.name,
                            "common_name": taxon.common_name,
                            "observation_count": taxon.observation_count,
                        }
                    )

            return ControllerResult.ok(
                data=str(output),
                message=f"Exported {len(taxa)} taxa to {output}",
            )
        except Exception as e:
            return ControllerResult.fail(f"Failed to export taxa: {e}")

    # =========================================================================
    # Project & Observation Info
    # =========================================================================

    def get_project_stats(
        self,
        project_id: str,
    ) -> ControllerResult[ProjectStats]:
        """
        Get statistics for an iNaturalist project.

        Args:
            project_id: Project ID or slug (e.g., "appalachia-bioacoustics")

        Returns:
            Result with project statistics
        """
        try:
            from bioamla.core.services.inaturalist import get_project_stats

            stats = get_project_stats(project_id, verbose=False)

            result = ProjectStats(
                id=stats["id"],
                title=stats["title"],
                description=stats["description"],
                slug=stats["slug"],
                observation_count=stats["observation_count"],
                species_count=stats["species_count"],
                observers_count=stats["observers_count"],
                created_at=stats["created_at"],
                project_type=stats["project_type"],
                place=stats["place"],
                url=stats["url"],
            )

            return ControllerResult.ok(
                data=result,
                message=f"Project: {result.title}",
                raw_stats=stats,
            )
        except Exception as e:
            return ControllerResult.fail(f"Failed to get project stats: {e}")

    def get_observation(
        self,
        observation_id: int,
    ) -> ControllerResult[ObservationInfo]:
        """
        Get information about a specific observation.

        Args:
            observation_id: The iNaturalist observation ID

        Returns:
            Result with observation information
        """
        try:
            from pyinaturalist import get_observations

            response = get_observations(id=observation_id)
            results = response.get("results", [])

            if not results:
                return ControllerResult.fail(f"Observation {observation_id} not found")

            obs = results[0]
            taxon = obs.get("taxon", {})

            result = ObservationInfo(
                id=obs.get("id"),
                taxon_name=taxon.get("name", "unknown"),
                common_name=taxon.get("preferred_common_name", ""),
                observed_on=obs.get("observed_on", ""),
                location=obs.get("location", ""),
                place_guess=obs.get("place_guess", ""),
                observer=obs.get("user", {}).get("login", ""),
                quality_grade=obs.get("quality_grade", ""),
                sounds=obs.get("sounds", []),
                url=f"https://www.inaturalist.org/observations/{observation_id}",
            )

            return ControllerResult.ok(
                data=result,
                message=f"Observation: {result.taxon_name}",
                raw_observation=obs,
            )
        except Exception as e:
            return ControllerResult.fail(f"Failed to get observation: {e}")

    def get_observation_sounds(
        self,
        observation_id: int,
    ) -> ControllerResult[List[Dict[str, Any]]]:
        """
        Get all sounds from a specific observation.

        Args:
            observation_id: The iNaturalist observation ID

        Returns:
            Result with list of sound dictionaries
        """
        try:
            from bioamla.core.services.inaturalist import get_observation_sounds

            sounds = get_observation_sounds(observation_id)

            return ControllerResult.ok(
                data=sounds,
                message=f"Found {len(sounds)} sounds",
            )
        except Exception as e:
            return ControllerResult.fail(f"Failed to get sounds: {e}")

    # =========================================================================
    # Utilities
    # =========================================================================

    def load_taxon_ids(
        self,
        csv_path: str,
    ) -> ControllerResult[List[int]]:
        """
        Load taxon IDs from a CSV file.

        Args:
            csv_path: Path to CSV file with 'taxon_id' column

        Returns:
            Result with list of taxon IDs
        """
        try:
            from bioamla.core.services.inaturalist import load_taxon_ids_from_csv

            taxon_ids = load_taxon_ids_from_csv(csv_path)

            return ControllerResult.ok(
                data=taxon_ids,
                message=f"Loaded {len(taxon_ids)} taxon IDs from {csv_path}",
            )
        except Exception as e:
            return ControllerResult.fail(f"Failed to load taxon IDs: {e}")

    def get_common_taxa(self) -> ControllerResult[Dict[str, int]]:
        """
        Get a dictionary of common taxon IDs.

        Returns:
            Result with dictionary mapping names to taxon IDs
        """
        common_taxa = {
            # Major groups
            "Aves": 3,  # Birds
            "Amphibia": 20978,  # Amphibians
            "Mammalia": 40151,  # Mammals
            "Insecta": 47158,  # Insects
            "Reptilia": 26036,  # Reptiles
            # Bird orders
            "Passeriformes": 7251,  # Songbirds
            "Strigiformes": 19350,  # Owls
            "Piciformes": 19893,  # Woodpeckers
            "Caprimulgiformes": 67569,  # Nightjars
            # Amphibian orders
            "Anura": 20979,  # Frogs and toads
            "Caudata": 20980,  # Salamanders
            # Common families
            "Corvidae": 7823,  # Crows, jays
            "Strigidae": 19374,  # Typical owls
            "Hylidae": 21013,  # Tree frogs
            "Ranidae": 21136,  # True frogs
            "Bufonidae": 20981,  # True toads
            # Common species
            "Strix varia": 19893,  # Barred Owl
            "Bubo virginianus": 20949,  # Great Horned Owl
            "Lithobates catesbeianus": 24230,  # American Bullfrog
            "Pseudacris crucifer": 24234,  # Spring Peeper
            "Anaxyrus americanus": 24225,  # American Toad
        }

        return ControllerResult.ok(
            data=common_taxa,
            message=f"Found {len(common_taxa)} common taxa",
        )
