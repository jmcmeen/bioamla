"""eBird observation data and taxonomy access.

eBird (https://ebird.org) provides bird observation data used for species
verification and geographic context. An API key is required — obtain one at
https://ebird.org/api/keygen and pass it to :class:`EBirdService` or set the
``EBIRD_API_KEY`` environment variable.

Failures raise :class:`~bioamla.exceptions.CatalogError`; a missing API key
raises :class:`~bioamla.exceptions.InvalidInputError`.
"""

import logging
import os
from typing import Any

import requests

from bioamla.catalogs._models import (
    EBIRD_API_URL,
    EBirdChecklist,
    EBirdHotspot,
    EBirdObservation,
    NearbyResult,
    RegionResult,
    ValidationResult,
)
from bioamla.exceptions import CatalogError, InvalidInputError

logger = logging.getLogger(__name__)


class EBirdService:
    """Client for eBird observation data and taxonomy.

    Args:
        api_key: eBird API key. If omitted, the ``EBIRD_API_KEY`` environment
            variable is used. Missing keys raise :class:`InvalidInputError`
            on first request.

    Example:
        >>> service = EBirdService(api_key="your_key")
        >>> result = service.get_recent_observations("US-CA")
        >>> print(result.total_count)
    """

    def __init__(self, api_key: str | None = None) -> None:
        self._api_key = api_key
        self._session: requests.Session | None = None

    def _get_api_key(self) -> str:
        if self._api_key:
            return self._api_key
        env_key = os.environ.get("EBIRD_API_KEY")
        if env_key:
            return env_key
        raise InvalidInputError(
            "eBird API key required. Set via EBIRD_API_KEY env var or pass to constructor."
        )

    def _get_session(self) -> requests.Session:
        if self._session is None:
            self._session = requests.Session()
            self._session.headers["X-eBirdApiToken"] = self._get_api_key()
        return self._session

    def _request(
        self,
        endpoint: str,
        params: dict[str, Any] | None = None,
    ) -> Any:
        session = self._get_session()
        url = f"{EBIRD_API_URL}/{endpoint}"
        response = session.get(url, params=params)
        response.raise_for_status()
        return response.json()

    def get_recent_observations(
        self,
        region_code: str,
        back: int = 14,
        max_results: int = 100,
        species_code: str | None = None,
    ) -> RegionResult:
        """Get recent observations for a region.

        Args:
            region_code: eBird region code (e.g., "US-CA", "US-CA-037").
            back: Days back to look (max 30).
            max_results: Maximum observations to return.
            species_code: Optional species code to filter by.

        Raises:
            CatalogError: on API failure.
        """
        try:
            if species_code:
                endpoint = f"data/obs/{region_code}/recent/{species_code}"
            else:
                endpoint = f"data/obs/{region_code}/recent"

            params = {"back": min(back, 30), "maxResults": max_results}
            data = self._request(endpoint, params)
            observations = [EBirdObservation.from_api_response(obs) for obs in data]
            return RegionResult(
                observations=observations,
                total_count=len(observations),
                region_code=region_code,
            )
        except CatalogError:
            raise
        except Exception as e:
            logger.error(f"Failed to get observations for {region_code}: {e}")
            raise CatalogError(f"Query failed: {e}") from e

    def get_nearby(
        self,
        lat: float,
        lng: float,
        distance_km: float = 25,
        days: int = 14,
        limit: int = 100,
    ) -> NearbyResult:
        """Get recent eBird observations near a location.

        Args:
            lat: Latitude.
            lng: Longitude.
            distance_km: Search radius in km (max 50).
            days: Days back to search (max 30).
            limit: Maximum results.

        Raises:
            CatalogError: on API failure.
        """
        try:
            endpoint = "data/obs/geo/recent"
            params = {
                "lat": lat,
                "lng": lng,
                "dist": min(distance_km, 50),
                "back": min(days, 30),
                "maxResults": limit,
            }
            data = self._request(endpoint, params)
            observations = [EBirdObservation.from_api_response(obs) for obs in data]
            return NearbyResult(
                observations=observations,
                total_count=len(observations),
                query_params={
                    "latitude": lat,
                    "longitude": lng,
                    "distance_km": distance_km,
                    "days": days,
                },
            )
        except CatalogError:
            raise
        except Exception as e:
            logger.error(f"Failed to get nearby observations: {e}")
            raise CatalogError(f"Query failed: {e}") from e

    def validate_species(
        self,
        species_code: str,
        lat: float,
        lng: float,
        distance_km: float = 50,
    ) -> ValidationResult:
        """Validate whether a species is expected at a location.

        Checks if the species has been recently observed near the given
        coordinates, which helps validate detection results.

        Raises:
            CatalogError: on API failure.
        """
        try:
            nearby = self.get_nearby(lat=lat, lng=lng, distance_km=distance_km, days=30, limit=100)
            observations = nearby.observations
            species_obs = [o for o in observations if o.species_code == species_code]
            all_species = {o.species_code for o in observations}
            return ValidationResult(
                species_code=species_code,
                is_valid=len(species_obs) > 0,
                nearby_observations=len(species_obs),
                total_species_in_area=len(all_species),
                most_recent_observation=species_obs[0].observation_date if species_obs else None,
            )
        except CatalogError:
            raise
        except Exception as e:
            logger.error(f"Species validation failed: {e}")
            raise CatalogError(f"Validation failed: {e}") from e

    def get_species_list(self, region_code: str) -> list[str]:
        """Get list of species codes observed in a region.

        Raises:
            CatalogError: on API failure.
        """
        try:
            return self._request(f"product/spplist/{region_code}")
        except CatalogError:
            raise
        except Exception as e:
            logger.error(f"Failed to get species list for {region_code}: {e}")
            raise CatalogError(f"Query failed: {e}") from e

    def get_taxonomy(
        self,
        species_codes: list[str] | None = None,
        category: str = "species",
    ) -> list[dict[str, Any]]:
        """Get eBird taxonomy data.

        Args:
            species_codes: Optional list of species codes to filter.
            category: Taxonomic category filter.

        Raises:
            CatalogError: on API failure.
        """
        try:
            params: dict[str, Any] = {"cat": category, "fmt": "json"}
            if species_codes:
                params["species"] = ",".join(species_codes)
            return self._request("ref/taxonomy/ebird", params)
        except CatalogError:
            raise
        except Exception as e:
            logger.error(f"Failed to get taxonomy: {e}")
            raise CatalogError(f"Query failed: {e}") from e

    def get_hotspots(self, region_code: str, back: int = 14) -> list[EBirdHotspot]:
        """Get eBird hotspots in a region.

        Args:
            region_code: eBird region code.
            back: Only include hotspots with observations in last N days.

        Raises:
            CatalogError: on API failure.
        """
        try:
            params = {"back": back, "fmt": "json"}
            data = self._request(f"ref/hotspot/{region_code}", params)
            return [EBirdHotspot.from_api_response(hs) for hs in data]
        except CatalogError:
            raise
        except Exception as e:
            logger.error(f"Failed to get hotspots for {region_code}: {e}")
            raise CatalogError(f"Query failed: {e}") from e

    def get_checklist(self, submission_id: str) -> EBirdChecklist:
        """Get details of a specific checklist.

        Raises:
            CatalogError: on API failure.
        """
        try:
            data = self._request(f"product/checklist/view/{submission_id}")
            observations = [
                EBirdObservation(
                    species_code=obs.get("speciesCode", ""),
                    common_name=obs.get("species", {}).get("comName", ""),
                    scientific_name=obs.get("species", {}).get("sciName", ""),
                    location_id=data.get("locId", ""),
                    location_name=data.get("loc", {}).get("name", ""),
                    observation_date=data.get("obsDt", ""),
                    how_many=obs.get("howManyStr"),
                )
                for obs in data.get("obs", [])
            ]
            duration_hrs = data.get("durationHrs")
            return EBirdChecklist(
                submission_id=data.get("subId", submission_id),
                location_id=data.get("locId", ""),
                location_name=data.get("loc", {}).get("name", ""),
                observation_date=data.get("obsDt", ""),
                observation_time=data.get("obsTime"),
                latitude=data.get("loc", {}).get("lat"),
                longitude=data.get("loc", {}).get("lng"),
                duration_minutes=int(duration_hrs * 60) if duration_hrs else None,
                distance_km=data.get("effortDistanceKm"),
                num_observers=data.get("numObservers"),
                species_count=len(observations),
                observations=observations,
            )
        except CatalogError:
            raise
        except Exception as e:
            logger.error(f"Failed to get checklist {submission_id}: {e}")
            raise CatalogError(f"Query failed: {e}") from e


def match_detections_to_ebird(
    detections: list[dict[str, Any]],
    service: EBirdService,
    latitude: float,
    longitude: float,
    species_mapping: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Match detection labels to eBird taxonomy and validate against nearby observations.

    Args:
        detections: List of detections with a ``label`` field.
        service: An :class:`EBirdService` instance with a valid API key.
        latitude: Location latitude.
        longitude: Location longitude.
        species_mapping: Optional mapping from detection labels to eBird species codes.

    Returns:
        Detections with eBird validation fields added.

    Raises:
        CatalogError: if the nearby-observations query fails.
    """
    nearby = service.get_nearby(lat=latitude, lng=longitude, distance_km=25, days=30)
    nearby_species = {obs.species_code: obs for obs in nearby.observations}

    results = []
    for det in detections:
        label = det.get("label", "")
        if species_mapping and label in species_mapping:
            species_code = species_mapping[label]
        else:
            species_code = label.lower().replace(" ", "")

        det_copy = det.copy()
        det_copy["ebird_validated"] = species_code in nearby_species
        det_copy["ebird_species_code"] = species_code
        det_copy["ebird_nearby_count"] = sum(
            1 for obs in nearby.observations if obs.species_code == species_code
        )
        results.append(det_copy)

    return results
