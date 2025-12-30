# services/ebird.py
"""
Service for eBird observation data operations.

Provides access to eBird observation data for species verification
and geographic context.

Example:
    >>> service = EBirdService(api_key="your_key")
    >>> result = service.get_nearby(lat=40.7128, lng=-74.0060)
    >>> if result.success:
    ...     for obs in result.data.observations:
    ...         print(f"{obs.common_name}: {obs.observation_date}")
"""

import logging
import os
from typing import Any, Dict, List, Optional

import requests

from bioamla.models.ebird import (
    EBIRD_API_URL,
    EBirdChecklist,
    EBirdHotspot,
    EBirdObservation,
    NearbyResult,
    RegionResult,
    ValidationResult,
)

from .base import BaseService, ServiceResult

logger = logging.getLogger(__name__)


class EBirdService(BaseService):
    """
    Service for eBird operations.

    Provides high-level methods for:
    - Getting recent observations by region
    - Getting nearby observations by coordinates
    - Validating species at locations
    - Accessing eBird hotspots and checklists

    Example:
        >>> service = EBirdService(api_key="your_key")
        >>> result = service.get_recent_observations("US-CA")
        >>> if result.success:
        ...     print(f"Found {result.data.total_count} observations")
    """

    def __init__(self, api_key: Optional[str] = None) -> None:
        """
        Initialize eBird service.

        Args:
            api_key: eBird API key. If not provided, will check EBIRD_API_KEY env var.
                     Get a key from https://ebird.org/api/keygen
        """
        super().__init__()
        self._api_key = api_key
        self._session: Optional[requests.Session] = None

    def _get_api_key(self) -> str:
        """Get API key from init or environment."""
        if self._api_key:
            return self._api_key

        env_key = os.environ.get("EBIRD_API_KEY")
        if env_key:
            return env_key

        raise ValueError(
            "eBird API key required. Set via EBIRD_API_KEY env var or pass to constructor."
        )

    def _get_session(self) -> requests.Session:
        """Get or create requests session."""
        if self._session is None:
            self._session = requests.Session()
            self._session.headers["X-eBirdApiToken"] = self._get_api_key()
        return self._session

    def _request(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """Make API request."""
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
        species_code: Optional[str] = None,
    ) -> ServiceResult[RegionResult]:
        """
        Get recent observations for a region.

        Args:
            region_code: eBird region code (e.g., "US-CA", "US-CA-037").
            back: Days back to look (max 30).
            max_results: Maximum observations to return.
            species_code: Optional species code to filter by.

        Returns:
            ServiceResult with RegionResult on success.

        Example:
            >>> result = service.get_recent_observations("US-CA", back=7)
            >>> if result.success:
            ...     for obs in result.data.observations[:5]:
            ...         print(f"{obs.common_name} at {obs.location_name}")
        """
        try:
            if species_code:
                endpoint = f"data/obs/{region_code}/recent/{species_code}"
            else:
                endpoint = f"data/obs/{region_code}/recent"

            params = {
                "back": min(back, 30),
                "maxResults": max_results,
            }

            data = self._request(endpoint, params)
            observations = [EBirdObservation.from_api_response(obs) for obs in data]

            result = RegionResult(
                observations=observations,
                total_count=len(observations),
                region_code=region_code,
            )

            return ServiceResult.ok(
                data=result,
                message=f"Found {len(observations)} recent observations in {region_code}",
            )

        except ValueError as e:
            return ServiceResult.fail(str(e))
        except Exception as e:
            logger.error(f"Failed to get observations for {region_code}: {e}")
            return ServiceResult.fail(f"Query failed: {e}")

    def get_nearby(
        self,
        lat: float,
        lng: float,
        distance_km: float = 25,
        days: int = 14,
        limit: int = 100,
    ) -> ServiceResult[NearbyResult]:
        """
        Get recent eBird observations near a location.

        Args:
            lat: Latitude.
            lng: Longitude.
            distance_km: Search radius in km (max 50).
            days: Days back to search (max 30).
            limit: Maximum results.

        Returns:
            ServiceResult with NearbyResult on success.

        Example:
            >>> result = service.get_nearby(lat=40.7128, lng=-74.0060)
            >>> if result.success:
            ...     print(f"Found {result.data.total_count} observations")
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

            result = NearbyResult(
                observations=observations,
                total_count=len(observations),
                query_params={
                    "latitude": lat,
                    "longitude": lng,
                    "distance_km": distance_km,
                    "days": days,
                },
            )

            return ServiceResult.ok(
                data=result,
                message=f"Found {len(observations)} recent observations",
            )

        except ValueError as e:
            return ServiceResult.fail(str(e))
        except Exception as e:
            logger.error(f"Failed to get nearby observations: {e}")
            return ServiceResult.fail(f"Query failed: {e}")

    def validate_species(
        self,
        species_code: str,
        lat: float,
        lng: float,
        distance_km: float = 50,
    ) -> ServiceResult[ValidationResult]:
        """
        Validate if a species is expected at a location.

        Checks if the species has been recently observed near the given
        coordinates, which helps validate detection results.

        Args:
            species_code: eBird species code.
            lat: Latitude.
            lng: Longitude.
            distance_km: Search radius in km.

        Returns:
            ServiceResult with ValidationResult on success.

        Example:
            >>> result = service.validate_species("amerob", lat=40.7, lng=-74.0)
            >>> if result.success:
            ...     print(f"Valid: {result.data.is_valid}")
        """
        try:
            # Get nearby observations
            nearby_result = self.get_nearby(
                lat=lat,
                lng=lng,
                distance_km=distance_km,
                days=30,
                limit=100,
            )

            if not nearby_result.success:
                return ServiceResult.fail(f"Failed to get nearby observations: {nearby_result.error}")

            observations = nearby_result.data.observations
            species_obs = [o for o in observations if o.species_code == species_code]
            all_species = {o.species_code for o in observations}

            result = ValidationResult(
                species_code=species_code,
                is_valid=len(species_obs) > 0,
                nearby_observations=len(species_obs),
                total_species_in_area=len(all_species),
                most_recent_observation=species_obs[0].observation_date if species_obs else None,
            )

            if result.is_valid:
                message = f"{species_code} is expected at this location ({len(species_obs)} recent obs)"
            else:
                message = f"{species_code} not recently observed at this location"

            return ServiceResult.ok(data=result, message=message)

        except ValueError as e:
            return ServiceResult.fail(str(e))
        except Exception as e:
            logger.error(f"Species validation failed: {e}")
            return ServiceResult.fail(f"Validation failed: {e}")

    def get_species_list(self, region_code: str) -> ServiceResult[List[str]]:
        """
        Get list of species observed in a region.

        Args:
            region_code: eBird region code.

        Returns:
            ServiceResult with list of species codes on success.

        Example:
            >>> result = service.get_species_list("US-CA")
            >>> if result.success:
            ...     print(f"Species in region: {len(result.data)}")
        """
        try:
            endpoint = f"product/spplist/{region_code}"
            data = self._request(endpoint)

            return ServiceResult.ok(
                data=data,
                message=f"Found {len(data)} species in {region_code}",
            )

        except ValueError as e:
            return ServiceResult.fail(str(e))
        except Exception as e:
            logger.error(f"Failed to get species list for {region_code}: {e}")
            return ServiceResult.fail(f"Query failed: {e}")

    def get_taxonomy(
        self,
        species_codes: Optional[List[str]] = None,
        category: str = "species",
    ) -> ServiceResult[List[Dict[str, Any]]]:
        """
        Get eBird taxonomy data.

        Args:
            species_codes: Optional list of species codes to filter.
            category: Taxonomic category filter.

        Returns:
            ServiceResult with taxonomy data on success.

        Example:
            >>> result = service.get_taxonomy(species_codes=["amerob", "barswa"])
            >>> if result.success:
            ...     for taxon in result.data:
            ...         print(f"{taxon['comName']}: {taxon['sciName']}")
        """
        try:
            endpoint = "ref/taxonomy/ebird"
            params: Dict[str, Any] = {"cat": category, "fmt": "json"}

            if species_codes:
                params["species"] = ",".join(species_codes)

            data = self._request(endpoint, params)

            return ServiceResult.ok(
                data=data,
                message=f"Retrieved {len(data)} taxonomy entries",
            )

        except ValueError as e:
            return ServiceResult.fail(str(e))
        except Exception as e:
            logger.error(f"Failed to get taxonomy: {e}")
            return ServiceResult.fail(f"Query failed: {e}")

    def get_hotspots(
        self,
        region_code: str,
        back: int = 14,
    ) -> ServiceResult[List[EBirdHotspot]]:
        """
        Get eBird hotspots in a region.

        Args:
            region_code: eBird region code.
            back: Only include hotspots with observations in last N days.

        Returns:
            ServiceResult with list of hotspots on success.

        Example:
            >>> result = service.get_hotspots("US-CA-037")
            >>> if result.success:
            ...     for hs in result.data[:5]:
            ...         print(f"{hs.loc_name}")
        """
        try:
            endpoint = f"ref/hotspot/{region_code}"
            params = {"back": back, "fmt": "json"}

            data = self._request(endpoint, params)
            hotspots = [EBirdHotspot.from_api_response(hs) for hs in data]

            return ServiceResult.ok(
                data=hotspots,
                message=f"Found {len(hotspots)} hotspots in {region_code}",
            )

        except ValueError as e:
            return ServiceResult.fail(str(e))
        except Exception as e:
            logger.error(f"Failed to get hotspots for {region_code}: {e}")
            return ServiceResult.fail(f"Query failed: {e}")

    def get_checklist(self, submission_id: str) -> ServiceResult[EBirdChecklist]:
        """
        Get details of a specific checklist.

        Args:
            submission_id: eBird submission ID.

        Returns:
            ServiceResult with EBirdChecklist on success.

        Example:
            >>> result = service.get_checklist("S123456789")
            >>> if result.success:
            ...     print(f"Species count: {result.data.species_count}")
        """
        try:
            endpoint = f"product/checklist/view/{submission_id}"
            data = self._request(endpoint)

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
            checklist = EBirdChecklist(
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

            return ServiceResult.ok(
                data=checklist,
                message=f"Checklist {submission_id}: {checklist.species_count} species",
            )

        except ValueError as e:
            return ServiceResult.fail(str(e))
        except Exception as e:
            logger.error(f"Failed to get checklist {submission_id}: {e}")
            return ServiceResult.fail(f"Query failed: {e}")


def match_detections_to_ebird(
    detections: List[Dict[str, Any]],
    service: EBirdService,
    latitude: float,
    longitude: float,
    species_mapping: Optional[Dict[str, str]] = None,
) -> List[Dict[str, Any]]:
    """
    Match detection labels to eBird taxonomy and validate.

    Args:
        detections: List of detections with 'label' field.
        service: EBirdService instance with valid API key.
        latitude: Location latitude.
        longitude: Location longitude.
        species_mapping: Optional mapping from detection labels to eBird species codes.

    Returns:
        Detections with eBird validation results added.

    Example:
        >>> detections = [{"label": "American Robin", "confidence": 0.95}]
        >>> service = EBirdService(api_key="your_key")
        >>> validated = match_detections_to_ebird(detections, service, 40.7, -74.0)
        >>> print(validated[0]["ebird_validated"])
    """
    # Get nearby observations for context
    nearby_result = service.get_nearby(
        lat=latitude,
        lng=longitude,
        distance_km=25,
        days=30,
    )

    if not nearby_result.success:
        logger.warning(f"Failed to get nearby observations: {nearby_result.error}")
        return detections

    nearby_species = {obs.species_code: obs for obs in nearby_result.data.observations}

    results = []
    for det in detections:
        label = det.get("label", "")

        # Map label to eBird code
        if species_mapping and label in species_mapping:
            species_code = species_mapping[label]
        else:
            # Try to match by name (simplified)
            species_code = label.lower().replace(" ", "")

        det_copy = det.copy()
        det_copy["ebird_validated"] = species_code in nearby_species
        det_copy["ebird_species_code"] = species_code
        det_copy["ebird_nearby_count"] = sum(
            1 for obs in nearby_result.data.observations if obs.species_code == species_code
        )

        results.append(det_copy)

    return results
