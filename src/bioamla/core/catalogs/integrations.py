"""
External Integrations Module
============================

This module provides integrations with external services:
- eBird checklist integration

Example:
    >>> from bioamla.integrations import EBirdClient
    >>> ebird = EBirdClient(api_key="your_key")
    >>> checklists = ebird.get_checklists_for_region("US-CA")
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

__all__ = [
    # eBird integration
    "EBirdObservation",
    "EBirdChecklist",
    "EBirdClient",
    "match_detections_to_ebird",
]


# =============================================================================
# eBird Integration
# =============================================================================


@dataclass
class EBirdObservation:
    """Represents an eBird observation."""

    species_code: str
    common_name: str
    scientific_name: str
    location_id: str
    location_name: str
    observation_date: str
    how_many: Optional[int] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    observation_valid: bool = True
    observation_reviewed: bool = False
    location_private: bool = False
    subid: Optional[str] = None
    obs_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "species_code": self.species_code,
            "common_name": self.common_name,
            "scientific_name": self.scientific_name,
            "location_id": self.location_id,
            "location_name": self.location_name,
            "observation_date": self.observation_date,
            "how_many": self.how_many,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "observation_valid": self.observation_valid,
            "subid": self.subid,
            "obs_id": self.obs_id,
        }

    @classmethod
    def from_api_response(cls, data: Dict[str, Any]) -> "EBirdObservation":
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
class EBirdChecklist:
    """Represents an eBird checklist."""

    submission_id: str
    location_id: str
    location_name: str
    observation_date: str
    observation_time: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    duration_minutes: Optional[int] = None
    distance_km: Optional[float] = None
    num_observers: Optional[int] = None
    species_count: int = 0
    observations: List[EBirdObservation] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "submission_id": self.submission_id,
            "location_id": self.location_id,
            "location_name": self.location_name,
            "observation_date": self.observation_date,
            "observation_time": self.observation_time,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "duration_minutes": self.duration_minutes,
            "distance_km": self.distance_km,
            "num_observers": self.num_observers,
            "species_count": self.species_count,
            "observations": [o.to_dict() for o in self.observations],
        }


class EBirdClient:
    """
    Client for eBird API.

    Provides access to eBird observation data for species verification
    and geographic context.
    """

    BASE_URL = "https://api.ebird.org/v2"

    def __init__(self, api_key: str):
        """
        Initialize eBird client.

        Args:
            api_key: eBird API key (get from https://ebird.org/api/keygen)
        """
        self.api_key = api_key
        self._session = None

    def _get_session(self):
        """Get or create requests session."""
        if self._session is None:
            try:
                import requests
            except ImportError as err:
                raise ImportError(
                    "requests is required. Install with: pip install requests"
                ) from err
            self._session = requests.Session()
            self._session.headers["X-eBirdApiToken"] = self.api_key
        return self._session

    def _request(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Any:
        """Make API request."""
        session = self._get_session()
        url = f"{self.BASE_URL}/{endpoint}"

        response = session.get(url, params=params)
        response.raise_for_status()

        return response.json()

    def get_recent_observations(
        self,
        region_code: str,
        back: int = 14,
        max_results: int = 100,
        species_code: Optional[str] = None,
    ) -> List[EBirdObservation]:
        """
        Get recent observations for a region.

        Args:
            region_code: eBird region code (e.g., "US-CA", "US-CA-037")
            back: Days back to look (max 30)
            max_results: Maximum observations to return
            species_code: Optional species code to filter by

        Returns:
            List of observations
        """
        if species_code:
            endpoint = f"data/obs/{region_code}/recent/{species_code}"
        else:
            endpoint = f"data/obs/{region_code}/recent"

        params = {
            "back": min(back, 30),
            "maxResults": max_results,
        }

        data = self._request(endpoint, params)

        return [EBirdObservation.from_api_response(obs) for obs in data]

    def get_nearby_observations(
        self,
        latitude: float,
        longitude: float,
        distance_km: float = 25,
        back: int = 14,
        max_results: int = 100,
    ) -> List[EBirdObservation]:
        """
        Get observations near a location.

        Args:
            latitude: Latitude
            longitude: Longitude
            distance_km: Search radius in km (max 50)
            back: Days back to look
            max_results: Maximum observations

        Returns:
            List of observations
        """
        endpoint = "data/obs/geo/recent"
        params = {
            "lat": latitude,
            "lng": longitude,
            "dist": min(distance_km, 50),
            "back": min(back, 30),
            "maxResults": max_results,
        }

        data = self._request(endpoint, params)

        return [EBirdObservation.from_api_response(obs) for obs in data]

    def get_species_list(self, region_code: str) -> List[Dict[str, str]]:
        """
        Get list of species observed in a region.

        Args:
            region_code: eBird region code

        Returns:
            List of species codes
        """
        endpoint = f"product/spplist/{region_code}"
        return self._request(endpoint)

    def get_taxonomy(
        self, species_codes: Optional[List[str]] = None, category: str = "species"
    ) -> List[Dict[str, Any]]:
        """
        Get eBird taxonomy data.

        Args:
            species_codes: Optional list of species codes to filter
            category: Taxonomic category filter

        Returns:
            Taxonomy data
        """
        endpoint = "ref/taxonomy/ebird"
        params = {"cat": category, "fmt": "json"}

        if species_codes:
            params["species"] = ",".join(species_codes)

        return self._request(endpoint, params)

    def get_hotspots(self, region_code: str, back: int = 14) -> List[Dict[str, Any]]:
        """
        Get eBird hotspots in a region.

        Args:
            region_code: eBird region code
            back: Only include hotspots with recent observations

        Returns:
            List of hotspot data
        """
        endpoint = f"ref/hotspot/{region_code}"
        params = {"back": back, "fmt": "json"}

        return self._request(endpoint, params)

    def get_checklist(self, submission_id: str) -> EBirdChecklist:
        """
        Get details of a specific checklist.

        Args:
            submission_id: eBird submission ID

        Returns:
            Checklist details
        """
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

        return EBirdChecklist(
            submission_id=data.get("subId", submission_id),
            location_id=data.get("locId", ""),
            location_name=data.get("loc", {}).get("name", ""),
            observation_date=data.get("obsDt", ""),
            observation_time=data.get("obsTime"),
            latitude=data.get("loc", {}).get("lat"),
            longitude=data.get("loc", {}).get("lng"),
            duration_minutes=data.get("durationHrs", 0) * 60 if data.get("durationHrs") else None,
            distance_km=data.get("effortDistanceKm"),
            num_observers=data.get("numObservers"),
            species_count=len(observations),
            observations=observations,
        )

    def validate_species_for_location(
        self,
        species_code: str,
        latitude: float,
        longitude: float,
        observation_date: Optional[str] = None,
        distance_km: float = 50,
    ) -> Dict[str, Any]:
        """
        Validate if a species is expected at a location.

        Args:
            species_code: eBird species code
            latitude: Latitude
            longitude: Longitude
            observation_date: Date of observation (for seasonal validation)
            distance_km: Search radius

        Returns:
            Validation result with nearby observations
        """
        # Get nearby observations of this species
        observations = self.get_nearby_observations(
            latitude=latitude,
            longitude=longitude,
            distance_km=distance_km,
            back=30,
        )

        species_obs = [o for o in observations if o.species_code == species_code]

        # Get all species in area for comparison
        all_species = {o.species_code for o in observations}

        return {
            "species_code": species_code,
            "is_valid": len(species_obs) > 0,
            "nearby_observations": len(species_obs),
            "total_species_in_area": len(all_species),
            "most_recent_observation": species_obs[0].observation_date if species_obs else None,
            "observations": [o.to_dict() for o in species_obs[:5]],
        }


def match_detections_to_ebird(
    detections: List[Dict[str, Any]],
    ebird_client: EBirdClient,
    latitude: float,
    longitude: float,
    species_mapping: Optional[Dict[str, str]] = None,
) -> List[Dict[str, Any]]:
    """
    Match detection labels to eBird taxonomy and validate.

    Args:
        detections: List of detections with 'label' field
        ebird_client: eBird API client
        latitude: Location latitude
        longitude: Location longitude
        species_mapping: Optional mapping from detection labels to eBird species codes

    Returns:
        Detections with eBird validation results
    """
    # Get nearby observations for context
    nearby_obs = ebird_client.get_nearby_observations(
        latitude=latitude,
        longitude=longitude,
        distance_km=25,
        back=30,
    )

    nearby_species = {o.species_code: o for o in nearby_obs}

    results = []
    for det in detections:
        label = det.get("label", "")

        # Map label to eBird code
        if species_mapping and label in species_mapping:
            species_code = species_mapping[label]
        else:
            # Try to match by name
            species_code = label.lower().replace(" ", "")

        det_copy = det.copy()
        det_copy["ebird_validated"] = species_code in nearby_species
        det_copy["ebird_species_code"] = species_code
        det_copy["ebird_nearby_count"] = len(
            [o for o in nearby_obs if o.species_code == species_code]
        )

        results.append(det_copy)

    return results
