# models/ebird.py
"""
Data models for eBird operations.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .base import ToDictMixin

# eBird API base URL
EBIRD_API_URL = "https://api.ebird.org/v2"


@dataclass
class EBirdObservation(ToDictMixin):
    """
    Information about an eBird observation.

    Attributes:
        species_code: eBird species code.
        common_name: Common species name.
        scientific_name: Scientific species name.
        location_id: eBird location identifier.
        location_name: Location name.
        observation_date: Date of observation.
        how_many: Count of individuals observed.
        latitude: Latitude coordinate.
        longitude: Longitude coordinate.
        observation_valid: Whether observation is validated.
        observation_reviewed: Whether observation has been reviewed.
        location_private: Whether location is private.
        subid: Submission ID.
        obs_id: Observation ID.
    """

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
class EBirdChecklist(ToDictMixin):
    """
    Information about an eBird checklist.

    Attributes:
        submission_id: eBird submission identifier.
        location_id: Location identifier.
        location_name: Location name.
        observation_date: Date of checklist.
        observation_time: Time of checklist.
        latitude: Latitude coordinate.
        longitude: Longitude coordinate.
        duration_minutes: Duration in minutes.
        distance_km: Distance covered in km.
        num_observers: Number of observers.
        species_count: Count of species observed.
        observations: List of observations in checklist.
    """

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


@dataclass
class EBirdHotspot(ToDictMixin):
    """
    Information about an eBird hotspot.

    Attributes:
        loc_id: Location identifier.
        loc_name: Location name.
        country_code: Country code.
        subnational1_code: State/province code.
        latitude: Latitude coordinate.
        longitude: Longitude coordinate.
        latest_obs_dt: Date of most recent observation.
        num_species_all_time: Total species observed at hotspot.
    """

    loc_id: str
    loc_name: str
    country_code: str
    subnational1_code: str
    latitude: float
    longitude: float
    latest_obs_dt: Optional[str] = None
    num_species_all_time: Optional[int] = None

    @classmethod
    def from_api_response(cls, data: Dict[str, Any]) -> "EBirdHotspot":
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
    most_recent_observation: Optional[str] = None


@dataclass
class NearbyResult(ToDictMixin):
    """Result of nearby observations query."""

    observations: List[EBirdObservation]
    total_count: int
    query_params: Dict[str, Any]


@dataclass
class RegionResult(ToDictMixin):
    """Result of regional observations query."""

    observations: List[EBirdObservation]
    total_count: int
    region_code: str
