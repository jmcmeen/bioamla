# services/ebird.py
"""
Service for eBird observation data operations.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .base import BaseService, ServiceResult, ToDictMixin


@dataclass
class EBirdObservation(ToDictMixin):
    """Information about an eBird observation."""

    species_code: str
    common_name: str
    scientific_name: str
    location_name: str
    observation_date: str
    how_many: Optional[int]
    latitude: Optional[float]
    longitude: Optional[float]


@dataclass
class ValidationResult(ToDictMixin):
    """Result of species validation at a location."""

    species_code: str
    is_valid: bool
    nearby_observations: int
    total_species_in_area: int
    most_recent_observation: Optional[str]


@dataclass
class NearbyResult(ToDictMixin):
    """Result of nearby observations query."""

    observations: List[EBirdObservation]
    total_count: int
    query_params: Dict[str, Any]


class EBirdService(BaseService):
    """
    Service for eBird operations.

    Provides high-level methods for:
    - Validating species at locations
    - Getting nearby observations
    """

    def __init__(self, api_key: Optional[str] = None) -> None:
        """
        Initialize eBird service.

        Args:
            api_key: eBird API key. If not provided, will check EBIRD_API_KEY env var.
        """
        super().__init__()
        self._api_key = api_key

    def _get_api_key(self) -> str:
        """Get API key from init or environment."""
        import os

        if self._api_key:
            return self._api_key

        env_key = os.environ.get("EBIRD_API_KEY")
        if env_key:
            return env_key

        raise ValueError("eBird API key required. Set via EBIRD_API_KEY env var or pass to constructor.")

    def validate_species(
        self,
        species_code: str,
        lat: float,
        lng: float,
        distance_km: float = 50,
    ) -> ServiceResult[ValidationResult]:
        """
        Validate if a species is expected at a location.

        Args:
            species_code: eBird species code
            lat: Latitude
            lng: Longitude
            distance_km: Search radius in km

        Returns:
            Result with validation data
        """
        try:
            from bioamla.core.catalogs.integrations import EBirdClient

            client = EBirdClient(api_key=self._get_api_key())
            data = client.validate_species_for_location(
                species_code=species_code,
                latitude=lat,
                longitude=lng,
                distance_km=distance_km,
            )

            result = ValidationResult(
                species_code=species_code,
                is_valid=data["is_valid"],
                nearby_observations=data["nearby_observations"],
                total_species_in_area=data["total_species_in_area"],
                most_recent_observation=data.get("most_recent_observation"),
            )

            if result.is_valid:
                message = f"{species_code} is expected at this location"
            else:
                message = f"{species_code} not recently observed at this location"

            return ServiceResult.ok(
                data=result,
                message=message,
            )
        except ValueError as e:
            return ServiceResult.fail(str(e))
        except Exception as e:
            return ServiceResult.fail(f"Validation failed: {e}")

    def get_nearby(
        self,
        lat: float,
        lng: float,
        distance_km: float = 25,
        days: int = 14,
        limit: int = 20,
    ) -> ServiceResult[NearbyResult]:
        """
        Get recent eBird observations near a location.

        Args:
            lat: Latitude
            lng: Longitude
            distance_km: Search radius in km
            days: Days back to search
            limit: Maximum results

        Returns:
            Result with nearby observations
        """
        try:
            from bioamla.core.catalogs.integrations import EBirdClient

            client = EBirdClient(api_key=self._get_api_key())
            observations = client.get_nearby_observations(
                latitude=lat,
                longitude=lng,
                distance_km=distance_km,
                back=days,
                max_results=limit,
            )

            obs_list = [
                EBirdObservation(
                    species_code=obs.species_code,
                    common_name=obs.common_name,
                    scientific_name=obs.scientific_name,
                    location_name=obs.location_name,
                    observation_date=obs.observation_date,
                    how_many=obs.how_many,
                    latitude=obs.latitude,
                    longitude=obs.longitude,
                )
                for obs in observations
            ]

            result = NearbyResult(
                observations=obs_list,
                total_count=len(obs_list),
                query_params={
                    "latitude": lat,
                    "longitude": lng,
                    "distance_km": distance_km,
                    "days": days,
                },
            )

            return ServiceResult.ok(
                data=result,
                message=f"Found {len(obs_list)} recent observations",
            )
        except ValueError as e:
            return ServiceResult.fail(str(e))
        except Exception as e:
            return ServiceResult.fail(f"Query failed: {e}")
