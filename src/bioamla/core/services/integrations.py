"""
External Integrations Module
============================

This module provides integrations with external services and databases:
- eBird checklist integration
- PostgreSQL database export

Example:
    >>> from bioamla.integrations import EBirdClient, PostgreSQLExporter
    >>> ebird = EBirdClient(api_key="your_key")
    >>> checklists = ebird.get_checklists_for_region("US-CA")
    >>>
    >>> exporter = PostgreSQLExporter(connection_string="postgresql://...")
    >>> exporter.export_detections(detections)
"""

import csv
import json
import logging
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from bioamla.core.files import TextFile

logger = logging.getLogger(__name__)

__all__ = [
    # eBird integration
    "EBirdObservation",
    "EBirdChecklist",
    "EBirdClient",
    "match_detections_to_ebird",
    # PostgreSQL integration
    "DatabaseConfig",
    "PostgreSQLExporter",
    "export_detections_to_postgres",
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


# =============================================================================
# PostgreSQL Database Export
# =============================================================================


@dataclass
class DatabaseConfig:
    """PostgreSQL database configuration."""

    host: str = "localhost"
    port: int = 5432
    database: str = "bioamla"
    user: str = "postgres"
    password: str = ""
    schema: str = "public"

    @property
    def connection_string(self) -> str:
        """Get connection string."""
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"


class PostgreSQLExporter:
    """
    Export data to PostgreSQL database.

    Creates tables and exports detections, annotations, and analysis results.
    """

    def __init__(
        self,
        config: Optional[DatabaseConfig] = None,
        connection_string: Optional[str] = None,
    ):
        """
        Initialize PostgreSQL exporter.

        Args:
            config: Database configuration
            connection_string: Direct connection string (overrides config)
        """
        self.config = config or DatabaseConfig()
        self._connection_string = connection_string or self.config.connection_string
        self._engine = None
        self._connection = None

    def _get_engine(self):
        """Get SQLAlchemy engine."""
        if self._engine is None:
            try:
                from sqlalchemy import create_engine
            except ImportError as err:
                raise ImportError(
                    "sqlalchemy and psycopg2 are required. "
                    "Install with: pip install sqlalchemy psycopg2-binary"
                ) from err
            self._engine = create_engine(self._connection_string)
        return self._engine

    def _get_connection(self):
        """Get database connection."""
        if self._connection is None:
            try:
                import psycopg2
            except ImportError as err:
                raise ImportError(
                    "psycopg2 is required. Install with: pip install psycopg2-binary"
                ) from err

            self._connection = psycopg2.connect(self._connection_string)
        return self._connection

    def create_tables(self) -> None:
        """Create database tables if they don't exist."""
        conn = self._get_connection()
        cursor = conn.cursor()

        # Detections table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS detections (
                id SERIAL PRIMARY KEY,
                filepath VARCHAR(1024),
                start_time FLOAT,
                end_time FLOAT,
                label VARCHAR(256),
                confidence FLOAT,
                low_freq FLOAT,
                high_freq FLOAT,
                detector VARCHAR(256),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                metadata JSONB
            )
        """)

        # Annotations table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS annotations (
                id SERIAL PRIMARY KEY,
                filepath VARCHAR(1024),
                start_time FLOAT,
                end_time FLOAT,
                low_freq FLOAT,
                high_freq FLOAT,
                label VARCHAR(256),
                annotator VARCHAR(256),
                confidence FLOAT,
                notes TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                metadata JSONB
            )
        """)

        # Audio files table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS audio_files (
                id SERIAL PRIMARY KEY,
                filepath VARCHAR(1024) UNIQUE,
                duration FLOAT,
                sample_rate INT,
                channels INT,
                file_size BIGINT,
                recorded_at TIMESTAMP,
                location_lat FLOAT,
                location_lng FLOAT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                metadata JSONB
            )
        """)

        # Analysis results table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS analysis_results (
                id SERIAL PRIMARY KEY,
                audio_file_id INT REFERENCES audio_files(id),
                analysis_type VARCHAR(256),
                results JSONB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Species observations table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS species_observations (
                id SERIAL PRIMARY KEY,
                species_code VARCHAR(32),
                common_name VARCHAR(256),
                scientific_name VARCHAR(256),
                observation_date DATE,
                location_lat FLOAT,
                location_lng FLOAT,
                location_name VARCHAR(512),
                detection_id INT REFERENCES detections(id),
                ebird_validated BOOLEAN DEFAULT FALSE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                metadata JSONB
            )
        """)

        # Create indexes
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_detections_label ON detections(label);
            CREATE INDEX IF NOT EXISTS idx_detections_filepath ON detections(filepath);
            CREATE INDEX IF NOT EXISTS idx_annotations_label ON annotations(label);
            CREATE INDEX IF NOT EXISTS idx_species_observations_species_code ON species_observations(species_code);
            CREATE INDEX IF NOT EXISTS idx_species_observations_date ON species_observations(observation_date);
        """)

        conn.commit()
        logger.info("Database tables created")

    def export_detections(
        self, detections: List[Dict[str, Any]], detector_name: str = "unknown"
    ) -> int:
        """
        Export detections to database.

        Args:
            detections: List of detection dictionaries
            detector_name: Name of detector used

        Returns:
            Number of detections exported
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        count = 0
        for det in detections:
            cursor.execute(
                """
                INSERT INTO detections
                (filepath, start_time, end_time, label, confidence,
                 low_freq, high_freq, detector, metadata)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
                (
                    det.get("filepath", ""),
                    det.get("start_time", 0),
                    det.get("end_time", 0),
                    det.get("label", ""),
                    det.get("confidence", 0),
                    det.get("low_freq"),
                    det.get("high_freq"),
                    detector_name,
                    json.dumps(det.get("metadata", {})),
                ),
            )
            count += 1

        conn.commit()
        logger.info(f"Exported {count} detections to database")
        return count

    def export_annotations(
        self, annotations: List[Dict[str, Any]], annotator: str = "unknown"
    ) -> int:
        """
        Export annotations to database.

        Args:
            annotations: List of annotation dictionaries
            annotator: Annotator identifier

        Returns:
            Number of annotations exported
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        count = 0
        for ann in annotations:
            cursor.execute(
                """
                INSERT INTO annotations
                (filepath, start_time, end_time, low_freq, high_freq,
                 label, annotator, confidence, notes, metadata)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
                (
                    ann.get("filepath", ""),
                    ann.get("start_time", 0),
                    ann.get("end_time", 0),
                    ann.get("low_freq"),
                    ann.get("high_freq"),
                    ann.get("label", ""),
                    annotator,
                    ann.get("confidence"),
                    ann.get("notes", ""),
                    json.dumps(ann.get("metadata", {})),
                ),
            )
            count += 1

        conn.commit()
        logger.info(f"Exported {count} annotations to database")
        return count

    def export_audio_file(
        self,
        filepath: str,
        duration: float,
        sample_rate: int,
        channels: int = 1,
        file_size: Optional[int] = None,
        recorded_at: Optional[datetime] = None,
        latitude: Optional[float] = None,
        longitude: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        """
        Export audio file metadata to database.

        Args:
            filepath: File path
            duration: Duration in seconds
            sample_rate: Sample rate
            channels: Number of channels
            file_size: File size in bytes
            recorded_at: Recording timestamp
            latitude: Recording location latitude
            longitude: Recording location longitude
            metadata: Additional metadata

        Returns:
            Database ID of inserted record
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO audio_files
            (filepath, duration, sample_rate, channels, file_size,
             recorded_at, location_lat, location_lng, metadata)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (filepath) DO UPDATE SET
                duration = EXCLUDED.duration,
                sample_rate = EXCLUDED.sample_rate,
                channels = EXCLUDED.channels,
                file_size = EXCLUDED.file_size,
                recorded_at = EXCLUDED.recorded_at,
                location_lat = EXCLUDED.location_lat,
                location_lng = EXCLUDED.location_lng,
                metadata = EXCLUDED.metadata
            RETURNING id
        """,
            (
                filepath,
                duration,
                sample_rate,
                channels,
                file_size,
                recorded_at,
                latitude,
                longitude,
                json.dumps(metadata or {}),
            ),
        )

        audio_file_id = cursor.fetchone()[0]
        conn.commit()

        return audio_file_id

    def export_species_observation(
        self,
        species_code: str,
        common_name: str,
        scientific_name: str,
        observation_date: date,
        latitude: Optional[float] = None,
        longitude: Optional[float] = None,
        location_name: Optional[str] = None,
        detection_id: Optional[int] = None,
        ebird_validated: bool = False,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        """
        Export species observation to database.

        Returns:
            Database ID of inserted record
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO species_observations
            (species_code, common_name, scientific_name, observation_date,
             location_lat, location_lng, location_name, detection_id,
             ebird_validated, metadata)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id
        """,
            (
                species_code,
                common_name,
                scientific_name,
                observation_date,
                latitude,
                longitude,
                location_name,
                detection_id,
                ebird_validated,
                json.dumps(metadata or {}),
            ),
        )

        obs_id = cursor.fetchone()[0]
        conn.commit()

        return obs_id

    def query_detections(
        self,
        label: Optional[str] = None,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        min_confidence: Optional[float] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        Query detections from database.

        Args:
            label: Filter by label
            start_date: Filter by start date
            end_date: Filter by end date
            min_confidence: Minimum confidence threshold
            limit: Maximum results

        Returns:
            List of detection records
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        query = "SELECT * FROM detections WHERE 1=1"
        params = []

        if label:
            query += " AND label = %s"
            params.append(label)

        if start_date:
            query += " AND created_at >= %s"
            params.append(start_date)

        if end_date:
            query += " AND created_at <= %s"
            params.append(end_date)

        if min_confidence:
            query += " AND confidence >= %s"
            params.append(min_confidence)

        query += " ORDER BY created_at DESC LIMIT %s"
        params.append(limit)

        cursor.execute(query, params)

        columns = [desc[0] for desc in cursor.description]
        results = []
        for row in cursor.fetchall():
            results.append(dict(zip(columns, row)))

        return results

    def get_statistics(self) -> Dict[str, Any]:
        """Get database statistics."""
        conn = self._get_connection()
        cursor = conn.cursor()

        stats = {}

        # Count tables
        for table in ["detections", "annotations", "audio_files", "species_observations"]:
            cursor.execute(f"SELECT COUNT(*) FROM {table}")
            stats[f"{table}_count"] = cursor.fetchone()[0]

        # Detections by label
        cursor.execute("""
            SELECT label, COUNT(*) as count
            FROM detections
            GROUP BY label
            ORDER BY count DESC
            LIMIT 20
        """)
        stats["detections_by_label"] = {row[0]: row[1] for row in cursor.fetchall()}

        # Species observations by month
        cursor.execute("""
            SELECT DATE_TRUNC('month', observation_date) as month, COUNT(*)
            FROM species_observations
            GROUP BY month
            ORDER BY month DESC
            LIMIT 12
        """)
        stats["observations_by_month"] = {str(row[0].date()): row[1] for row in cursor.fetchall()}

        return stats

    def export_to_csv(
        self, table: str, output_path: str, query_filter: Optional[str] = None
    ) -> str:
        """
        Export table to CSV.

        Args:
            table: Table name
            output_path: Output file path
            query_filter: Optional WHERE clause

        Returns:
            Path to exported file
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        query = f"SELECT * FROM {table}"
        if query_filter:
            query += f" WHERE {query_filter}"

        cursor.execute(query)

        columns = [desc[0] for desc in cursor.description]

        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with TextFile(output_file, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f.handle)
            writer.writerow(columns)

            for row in cursor.fetchall():
                writer.writerow(row)

        logger.info(f"Exported {table} to {output_path}")
        return str(output_file)

    def close(self) -> None:
        """Close database connections."""
        if self._connection is not None:
            self._connection.close()
            self._connection = None

        if self._engine is not None:
            self._engine.dispose()
            self._engine = None


def export_detections_to_postgres(
    detections: List[Dict[str, Any]],
    connection_string: str,
    detector_name: str = "unknown",
    create_tables: bool = True,
) -> int:
    """
    Convenience function to export detections to PostgreSQL.

    Args:
        detections: List of detections
        connection_string: PostgreSQL connection string
        detector_name: Name of detector
        create_tables: Whether to create tables if needed

    Returns:
        Number of detections exported
    """
    exporter = PostgreSQLExporter(connection_string=connection_string)

    if create_tables:
        exporter.create_tables()

    count = exporter.export_detections(detections, detector_name)
    exporter.close()

    return count
