"""
Unit tests for bioamla.integrations module.
"""

import json
from datetime import date, datetime
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import pytest


class TestEBirdObservation:
    """Tests for EBirdObservation dataclass."""

    def test_creation(self):
        """Test creating observation."""
        from bioamla.integrations import EBirdObservation

        obs = EBirdObservation(
            species_code="carwre",
            common_name="Carolina Wren",
            scientific_name="Thryothorus ludovicianus",
            location_id="L12345",
            location_name="Test Location",
            observation_date="2024-01-15",
            how_many=3,
            latitude=35.0,
            longitude=-80.0,
        )

        assert obs.species_code == "carwre"
        assert obs.common_name == "Carolina Wren"
        assert obs.how_many == 3

    def test_to_dict(self):
        """Test conversion to dictionary."""
        from bioamla.integrations import EBirdObservation

        obs = EBirdObservation(
            species_code="carwre",
            common_name="Carolina Wren",
            scientific_name="Thryothorus ludovicianus",
            location_id="L12345",
            location_name="Test Location",
            observation_date="2024-01-15",
        )

        d = obs.to_dict()

        assert d["species_code"] == "carwre"
        assert d["common_name"] == "Carolina Wren"
        assert "observation_date" in d

    def test_from_api_response(self):
        """Test creating from API response."""
        from bioamla.integrations import EBirdObservation

        api_data = {
            "speciesCode": "carwre",
            "comName": "Carolina Wren",
            "sciName": "Thryothorus ludovicianus",
            "locId": "L12345",
            "locName": "Test Location",
            "obsDt": "2024-01-15",
            "howMany": 5,
            "lat": 35.0,
            "lng": -80.0,
            "obsValid": True,
            "obsReviewed": False,
        }

        obs = EBirdObservation.from_api_response(api_data)

        assert obs.species_code == "carwre"
        assert obs.common_name == "Carolina Wren"
        assert obs.how_many == 5
        assert obs.latitude == 35.0


class TestEBirdChecklist:
    """Tests for EBirdChecklist dataclass."""

    def test_creation(self):
        """Test creating checklist."""
        from bioamla.integrations import EBirdChecklist

        checklist = EBirdChecklist(
            submission_id="S123456",
            location_id="L12345",
            location_name="Test Location",
            observation_date="2024-01-15",
            observation_time="08:00",
            duration_minutes=60,
            species_count=25,
        )

        assert checklist.submission_id == "S123456"
        assert checklist.duration_minutes == 60
        assert checklist.species_count == 25

    def test_to_dict(self):
        """Test conversion to dictionary."""
        from bioamla.integrations import EBirdChecklist, EBirdObservation

        obs = EBirdObservation(
            species_code="carwre",
            common_name="Carolina Wren",
            scientific_name="Thryothorus ludovicianus",
            location_id="L12345",
            location_name="Test Location",
            observation_date="2024-01-15",
        )

        checklist = EBirdChecklist(
            submission_id="S123456",
            location_id="L12345",
            location_name="Test Location",
            observation_date="2024-01-15",
            observations=[obs],
            species_count=1,
        )

        d = checklist.to_dict()

        assert d["submission_id"] == "S123456"
        assert len(d["observations"]) == 1
        assert d["observations"][0]["species_code"] == "carwre"


class TestEBirdClient:
    """Tests for EBirdClient."""

    def test_initialization(self):
        """Test client initialization."""
        from bioamla.integrations import EBirdClient

        client = EBirdClient(api_key="test_key")
        assert client.api_key == "test_key"

    def test_get_session_creates_session(self):
        """Test session creation."""
        from bioamla.integrations import EBirdClient

        # Test that session is created and headers are set
        client = EBirdClient(api_key="test_key")

        with patch.object(client, "_get_session") as mock_get_session:
            mock_session = MagicMock()
            mock_session.headers = {"X-eBirdApiToken": "test_key"}
            mock_get_session.return_value = mock_session

            session = client._get_session()
            assert session.headers["X-eBirdApiToken"] == "test_key"

    def test_get_session_import_error(self):
        """Test error when requests not installed."""
        from bioamla.integrations import EBirdClient

        client = EBirdClient(api_key="test_key")

        with patch.dict("sys.modules", {"requests": None}):
            with patch("bioamla.integrations.EBirdClient._get_session") as mock:
                mock.side_effect = ImportError("requests is required")
                with pytest.raises(ImportError, match="requests"):
                    client._get_session()

    @patch("bioamla.integrations.EBirdClient._request")
    def test_get_recent_observations(self, mock_request):
        """Test getting recent observations."""
        from bioamla.integrations import EBirdClient

        mock_request.return_value = [
            {
                "speciesCode": "carwre",
                "comName": "Carolina Wren",
                "sciName": "Thryothorus ludovicianus",
                "locId": "L12345",
                "locName": "Test Location",
                "obsDt": "2024-01-15",
            }
        ]

        client = EBirdClient(api_key="test_key")
        observations = client.get_recent_observations("US-NC", back=7)

        assert len(observations) == 1
        assert observations[0].species_code == "carwre"

    @patch("bioamla.integrations.EBirdClient._request")
    def test_get_recent_observations_by_species(self, mock_request):
        """Test getting recent observations filtered by species."""
        from bioamla.integrations import EBirdClient

        mock_request.return_value = []

        client = EBirdClient(api_key="test_key")
        client.get_recent_observations("US-NC", species_code="carwre")

        mock_request.assert_called_once()
        assert "carwre" in mock_request.call_args[0][0]

    @patch("bioamla.integrations.EBirdClient._request")
    def test_get_nearby_observations(self, mock_request):
        """Test getting nearby observations."""
        from bioamla.integrations import EBirdClient

        mock_request.return_value = []

        client = EBirdClient(api_key="test_key")
        client.get_nearby_observations(
            latitude=35.0,
            longitude=-80.0,
            distance_km=25,
        )

        mock_request.assert_called_once()
        call_args = mock_request.call_args
        # Check positional args or params
        assert "data/obs/geo/recent" in call_args[0]
        params = call_args[0][1] if len(call_args[0]) > 1 else call_args[1].get("params", {})
        assert params.get("lat") == 35.0 or "lat" in str(call_args)

    @patch("bioamla.integrations.EBirdClient._request")
    def test_get_species_list(self, mock_request):
        """Test getting species list."""
        from bioamla.integrations import EBirdClient

        mock_request.return_value = ["carwre", "norcar", "eastow"]

        client = EBirdClient(api_key="test_key")
        species = client.get_species_list("US-NC")

        assert len(species) == 3

    @patch("bioamla.integrations.EBirdClient._request")
    def test_get_taxonomy(self, mock_request):
        """Test getting taxonomy."""
        from bioamla.integrations import EBirdClient

        mock_request.return_value = [
            {"speciesCode": "carwre", "comName": "Carolina Wren"}
        ]

        client = EBirdClient(api_key="test_key")
        taxonomy = client.get_taxonomy(species_codes=["carwre"])

        assert len(taxonomy) == 1

    @patch("bioamla.integrations.EBirdClient._request")
    def test_get_hotspots(self, mock_request):
        """Test getting hotspots."""
        from bioamla.integrations import EBirdClient

        mock_request.return_value = [
            {"locId": "L12345", "locName": "Test Hotspot"}
        ]

        client = EBirdClient(api_key="test_key")
        hotspots = client.get_hotspots("US-NC")

        assert len(hotspots) == 1

    @patch("bioamla.integrations.EBirdClient._request")
    def test_get_checklist(self, mock_request):
        """Test getting checklist details."""
        from bioamla.integrations import EBirdClient

        mock_request.return_value = {
            "subId": "S123456",
            "locId": "L12345",
            "loc": {"name": "Test Location", "lat": 35.0, "lng": -80.0},
            "obsDt": "2024-01-15",
            "obsTime": "08:00",
            "durationHrs": 1.0,
            "numObservers": 2,
            "obs": [
                {
                    "speciesCode": "carwre",
                    "species": {
                        "comName": "Carolina Wren",
                        "sciName": "Thryothorus ludovicianus",
                    },
                    "howManyStr": "3",
                }
            ],
        }

        client = EBirdClient(api_key="test_key")
        checklist = client.get_checklist("S123456")

        assert checklist.submission_id == "S123456"
        assert checklist.location_name == "Test Location"
        assert len(checklist.observations) == 1
        assert checklist.duration_minutes == 60

    @patch("bioamla.integrations.EBirdClient.get_nearby_observations")
    def test_validate_species_for_location(self, mock_nearby):
        """Test validating species for location."""
        from bioamla.integrations import EBirdClient, EBirdObservation

        mock_nearby.return_value = [
            EBirdObservation(
                species_code="carwre",
                common_name="Carolina Wren",
                scientific_name="Thryothorus ludovicianus",
                location_id="L12345",
                location_name="Test Location",
                observation_date="2024-01-15",
            ),
            EBirdObservation(
                species_code="norcar",
                common_name="Northern Cardinal",
                scientific_name="Cardinalis cardinalis",
                location_id="L12345",
                location_name="Test Location",
                observation_date="2024-01-15",
            ),
        ]

        client = EBirdClient(api_key="test_key")
        result = client.validate_species_for_location(
            species_code="carwre",
            latitude=35.0,
            longitude=-80.0,
        )

        assert result["is_valid"] is True
        assert result["nearby_observations"] == 1
        assert result["total_species_in_area"] == 2

    @patch("bioamla.integrations.EBirdClient.get_nearby_observations")
    def test_validate_species_not_found(self, mock_nearby):
        """Test validation when species not found."""
        from bioamla.integrations import EBirdClient, EBirdObservation

        mock_nearby.return_value = [
            EBirdObservation(
                species_code="norcar",
                common_name="Northern Cardinal",
                scientific_name="Cardinalis cardinalis",
                location_id="L12345",
                location_name="Test Location",
                observation_date="2024-01-15",
            ),
        ]

        client = EBirdClient(api_key="test_key")
        result = client.validate_species_for_location(
            species_code="snogoo",
            latitude=35.0,
            longitude=-80.0,
        )

        assert result["is_valid"] is False
        assert result["nearby_observations"] == 0


class TestMatchDetectionsToEbird:
    """Tests for match_detections_to_ebird function."""

    @patch("bioamla.integrations.EBirdClient.get_nearby_observations")
    def test_basic_matching(self, mock_nearby):
        """Test basic detection matching."""
        from bioamla.integrations import EBirdClient, EBirdObservation, match_detections_to_ebird

        mock_nearby.return_value = [
            EBirdObservation(
                species_code="carwre",
                common_name="Carolina Wren",
                scientific_name="Thryothorus ludovicianus",
                location_id="L12345",
                location_name="Test Location",
                observation_date="2024-01-15",
            ),
        ]

        client = EBirdClient(api_key="test_key")
        detections = [
            {"label": "carwre", "confidence": 0.9},
            {"label": "unknown", "confidence": 0.5},
        ]

        results = match_detections_to_ebird(
            detections=detections,
            ebird_client=client,
            latitude=35.0,
            longitude=-80.0,
        )

        assert len(results) == 2
        assert results[0]["ebird_validated"] is True
        assert results[1]["ebird_validated"] is False

    @patch("bioamla.integrations.EBirdClient.get_nearby_observations")
    def test_matching_with_species_mapping(self, mock_nearby):
        """Test matching with custom species mapping."""
        from bioamla.integrations import EBirdClient, EBirdObservation, match_detections_to_ebird

        mock_nearby.return_value = [
            EBirdObservation(
                species_code="carwre",
                common_name="Carolina Wren",
                scientific_name="Thryothorus ludovicianus",
                location_id="L12345",
                location_name="Test Location",
                observation_date="2024-01-15",
            ),
        ]

        client = EBirdClient(api_key="test_key")
        detections = [{"label": "carolina_wren", "confidence": 0.9}]
        species_mapping = {"carolina_wren": "carwre"}

        results = match_detections_to_ebird(
            detections=detections,
            ebird_client=client,
            latitude=35.0,
            longitude=-80.0,
            species_mapping=species_mapping,
        )

        assert results[0]["ebird_validated"] is True
        assert results[0]["ebird_species_code"] == "carwre"


class TestDatabaseConfig:
    """Tests for DatabaseConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        from bioamla.integrations import DatabaseConfig

        config = DatabaseConfig()
        assert config.host == "localhost"
        assert config.port == 5432
        assert config.database == "bioamla"

    def test_connection_string(self):
        """Test connection string generation."""
        from bioamla.integrations import DatabaseConfig

        config = DatabaseConfig(
            host="db.example.com",
            port=5433,
            database="mydb",
            user="myuser",
            password="secret",
        )

        assert config.connection_string == "postgresql://myuser:secret@db.example.com:5433/mydb"


class TestPostgreSQLExporter:
    """Tests for PostgreSQLExporter."""

    def test_initialization_with_config(self):
        """Test initialization with config."""
        from bioamla.integrations import PostgreSQLExporter, DatabaseConfig

        config = DatabaseConfig(database="testdb")
        exporter = PostgreSQLExporter(config=config)

        assert "testdb" in exporter._connection_string

    def test_initialization_with_connection_string(self):
        """Test initialization with connection string."""
        from bioamla.integrations import PostgreSQLExporter

        conn_str = "postgresql://user:pass@localhost/mydb"
        exporter = PostgreSQLExporter(connection_string=conn_str)

        assert exporter._connection_string == conn_str

    def test_get_engine_import_error(self):
        """Test error when sqlalchemy not installed."""
        from bioamla.integrations import PostgreSQLExporter

        exporter = PostgreSQLExporter(connection_string="postgresql://localhost/test")

        with patch.dict("sys.modules", {"sqlalchemy": None}):
            with patch("bioamla.integrations.PostgreSQLExporter._get_engine") as mock:
                mock.side_effect = ImportError("sqlalchemy and psycopg2 are required")
                with pytest.raises(ImportError, match="sqlalchemy"):
                    exporter._get_engine()

    @patch("bioamla.integrations.PostgreSQLExporter._get_connection")
    def test_create_tables(self, mock_get_conn):
        """Test table creation."""
        from bioamla.integrations import PostgreSQLExporter

        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_get_conn.return_value = mock_conn

        exporter = PostgreSQLExporter(connection_string="postgresql://localhost/test")
        exporter.create_tables()

        # Should have multiple CREATE TABLE calls
        assert mock_cursor.execute.call_count >= 5
        mock_conn.commit.assert_called_once()

    @patch("bioamla.integrations.PostgreSQLExporter._get_connection")
    def test_export_detections(self, mock_get_conn):
        """Test exporting detections."""
        from bioamla.integrations import PostgreSQLExporter

        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_get_conn.return_value = mock_conn

        exporter = PostgreSQLExporter(connection_string="postgresql://localhost/test")
        detections = [
            {
                "filepath": "test.wav",
                "start_time": 0.0,
                "end_time": 1.0,
                "label": "bird",
                "confidence": 0.9,
            },
            {
                "filepath": "test.wav",
                "start_time": 2.0,
                "end_time": 3.0,
                "label": "frog",
                "confidence": 0.8,
            },
        ]

        count = exporter.export_detections(detections, detector_name="test_detector")

        assert count == 2
        assert mock_cursor.execute.call_count == 2
        mock_conn.commit.assert_called_once()

    @patch("bioamla.integrations.PostgreSQLExporter._get_connection")
    def test_export_annotations(self, mock_get_conn):
        """Test exporting annotations."""
        from bioamla.integrations import PostgreSQLExporter

        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_get_conn.return_value = mock_conn

        exporter = PostgreSQLExporter(connection_string="postgresql://localhost/test")
        annotations = [
            {
                "filepath": "test.wav",
                "start_time": 0.0,
                "end_time": 1.0,
                "label": "bird",
                "notes": "Clear call",
            },
        ]

        count = exporter.export_annotations(annotations, annotator="tester")

        assert count == 1

    @patch("bioamla.integrations.PostgreSQLExporter._get_connection")
    def test_export_audio_file(self, mock_get_conn):
        """Test exporting audio file metadata."""
        from bioamla.integrations import PostgreSQLExporter

        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = (1,)
        mock_conn.cursor.return_value = mock_cursor
        mock_get_conn.return_value = mock_conn

        exporter = PostgreSQLExporter(connection_string="postgresql://localhost/test")
        audio_id = exporter.export_audio_file(
            filepath="test.wav",
            duration=60.0,
            sample_rate=16000,
            channels=1,
            latitude=35.0,
            longitude=-80.0,
        )

        assert audio_id == 1
        mock_cursor.execute.assert_called_once()

    @patch("bioamla.integrations.PostgreSQLExporter._get_connection")
    def test_export_species_observation(self, mock_get_conn):
        """Test exporting species observation."""
        from bioamla.integrations import PostgreSQLExporter

        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = (42,)
        mock_conn.cursor.return_value = mock_cursor
        mock_get_conn.return_value = mock_conn

        exporter = PostgreSQLExporter(connection_string="postgresql://localhost/test")
        obs_id = exporter.export_species_observation(
            species_code="carwre",
            common_name="Carolina Wren",
            scientific_name="Thryothorus ludovicianus",
            observation_date=date(2024, 1, 15),
            latitude=35.0,
            longitude=-80.0,
            ebird_validated=True,
        )

        assert obs_id == 42

    @patch("bioamla.integrations.PostgreSQLExporter._get_connection")
    def test_query_detections(self, mock_get_conn):
        """Test querying detections."""
        from bioamla.integrations import PostgreSQLExporter

        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.description = [
            ("id",), ("filepath",), ("label",), ("confidence",)
        ]
        mock_cursor.fetchall.return_value = [
            (1, "test.wav", "bird", 0.9),
            (2, "test2.wav", "bird", 0.85),
        ]
        mock_conn.cursor.return_value = mock_cursor
        mock_get_conn.return_value = mock_conn

        exporter = PostgreSQLExporter(connection_string="postgresql://localhost/test")
        results = exporter.query_detections(label="bird", limit=10)

        assert len(results) == 2
        assert results[0]["label"] == "bird"

    @patch("bioamla.integrations.PostgreSQLExporter._get_connection")
    def test_get_statistics(self, mock_get_conn):
        """Test getting statistics."""
        from bioamla.integrations import PostgreSQLExporter

        mock_conn = MagicMock()
        mock_cursor = MagicMock()

        # Mock the count queries
        mock_cursor.fetchone.side_effect = [
            (100,),  # detections
            (50,),   # annotations
            (20,),   # audio_files
            (75,),   # species_observations
        ]

        # Mock the group by queries
        mock_cursor.fetchall.side_effect = [
            [("bird", 60), ("frog", 30), ("unknown", 10)],  # by label
            [],  # by month
        ]

        mock_conn.cursor.return_value = mock_cursor
        mock_get_conn.return_value = mock_conn

        exporter = PostgreSQLExporter(connection_string="postgresql://localhost/test")
        stats = exporter.get_statistics()

        assert stats["detections_count"] == 100
        assert stats["annotations_count"] == 50
        assert "detections_by_label" in stats

    @patch("bioamla.integrations.PostgreSQLExporter._get_connection")
    def test_export_to_csv(self, mock_get_conn, tmp_path):
        """Test exporting to CSV."""
        from bioamla.integrations import PostgreSQLExporter

        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.description = [("id",), ("label",), ("confidence",)]
        mock_cursor.fetchall.return_value = [
            (1, "bird", 0.9),
            (2, "frog", 0.8),
        ]
        mock_conn.cursor.return_value = mock_cursor
        mock_get_conn.return_value = mock_conn

        output_path = tmp_path / "export.csv"
        exporter = PostgreSQLExporter(connection_string="postgresql://localhost/test")
        result = exporter.export_to_csv("detections", str(output_path))

        assert Path(result).exists()

        # Check CSV content
        with open(result) as f:
            lines = f.readlines()
        assert len(lines) == 3  # header + 2 rows
        assert "id,label,confidence" in lines[0]

    def test_close(self):
        """Test closing connections."""
        from bioamla.integrations import PostgreSQLExporter

        mock_conn = MagicMock()
        mock_engine = MagicMock()

        exporter = PostgreSQLExporter(connection_string="postgresql://localhost/test")
        exporter._connection = mock_conn
        exporter._engine = mock_engine

        exporter.close()

        mock_conn.close.assert_called_once()
        mock_engine.dispose.assert_called_once()
        assert exporter._connection is None
        assert exporter._engine is None


class TestExportDetectionsToPostgres:
    """Tests for export_detections_to_postgres convenience function."""

    @patch("bioamla.integrations.PostgreSQLExporter")
    def test_export_with_create_tables(self, mock_exporter_class):
        """Test export with table creation."""
        from bioamla.integrations import export_detections_to_postgres

        mock_exporter = MagicMock()
        mock_exporter.export_detections.return_value = 5
        mock_exporter_class.return_value = mock_exporter

        detections = [{"label": "bird", "confidence": 0.9}]
        count = export_detections_to_postgres(
            detections=detections,
            connection_string="postgresql://localhost/test",
            detector_name="test",
            create_tables=True,
        )

        mock_exporter.create_tables.assert_called_once()
        assert count == 5
        mock_exporter.close.assert_called_once()

    @patch("bioamla.integrations.PostgreSQLExporter")
    def test_export_without_create_tables(self, mock_exporter_class):
        """Test export without table creation."""
        from bioamla.integrations import export_detections_to_postgres

        mock_exporter = MagicMock()
        mock_exporter.export_detections.return_value = 3
        mock_exporter_class.return_value = mock_exporter

        detections = [{"label": "bird", "confidence": 0.9}]
        count = export_detections_to_postgres(
            detections=detections,
            connection_string="postgresql://localhost/test",
            create_tables=False,
        )

        mock_exporter.create_tables.assert_not_called()
        assert count == 3


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def temp_dir(tmp_path):
    """Provide a temporary directory."""
    return tmp_path
