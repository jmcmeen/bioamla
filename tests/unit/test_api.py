"""
Unit tests for bioamla.api module.
"""

import json
import pickle
import tempfile
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from bioamla.api.base import APICache, APIClient, RateLimiter, cached, rate_limited


class TestRateLimiter:
    """Tests for RateLimiter class."""

    def test_basic_rate_limiting(self):
        """Test that rate limiter throttles requests."""
        limiter = RateLimiter(requests_per_second=10.0, burst_size=1)

        # First request should be immediate
        wait1 = limiter.acquire()
        assert wait1 == 0.0

        # Second request should wait
        wait2 = limiter.acquire()
        assert wait2 > 0

    def test_burst_size(self):
        """Test burst size allows multiple immediate requests."""
        limiter = RateLimiter(requests_per_second=1.0, burst_size=3)

        # Should allow 3 immediate requests
        for _ in range(3):
            wait = limiter.acquire()
            assert wait == 0.0

        # Fourth should wait
        wait = limiter.acquire()
        assert wait > 0

    def test_try_acquire_non_blocking(self):
        """Test try_acquire doesn't block."""
        limiter = RateLimiter(requests_per_second=10.0, burst_size=1)

        # First should succeed
        assert limiter.try_acquire() is True

        # Second should fail without blocking
        start = time.time()
        result = limiter.try_acquire()
        elapsed = time.time() - start

        assert result is False
        assert elapsed < 0.1  # Should be nearly instant

    def test_thread_safety(self):
        """Test rate limiter is thread-safe."""
        limiter = RateLimiter(requests_per_second=100.0, burst_size=5)
        acquired = []

        def acquire_tokens():
            for _ in range(10):
                limiter.acquire()
                acquired.append(1)

        threads = [threading.Thread(target=acquire_tokens) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(acquired) == 30


class TestAPICache:
    """Tests for APICache class."""

    def test_set_and_get(self, temp_dir):
        """Test basic set and get operations."""
        cache = APICache(cache_dir=temp_dir, default_ttl=3600)

        cache.set("key1", {"data": "value1"})
        result = cache.get("key1")

        assert result == {"data": "value1"}

    def test_cache_miss(self, temp_dir):
        """Test cache miss returns None."""
        cache = APICache(cache_dir=temp_dir)

        result = cache.get("nonexistent_key")
        assert result is None

    def test_ttl_expiration(self, temp_dir):
        """Test cache entries expire after TTL."""
        cache = APICache(cache_dir=temp_dir, default_ttl=1)

        cache.set("key", "value", ttl=1)
        assert cache.get("key") == "value"

        time.sleep(1.5)
        assert cache.get("key") is None

    def test_custom_ttl(self, temp_dir):
        """Test custom TTL overrides default."""
        cache = APICache(cache_dir=temp_dir, default_ttl=3600)

        cache.set("key", "value", ttl=1)
        time.sleep(1.5)
        assert cache.get("key") is None

    def test_delete(self, temp_dir):
        """Test delete removes cache entry."""
        cache = APICache(cache_dir=temp_dir)

        cache.set("key", "value")
        assert cache.get("key") == "value"

        deleted = cache.delete("key")
        assert deleted is True
        assert cache.get("key") is None

    def test_delete_nonexistent(self, temp_dir):
        """Test delete returns False for nonexistent key."""
        cache = APICache(cache_dir=temp_dir)

        deleted = cache.delete("nonexistent")
        assert deleted is False

    def test_clear(self, temp_dir):
        """Test clear removes all entries."""
        cache = APICache(cache_dir=temp_dir)

        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.set("key3", "value3")

        count = cache.clear()
        assert count >= 3

        assert cache.get("key1") is None
        assert cache.get("key2") is None
        assert cache.get("key3") is None

    def test_complex_data(self, temp_dir):
        """Test caching complex data structures."""
        cache = APICache(cache_dir=temp_dir)

        data = {
            "list": [1, 2, 3],
            "nested": {"a": 1, "b": 2},
            "string": "hello",
            "number": 42,
        }

        cache.set("complex", data)
        result = cache.get("complex")

        assert result == data


class TestAPIClient:
    """Tests for APIClient class."""

    @patch("requests.Session")
    def test_get_request(self, mock_session_cls):
        """Test basic GET request."""
        mock_session = MagicMock()
        mock_response = MagicMock()
        mock_response.json.return_value = {"data": "value"}
        mock_session.get.return_value = mock_response
        mock_session_cls.return_value = mock_session

        client = APIClient(base_url="https://api.example.com")
        client.session = mock_session

        result = client.get("/endpoint", params={"q": "test"})

        assert result == {"data": "value"}
        mock_session.get.assert_called_once()

    @patch("requests.Session")
    def test_get_with_cache(self, mock_session_cls, temp_dir):
        """Test GET request uses cache."""
        mock_session = MagicMock()
        mock_response = MagicMock()
        mock_response.json.return_value = {"data": "cached"}
        mock_session.get.return_value = mock_response
        mock_session_cls.return_value = mock_session

        cache = APICache(cache_dir=temp_dir)
        client = APIClient(cache=cache)
        client.session = mock_session

        # First request
        result1 = client.get("https://api.example.com/data")
        assert result1 == {"data": "cached"}

        # Second request should use cache
        mock_session.get.reset_mock()
        result2 = client.get("https://api.example.com/data")
        assert result2 == {"data": "cached"}
        mock_session.get.assert_not_called()

    @patch("requests.Session")
    def test_get_with_rate_limiter(self, mock_session_cls):
        """Test GET request respects rate limiting."""
        mock_session = MagicMock()
        mock_response = MagicMock()
        mock_response.json.return_value = {}
        mock_session.get.return_value = mock_response
        mock_session_cls.return_value = mock_session

        limiter = RateLimiter(requests_per_second=5.0)
        client = APIClient(rate_limiter=limiter)
        client.session = mock_session

        start = time.time()
        for _ in range(3):
            client.get("https://api.example.com/data", use_cache=False)
        elapsed = time.time() - start

        # Should have some delay due to rate limiting
        assert elapsed >= 0.3

    @patch("requests.Session")
    def test_context_manager(self, mock_session_cls):
        """Test client works as context manager."""
        mock_session = MagicMock()
        mock_session_cls.return_value = mock_session

        with APIClient() as client:
            pass

        mock_session.close.assert_called_once()


class TestRateLimitedDecorator:
    """Tests for rate_limited decorator."""

    def test_decorator_limits_calls(self):
        """Test decorator rate limits function calls."""
        call_times = []

        @rate_limited(requests_per_second=10.0)
        def test_func():
            call_times.append(time.time())
            return "result"

        # Make several calls
        for _ in range(3):
            test_func()

        # Check timing
        if len(call_times) >= 2:
            time_diff = call_times[-1] - call_times[0]
            assert time_diff >= 0.1


class TestCachedDecorator:
    """Tests for cached decorator."""

    def test_decorator_caches_results(self, temp_dir):
        """Test decorator caches function results."""
        call_count = 0

        @cached(ttl=3600, cache_dir=str(temp_dir))
        def expensive_func(arg):
            nonlocal call_count
            call_count += 1
            return f"result_{arg}"

        # First call
        result1 = expensive_func("test")
        assert result1 == "result_test"
        assert call_count == 1

        # Second call should use cache
        result2 = expensive_func("test")
        assert result2 == "result_test"
        assert call_count == 1

        # Different arg should call function
        result3 = expensive_func("other")
        assert result3 == "result_other"
        assert call_count == 2


class TestXenoCanto:
    """Tests for xeno_canto module."""

    def test_xc_recording_from_api_response(self):
        """Test creating XCRecording from API data."""
        from bioamla.api.xeno_canto import XCRecording

        data = {
            "id": "12345",
            "gen": "Turdus",
            "sp": "migratorius",
            "en": "American Robin",
            "rec": "John Doe",
            "cnt": "United States",
            "loc": "New York",
            "lat": "40.7",
            "lng": "-74.0",
            "type": "song",
            "q": "A",
            "length": "1:30",
            "date": "2023-05-01",
            "url": "https://xeno-canto.org/12345",
            "file": "https://xeno-canto.org/sounds/12345.mp3",
            "lic": "CC-BY-NC",
        }

        recording = XCRecording.from_api_response(data)

        assert recording.id == "12345"
        assert recording.genus == "Turdus"
        assert recording.species == "migratorius"
        assert recording.scientific_name == "Turdus migratorius"
        assert recording.common_name == "American Robin"
        assert recording.quality == "A"
        assert recording.latitude == 40.7
        assert recording.longitude == -74.0

    def test_xc_recording_to_dict(self):
        """Test converting XCRecording to dictionary."""
        from bioamla.api.xeno_canto import XCRecording

        recording = XCRecording(
            id="12345",
            genus="Turdus",
            species="migratorius",
            common_name="American Robin",
        )

        d = recording.to_dict()

        assert d["id"] == "12345"
        assert d["scientific_name"] == "Turdus migratorius"
        assert d["common_name"] == "American Robin"

    @patch("bioamla.api.xeno_canto._client")
    def test_search_requires_parameters(self, mock_client):
        """Test search raises error without parameters."""
        from bioamla.api.xeno_canto import search

        with pytest.raises(ValueError, match="At least one search parameter"):
            search()


class TestMacaulayLibrary:
    """Tests for macaulay module."""

    def test_ml_asset_from_api_response(self):
        """Test creating MLAsset from API data."""
        from bioamla.api.macaulay import MLAsset

        data = {
            "assetId": "123456",
            "catalogId": "ML123456",
            "speciesCode": "amerob",
            "commonName": "American Robin",
            "sciName": "Turdus migratorius",
            "mediaType": "audio",
            "rating": 4,
            "location": "New York",
            "country": "United States",
            "latitude": 40.7,
            "longitude": -74.0,
            "duration": 30.5,
        }

        asset = MLAsset.from_api_response(data)

        assert asset.asset_id == "123456"
        assert asset.catalog_id == "ML123456"
        assert asset.species_code == "amerob"
        assert asset.scientific_name == "Turdus migratorius"
        assert asset.rating == 4
        assert asset.duration == 30.5

    def test_ml_asset_to_dict(self):
        """Test converting MLAsset to dictionary."""
        from bioamla.api.macaulay import MLAsset

        asset = MLAsset(
            asset_id="123456",
            catalog_id="ML123456",
            scientific_name="Turdus migratorius",
            common_name="American Robin",
            media_type="audio",
        )

        d = asset.to_dict()

        assert d["asset_id"] == "123456"
        assert d["scientific_name"] == "Turdus migratorius"
        assert d["media_type"] == "audio"

    @patch("bioamla.api.macaulay._client")
    def test_search_requires_filter(self, mock_client):
        """Test search raises error without filters."""
        from bioamla.api.macaulay import search

        with pytest.raises(ValueError, match="At least one search filter"):
            search()


class TestSpecies:
    """Tests for species module."""

    def test_species_info_to_dict(self):
        """Test SpeciesInfo to_dict method."""
        from bioamla.api.species import SpeciesInfo

        info = SpeciesInfo(
            scientific_name="Turdus migratorius",
            common_name="American Robin",
            species_code="amerob",
            family="Thrushes",
            genus="Turdus",
            species="migratorius",
        )

        d = info.to_dict()

        assert d["scientific_name"] == "Turdus migratorius"
        assert d["common_name"] == "American Robin"
        assert d["species_code"] == "amerob"
        assert d["family"] == "Thrushes"

    def test_normalize_name(self):
        """Test name normalization."""
        from bioamla.api.species import _normalize_name

        assert _normalize_name("Turdus migratorius") == "turdus migratorius"
        assert _normalize_name("American Robin") == "american robin"
        assert _normalize_name("  Test Name  ") == "test name"
        assert _normalize_name("Name-with-dashes") == "namewithdashes"

    @patch("bioamla.api.species._taxonomy_cache", {})
    @patch("bioamla.api.species._taxonomy_loaded", False)
    def test_scientific_to_common_cache_lookup(self):
        """Test scientific_to_common uses cache."""
        from bioamla.api import species

        # Pre-populate cache
        species._taxonomy_cache["turdus migratorius"] = {
            "scientific_name": "Turdus migratorius",
            "common_name": "American Robin",
            "species_code": "amerob",
        }
        species._taxonomy_loaded = True

        result = species.scientific_to_common("Turdus migratorius", fallback_inat=False)
        assert result == "American Robin"

    @patch("bioamla.api.species._taxonomy_cache", {})
    @patch("bioamla.api.species._taxonomy_loaded", False)
    def test_common_to_scientific_cache_lookup(self):
        """Test common_to_scientific uses cache."""
        from bioamla.api import species

        # Pre-populate cache
        species._taxonomy_cache["american robin"] = {
            "scientific_name": "Turdus migratorius",
            "common_name": "American Robin",
            "species_code": "amerob",
        }
        species._taxonomy_loaded = True

        result = species.common_to_scientific("American Robin", fallback_inat=False)
        assert result == "Turdus migratorius"

    def test_batch_convert(self):
        """Test batch_convert function."""
        from bioamla.api.species import batch_convert, _taxonomy_cache, _normalize_name

        # Pre-populate cache
        _taxonomy_cache[_normalize_name("Turdus migratorius")] = {
            "scientific_name": "Turdus migratorius",
            "common_name": "American Robin",
        }
        _taxonomy_cache[_normalize_name("Strix varia")] = {
            "scientific_name": "Strix varia",
            "common_name": "Barred Owl",
        }

        from bioamla.api import species
        species._taxonomy_loaded = True

        results = batch_convert(
            ["Turdus migratorius", "Strix varia"],
            direction="scientific_to_common"
        )

        assert results["Turdus migratorius"] == "American Robin"
        assert results["Strix varia"] == "Barred Owl"


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def temp_dir(tmp_path):
    """Provide a temporary directory."""
    return tmp_path
