"""
API Base Utilities
==================

Provides rate limiting, caching, and common HTTP client functionality
for all API integrations.

Features:
- Thread-safe rate limiting with configurable requests per second
- Disk-based caching with TTL support
- Automatic retry with exponential backoff
- Unified HTTP client with timeout and error handling
- @config_aware decorator for configuration-driven API methods
"""

import functools
import hashlib
import inspect
import json
import pickle
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Optional, TypeVar, Union

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from bioamla.core.files import BinaryFile, TextFile
from bioamla.core.logger import get_logger

logger = get_logger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


# =============================================================================
# Configuration-Aware Decorator
# =============================================================================


def config_aware(
    section: str,
    mapping: Optional[Dict[str, str]] = None,
) -> Callable[[F], F]:
    """
    Decorator to make API methods configuration-aware.

    This decorator allows function parameters to automatically use
    configuration values as defaults. When a parameter is not explicitly
    provided (i.e., remains at its default value of None), the decorator
    looks up the corresponding configuration key and uses that value.

    Args:
        section: Configuration section to read from (e.g., "audio", "inference")
        mapping: Optional mapping of parameter names to config keys.
                 If not provided, parameter names are used as config keys.

    Returns:
        Decorated function that pulls defaults from configuration

    Example:
        @config_aware("audio")
        def process_audio(
            filepath: str,
            sample_rate: Optional[int] = None,
            mono: Optional[bool] = None,
        ) -> AudioData:
            # If sample_rate is None, it will be read from config.audio.sample_rate
            # If mono is None, it will be read from config.audio.mono
            ...

        # With custom mapping
        @config_aware("inference", mapping={"num_results": "top_k"})
        def predict(
            audio_path: str,
            num_results: Optional[int] = None,  # Maps to config.inference.top_k
        ):
            ...

    Notes:
        - Only parameters with None defaults are candidates for config lookup
        - Explicit parameter values always override configuration values
        - If a config key is not found, the original default (None) is preserved
        - The decorator inspects function signature to find configurable params
    """

    def decorator(func: F) -> F:
        # Get function signature for parameter inspection
        sig = inspect.signature(func)
        params = sig.parameters

        # Build list of configurable parameters (those with None defaults)
        configurable_params: Dict[str, str] = {}
        for param_name, param in params.items():
            if param.default is None:
                # Map parameter name to config key
                config_key = (mapping or {}).get(param_name, param_name)
                configurable_params[param_name] = config_key

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Import config here to avoid circular imports
            from bioamla.core.config import get_config

            config = get_config()
            section_config = getattr(config, section, {}) or {}

            # Get bound arguments to see which were explicitly provided
            try:
                bound = sig.bind_partial(*args, **kwargs)
            except TypeError:
                # If binding fails, just call the function normally
                return func(*args, **kwargs)

            # Fill in configurable parameters that weren't explicitly provided
            for param_name, config_key in configurable_params.items():
                # Only fill if not already in kwargs and not in bound.arguments
                if param_name not in bound.arguments:
                    config_value = section_config.get(config_key)
                    if config_value is not None:
                        kwargs[param_name] = config_value
                        logger.debug(
                            f"Using config value for {param_name}: "
                            f"[{section}].{config_key} = {config_value}"
                        )

            return func(*args, **kwargs)

        return wrapper  # type: ignore

    return decorator


def config_aware_class(section: str) -> Callable[[type], type]:
    """
    Class decorator to make all public methods configuration-aware.

    This decorator applies @config_aware to all public methods of a class,
    allowing them to pull default values from the specified config section.

    Args:
        section: Configuration section to use for all methods

    Returns:
        Decorated class

    Example:
        @config_aware_class("audio")
        class AudioProcessor:
            def process(
                self,
                filepath: str,
                sample_rate: Optional[int] = None,
            ):
                # sample_rate will come from config.audio.sample_rate if not provided
                ...

            def analyze(
                self,
                filepath: str,
                n_fft: Optional[int] = None,
            ):
                # n_fft will come from config.audio.n_fft if not provided
                ...
    """

    def decorator(cls: type) -> type:
        for name in dir(cls):
            if name.startswith("_"):
                continue

            method = getattr(cls, name)
            if callable(method) and not isinstance(method, type):
                # Apply config_aware to the method
                wrapped = config_aware(section)(method)
                setattr(cls, name, wrapped)

        return cls

    return decorator


class ConfigAwareMixin:
    """
    Mixin class that provides configuration awareness to subclasses.

    Subclasses can define a `_config_section` attribute to specify
    which configuration section to use for default values.

    Example:
        class AudioProcessor(ConfigAwareMixin):
            _config_section = "audio"

            def process(self, sample_rate: Optional[int] = None):
                # Get sample_rate from config if not provided
                sample_rate = self._get_config_default("sample_rate", sample_rate)
                ...
    """

    _config_section: str = ""

    def _get_config_default(
        self,
        key: str,
        value: Any,
        section: Optional[str] = None,
    ) -> Any:
        """
        Get a configuration default if value is None.

        Args:
            key: Configuration key to look up
            value: Current value (if None, will be replaced by config value)
            section: Optional section override (defaults to _config_section)

        Returns:
            The value if not None, otherwise the configuration value
        """
        if value is not None:
            return value

        from bioamla.core.config import get_config

        config = get_config()
        section_name = section or self._config_section
        section_config = getattr(config, section_name, {}) or {}
        config_value = section_config.get(key)

        if config_value is not None:
            logger.debug(f"Using config default for {key}: [{section_name}].{key} = {config_value}")

        return config_value

    def _get_config_defaults(
        self,
        section: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Get multiple configuration defaults at once.

        Args:
            section: Optional section override
            **kwargs: Parameter name -> current value mappings

        Returns:
            Dictionary with config values filled in for None values

        Example:
            params = self._get_config_defaults(
                sample_rate=sample_rate,  # None -> uses config
                normalize=normalize,       # None -> uses config
                mono=True,                # Explicit -> preserved
            )
        """
        result = {}
        for key, value in kwargs.items():
            result[key] = self._get_config_default(key, value, section)
        return result


@dataclass
class RateLimiter:
    """
    Thread-safe rate limiter using token bucket algorithm.

    Limits the number of requests that can be made within a time window
    to comply with API rate limits.

    Thread Safety:
        This class is fully thread-safe. All public methods use internal
        locking via ``threading.Lock`` to ensure safe concurrent access
        from multiple threads. Safe to share a single instance across
        worker threads.

    Args:
        requests_per_second: Maximum requests allowed per second.
        burst_size: Maximum burst of requests allowed (default: 1).

    Example:
        >>> limiter = RateLimiter(requests_per_second=1.0)
        >>> limiter.acquire()  # Blocks if rate limit exceeded
        >>> # Make API request here
    """

    requests_per_second: float = 1.0
    burst_size: int = 1
    _tokens: float = field(default=0.0, init=False, repr=False)
    _last_update: float = field(default=0.0, init=False, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    def __post_init__(self):
        self._tokens = float(self.burst_size)
        self._last_update = time.monotonic()

    def acquire(self, tokens: int = 1) -> float:
        """
        Acquire tokens, blocking if necessary.

        Args:
            tokens: Number of tokens to acquire.

        Returns:
            Time waited in seconds.
        """
        with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_update
            self._tokens = min(self.burst_size, self._tokens + elapsed * self.requests_per_second)
            self._last_update = now

            if self._tokens >= tokens:
                self._tokens -= tokens
                return 0.0

            wait_time = (tokens - self._tokens) / self.requests_per_second
            time.sleep(wait_time)
            self._tokens = 0
            self._last_update = time.monotonic()
            return wait_time

    def try_acquire(self, tokens: int = 1) -> bool:
        """
        Try to acquire tokens without blocking.

        Args:
            tokens: Number of tokens to acquire.

        Returns:
            True if tokens were acquired, False otherwise.
        """
        with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_update
            self._tokens = min(self.burst_size, self._tokens + elapsed * self.requests_per_second)
            self._last_update = now

            if self._tokens >= tokens:
                self._tokens -= tokens
                return True
            return False


class APICache:
    """
    Disk-based cache with TTL support for API responses.

    Stores cached responses as JSON or pickle files in a cache directory.
    Automatically cleans up expired entries on access.

    Thread Safety:
        This class is thread-safe for concurrent read/write operations.
        Uses ``threading.Lock`` internally to protect cache modifications.
        Multiple threads can safely call ``get()``, ``set()``, and ``delete()``
        concurrently.

    Args:
        cache_dir: Directory to store cache files (default: ~/.cache/bioamla).
        default_ttl: Default time-to-live in seconds (default: 3600 = 1 hour).
        max_size_mb: Maximum cache size in megabytes (default: 100).

    Example:
        >>> cache = APICache(default_ttl=3600)
        >>> cache.set("key", {"data": "value"})
        >>> result = cache.get("key")
    """

    def __init__(
        self,
        cache_dir: Optional[Union[str, Path]] = None,
        default_ttl: int = 3600,
        max_size_mb: int = 100,
    ):
        if cache_dir is None:
            cache_dir = Path.home() / ".cache" / "bioamla" / "api"
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.default_ttl = default_ttl
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self._lock = threading.Lock()

    def _get_cache_path(self, key: str) -> Path:
        """Get the cache file path for a key."""
        key_hash = hashlib.md5(key.encode()).hexdigest()
        return self.cache_dir / f"{key_hash}.cache"

    def _get_meta_path(self, key: str) -> Path:
        """Get the metadata file path for a key."""
        key_hash = hashlib.md5(key.encode()).hexdigest()
        return self.cache_dir / f"{key_hash}.meta"

    def get(self, key: str) -> Optional[Any]:
        """
        Get a cached value if it exists and hasn't expired.

        Args:
            key: Cache key.

        Returns:
            Cached value or None if not found/expired.
        """
        cache_path = self._get_cache_path(key)
        meta_path = self._get_meta_path(key)

        if not cache_path.exists() or not meta_path.exists():
            return None

        try:
            with TextFile(meta_path, mode="r") as f:
                meta = json.load(f.handle)

            # Check expiration
            if time.time() > meta.get("expires_at", 0):
                self.delete(key)
                return None

            # Load cached data
            with BinaryFile(cache_path, mode="rb") as f:
                return pickle.load(f.handle)

        except (json.JSONDecodeError, pickle.PickleError, OSError) as e:
            logger.warning(f"Cache read error for {key}: {e}")
            self.delete(key)
            return None

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """
        Store a value in the cache.

        Args:
            key: Cache key.
            value: Value to cache.
            ttl: Time-to-live in seconds (uses default if not specified).
        """
        if ttl is None:
            ttl = self.default_ttl

        cache_path = self._get_cache_path(key)
        meta_path = self._get_meta_path(key)

        with self._lock:
            try:
                # Check cache size and clean if necessary
                self._maybe_clean()

                # Write cache data
                with BinaryFile(cache_path, mode="wb") as f:
                    pickle.dump(value, f.handle)

                # Write metadata
                meta = {
                    "key": key,
                    "created_at": time.time(),
                    "expires_at": time.time() + ttl,
                }
                with TextFile(meta_path, mode="w") as f:
                    json.dump(meta, f.handle)

            except (pickle.PickleError, OSError) as e:
                logger.warning(f"Cache write error for {key}: {e}")

    def delete(self, key: str) -> bool:
        """
        Delete a cached value.

        Args:
            key: Cache key.

        Returns:
            True if deleted, False if not found.
        """
        cache_path = self._get_cache_path(key)
        meta_path = self._get_meta_path(key)

        deleted = False
        with self._lock:
            if cache_path.exists():
                cache_path.unlink()
                deleted = True
            if meta_path.exists():
                meta_path.unlink()
                deleted = True
        return deleted

    def clear(self) -> int:
        """
        Clear all cached entries.

        Returns:
            Number of entries cleared.
        """
        count = 0
        with self._lock:
            for path in self.cache_dir.glob("*.cache"):
                path.unlink()
                count += 1
            for path in self.cache_dir.glob("*.meta"):
                path.unlink()
        return count

    def _maybe_clean(self) -> None:
        """Clean expired entries and enforce size limit."""
        # Clean expired entries
        now = time.time()
        for meta_path in self.cache_dir.glob("*.meta"):
            try:
                with TextFile(meta_path, mode="r") as f:
                    meta = json.load(f.handle)
                if now > meta.get("expires_at", 0):
                    cache_path = meta_path.with_suffix(".cache")
                    if cache_path.exists():
                        cache_path.unlink()
                    meta_path.unlink()
            except (json.JSONDecodeError, OSError):
                pass

        # Check total size
        total_size = sum(f.stat().st_size for f in self.cache_dir.glob("*.cache") if f.exists())

        if total_size > self.max_size_bytes:
            # Remove oldest entries until under limit
            entries = []
            for meta_path in self.cache_dir.glob("*.meta"):
                try:
                    with TextFile(meta_path, mode="r") as f:
                        meta = json.load(f.handle)
                    cache_path = meta_path.with_suffix(".cache")
                    if cache_path.exists():
                        entries.append((meta.get("created_at", 0), cache_path, meta_path))
                except (json.JSONDecodeError, OSError):
                    pass

            # Sort by creation time (oldest first)
            entries.sort(key=lambda x: x[0])

            for _, cache_path, meta_path in entries:
                if total_size <= self.max_size_bytes * 0.8:  # Clean to 80%
                    break
                try:
                    size = cache_path.stat().st_size
                    cache_path.unlink()
                    meta_path.unlink()
                    total_size -= size
                except OSError:
                    pass


class APIClient:
    """
    HTTP client with retry, rate limiting, and caching support.

    Provides a unified interface for making API requests with automatic
    error handling, exponential backoff retry, and optional caching.

    Args:
        base_url: Base URL for API requests.
        rate_limiter: RateLimiter instance for rate limiting.
        cache: APICache instance for caching responses.
        timeout: Request timeout in seconds (default: 30).
        max_retries: Maximum number of retries (default: 3).
        user_agent: User-Agent header value.

    Example:
        >>> client = APIClient(
        ...     base_url="https://api.example.com",
        ...     rate_limiter=RateLimiter(requests_per_second=1.0)
        ... )
        >>> response = client.get("/endpoint", params={"q": "search"})
    """

    def __init__(
        self,
        base_url: str = "",
        rate_limiter: Optional[RateLimiter] = None,
        cache: Optional[APICache] = None,
        timeout: int = 30,
        max_retries: int = 3,
        user_agent: str = "bioamla/1.0",
    ):
        self.base_url = base_url.rstrip("/")
        self.rate_limiter = rate_limiter
        self.cache = cache
        self.timeout = timeout
        self.user_agent = user_agent

        # Configure session with retries
        self.session = requests.Session()
        retry_strategy = Retry(
            total=max_retries,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET", "POST"],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        self.session.headers.update({"User-Agent": user_agent})

    def _make_cache_key(self, method: str, url: str, params: Optional[Dict] = None) -> str:
        """Generate a cache key for a request."""
        key_parts = [method, url]
        if params:
            key_parts.append(json.dumps(params, sort_keys=True))
        return ":".join(key_parts)

    def get(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        cache_ttl: Optional[int] = None,
        use_cache: bool = True,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Make a GET request.

        Args:
            endpoint: API endpoint (appended to base_url).
            params: Query parameters.
            cache_ttl: Cache TTL in seconds (None uses default).
            use_cache: Whether to use caching.
            **kwargs: Additional arguments passed to requests.

        Returns:
            Response JSON as dictionary.

        Raises:
            requests.RequestException: On request failure.
        """
        url = f"{self.base_url}{endpoint}" if self.base_url else endpoint

        # Check cache first
        if use_cache and self.cache:
            cache_key = self._make_cache_key("GET", url, params)
            cached_response = self.cache.get(cache_key)
            if cached_response is not None:
                logger.debug(f"Cache hit for {url}")
                return cached_response

        # Rate limiting
        if self.rate_limiter:
            wait_time = self.rate_limiter.acquire()
            if wait_time > 0:
                logger.debug(f"Rate limited, waited {wait_time:.2f}s")

        # Make request
        response = self.session.get(
            url,
            params=params,
            timeout=self.timeout,
            **kwargs,
        )
        response.raise_for_status()
        data = response.json()

        # Cache response
        if use_cache and self.cache:
            self.cache.set(cache_key, data, cache_ttl)

        return data

    def post(
        self,
        endpoint: str,
        data: Optional[Dict[str, Any]] = None,
        json_data: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Make a POST request.

        Args:
            endpoint: API endpoint.
            data: Form data.
            json_data: JSON data.
            **kwargs: Additional arguments passed to requests.

        Returns:
            Response JSON as dictionary.
        """
        url = f"{self.base_url}{endpoint}" if self.base_url else endpoint

        if self.rate_limiter:
            self.rate_limiter.acquire()

        response = self.session.post(
            url,
            data=data,
            json=json_data,
            timeout=self.timeout,
            **kwargs,
        )
        response.raise_for_status()
        return response.json()

    def download(
        self,
        url: str,
        filepath: Union[str, Path],
        chunk_size: int = 8192,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> Path:
        """
        Download a file from URL.

        Args:
            url: URL to download from.
            filepath: Local path to save file.
            chunk_size: Download chunk size in bytes.
            progress_callback: Callback(downloaded_bytes, total_bytes).

        Returns:
            Path to downloaded file.
        """
        filepath = Path(filepath)

        if self.rate_limiter:
            self.rate_limiter.acquire()

        response = self.session.get(url, stream=True, timeout=self.timeout)
        response.raise_for_status()

        total_size = int(response.headers.get("content-length", 0))
        downloaded = 0

        filepath.parent.mkdir(parents=True, exist_ok=True)

        with BinaryFile(filepath, mode="wb") as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                f.write(chunk)
                downloaded += len(chunk)
                if progress_callback:
                    progress_callback(downloaded, total_size)

        return filepath

    def close(self) -> None:
        """Close the HTTP session."""
        self.session.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False


def rate_limited(requests_per_second: float = 1.0) -> Callable[[F], F]:
    """
    Decorator to rate limit a function.

    Args:
        requests_per_second: Maximum calls per second.

    Returns:
        Decorated function.

    Example:
        >>> @rate_limited(0.5)  # 1 call per 2 seconds
        ... def fetch_data():
        ...     return requests.get("https://api.example.com/data")
    """
    limiter = RateLimiter(requests_per_second=requests_per_second)

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            limiter.acquire()
            return func(*args, **kwargs)

        return wrapper  # type: ignore

    return decorator


def cached(ttl: int = 3600, cache_dir: Optional[str] = None) -> Callable[[F], F]:
    """
    Decorator to cache function results.

    Args:
        ttl: Cache time-to-live in seconds.
        cache_dir: Cache directory (uses default if None).

    Returns:
        Decorated function.

    Example:
        >>> @cached(ttl=3600)
        ... def expensive_api_call(query: str):
        ...     return requests.get(f"https://api.example.com?q={query}").json()
    """
    cache = APICache(cache_dir=cache_dir, default_ttl=ttl)

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Create cache key from function name and arguments
            key_parts = [func.__name__]
            key_parts.extend(str(arg) for arg in args)
            key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
            key = ":".join(key_parts)

            # Check cache
            result = cache.get(key)
            if result is not None:
                return result

            # Call function and cache result
            result = func(*args, **kwargs)
            cache.set(key, result)
            return result

        return wrapper  # type: ignore

    return decorator
